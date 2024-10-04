/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/string_parsing.hpp"
#include "nested_json.hpp"

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <deque>

namespace cudf::io::json::detail {

/**
 * @brief Get the column indices for the values column for array of arrays rows
 *
 * @param row_array_children_level The level of the row array's children
 * @param d_tree The tree metadata
 * @param col_ids The column ids
 * @param num_columns The number of columns
 * @param stream The stream to use
 * @return The value columns' indices
 */
rmm::device_uvector<NodeIndexT> get_values_column_indices(TreeDepthT const row_array_children_level,
                                                          tree_meta_t const& d_tree,
                                                          device_span<NodeIndexT const> col_ids,
                                                          size_type const num_columns,
                                                          rmm::cuda_stream_view stream)
{
  auto [level2_nodes, level2_indices] = get_array_children_indices(
    row_array_children_level, d_tree.node_levels, d_tree.parent_node_ids, stream);
  auto col_id_location = thrust::make_permutation_iterator(col_ids.begin(), level2_nodes.begin());
  rmm::device_uvector<NodeIndexT> values_column_indices(num_columns, stream);
  thrust::scatter(rmm::exec_policy_nosync(stream),
                  level2_indices.begin(),
                  level2_indices.end(),
                  col_id_location,
                  values_column_indices.begin());
  return values_column_indices;
}

/**
 * @brief Copies strings specified by pair of begin, end offsets to host vector of strings.
 *
 * @param input String device buffer
 * @param node_range_begin Begin offset of the strings
 * @param node_range_end End offset of the strings
 * @param stream CUDA stream
 * @return Vector of strings
 */
std::vector<std::string> copy_strings_to_host_sync(
  device_span<SymbolT const> input,
  device_span<SymbolOffsetT const> node_range_begin,
  device_span<SymbolOffsetT const> node_range_end,
  rmm::cuda_stream_view stream)
{
  auto const num_strings = node_range_begin.size();
  rmm::device_uvector<size_type> string_offsets(num_strings, stream);
  rmm::device_uvector<size_type> string_lengths(num_strings, stream);
  auto d_offset_pairs = thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  thrust::transform(rmm::exec_policy_nosync(stream),
                    d_offset_pairs,
                    d_offset_pairs + num_strings,
                    thrust::make_zip_iterator(string_offsets.begin(), string_lengths.begin()),
                    [] __device__(auto const& offsets) {
                      // Note: first character for non-field columns
                      return thrust::make_tuple(
                        static_cast<size_type>(thrust::get<0>(offsets)),
                        static_cast<size_type>(thrust::get<1>(offsets) - thrust::get<0>(offsets)));
                    });

  cudf::io::parse_options_view options_view{};
  options_view.quotechar  = '\0';  // no quotes
  options_view.keepquotes = true;
  auto d_offset_length_it =
    thrust::make_zip_iterator(string_offsets.begin(), string_lengths.begin());
  auto d_column_names = parse_data(input.data(),
                                   d_offset_length_it,
                                   num_strings,
                                   data_type{type_id::STRING},
                                   rmm::device_buffer{},
                                   0,
                                   options_view,
                                   stream,
                                   cudf::get_current_device_resource_ref());
  auto to_host        = [stream](auto const& col) {
    if (col.is_empty()) return std::vector<std::string>{};
    auto const scv     = cudf::strings_column_view(col);
    auto const h_chars = cudf::detail::make_host_vector_async<char>(
      cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
    auto const h_offsets = cudf::detail::make_host_vector_async(
      cudf::device_span<cudf::size_type const>(scv.offsets().data<cudf::size_type>() + scv.offset(),
                                               scv.size() + 1),
      stream);
    stream.synchronize();

    // build std::string vector from chars and offsets
    std::vector<std::string> host_data;
    host_data.reserve(col.size());
    std::transform(
      std::begin(h_offsets),
      std::end(h_offsets) - 1,
      std::begin(h_offsets) + 1,
      std::back_inserter(host_data),
      [&](auto start, auto end) { return std::string(h_chars.data() + start, end - start); });
    return host_data;
  };
  return to_host(d_column_names->view());
}

/**
 * @brief Checks if all strings in each string column in the tree are nulls.
 * For non-string columns, it's set as true. If any of rows in a string column is false, it's set as
 * false.
 *
 * @param input Input JSON string device data
 * @param d_column_tree column tree representation of JSON string
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param options Parsing options specifying the parsing behaviour
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Array of bytes where each byte indicate if it is all nulls string column.
 */
rmm::device_uvector<uint8_t> is_all_nulls_each_column(device_span<SymbolT const> input,
                                                      tree_meta_t const& d_column_tree,
                                                      tree_meta_t const& tree,
                                                      device_span<NodeIndexT const> col_ids,
                                                      cudf::io::json_reader_options const& options,
                                                      rmm::cuda_stream_view stream)
{
  auto const num_nodes = col_ids.size();
  auto const num_cols  = d_column_tree.node_categories.size();
  rmm::device_uvector<uint8_t> is_all_nulls(num_cols, stream);
  thrust::fill(rmm::exec_policy_nosync(stream), is_all_nulls.begin(), is_all_nulls.end(), true);

  auto parse_opt = parsing_options(options, stream);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<size_type>(0),
    num_nodes,
    [options           = parse_opt.view(),
     data              = input.data(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     range_begin       = tree.node_range_begin.begin(),
     range_end         = tree.node_range_end.begin(),
     is_all_nulls      = is_all_nulls.begin()] __device__(size_type i) {
      auto const node_category = column_categories[col_ids[i]];
      if (node_category == NC_STR or node_category == NC_VAL) {
        auto const is_null_literal = serialized_trie_contains(
          options.trie_na,
          {data + range_begin[i], static_cast<size_t>(range_end[i] - range_begin[i])});
        if (!is_null_literal) is_all_nulls[col_ids[i]] = false;
      }
    });
  return is_all_nulls;
}

NodeIndexT get_row_array_parent_col_id(device_span<NodeIndexT const> col_ids,
                                       bool is_enabled_lines,
                                       rmm::cuda_stream_view stream)
{
  NodeIndexT value = parent_node_sentinel;
  if (!col_ids.empty()) {
    auto const list_node_index = is_enabled_lines ? 0 : 1;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&value,
                                  col_ids.data() + list_node_index,
                                  sizeof(NodeIndexT),
                                  cudaMemcpyDefault,
                                  stream.value()));
    stream.synchronize();
  }
  return value;
}
/**
 * @brief Holds member data pointers of `d_json_column`
 *
 */
struct json_column_data {
  using row_offset_t = json_column::row_offset_t;
  row_offset_t* string_offsets;
  row_offset_t* string_lengths;
  row_offset_t* child_offsets;
  bitmask_type* validity;
};

using hashmap_of_device_columns =
  std::unordered_map<NodeIndexT, std::reference_wrapper<device_json_column>>;

std::pair<cudf::detail::host_vector<bool>, hashmap_of_device_columns> build_tree(
  device_json_column& root,
  host_span<uint8_t const> is_str_column_all_nulls,
  tree_meta_t& d_column_tree,
  device_span<NodeIndexT const> d_unique_col_ids,
  device_span<size_type const> d_max_row_offsets,
  std::vector<std::string> const& column_names,
  NodeIndexT row_array_parent_col_id,
  bool is_array_of_arrays,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
void scatter_offsets(tree_meta_t const& tree,
                     device_span<NodeIndexT const> col_ids,
                     device_span<size_type const> row_offsets,
                     device_span<size_type> node_ids,
                     device_span<size_type> sorted_col_ids,  // Reuse this for parent_col_ids
                     tree_meta_t const& d_column_tree,
                     host_span<const bool> ignore_vals,
                     hashmap_of_device_columns const& columns,
                     rmm::cuda_stream_view stream);

/**
 * @brief Constructs `d_json_column` from node tree representation
 * Newly constructed columns are inserted into `root`'s children.
 * `root` must be a list type.
 *
 * @param input Input JSON string device data
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param row_offsets Row offsets of the nodes in the tree
 * @param root Root node of the `d_json_column` tree
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param options Parsing options specifying the parsing behaviour
 * options affecting behaviour are
 *   is_enabled_lines: Whether the input is a line-delimited JSON
 *   is_enabled_mixed_types_as_string: Whether to enable reading mixed types as string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the device memory
 * of child_offets and validity members of `d_json_column`
 */
void make_device_json_column(device_span<SymbolT const> input,
                             tree_meta_t const& tree,
                             device_span<NodeIndexT const> col_ids,
                             device_span<size_type const> row_offsets,
                             device_json_column& root,
                             bool is_array_of_arrays,
                             cudf::io::json_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const is_enabled_lines                 = options.is_enabled_lines();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  // make a copy
  auto sorted_col_ids = cudf::detail::make_device_uvector_async(
    col_ids, stream, cudf::get_current_device_resource_ref());

  // sort by {col_id} on {node_ids} stable
  rmm::device_uvector<NodeIndexT> node_ids(col_ids.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                             sorted_col_ids.begin(),
                             sorted_col_ids.end(),
                             node_ids.begin());

  NodeIndexT const row_array_parent_col_id =
    get_row_array_parent_col_id(col_ids, is_enabled_lines, stream);

  // 1. gather column information.
  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    reduce_to_column_tree(tree,
                          col_ids,
                          sorted_col_ids,
                          node_ids,
                          row_offsets,
                          is_array_of_arrays,
                          row_array_parent_col_id,
                          stream);
  auto num_columns                      = d_unique_col_ids.size();
  std::vector<std::string> column_names = copy_strings_to_host_sync(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);
  // array of arrays column names
  if (is_array_of_arrays) {
    auto const unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
    auto const column_parent_ids =
      cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
    TreeDepthT const row_array_children_level = is_enabled_lines ? 1 : 2;
    auto values_column_indices =
      get_values_column_indices(row_array_children_level, tree, col_ids, num_columns, stream);
    auto h_values_column_indices =
      cudf::detail::make_host_vector_sync(values_column_indices, stream);
    std::transform(unique_col_ids.begin(),
                   unique_col_ids.end(),
                   column_names.cbegin(),
                   column_names.begin(),
                   [&h_values_column_indices, &column_parent_ids, row_array_parent_col_id](
                     auto col_id, auto name) mutable {
                     return column_parent_ids[col_id] == row_array_parent_col_id
                              ? std::to_string(h_values_column_indices[col_id])
                              : name;
                   });
  }

  auto const is_str_column_all_nulls = [&, &column_tree = d_column_tree]() {
    if (is_enabled_mixed_types_as_string) {
      return cudf::detail::make_std_vector_sync(
        is_all_nulls_each_column(input, column_tree, tree, col_ids, options, stream), stream);
    }
    return std::vector<uint8_t>();
  }();
  auto const [ignore_vals, columns] = build_tree(root,
                                                 is_str_column_all_nulls,
                                                 d_column_tree,
                                                 d_unique_col_ids,
                                                 d_max_row_offsets,
                                                 column_names,
                                                 row_array_parent_col_id,
                                                 is_array_of_arrays,
                                                 options,
                                                 stream,
                                                 mr);

  scatter_offsets(tree,
                  col_ids,
                  row_offsets,
                  node_ids,
                  sorted_col_ids,
                  d_column_tree,
                  ignore_vals,
                  columns,
                  stream);
}

std::pair<cudf::detail::host_vector<bool>, hashmap_of_device_columns> build_tree(
  device_json_column& root,
  host_span<uint8_t const> is_str_column_all_nulls,
  tree_meta_t& d_column_tree,
  device_span<NodeIndexT const> d_unique_col_ids,
  device_span<size_type const> d_max_row_offsets,
  std::vector<std::string> const& column_names,
  NodeIndexT row_array_parent_col_id,
  bool is_array_of_arrays,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  auto unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
  auto column_categories =
    cudf::detail::make_host_vector_async(d_column_tree.node_categories, stream);
  auto const column_parent_ids =
    cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
  auto column_range_beg =
    cudf::detail::make_host_vector_async(d_column_tree.node_range_begin, stream);
  auto const max_row_offsets = cudf::detail::make_host_vector_async(d_max_row_offsets, stream);
  auto num_columns           = d_unique_col_ids.size();
  stream.synchronize();

  auto to_json_col_type = [](auto category) {
    switch (category) {
      case NC_STRUCT: return json_col_t::StructColumn;
      case NC_LIST: return json_col_t::ListColumn;
      case NC_STR: [[fallthrough]];
      case NC_VAL: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    }
  };
  auto init_to_zero = [stream](auto& v) {
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), v.begin(), v.end(), 0);
  };

  auto initialize_json_columns = [&](auto i, auto& col, auto column_category) {
    if (column_category == NC_ERR || column_category == NC_FN) {
      return;
    } else if (column_category == NC_VAL || column_category == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      init_to_zero(col.string_offsets);
      init_to_zero(col.string_lengths);
    } else if (column_category == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      init_to_zero(col.child_offsets);
    }
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = to_json_col_type(column_category);
  };

  auto reinitialize_as_string = [&](auto i, auto& col) {
    col.string_offsets.resize(max_row_offsets[i] + 1, stream);
    col.string_lengths.resize(max_row_offsets[i] + 1, stream);
    init_to_zero(col.string_offsets);
    init_to_zero(col.string_lengths);
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = json_col_t::StringColumn;
    // destroy references of all child columns after this step, by calling remove_child_columns
  };

  path_from_tree tree_path{column_categories,
                           column_parent_ids,
                           column_names,
                           is_array_of_arrays,
                           row_array_parent_col_id};

  // 2. generate nested columns tree and its device_memory
  // reorder unique_col_ids w.r.t. column_range_begin for order of column to be in field order.
  auto h_range_col_id_it =
    thrust::make_zip_iterator(column_range_beg.begin(), unique_col_ids.begin());
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<0>(a) < thrust::get<0>(b);
  });

  // use hash map because we may skip field name's col_ids
  hashmap_of_device_columns columns;
  // map{parent_col_id, child_col_name}> = child_col_id, used for null value column tracking
  std::map<std::pair<NodeIndexT, std::string>, NodeIndexT> mapped_columns;
  // find column_ids which are values, but should be ignored in validity
  auto ignore_vals = cudf::detail::make_host_vector<bool>(num_columns, stream);
  std::fill(ignore_vals.begin(), ignore_vals.end(), false);
  std::vector<uint8_t> is_mixed_type_column(num_columns, 0);
  std::vector<uint8_t> is_pruned(num_columns, 0);
  // for columns that are not mixed type but have been forced as string
  std::vector<bool> forced_as_string_column(num_columns);
  columns.try_emplace(parent_node_sentinel, std::ref(root));

  std::function<void(NodeIndexT, device_json_column&)> remove_child_columns =
    [&](NodeIndexT this_col_id, device_json_column& col) {
      for (auto const& col_name : col.column_order) {
        auto child_id                  = mapped_columns[{this_col_id, col_name}];
        is_mixed_type_column[child_id] = 1;
        remove_child_columns(child_id, col.child_columns.at(col_name));
        mapped_columns.erase({this_col_id, col_name});
        columns.erase(child_id);
      }
      col.child_columns.clear();  // their references are deleted above.
      col.column_order.clear();
    };

  auto name_and_parent_index = [&is_array_of_arrays,
                                &row_array_parent_col_id,
                                &column_parent_ids,
                                &column_categories,
                                &column_names](auto this_col_id) {
    std::string name   = "";
    auto parent_col_id = column_parent_ids[this_col_id];
    if (parent_col_id == parent_node_sentinel || column_categories[parent_col_id] == NC_LIST) {
      if (is_array_of_arrays && parent_col_id == row_array_parent_col_id) {
        name = column_names[this_col_id];
      } else {
        name = list_child_name;
      }
    } else if (column_categories[parent_col_id] == NC_FN) {
      auto field_name_col_id = parent_col_id;
      parent_col_id          = column_parent_ids[parent_col_id];
      name                   = column_names[field_name_col_id];
    } else {
      CUDF_FAIL("Unexpected parent column category");
    }
    return std::pair{name, parent_col_id};
  };

  // Prune columns that are not required to be parsed.
  if (options.is_enabled_prune_columns()) {
    for (auto const this_col_id : unique_col_ids) {
      if (column_categories[this_col_id] == NC_ERR || column_categories[this_col_id] == NC_FN) {
        continue;
      }
      // Struct, List, String, Value
      auto [name, parent_col_id] = name_and_parent_index(this_col_id);
      // get path of this column, and get its dtype if present in options
      auto const nt                             = tree_path.get_path(this_col_id);
      std::optional<data_type> const user_dtype = get_path_data_type(nt, options);
      if (!user_dtype.has_value() and parent_col_id != parent_node_sentinel) {
        is_pruned[this_col_id] = 1;
        continue;
      } else {
        // make sure all its parents are not pruned.
        while (parent_col_id != parent_node_sentinel and is_pruned[parent_col_id] == 1) {
          is_pruned[parent_col_id] = 0;
          parent_col_id            = column_parent_ids[parent_col_id];
        }
      }
    }
  }

  // Build the column tree, also, handles mixed types.
  for (auto const this_col_id : unique_col_ids) {
    if (column_categories[this_col_id] == NC_ERR || column_categories[this_col_id] == NC_FN) {
      continue;
    }
    // Struct, List, String, Value
    auto [name, parent_col_id] = name_and_parent_index(this_col_id);

    // if parent is mixed type column or this column is pruned or if parent
    // has been forced as string, ignore this column.
    if (parent_col_id != parent_node_sentinel &&
          (is_mixed_type_column[parent_col_id] || is_pruned[this_col_id]) ||
        forced_as_string_column[parent_col_id]) {
      ignore_vals[this_col_id] = true;
      if (is_mixed_type_column[parent_col_id]) { is_mixed_type_column[this_col_id] = 1; }
      if (forced_as_string_column[parent_col_id]) { forced_as_string_column[this_col_id] = true; }
      continue;
    }

    // If the child is already found,
    // replace if this column is a nested column and the existing was a value column
    // ignore this column if this column is a value column and the existing was a nested column
    auto it = columns.find(parent_col_id);
    CUDF_EXPECTS(it != columns.end(), "Parent column not found");
    auto& parent_col = it->second.get();
    bool replaced    = false;
    if (mapped_columns.count({parent_col_id, name}) > 0) {
      auto const old_col_id = mapped_columns[{parent_col_id, name}];
      // If mixed type as string is enabled, make both of them strings and merge them.
      // All child columns will be ignored when parsing.
      if (is_enabled_mixed_types_as_string) {
        bool const is_mixed_type = [&]() {
          // If new or old is STR and they are all not null, make it mixed type, else ignore.
          if (column_categories[this_col_id] == NC_VAL ||
              column_categories[this_col_id] == NC_STR) {
            if (is_str_column_all_nulls[this_col_id]) return false;
          }
          if (column_categories[old_col_id] == NC_VAL || column_categories[old_col_id] == NC_STR) {
            if (is_str_column_all_nulls[old_col_id]) return false;
          }
          return true;
        }();
        if (is_mixed_type) {
          is_mixed_type_column[this_col_id] = 1;
          is_mixed_type_column[old_col_id]  = 1;
          // if old col type (not cat) is list or struct, replace with string.
          auto& col = columns.at(old_col_id).get();
          if (col.type == json_col_t::ListColumn or col.type == json_col_t::StructColumn) {
            reinitialize_as_string(old_col_id, col);
            remove_child_columns(old_col_id, col);
            // all its children (which are already inserted) are ignored later.
          }
          col.forced_as_string_column = true;
          columns.try_emplace(this_col_id, columns.at(old_col_id));
          continue;
        }
      }

      if (column_categories[this_col_id] == NC_VAL || column_categories[this_col_id] == NC_STR) {
        ignore_vals[this_col_id] = true;
        continue;
      }
      if (column_categories[old_col_id] == NC_VAL || column_categories[old_col_id] == NC_STR) {
        // remap
        ignore_vals[old_col_id] = true;
        mapped_columns.erase({parent_col_id, name});
        columns.erase(old_col_id);
        parent_col.child_columns.erase(name);
        replaced = true;  // to skip duplicate name in column_order
      } else {
        // If this is a nested column but we're trying to insert either (a) a list node into a
        // struct column or (b) a struct node into a list column, we fail
        CUDF_EXPECTS(not((column_categories[old_col_id] == NC_LIST and
                          column_categories[this_col_id] == NC_STRUCT) or
                         (column_categories[old_col_id] == NC_STRUCT and
                          column_categories[this_col_id] == NC_LIST)),
                     "A mix of lists and structs within the same column is not supported");
      }
    }

    auto this_column_category = column_categories[this_col_id];
    // get path of this column, check if it is a struct/list forced as string, and enforce it
    auto const nt                             = tree_path.get_path(this_col_id);
    std::optional<data_type> const user_dtype = get_path_data_type(nt, options);
    if ((column_categories[this_col_id] == NC_STRUCT or
         column_categories[this_col_id] == NC_LIST) and
        user_dtype.has_value() and user_dtype.value().id() == type_id::STRING) {
      this_column_category = NC_STR;
    }

    CUDF_EXPECTS(parent_col.child_columns.count(name) == 0, "duplicate column name: " + name);
    // move into parent
    device_json_column col(stream, mr);
    initialize_json_columns(this_col_id, col, this_column_category);
    if ((column_categories[this_col_id] == NC_STRUCT or
         column_categories[this_col_id] == NC_LIST) and
        user_dtype.has_value() and user_dtype.value().id() == type_id::STRING) {
      col.forced_as_string_column          = true;
      forced_as_string_column[this_col_id] = true;
    }

    auto inserted = parent_col.child_columns.try_emplace(name, std::move(col)).second;
    CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
    if (not replaced) parent_col.column_order.push_back(name);
    columns.try_emplace(this_col_id, std::ref(parent_col.child_columns.at(name)));
    mapped_columns.try_emplace(std::make_pair(parent_col_id, name), this_col_id);
  }

  if (is_enabled_mixed_types_as_string) {
    // ignore all children of mixed type columns
    for (auto const this_col_id : unique_col_ids) {
      auto parent_col_id = column_parent_ids[this_col_id];
      if (parent_col_id != parent_node_sentinel and is_mixed_type_column[parent_col_id] == 1) {
        is_mixed_type_column[this_col_id] = 1;
        ignore_vals[this_col_id]          = true;
        columns.erase(this_col_id);
      }
      // Convert only mixed type columns as string (so to copy), but not its children
      if (parent_col_id != parent_node_sentinel and is_mixed_type_column[parent_col_id] == 0 and
          is_mixed_type_column[this_col_id] == 1)
        column_categories[this_col_id] = NC_STR;
    }
    cudf::detail::cuda_memcpy_async<NodeT>(
      d_column_tree.node_categories, column_categories, stream);
  }

  // ignore all children of columns forced as string
  for (auto const this_col_id : unique_col_ids) {
    auto parent_col_id = column_parent_ids[this_col_id];
    if (parent_col_id != parent_node_sentinel and forced_as_string_column[parent_col_id]) {
      forced_as_string_column[this_col_id] = true;
      ignore_vals[this_col_id]             = true;
    }
    // Convert only mixed type columns as string (so to copy), but not its children
    if (parent_col_id != parent_node_sentinel and not forced_as_string_column[parent_col_id] and
        forced_as_string_column[this_col_id])
      column_categories[this_col_id] = NC_STR;
  }
  cudf::detail::cuda_memcpy_async<NodeT>(d_column_tree.node_categories, column_categories, stream);

  // restore unique_col_ids order
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<1>(a) < thrust::get<1>(b);
  });
  return {ignore_vals, columns};
}

void scatter_offsets(tree_meta_t const& tree,
                     device_span<NodeIndexT const> col_ids,
                     device_span<size_type const> row_offsets,
                     device_span<size_type> node_ids,
                     device_span<size_type> sorted_col_ids,  // Reuse this for parent_col_ids
                     tree_meta_t const& d_column_tree,
                     host_span<const bool> ignore_vals,
                     hashmap_of_device_columns const& columns,
                     rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes   = col_ids.size();
  auto const num_columns = d_column_tree.node_categories.size();
  // move columns data to device.
  auto columns_data = cudf::detail::make_host_vector<json_column_data>(num_columns, stream);
  for (auto& [col_id, col_ref] : columns) {
    if (col_id == parent_node_sentinel) continue;
    auto& col            = col_ref.get();
    columns_data[col_id] = json_column_data{col.string_offsets.data(),
                                            col.string_lengths.data(),
                                            col.child_offsets.data(),
                                            static_cast<bitmask_type*>(col.validity.data())};
  }

  auto d_ignore_vals = cudf::detail::make_device_uvector_async(
    ignore_vals, stream, cudf::get_current_device_resource_ref());
  auto d_columns_data = cudf::detail::make_device_uvector_async(
    columns_data, stream, cudf::get_current_device_resource_ref());

  // 3. scatter string offsets to respective columns, set validity bits
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<size_type>(0),
    num_nodes,
    [column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     row_offsets       = row_offsets.begin(),
     range_begin       = tree.node_range_begin.begin(),
     range_end         = tree.node_range_end.begin(),
     d_ignore_vals     = d_ignore_vals.begin(),
     d_columns_data    = d_columns_data.begin()] __device__(size_type i) {
      if (d_ignore_vals[col_ids[i]]) return;
      auto const node_category = column_categories[col_ids[i]];
      switch (node_category) {
        case NC_STRUCT: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_LIST: set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]); break;
        case NC_STR: [[fallthrough]];
        case NC_VAL:
          if (d_ignore_vals[col_ids[i]]) break;
          set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]);
          d_columns_data[col_ids[i]].string_offsets[row_offsets[i]] = range_begin[i];
          d_columns_data[col_ids[i]].string_lengths[row_offsets[i]] = range_end[i] - range_begin[i];
          break;
        default: break;
      }
    });

  // 4. scatter List offset
  // copy_if only node's whose parent is list, (node_id, parent_col_id)
  // stable_sort by parent_col_id of {node_id}.
  // For all unique parent_node_id of (i==0, i-1!=i), write start offset.
  //                                  (i==last, i+1!=i), write end offset.
  //    unique_copy_by_key {parent_node_id} {row_offset} to
  //    col[parent_col_id].child_offsets[row_offset[parent_node_id]]

  auto& parent_col_ids = sorted_col_ids;  // reuse sorted_col_ids
  auto parent_col_id   = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<NodeIndexT>(
      [col_ids         = col_ids.begin(),
       parent_node_ids = tree.parent_node_ids.begin()] __device__(size_type node_id) {
        return parent_node_ids[node_id] == parent_node_sentinel ? parent_node_sentinel
                                                                  : col_ids[parent_node_ids[node_id]];
      }));
  auto const list_children_end = thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    thrust::make_zip_iterator(thrust::make_counting_iterator<size_type>(0), parent_col_id),
    thrust::make_zip_iterator(thrust::make_counting_iterator<size_type>(0), parent_col_id) +
      num_nodes,
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_zip_iterator(node_ids.begin(), parent_col_ids.begin()),
    [d_ignore_vals     = d_ignore_vals.begin(),
     parent_node_ids   = tree.parent_node_ids.begin(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      return parent_node_id != parent_node_sentinel and
             column_categories[col_ids[parent_node_id]] == NC_LIST and
             (!d_ignore_vals[col_ids[parent_node_id]]);
    });

  auto const num_list_children =
    list_children_end - thrust::make_zip_iterator(node_ids.begin(), parent_col_ids.begin());
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                             parent_col_ids.begin(),
                             parent_col_ids.begin() + num_list_children,
                             node_ids.begin());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator<size_type>(0),
    num_list_children,
    [node_ids        = node_ids.begin(),
     parent_node_ids = tree.parent_node_ids.begin(),
     parent_col_ids  = parent_col_ids.begin(),
     row_offsets     = row_offsets.begin(),
     d_columns_data  = d_columns_data.begin(),
     num_list_children] __device__(size_type i) {
      auto const node_id        = node_ids[i];
      auto const parent_node_id = parent_node_ids[node_id];
      // scatter to list_offset
      if (i == 0 or parent_node_ids[node_ids[i - 1]] != parent_node_id) {
        d_columns_data[parent_col_ids[i]].child_offsets[row_offsets[parent_node_id]] =
          row_offsets[node_id];
      }
      // last value of list child_offset is its size.
      if (i == num_list_children - 1 or parent_node_ids[node_ids[i + 1]] != parent_node_id) {
        d_columns_data[parent_col_ids[i]].child_offsets[row_offsets[parent_node_id] + 1] =
          row_offsets[node_id] + 1;
      }
    });

  // 5. scan on offsets.
  for (auto& [id, col_ref] : columns) {
    auto& col = col_ref.get();
    if (col.type == json_col_t::StringColumn) {
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                             col.string_offsets.begin(),
                             col.string_offsets.end(),
                             col.string_offsets.begin(),
                             thrust::maximum<json_column::row_offset_t>{});
    } else if (col.type == json_col_t::ListColumn) {
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                             col.child_offsets.begin(),
                             col.child_offsets.end(),
                             col.child_offsets.begin(),
                             thrust::maximum<json_column::row_offset_t>{});
    }
  }
  stream.synchronize();
}

namespace experimental {

std::map<std::string, schema_element> unified_schema(cudf::io::json_reader_options const& options)
{
  return std::visit(
    cudf::detail::visitor_overload{
      [](std::vector<data_type> const& user_dtypes) {
        std::map<std::string, schema_element> dnew;
        std::transform(thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(user_dtypes.size()),
                       std::inserter(dnew, dnew.end()),
                       [&user_dtypes](auto i) {
                         return std::pair(std::to_string(i), schema_element{user_dtypes[i]});
                       });
        return dnew;
      },
      [](std::map<std::string, data_type> const& user_dtypes) {
        std::map<std::string, schema_element> dnew;
        std::transform(user_dtypes.begin(),
                       user_dtypes.end(),
                       std::inserter(dnew, dnew.end()),
                       [](auto key_dtype) {
                         return std::pair(key_dtype.first, schema_element{key_dtype.second});
                       });
        return dnew;
      },
      [](std::map<std::string, schema_element> const& user_dtypes) { return user_dtypes; }},
    options.get_dtypes());
}

std::pair<cudf::detail::host_vector<bool>, hashmap_of_device_columns> build_tree(
  device_json_column& root,
  host_span<uint8_t const> is_str_column_all_nulls,
  tree_meta_t& d_column_tree,
  device_span<NodeIndexT const> d_unique_col_ids,
  device_span<size_type const> d_max_row_offsets,
  std::vector<std::string> const& column_names,
  NodeIndexT row_array_parent_col_id,
  bool is_array_of_arrays,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Constructs `d_json_column` from node tree representation
 * Newly constructed columns are inserted into `root`'s children.
 * `root` must be a list type.
 *
 * @param input Input JSON string device data
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param row_offsets Row offsets of the nodes in the tree
 * @param root Root node of the `d_json_column` tree
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param options Parsing options specifying the parsing behaviour
 * options affecting behaviour are
 *   is_enabled_lines: Whether the input is a line-delimited JSON
 *   is_enabled_mixed_types_as_string: Whether to enable reading mixed types as string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the device memory
 * of child_offets and validity members of `d_json_column`
 */
void make_device_json_column(device_span<SymbolT const> input,
                             tree_meta_t const& tree,
                             device_span<NodeIndexT const> col_ids,
                             device_span<size_type const> row_offsets,
                             device_json_column& root,
                             bool is_array_of_arrays,
                             cudf::io::json_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const is_enabled_lines                 = options.is_enabled_lines();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  // make a copy
  auto sorted_col_ids = cudf::detail::make_device_uvector_async(
    col_ids, stream, cudf::get_current_device_resource_ref());

  // sort by {col_id} on {node_ids} stable
  rmm::device_uvector<NodeIndexT> node_ids(col_ids.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                             sorted_col_ids.begin(),
                             sorted_col_ids.end(),
                             node_ids.begin());

  NodeIndexT const row_array_parent_col_id =
    get_row_array_parent_col_id(col_ids, is_enabled_lines, stream);

  // 1. gather column information.
  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    reduce_to_column_tree(tree,
                          col_ids,
                          sorted_col_ids,
                          node_ids,
                          row_offsets,
                          is_array_of_arrays,
                          row_array_parent_col_id,
                          stream);

  auto num_columns                      = d_unique_col_ids.size();
  std::vector<std::string> column_names = copy_strings_to_host_sync(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);
  // array of arrays column names
  if (is_array_of_arrays) {
    auto const unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
    auto const column_parent_ids =
      cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
    TreeDepthT const row_array_children_level = is_enabled_lines ? 1 : 2;
    auto values_column_indices =
      get_values_column_indices(row_array_children_level, tree, col_ids, num_columns, stream);
    auto h_values_column_indices =
      cudf::detail::make_host_vector_sync(values_column_indices, stream);
    std::transform(unique_col_ids.begin(),
                   unique_col_ids.end(),
                   column_names.cbegin(),
                   column_names.begin(),
                   [&h_values_column_indices, &column_parent_ids, row_array_parent_col_id](
                     auto col_id, auto name) mutable {
                     return column_parent_ids[col_id] == row_array_parent_col_id
                              ? std::to_string(h_values_column_indices[col_id])
                              : name;
                   });
  }

  auto const is_str_column_all_nulls = [&, &column_tree = d_column_tree]() {
    if (is_enabled_mixed_types_as_string) {
      return cudf::detail::make_std_vector_sync(
        is_all_nulls_each_column(input, column_tree, tree, col_ids, options, stream), stream);
    }
    return std::vector<uint8_t>();
  }();
  auto const [ignore_vals, columns] = build_tree(root,
                                                 is_str_column_all_nulls,
                                                 d_column_tree,
                                                 d_unique_col_ids,
                                                 d_max_row_offsets,
                                                 column_names,
                                                 row_array_parent_col_id,
                                                 is_array_of_arrays,
                                                 options,
                                                 stream,
                                                 mr);
  if (ignore_vals.empty()) return;
  scatter_offsets(tree,
                  col_ids,
                  row_offsets,
                  node_ids,
                  sorted_col_ids,
                  d_column_tree,
                  ignore_vals,
                  columns,
                  stream);
}

std::pair<cudf::detail::host_vector<bool>, hashmap_of_device_columns> build_tree(
  device_json_column& root,
  host_span<uint8_t const> is_str_column_all_nulls,
  tree_meta_t& d_column_tree,
  device_span<NodeIndexT const> d_unique_col_ids,
  device_span<size_type const> d_max_row_offsets,
  std::vector<std::string> const& column_names,
  NodeIndexT row_array_parent_col_id,
  bool is_array_of_arrays,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const is_enabled_lines                 = options.is_enabled_lines();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  auto unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
  auto column_categories =
    cudf::detail::make_host_vector_async(d_column_tree.node_categories, stream);
  auto const column_parent_ids =
    cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
  auto column_range_beg =
    cudf::detail::make_host_vector_async(d_column_tree.node_range_begin, stream);
  auto const max_row_offsets = cudf::detail::make_host_vector_async(d_max_row_offsets, stream);
  auto num_columns           = d_unique_col_ids.size();
  stream.synchronize();

  auto to_json_col_type = [](auto category) {
    switch (category) {
      case NC_STRUCT: return json_col_t::StructColumn;
      case NC_LIST: return json_col_t::ListColumn;
      case NC_STR: [[fallthrough]];
      case NC_VAL: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    }
  };

  auto initialize_json_columns = [&](auto i, auto& col_ref, auto column_category) {
    auto& col = col_ref.get();
    if (col.type != json_col_t::Unknown) { return; }
    if (column_category == NC_ERR || column_category == NC_FN) {
      return;
    } else if (column_category == NC_VAL || column_category == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      thrust::fill(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(col.string_offsets.begin(), col.string_lengths.begin()),
        thrust::make_zip_iterator(col.string_offsets.end(), col.string_lengths.end()),
        thrust::make_tuple(0, 0));
    } else if (column_category == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      thrust::uninitialized_fill(
        rmm::exec_policy_nosync(stream), col.child_offsets.begin(), col.child_offsets.end(), 0);
    }
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = to_json_col_type(column_category);
  };

  // 2. generate nested columns tree and its device_memory
  // reorder unique_col_ids w.r.t. column_range_begin for order of column to be in field order.
  auto h_range_col_id_it =
    thrust::make_zip_iterator(column_range_beg.begin(), unique_col_ids.begin());
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return thrust::get<0>(a) < thrust::get<0>(b);
  });
  // adjacency list construction
  std::map<NodeIndexT, std::vector<NodeIndexT>> adj;
  for (auto const this_col_id : unique_col_ids) {
    auto parent_col_id = column_parent_ids[this_col_id];
    adj[parent_col_id].push_back(this_col_id);
  }

  // Pruning
  auto is_pruned = cudf::detail::make_host_vector<bool>(num_columns, stream);
  std::fill_n(is_pruned.begin(), num_columns, options.is_enabled_prune_columns());

  // prune all children of a column, but not self.
  auto ignore_all_children = [&](auto parent_col_id) {
    std::deque<NodeIndexT> offspring;
    if (adj.count(parent_col_id)) {
      for (auto const& child : adj[parent_col_id]) {
        offspring.push_back(child);
      }
    }
    while (!offspring.empty()) {
      auto this_id = offspring.front();
      offspring.pop_front();
      is_pruned[this_id] = true;
      if (adj.count(this_id)) {
        for (auto const& child : adj[this_id]) {
          offspring.push_back(child);
        }
      }
    }
  };

  // Pruning: iterate through schema and mark only those columns and enforce type.
  // NoPruning: iterate through schema and enforce type.

  if (adj[parent_node_sentinel].empty())
    return {cudf::detail::make_host_vector<bool>(0, stream), {}};  // for empty file
  CUDF_EXPECTS(adj[parent_node_sentinel].size() == 1, "Should be 1");
  auto expected_types = cudf::detail::make_host_vector<NodeT>(num_columns, stream);
  std::fill_n(expected_types.begin(), num_columns, NUM_NODE_CLASSES);

  auto lookup_names = [&column_names](auto child_ids, auto name) {
    for (auto const& child_id : child_ids) {
      if (column_names[child_id] == name) return child_id;
    }
    return -1;
  };
  // recursive lambda on schema to mark columns as pruned.
  std::function<void(NodeIndexT root, schema_element const& schema)> mark_is_pruned;
  mark_is_pruned = [&is_pruned,
                    &mark_is_pruned,
                    &adj,
                    &lookup_names,
                    &column_categories,
                    &expected_types,
                    &ignore_all_children](NodeIndexT root, schema_element const& schema) -> void {
    if (root == -1) return;
    bool pass =
      (schema.type == data_type{type_id::STRUCT} and column_categories[root] == NC_STRUCT) or
      (schema.type == data_type{type_id::LIST} and column_categories[root] == NC_LIST) or
      (schema.type != data_type{type_id::STRUCT} and schema.type != data_type{type_id::LIST} and
       column_categories[root] != NC_FN);
    if (!pass) {
      // ignore all children of this column and prune this column.
      is_pruned[root] = true;
      ignore_all_children(root);
      return;
    }
    is_pruned[root]    = false;
    auto expected_type = [](auto type, auto cat) {
      if (type == data_type{type_id::STRUCT} and cat == NC_STRUCT) return NC_STRUCT;
      if (type == data_type{type_id::LIST} and cat == NC_LIST) return NC_LIST;
      if (type != data_type{type_id::STRUCT} and type != data_type{type_id::LIST}) return NC_STR;
      return NC_ERR;
    }(schema.type, column_categories[root]);
    expected_types[root] = expected_type;  // forced type.
    // ignore children of nested columns, but not self.
    if (expected_type == NC_STR and
        (column_categories[root] == NC_STRUCT or column_categories[root] == NC_LIST))
      ignore_all_children(root);
    if (not(schema.type == data_type{type_id::STRUCT} or schema.type == data_type{type_id::LIST}))
      return;  // no children to mark for non-nested.
    auto child_ids = adj.count(root) ? adj[root] : std::vector<NodeIndexT>{};
    if (schema.type == data_type{type_id::STRUCT}) {
      for (auto const& key_pair : schema.child_types) {
        auto col_id = lookup_names(child_ids, key_pair.first);
        if (col_id == -1) continue;
        is_pruned[col_id] = false;
        for (auto const& child_id : adj[col_id])  // children of field (>1 if mixed)
          mark_is_pruned(child_id, key_pair.second);
      }
    } else if (schema.type == data_type{type_id::LIST}) {
      // partial solution for list children to have any name.
      auto this_list_child_name =
        schema.child_types.size() == 1 ? schema.child_types.begin()->first : list_child_name;
      if (schema.child_types.count(this_list_child_name) == 0) return;
      auto list_child = schema.child_types.at(this_list_child_name);
      for (auto const& child_id : child_ids)
        mark_is_pruned(child_id, list_child);
    }
  };
  if (is_array_of_arrays) {
    if (adj[adj[parent_node_sentinel][0]].empty())
      return {cudf::detail::make_host_vector<bool>(0, stream), {}};
    auto root_list_col_id =
      is_enabled_lines ? adj[parent_node_sentinel][0] : adj[adj[parent_node_sentinel][0]][0];
    // mark root and row array col_id as not pruned.
    if (!is_enabled_lines) {
      auto top_level_list_id       = adj[parent_node_sentinel][0];
      is_pruned[top_level_list_id] = false;
    }
    is_pruned[root_list_col_id] = false;
    std::visit(cudf::detail::visitor_overload{
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   std::vector<data_type> const& user_dtypes) -> void {
                   for (size_t i = 0; i < adj[root_list_col_id].size() && i < user_dtypes.size();
                        i++) {
                     NodeIndexT const first_child_id = adj[root_list_col_id][i];
                     auto name                       = column_names[first_child_id];
                     auto value_id                   = std::stol(name);
                     if (value_id >= 0 and value_id < static_cast<long>(user_dtypes.size()))
                       mark_is_pruned(first_child_id, schema_element{user_dtypes[value_id]});
                     // Note: mixed type - forced type, will work here.
                   }
                 },
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   std::map<std::string, data_type> const& user_dtypes) -> void {
                   for (size_t i = 0; i < adj[root_list_col_id].size(); i++) {
                     auto const first_child_id = adj[root_list_col_id][i];
                     auto name                 = column_names[first_child_id];
                     if (user_dtypes.count(name))
                       mark_is_pruned(first_child_id, schema_element{user_dtypes.at(name)});
                   }
                 },
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   std::map<std::string, schema_element> const& user_dtypes) -> void {
                   for (size_t i = 0; i < adj[root_list_col_id].size(); i++) {
                     auto const first_child_id = adj[root_list_col_id][i];
                     auto name                 = column_names[first_child_id];
                     if (user_dtypes.count(name))
                       mark_is_pruned(first_child_id, user_dtypes.at(name));
                   }
                 }},
               options.get_dtypes());
  } else {
    auto root_struct_col_id =
      is_enabled_lines
        ? adj[parent_node_sentinel][0]
        : (adj[adj[parent_node_sentinel][0]].empty() ? -1 : adj[adj[parent_node_sentinel][0]][0]);
    // mark root and row struct col_id as not pruned.
    if (!is_enabled_lines) {
      auto top_level_list_id       = adj[parent_node_sentinel][0];
      is_pruned[top_level_list_id] = false;
    }
    is_pruned[root_struct_col_id] = false;
    schema_element u_schema{data_type{type_id::STRUCT}};
    u_schema.child_types = unified_schema(options);
    std::visit(
      cudf::detail::visitor_overload{
        [&is_pruned, &root_struct_col_id, &adj, &mark_is_pruned](
          std::vector<data_type> const& user_dtypes) -> void {
          for (size_t i = 0; i < adj[root_struct_col_id].size() && i < user_dtypes.size(); i++) {
            NodeIndexT const first_field_id = adj[root_struct_col_id][i];
            is_pruned[first_field_id]       = false;
            for (auto const& child_id : adj[first_field_id])  // children of field (>1 if mixed)
              mark_is_pruned(child_id, schema_element{user_dtypes[i]});
          }
        },
        [&root_struct_col_id, &adj, &mark_is_pruned, &u_schema](
          std::map<std::string, data_type> const& user_dtypes) -> void {
          mark_is_pruned(root_struct_col_id, u_schema);
        },
        [&root_struct_col_id, &adj, &mark_is_pruned, &u_schema](
          std::map<std::string, schema_element> const& user_dtypes) -> void {
          mark_is_pruned(root_struct_col_id, u_schema);
        }},
      options.get_dtypes());
  }
  // Useful for array of arrays
  auto named_level =
    is_enabled_lines
      ? adj[parent_node_sentinel][0]
      : (adj[adj[parent_node_sentinel][0]].empty() ? -1 : adj[adj[parent_node_sentinel][0]][0]);

  auto handle_mixed_types = [&column_categories,
                             &is_str_column_all_nulls,
                             &is_pruned,
                             &expected_types,
                             &is_enabled_mixed_types_as_string,
                             &ignore_all_children](std::vector<NodeIndexT>& child_ids) {
    // do these on unpruned columns only.
    // when mixed types is disabled, ignore string sibling of nested column.
    // when mixed types is disabled, and both list and struct columns are siblings, error out.
    // when mixed types is enabled, force string type on all columns

    // Remove pruned children (forced type will not clash here because other types are already
    // pruned)
    child_ids.erase(
      std::remove_if(child_ids.begin(),
                     child_ids.end(),
                     [&is_pruned](NodeIndexT child_id) { return is_pruned[child_id]; }),
      child_ids.end());
    // find string id, struct id, list id.
    NodeIndexT str_col_id{-1}, struct_col_id{-1}, list_col_id{-1};
    for (auto const& child_id : child_ids) {
      if (column_categories[child_id] == NC_VAL || column_categories[child_id] == NC_STR)
        str_col_id = child_id;
      else if (column_categories[child_id] == NC_STRUCT)
        struct_col_id = child_id;
      else if (column_categories[child_id] == NC_LIST)
        list_col_id = child_id;
    }
    // conditions for handling mixed types.
    if (is_enabled_mixed_types_as_string) {
      if (struct_col_id != -1 and list_col_id != -1) {
        expected_types[struct_col_id] = NC_STR;
        expected_types[list_col_id]   = NC_STR;
        // ignore children of nested columns.
        ignore_all_children(struct_col_id);
        ignore_all_children(list_col_id);
      }
      if ((struct_col_id != -1 or list_col_id != -1) and str_col_id != -1) {
        if (is_str_column_all_nulls[str_col_id])
          is_pruned[str_col_id] = true;
        else {
          // ignore children of nested columns.
          if (struct_col_id != -1) {
            expected_types[struct_col_id] = NC_STR;
            ignore_all_children(struct_col_id);
          }
          if (list_col_id != -1) {
            expected_types[list_col_id] = NC_STR;
            ignore_all_children(list_col_id);
          }
        }
      }
    } else {
      // if both are present, error out.
      CUDF_EXPECTS(struct_col_id == -1 or list_col_id == -1,
                   "A mix of lists and structs within the same column is not supported");
      // either one only: so ignore str column.
      if ((struct_col_id != -1 or list_col_id != -1) and str_col_id != -1) {
        is_pruned[str_col_id] = true;
      }
    }
  };

  using dev_ref = std::reference_wrapper<device_json_column>;
  std::unordered_map<NodeIndexT, dev_ref> columns;
  columns.try_emplace(parent_node_sentinel, std::ref(root));
  // convert adjaceny list to tree.
  dev_ref parent_ref = std::ref(root);
  // creates children column
  std::function<void(NodeIndexT, dev_ref)> construct_tree;
  construct_tree = [&](NodeIndexT root, dev_ref ref) -> void {
    if (is_pruned[root]) return;
    auto expected_category =
      expected_types[root] == NUM_NODE_CLASSES ? column_categories[root] : expected_types[root];
    initialize_json_columns(root, ref, expected_category);
    auto child_ids = adj.count(root) ? adj[root] : std::vector<NodeIndexT>{};
    if (expected_category == NC_STRUCT) {
      // find field column ids, and its children and create columns.
      for (auto const& field_id : child_ids) {
        auto name = column_names[field_id];
        if (is_pruned[field_id]) continue;
        auto inserted =
          ref.get().child_columns.try_emplace(name, device_json_column(stream, mr)).second;
        ref.get().column_order.emplace_back(name);
        CUDF_EXPECTS(inserted,
                     "struct child column insertion failed, duplicate column name in the parent");
        auto this_ref = std::ref(ref.get().child_columns.at(name));
        // Mixed type handling
        auto& value_col_ids = adj[field_id];
        handle_mixed_types(value_col_ids);
        if (value_col_ids.empty()) {
          // If no column is present, remove the uninitialized column.
          ref.get().child_columns.erase(name);
          ref.get().column_order.pop_back();
          continue;
        }
        for (auto const& child_id : value_col_ids)  // children of field (>1 if mixed)
        {
          if (is_pruned[child_id]) continue;
          columns.try_emplace(child_id, this_ref);
          construct_tree(child_id, this_ref);
        }
      }
    } else if (expected_category == NC_LIST) {
      // array of arrays interpreted as array of structs.
      if (is_array_of_arrays and root == named_level) {
        // create column names
        std::map<NodeIndexT, std::vector<NodeIndexT>> array_values;
        for (auto const& child_id : child_ids) {
          if (is_pruned[child_id]) continue;
          auto name = column_names[child_id];
          array_values[std::stoi(name)].push_back(child_id);
        }
        //
        for (auto const& value_id_pair : array_values) {
          auto [value_id, value_col_ids] = value_id_pair;
          auto name                      = std::to_string(value_id);
          auto inserted =
            ref.get().child_columns.try_emplace(name, device_json_column(stream, mr)).second;
          ref.get().column_order.emplace_back(name);
          CUDF_EXPECTS(inserted,
                       "list child column insertion failed, duplicate column name in the parent");
          auto this_ref = std::ref(ref.get().child_columns.at(name));
          handle_mixed_types(value_col_ids);
          if (value_col_ids.empty()) {
            // If no column is present, remove the uninitialized column.
            ref.get().child_columns.erase(name);
            ref.get().column_order.pop_back();
            continue;
          }
          for (auto const& child_id : value_col_ids)  // children of field (>1 if mixed)
          {
            if (is_pruned[child_id]) continue;
            columns.try_emplace(child_id, this_ref);
            construct_tree(child_id, this_ref);
          }
        }
      } else {
        if (child_ids.empty()) return;
        auto inserted =
          ref.get()
            .child_columns.try_emplace(list_child_name, device_json_column(stream, mr))
            .second;
        CUDF_EXPECTS(inserted,
                     "list child column insertion failed, duplicate column name in the parent");
        ref.get().column_order.emplace_back(list_child_name);
        auto this_ref = std::ref(ref.get().child_columns.at(list_child_name));
        // Mixed type handling
        handle_mixed_types(child_ids);
        if (child_ids.empty()) {
          // If no column is present, remove the uninitialized column.
          ref.get().child_columns.erase(list_child_name);
        }
        for (auto const& child_id : child_ids) {
          if (is_pruned[child_id]) continue;
          columns.try_emplace(child_id, this_ref);
          construct_tree(child_id, this_ref);
        }
      }
    }
  };
  auto inserted = parent_ref.get()
                    .child_columns.try_emplace(list_child_name, device_json_column(stream, mr))
                    .second;
  CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
  parent_ref = std::ref(parent_ref.get().child_columns.at(list_child_name));
  columns.try_emplace(adj[parent_node_sentinel][0], parent_ref);
  construct_tree(adj[parent_node_sentinel][0], parent_ref);

  // Forced string type due to input schema and mixed type as string.
  for (size_t i = 0; i < expected_types.size(); i++) {
    if (expected_types[i] == NC_STR) {
      if (columns.count(i)) { columns.at(i).get().forced_as_string_column = true; }
    }
  }
  std::transform(expected_types.cbegin(),
                 expected_types.cend(),
                 column_categories.cbegin(),
                 expected_types.begin(),
                 [](auto exp, auto cat) { return exp == NUM_NODE_CLASSES ? cat : exp; });
  cudaMemcpyAsync(d_column_tree.node_categories.begin(),
                  expected_types.data(),
                  expected_types.size() * sizeof(column_categories[0]),
                  cudaMemcpyDefault,
                  stream.value());

  return {is_pruned, columns};
}
}  // namespace experimental

}  // namespace cudf::io::json::detail
