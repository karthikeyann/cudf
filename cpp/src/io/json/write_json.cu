/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * @file write_json.cu
 * @brief cuDF-IO JSON writer implementation
 */

#include <io/csv/durations.hpp>
#include <lists/utilities.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/data_casting.cuh>
#include <cudf/io/detail/json.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace cudf::io::json::detail {

std::unique_ptr<column> make_column_names_column(host_span<column_name_info const> column_names,
                                                 size_type num_columns,
                                                 rmm::cuda_stream_view stream);
namespace {

/**
 * @brief Functor to modify a string column for JSON format.
 *
 * This will convert escape characters and wrap quotes around strings.
 */
struct escape_strings_fn {
  column_device_view const d_column;
  bool const append_colon{false};
  offset_type* d_offsets{};
  char* d_chars{};

  __device__ void write_char(char_utf8 chr, char*& d_buffer, offset_type& bytes)
  {
    if (d_buffer)
      d_buffer += cudf::strings::detail::from_char_utf8(chr, d_buffer);
    else
      bytes += cudf::strings::detail::bytes_in_char_utf8(chr);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    auto const d_str = d_column.element<string_view>(idx);

    // entire string must be double-quoted.
    constexpr char_utf8 const quote = '\"';  // wrap quotes
    bool constexpr quote_row        = true;

    char* d_buffer    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    offset_type bytes = 0;

    if (quote_row) write_char(quote, d_buffer, bytes);
    for (auto chr : d_str) {
      auto escaped_chars = cudf::io::json::experimental::detail::get_escaped_char(chr);
      if (escaped_chars.first == '\0') {
        write_char(escaped_chars.second, d_buffer, bytes);
      } else {
        write_char(escaped_chars.first, d_buffer, bytes);
        write_char(escaped_chars.second, d_buffer, bytes);
      }
    }
    if (quote_row) write_char(quote, d_buffer, bytes);
    constexpr char_utf8 const colon = ':';  // append colon
    if (append_colon) write_char(colon, d_buffer, bytes);

    if (!d_chars) d_offsets[idx] = bytes;
  }

  std::unique_ptr<column> get_escaped_strings(column_view const& column_v,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
  {
    auto children =
      cudf::strings::detail::make_strings_children(*this, column_v.size(), stream, mr);

    return make_strings_column(column_v.size(),
                               std::move(children.first),
                               std::move(children.second),
                               column_v.null_count(),
                               cudf::detail::copy_bitmask(column_v, stream, mr));
  }
};

/**
 * @brief Concatenate the strings from each row of the given table as structs in JSON string
 *
 * Each row will be struct with field name as column names and values from each column in the table.
 *
 * @param strings_columns Table of strings columns
 * @param column_names Column of names for each column in the table
 * @param row_prefix Prepend this string to each row
 * @param row_suffix  Append this string to each row
 * @param column_name_separator  Separator between column name and value
 * @param value_separator Separator between values
 * @param narep Null-String replacement
 * @param include_nulls Include null string entries in the output
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation.
 * @return New strings column of JSON structs in each row
 */
std::unique_ptr<column> struct_to_strings(table_view const& strings_columns,
                                          column_view const& column_names,
                                          string_view const row_prefix,
                                          string_view const row_suffix,
                                          string_view const column_name_separator,
                                          string_view const value_separator,
                                          string_scalar const& narep,
                                          bool include_nulls,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(column_names.type().id() == type_id::STRING, "Column names must be of type string");
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns == column_names.size(),
               "Number of column names should be equal to number of columns in the table");
  auto const strings_count = strings_columns.num_rows();
  if (strings_count == 0)  // empty begets empty
    return make_empty_column(type_id::STRING);
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto const& c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");
  auto constexpr str_views_per_column = 3;  // (for each "column_name:", "value",  "separator")
  auto const num_str_views_per_row    = strings_columns.num_columns() * str_views_per_column + 1;
  // Eg. { col1: value , col2: value , col3: value } = 1+ 3+3+(3-1) +1 = 10

  auto tbl_device_view = cudf::table_device_view::create(strings_columns, stream);
  auto d_column_names  = column_device_view::create(column_names, stream);

  // (num_columns*3+1)*num_rows size (very high!) IDEA: chunk it here? but maximize parallelism?
  auto const string_views_total = num_str_views_per_row * strings_columns.num_rows();
  auto const total_rows         = strings_columns.num_rows() * strings_columns.num_columns();
  using str_pair                = thrust::pair<const char*, size_type>;
  rmm::device_uvector<str_pair> d_strviews(string_views_total, stream);

  // scatter row_prefix, row_suffix, column_name:, value, value_separator as string_views
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(total_rows),
    [tbl       = *tbl_device_view,
     col_names = *d_column_names,
     str_views_per_column,
     num_str_views_per_row,
     row_prefix,
     row_suffix,
     value_separator,
     narep = narep.value(stream),
     include_nulls,
     d_strviews = d_strviews.begin()] __device__(auto idx) {
      auto const row        = idx / tbl.num_columns();
      auto const col        = idx % tbl.num_columns();
      auto const d_str_null = tbl.column(col).is_null(row);
      auto const this_index = row * num_str_views_per_row + col * str_views_per_column + 1;
      // prefix
      if (col == 0) {
        d_strviews[this_index - 1] = str_pair(row_prefix.data(), row_prefix.size_bytes());
      }
      if (!include_nulls && d_str_null) {
        if (col != 0) d_strviews[this_index - 1] = str_pair(nullptr, 0);
        d_strviews[this_index]     = str_pair(nullptr, 0);
        d_strviews[this_index + 1] = str_pair(nullptr, 0);
      } else {
        // if previous column was null, then we skip the value separator
        if (col != 0)
          if (tbl.column(col - 1).is_null(row) && !include_nulls)
            d_strviews[this_index - 1] = str_pair(nullptr, 0);
          else
            d_strviews[this_index - 1] =
              str_pair(value_separator.data(), value_separator.size_bytes());
        auto const d_col_name = col_names.element<string_view>(col);
        auto const d_str = d_str_null ? narep : tbl.column(col).template element<string_view>(row);
        // column_name: value
        d_strviews[this_index]     = str_pair(d_col_name.data(), d_col_name.size_bytes());
        d_strviews[this_index + 1] = str_pair(d_str.data(), d_str.size_bytes());
      }
      // suffix
      if (col == tbl.num_columns() - 1) {
        d_strviews[this_index + 2] = str_pair(row_suffix.data(), row_suffix.size_bytes());
      }
    });
  auto joined_col = make_strings_column(d_strviews, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(strings_columns.num_rows() + 1, stream, mr);
  auto const d_strview_offsets = cudf::detail::make_counting_transform_iterator(
    0, [num_str_views_per_row] __device__(size_type const i) { return i * num_str_views_per_row; });
  thrust::gather(rmm::exec_policy(stream),
                 d_strview_offsets,
                 d_strview_offsets + row_string_offsets.size(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  return make_strings_column(
    strings_columns.num_rows(),
    std::make_unique<cudf::column>(std::move(row_string_offsets)),
    std::move(joined_col->release().children[strings_column_view::chars_column_index]),
    0,
    {});
}

/**
 * @brief Concatenates a list of strings columns into a single strings column.
 *
 * @param lists_strings Column containing lists of strings to concatenate.
 * @param list_prefix String to place before each list. (typically [)
 * @param list_suffix String to place after each list. (typically ])
 * @param element_separator String that should inserted between strings of each list row.
 * @param element_narep String that should be used in place of any null strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with concatenated results.
 */
std::unique_ptr<column> join_list_of_strings(lists_column_view const& lists_strings,
                                             string_view const list_prefix,
                                             string_view const list_suffix,
                                             string_view const element_separator,
                                             string_view const element_narep,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // create string_views of the list elements, and the list separators and list prefix/suffix.
  // then concatenates them all together.
  // gather offset of first string_view of each row as offsets for output string column.
  auto const offsets          = lists_strings.offsets();
  auto const strings_children = lists_strings.get_sliced_child(stream);
  auto const null_count       = lists_strings.null_count();
  auto const num_lists        = lists_strings.size();
  auto const num_strings      = strings_children.size();
  auto const num_offsets      = offsets.size();

  rmm::device_uvector<size_type> d_strview_offsets(num_offsets, stream);
  auto num_str_views_per_list = cudf::detail::make_counting_transform_iterator(
    0, [offsets = offsets.begin<size_type>(), num_offsets] __device__(size_type idx) {
      if (idx + 1 >= num_offsets) return 0;
      auto const length = offsets[idx + 1] - offsets[idx];
      return length == 0 ? 2 : (2 + length + length - 1);
    });
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_str_views_per_list,
                         num_str_views_per_list + num_offsets,
                         d_strview_offsets.begin());
  auto const string_views_total = d_strview_offsets.back_element(stream);

  using str_pair = thrust::pair<const char*, size_type>;
  rmm::device_uvector<str_pair> d_strviews(string_views_total, stream);
  // scatter null_list and list_prefix, list_suffix
  auto col_device_view = cudf::column_device_view::create(lists_strings.parent(), stream);
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_lists),
    [col = *col_device_view,
     list_prefix,
     list_suffix,
     d_strview_counts = d_strview_offsets.begin(),
     d_strviews       = d_strviews.begin()] __device__(auto idx) {
      if (col.is_null(idx)) {
        d_strviews[d_strview_counts[idx]]     = str_pair{nullptr, 0};
        d_strviews[d_strview_counts[idx] + 1] = str_pair{nullptr, 0};
      } else {
        // [ ]
        d_strviews[d_strview_counts[idx]] = str_pair{list_prefix.data(), list_prefix.size_bytes()};
        d_strviews[d_strview_counts[idx + 1] - 1] =
          str_pair{list_suffix.data(), list_suffix.size_bytes()};
      }
    });

  // scatter string and separator
  auto labels = cudf::lists::detail::generate_labels(
    lists_strings, num_strings, stream, rmm::mr::get_current_device_resource());
  auto d_strings_children = cudf::column_device_view::create(strings_children, stream);
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(num_strings),
                   [col                = *col_device_view,
                    d_strview_counts   = d_strview_offsets.begin(),
                    d_strviews         = d_strviews.begin(),
                    labels             = labels->view().begin<size_type>(),
                    list_offsets       = offsets.begin<size_type>(),
                    d_strings_children = *d_strings_children,
                    element_separator,
                    element_narep] __device__(auto idx) {
                     auto const label         = labels[idx];
                     auto const sublist_index = idx - list_offsets[label];
                     auto const strview_index = d_strview_counts[label] + sublist_index * 2 + 1;
                     // value or na_rep
                     auto const strview = d_strings_children.element<cudf::string_view>(idx);
                     d_strviews[strview_index] =
                       d_strings_children.is_null(idx)
                         ? str_pair{element_narep.data(), element_narep.size_bytes()}
                         : str_pair{strview.data(), strview.size_bytes()};
                     // separator
                     if (sublist_index != 0) {
                       d_strviews[strview_index - 1] =
                         str_pair{element_separator.data(), element_separator.size_bytes()};
                     }
                   });

  auto joined_col = make_strings_column(d_strviews, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(num_offsets, stream, mr);
  thrust::gather(rmm::exec_policy(stream),
                 d_strview_offsets.begin(),
                 d_strview_offsets.end(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  return make_strings_column(
    num_lists,
    std::make_unique<cudf::column>(std::move(row_string_offsets)),
    std::move(joined_col->release().children[strings_column_view::chars_column_index]),
    lists_strings.null_count(),
    cudf::detail::copy_bitmask(lists_strings.parent(), stream, mr));
}

/**
 * @brief Functor to convert a column to string representation for JSON format.
 */
struct column_to_strings_fn {
  /**
   * @brief Returns true if the specified type is not supported by the JSON writer.
   */
  template <typename column_type>
  constexpr static bool is_not_handled()
  {
    // Note: the case (not std::is_same_v<column_type, bool>)  is already covered by is_integral)
    return not((std::is_same_v<column_type, cudf::string_view>) ||
               (std::is_integral_v<column_type>) || (std::is_floating_point_v<column_type>) ||
               (cudf::is_fixed_point<column_type>()) || (cudf::is_timestamp<column_type>()) ||
               (cudf::is_duration<column_type>()));
  }

  explicit column_to_strings_fn(json_writer_options const& options,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
    : options_(options), stream_(stream), mr_(mr), narep(options.get_na_rep())
  {
  }

  // unsupported type of column:
  template <typename column_type>
  std::enable_if_t<is_not_handled<column_type>(), std::unique_ptr<column>> operator()(
    column_view const&) const
  {
    CUDF_FAIL("Unsupported column type.");
  }

  // Note: `null` replacement with `na_rep` deferred to `concatenate()`
  // instead of column-wise; might be faster.

  // bools:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, bool>, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_booleans(
      column, options_.get_true_value(), options_.get_false_value(), stream_, mr_);
  }

  // strings:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::string_view>, std::unique_ptr<column>>
  operator()(column_view const& column_v) const
  {
    auto d_column = column_device_view::create(column_v, stream_);
    return escape_strings_fn{*d_column}.get_escaped_strings(column_v, stream_, mr_);
  }

  // ints:
  template <typename column_type>
  std::enable_if_t<std::is_integral_v<column_type> && !std::is_same_v<column_type, bool>,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    return cudf::strings::detail::from_integers(column, stream_, mr_);
  }

  // floats:
  template <typename column_type>
  std::enable_if_t<std::is_floating_point_v<column_type>, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_floats(column, stream_, mr_);
  }

  // fixed point:
  template <typename column_type>
  std::enable_if_t<cudf::is_fixed_point<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_fixed_point(column, stream_, mr_);
  }

  // timestamps:
  template <typename column_type>
  std::enable_if_t<cudf::is_timestamp<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    std::string format = [&]() {
      if (std::is_same_v<cudf::timestamp_s, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%SZ"};
      } else if (std::is_same_v<cudf::timestamp_ms, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%3fZ"};
      } else if (std::is_same_v<cudf::timestamp_us, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%6fZ"};
      } else if (std::is_same_v<cudf::timestamp_ns, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%9fZ"};
      } else {
        return std::string{"%Y-%m-%d"};
      }
    }();

    // Since format uses ":", we need to add quotes to the format
    format = "\"" + format + "\"";

    return cudf::strings::detail::from_timestamps(
      column,
      format,
      strings_column_view(column_view{data_type{type_id::STRING}, 0, nullptr}),
      stream_,
      mr_);
  }

  template <typename column_type>
  std::enable_if_t<cudf::is_duration<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    auto duration_string = cudf::io::detail::csv::pandas_format_durations(column, stream_, mr_);
    auto quotes = make_column_from_scalar(string_scalar{"\""}, column.size(), stream_, mr_);
    return cudf::strings::detail::concatenate(
      table_view{{quotes->view(), duration_string->view(), quotes->view()}},
      string_scalar(""),
      string_scalar("", false),
      strings::separator_on_nulls::YES,
      stream_,
      mr_);
  }

  // lists:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::list_view>, std::unique_ptr<column>>
  operator()(column_view const& column, host_span<column_name_info const> children_names) const
  {
    auto child_view            = lists_column_view(column).get_sliced_child(stream_);
    auto constexpr child_index = lists_column_view::child_column_index;

    auto child_string_with_null = [&]() {
      if (child_view.type().id() == type_id::STRUCT) {
        return (*this).template operator()<cudf::struct_view>(
          child_view,
          children_names.size() > child_index ? children_names[child_index].children
                                              : std::vector<column_name_info>{});
      } else if (child_view.type().id() == type_id::LIST) {
        return (*this).template operator()<cudf::list_view>(child_view,
                                                            children_names.size() > child_index
                                                              ? children_names[child_index].children
                                                              : std::vector<column_name_info>{});
      } else {
        return cudf::type_dispatcher(child_view.type(), *this, child_view);
      }
    };
    auto new_offsets = cudf::lists::detail::get_normalized_offsets(
      lists_column_view(column), stream_, rmm::mr::get_current_device_resource());
    auto const list_child_string = make_lists_column(
      column.size(),
      std::move(new_offsets),
      std::move(child_string_with_null()),
      column.null_count(),
      cudf::detail::copy_bitmask(column, stream_, rmm::mr::get_current_device_resource()),
      stream_);
    auto l_pre = string_scalar{"["}, l_post = string_scalar{"]"}, sep = string_scalar{","},
         elem_narep = string_scalar{"null"};
    return join_list_of_strings(lists_column_view(*list_child_string),
                                l_pre.value(stream_),
                                l_post.value(stream_),
                                sep.value(stream_),
                                elem_narep.value(stream_),
                                stream_,
                                mr_);
  }

  // structs:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::struct_view>, std::unique_ptr<column>>
  operator()(column_view const& column, host_span<column_name_info const> children_names) const
  {
    auto const child_it = cudf::detail::make_counting_transform_iterator(
      0, [structs_view = structs_column_view{column}](auto const child_idx) {
        return structs_view.get_sliced_child(child_idx);
      });
    auto col_string = operator()(
      child_it, child_it + column.num_children(), children_names, row_end_wrap.value(stream_));
    col_string->set_null_mask(cudf::detail::copy_bitmask(column, stream_, mr_),
                              column.null_count());
    return col_string;
  }

  // Table:
  template <typename column_iterator>
  std::unique_ptr<column> operator()(column_iterator column_begin,
                                     column_iterator column_end,
                                     host_span<column_name_info const> children_names,
                                     cudf::string_view const row_end_wrap_value) const
  {
    auto const num_columns = std::distance(column_begin, column_end);
    auto column_names      = make_column_names_column(children_names, num_columns, stream_);
    auto column_names_view = column_names->view();
    std::vector<std::unique_ptr<cudf::column>> str_column_vec;

    // populate vector of string-converted columns:
    //
    auto i_col_begin =
      thrust::make_zip_iterator(thrust::counting_iterator<size_t>(0), column_begin);
    std::transform(i_col_begin,
                   i_col_begin + num_columns,
                   std::back_inserter(str_column_vec),
                   [this, &children_names](auto const& i_current_col) {
                     auto const i            = thrust::get<0>(i_current_col);
                     auto const& current_col = thrust::get<1>(i_current_col);
                     // Struct needs children's column names
                     if (current_col.type().id() == type_id::STRUCT) {
                       return (*this).template operator()<cudf::struct_view>(
                         current_col,
                         children_names.size() > i ? children_names[i].children
                                                   : std::vector<column_name_info>{});
                     } else if (current_col.type().id() == type_id::LIST) {
                       return (*this).template operator()<cudf::list_view>(
                         current_col,
                         children_names.size() > i ? children_names[i].children
                                                   : std::vector<column_name_info>{});
                     } else {
                       return cudf::type_dispatcher(current_col.type(), *this, current_col);
                     }
                   });

    // create string table view from str_column_vec:
    //
    auto str_table_ptr  = std::make_unique<cudf::table>(std::move(str_column_vec));
    auto str_table_view = str_table_ptr->view();

    // concatenate columns in each row into one big string column
    // (using null representation and delimiter):
    //
    return struct_to_strings(str_table_view,
                             column_names_view,
                             row_begin_wrap.value(stream_),
                             row_end_wrap_value,
                             column_seperator.value(stream_),
                             value_seperator.value(stream_),
                             narep,
                             options_.is_enabled_include_nulls(),
                             stream_,
                             rmm::mr::get_current_device_resource());
  }

 private:
  json_writer_options const& options_;
  rmm::cuda_stream_view stream_;
  rmm::mr::device_memory_resource* mr_;
  string_scalar const column_seperator{":"};
  string_scalar const value_seperator{","};
  string_scalar const row_begin_wrap{"{"};
  string_scalar const row_end_wrap{"}"};
  string_scalar const narep;
};

}  // namespace

std::unique_ptr<column> make_strings_column_from_host(host_span<std::string const> host_strings,
                                                      rmm::cuda_stream_view stream)
{
  std::string const host_chars =
    std::accumulate(host_strings.begin(), host_strings.end(), std::string(""));
  auto d_chars = cudf::detail::make_device_uvector_async(
    host_chars, stream, rmm::mr::get_current_device_resource());
  std::vector<cudf::size_type> offsets(host_strings.size() + 1, 0);
  std::transform_inclusive_scan(host_strings.begin(),
                                host_strings.end(),
                                offsets.begin() + 1,
                                std::plus<cudf::size_type>{},
                                [](auto& str) { return str.size(); });
  auto d_offsets =
    cudf::detail::make_device_uvector_sync(offsets, stream, rmm::mr::get_current_device_resource());
  return cudf::make_strings_column(
    host_strings.size(), std::move(d_offsets), std::move(d_chars), {}, 0);
}

std::unique_ptr<column> make_column_names_column(host_span<column_name_info const> column_names,
                                                 size_type num_columns,
                                                 rmm::cuda_stream_view stream)
{
  std::vector<std::string> unescaped_column_names;
  if (column_names.empty()) {
    std::generate_n(std::back_inserter(unescaped_column_names), num_columns, [v = 0]() mutable {
      return std::to_string(v++);
    });
  } else {
    std::transform(column_names.begin(),
                   column_names.end(),
                   std::back_inserter(unescaped_column_names),
                   [](column_name_info const& name_info) { return name_info.name; });
  }
  auto unescaped_string_col = make_strings_column_from_host(unescaped_column_names, stream);
  auto d_column             = column_device_view::create(*unescaped_string_col, stream);
  return escape_strings_fn{*d_column, true}.get_escaped_strings(
    *unescaped_string_col, stream, rmm::mr::get_current_device_resource());
}

void write_chunked(data_sink* out_sink,
                   strings_column_view const& str_column_view,
                   int const skip_last_chars,
                   json_writer_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(str_column_view.size() > 0, "Unexpected empty strings column.");

  auto const total_num_bytes = str_column_view.chars_size() - skip_last_chars;
  char const* ptr_all_bytes  = str_column_view.chars_begin();

  if (out_sink->is_device_write_preferred(total_num_bytes)) {
    // Direct write from device memory
    out_sink->device_write(ptr_all_bytes, total_num_bytes, stream);
  } else {
    // copy the bytes to host to write them out
    auto const h_bytes = cudf::detail::make_host_vector_sync(
      device_span<char const>(ptr_all_bytes, total_num_bytes), stream);

    out_sink->host_write(h_bytes.data(), total_num_bytes);
  }
}

void write_json(data_sink* out_sink,
                table_view const& table,
                json_writer_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  std::vector<column_name_info> user_column_names = [&]() {
    auto const& metadata = options.get_metadata();
    if (metadata.has_value() and not metadata->schema_info.empty()) {
      return metadata->schema_info;
    } else {
      std::vector<column_name_info> names;
      // generate strings 0 to table.num_columns()
      std::transform(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(table.num_columns()),
                     std::back_inserter(names),
                     [](auto i) { return column_name_info{std::to_string(i)}; });
      return names;
    }
  }();
  auto const line_terminator = std::string(options.is_enabled_lines() ? "\n" : ",");
  string_scalar d_line_terminator_with_row_end{"}" + line_terminator};
  string_scalar d_line_terminator{line_terminator};

  // write header: required for non-record oriented output
  // header varies depending on orient.
  // write_chunked_begin(out_sink, table, user_column_names, options, stream, mr);
  // TODO This should go into the write_chunked_begin function
  std::string const list_braces{"[]"};
  string_scalar d_list_braces{list_braces};
  if (!options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_list_braces.data(), 1, stream);
    } else {
      out_sink->host_write(list_braces.data(), 1);
    }
  }

  if (table.num_rows() > 0) {
    auto n_rows_per_chunk = options.get_rows_per_chunk();

    // This outputs the JSON in row chunks to save memory.
    // Maybe we can use the total_rows*count calculation and a memory threshold
    // instead of an arbitrary chunk count.
    // The entire JSON chunk must fit in CPU memory before writing it out.
    //
    if (n_rows_per_chunk % 8)  // must be divisible by 8
      n_rows_per_chunk += 8 - (n_rows_per_chunk % 8);

    CUDF_EXPECTS(n_rows_per_chunk >= 8, "write_json: invalid chunk_rows; must be at least 8");

    auto num_rows = table.num_rows();
    std::vector<table_view> vector_views;

    if (num_rows <= n_rows_per_chunk) {
      vector_views.push_back(table);
    } else {
      auto const n_chunks = num_rows / n_rows_per_chunk;
      std::vector<size_type> splits(n_chunks);
      thrust::tabulate(splits.begin(), splits.end(), [n_rows_per_chunk](auto idx) {
        return (idx + 1) * n_rows_per_chunk;
      });

      // split table_view into chunks:
      vector_views = cudf::detail::split(table, splits, stream);
    }

    // convert each chunk to JSON:
    column_to_strings_fn converter{options, stream, rmm::mr::get_current_device_resource()};

    for (auto&& sub_view : vector_views) {
      // Skip if the table has no rows
      if (sub_view.num_rows() == 0) continue;
      std::vector<std::unique_ptr<column>> str_column_vec;

      // struct converter for the table
      auto str_concat_col = converter(sub_view.begin(),
                                      sub_view.end(),
                                      user_column_names,
                                      d_line_terminator_with_row_end.value(stream));

      // Needs line_terminator at the end, to separate from next chunk
      bool const include_line_terminator =
        (&sub_view != &vector_views.back()) or options.is_enabled_lines();
      auto const skip_last_chars = (include_line_terminator ? 0 : line_terminator.size());
      write_chunked(out_sink, str_concat_col->view(), skip_last_chars, options, stream, mr);
    }
  } else {
    if (options.is_enabled_lines()) {
      if (out_sink->is_device_write_preferred(1)) {
        out_sink->device_write(d_line_terminator.data(), d_line_terminator.size(), stream);
      } else {
        out_sink->host_write(line_terminator.data(), line_terminator.size());
      }
    }
  }
  // TODO write_chunked_end(out_sink, options, stream, mr);
  if (!options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_list_braces.data() + 1, 1, stream);
    } else {
      out_sink->host_write(list_braces.data() + 1, 1);
    }
  }
}

}  // namespace cudf::io::json::detail
