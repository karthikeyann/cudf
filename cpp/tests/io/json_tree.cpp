/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <io/json/nested_json.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace cuio_json = cudf::io::json;

namespace cudf::io::json {
// Host copy of tree_meta_t
struct tree_meta_t2 {
  std::vector<NodeT> node_categories;
  std::vector<NodeIndexT> parent_node_ids;
  std::vector<TreeDepthT> node_levels;
  std::vector<SymbolOffsetT> node_range_begin;
  std::vector<SymbolOffsetT> node_range_end;
};
}  // namespace cudf::io::json

namespace {
std::string get_node_string(std::size_t const node_id,
                            cuio_json::tree_meta_t2 const& tree_rep,
                            std::string const& json_input)
{
  auto node_to_str = [] __host__ __device__(cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::NC_STRUCT: return "STRUCT";
      case cuio_json::NC_LIST: return "LIST";
      case cuio_json::NC_FN: return "FN";
      case cuio_json::NC_STR: return "STR";
      case cuio_json::NC_VAL: return "VAL";
      case cuio_json::NC_ERR: return "ERR";
      default: return "N/A";
    };
  };

  return "<" + std::to_string(node_id) + ":" + node_to_str(tree_rep.node_categories[node_id]) +
         ":[" + std::to_string(tree_rep.node_range_begin[node_id]) + ", " +
         std::to_string(tree_rep.node_range_end[node_id]) + ") '" +
         json_input.substr(tree_rep.node_range_begin[node_id],
                           tree_rep.node_range_end[node_id] - tree_rep.node_range_begin[node_id]) +
         "'>";
}

void print_tree_representation(std::string const& json_input,
                               cuio_json::tree_meta_t2 const& tree_rep)
{
  for (std::size_t i = 0; i < tree_rep.node_categories.size(); i++) {
    std::size_t parent_id = tree_rep.parent_node_ids[i];
    std::stack<std::size_t> path;
    path.push(i);
    while (parent_id != cuio_json::parent_node_sentinel) {
      path.push(parent_id);
      parent_id = tree_rep.parent_node_ids[parent_id];
    }

    while (path.size()) {
      auto const node_id = path.top();
      std::cout << get_node_string(node_id, tree_rep, json_input)
                << (path.size() > 1 ? " -> " : "");
      path.pop();
    }
    std::cout << "\n";
  }
}
}  // namespace

// cudf::io::json::
namespace cudf::io::json {
namespace test {

tree_meta_t2 to_cpu_tree(tree_meta_t const& d_value, rmm::cuda_stream_view stream)
{
  return {cudf::detail::make_std_vector_async(d_value.node_categories, stream),
          cudf::detail::make_std_vector_async(d_value.parent_node_ids, stream),
          cudf::detail::make_std_vector_async(d_value.node_levels, stream),
          cudf::detail::make_std_vector_async(d_value.node_range_begin, stream),
          cudf::detail::make_std_vector_async(d_value.node_range_end, stream)};
}

void compare_trees(tree_meta_t2 const& cpu_tree, tree_meta_t const& d_gpu_tree)
{
  auto gpu_tree = to_cpu_tree(d_gpu_tree, cudf::default_stream_value);
  // DEBUG prints
  auto to_cat = [](auto v) -> std::string {
    switch (v) {
      case NC_STRUCT: return " S";
      case NC_LIST: return " L";
      case NC_STR: return " \"";
      case NC_VAL: return " V";
      case NC_FN: return " F";
      case NC_ERR: return "ER";
      default: return "UN";
    };
  };
  auto to_int    = [](auto v) { return std::to_string(static_cast<int>(v)); };
  bool mismatch  = false;
  auto print_vec = [&](auto const& cpu, auto const& gpu, auto const name, auto converter) {
    if (not cpu.empty()) {
      for (auto const& v : cpu)
        printf("%3s,", converter(v).c_str());
      std::cout << name << "(CPU):" << std::endl;
    }
    if (not cpu.empty()) {
      for (auto const& v : gpu)
        printf("%3s,", converter(v).c_str());
      std::cout << name << "(GPU):" << std::endl;
    }
    if (not cpu.empty() and not gpu.empty()) {
      if (!std::equal(gpu.begin(), gpu.end(), cpu.begin())) {
        for (auto i = 0lu; i < cpu.size(); i++) {
          mismatch |= (gpu[i] != cpu[i]);
          printf("%3s,", (gpu[i] == cpu[i] ? " " : "x"));
        }
        std::cout << std::endl;
      }
    }
  };
#define PRINT_VEC(vec) print_vec(cpu_tree.vec, gpu_tree.vec, #vec, to_int);
  for (int i = 0; i < int(gpu_tree.node_categories.size()); i++)
    printf("%3d,", i);
  printf(" node_id\n");
  print_vec(
    cpu_tree.node_categories, gpu_tree.node_categories, "node_categories", to_cat);  // Works
  PRINT_VEC(node_levels);                                                            // Works
  // PRINT_VEC(node_range_begin);  // Works
  // PRINT_VEC(node_range_end);    // Works
  PRINT_VEC(parent_node_ids);  // Works
  CUDF_EXPECTS(!mismatch, "Mismatch in GPU and CPU tree representation");
  // std::cout << "Mismatch: " << mismatch << std::endl;
#undef PRINT_VEC
}

tree_meta_t2 get_tree_representation_cpu(host_span<SymbolT const> input,
                                         rmm::cuda_stream_view stream)
{
  constexpr std::size_t single_item = 1;
  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream};

  rmm::device_uvector<SymbolT> d_input{input.size(), stream};
  cudaMemcpyAsync(
    d_input.data(), input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice, stream);

  // Parse the JSON and get the token stream
  cudf::io::json::detail::get_token_stream(
    cudf::device_span<SymbolT>{d_input.data(), d_input.size()},
    tokens_gpu.device_ptr(),
    token_indices_gpu.device_ptr(),
    num_tokens_out.device_ptr(),
    stream);

  // Copy the JSON tokens to the host
  token_indices_gpu.device_to_host(stream);
  tokens_gpu.device_to_host(stream);
  num_tokens_out.device_to_host(stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

  // DEBUG print
  [[maybe_unused]] auto to_token_str = [](PdaTokenT token) {
    switch (token) {
      case token_t::StructBegin: return " {";
      case token_t::StructEnd: return " }";
      case token_t::ListBegin: return " [";
      case token_t::ListEnd: return " ]";
      case token_t::FieldNameBegin: return "FB";
      case token_t::FieldNameEnd: return "FE";
      case token_t::StringBegin: return "SB";
      case token_t::StringEnd: return "SE";
      case token_t::ErrorBegin: return "er";
      case token_t::ValueBegin: return "VB";
      case token_t::ValueEnd: return "VE";
      case token_t::StructMemberBegin: return " <";
      case token_t::StructMemberEnd: return " >";
      default: return ".";
    }
  };
  std::cout << "Tokens: \n";
  for (auto i = 0u; i < num_tokens_out[0]; i++) {
    std::cout << to_token_str(tokens_gpu[i]) << " ";
  }
  std::cout << std::endl;

  // Whether a token does represent a node in the tree representation
  auto is_node = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // The node that a token represents
  auto token_to_node = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin: return NC_STRUCT;
      case token_t::ListBegin: return NC_LIST;
      case token_t::StringBegin: return NC_STR;
      case token_t::ValueBegin: return NC_VAL;
      case token_t::FieldNameBegin: return NC_FN;
      default: return NC_ERR;
    };
  };

  auto get_token_index = [](PdaTokenT const token, SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT skip_quote_char = 1;
    switch (token) {
      case token_t::StringBegin: return token_index + skip_quote_char;
      case token_t::FieldNameBegin: return token_index + skip_quote_char;
      default: return token_index;
    };
  };

  // Whether a token expects to be followed by its respective end-of-* token partner
  auto is_begin_of_section = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin: return true;
      default: return false;
    };
  };

  // The end-of-* partner token for a given beginning-of-* token
  auto end_of_partner = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin: return token_t::StringEnd;
      case token_t::ValueBegin: return token_t::ValueEnd;
      case token_t::FieldNameBegin: return token_t::FieldNameEnd;
      default: return token_t::ErrorBegin;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  // The node id sitting on top of the stack becomes the node's parent
  // The full stack represents the path from the root to the current node
  std::stack<std::pair<NodeIndexT, bool>> parent_stack;

  constexpr bool field_name_node    = true;
  constexpr bool no_field_name_node = false;

  std::vector<NodeT> node_categories;
  std::vector<NodeIndexT> parent_node_ids;
  std::vector<TreeDepthT> node_levels;
  std::vector<SymbolOffsetT> node_range_begin;
  std::vector<SymbolOffsetT> node_range_end;

  std::size_t node_id = 0;
  for (std::size_t i = 0; i < num_tokens_out[0]; i++) {
    auto token = tokens_gpu[i];

    // The section from the original JSON input that this token demarcates
    std::size_t range_begin = get_token_index(token, token_indices_gpu[i]);
    std::size_t range_end   = range_begin + 1;

    // Identify this node's parent node id
    std::size_t parent_node_id =
      (parent_stack.size() > 0) ? parent_stack.top().first : parent_node_sentinel;

    // If this token is the beginning-of-{value, string, field name}, also consume the next end-of-*
    // token
    if (is_begin_of_section(token)) {
      if ((i + 1) < num_tokens_out[0] && end_of_partner(token) == tokens_gpu[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = token_indices_gpu[i + 1];
        // We can skip the subsequent end-of-* token
        i++;
      }
    }

    // Emit node if this token becomes a node in the tree
    if (is_node(token)) {
      node_categories.push_back(token_to_node(token));
      parent_node_ids.push_back(parent_node_id);
      node_levels.push_back(parent_stack.size());
      node_range_begin.push_back(range_begin);
      node_range_end.push_back(range_end);
    }

    // Modify the stack if needed
    if (token == token_t::FieldNameBegin) {
      parent_stack.push({node_id, field_name_node});
    } else {
      if (does_push(token)) {
        parent_stack.push({node_id, no_field_name_node});
      } else if (does_pop(token)) {
        CUDF_EXPECTS(parent_stack.size() >= 1, "Invalid JSON input.");
        parent_stack.pop();
      }

      // If what we're left with is a field name on top of stack, we need to pop it
      if (parent_stack.size() >= 1 && parent_stack.top().second == field_name_node) {
        parent_stack.pop();
      }
    }

    // Update node_id
    if (is_node(token)) { node_id++; }
  }

  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

}  // namespace test
}  // namespace cudf::io::json

// Base test fixture for tests
struct JsonTest : public cudf::test::BaseFixture {
};

TEST_F(JsonTest, TreeRepresentation)
{
  auto stream = cudf::default_stream_value;

  // Test input
  std::string input = R"(  [{)"
                      R"("category": "reference",)"
                      R"("index:": [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "[Sayings of the Century]",)"
                      R"("price": 8.95)"
                      R"(},  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index": [4,{},null,{"a":[{ }, {}] } ],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "{}[], <=semantic-symbols-string",)"
                      R"("price": 8.95)"
                      R"(}] )";
  cudf::string_scalar d_input(input, true, stream);

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    {d_input.data(), static_cast<size_t>(d_input.size())}, stream);
  // host tree generation
  auto tree_rep = cuio_json::test::get_tree_representation_cpu(input, stream);
  cudf::io::json::test::compare_trees(tree_rep, gpu_tree);

  // Print tree representation
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) { print_tree_representation(input, tree_rep); }

  // Golden sample of node categories
  std::vector<cuio_json::node_t> golden_node_categories = {
    cuio_json::NC_LIST, cuio_json::NC_STRUCT, cuio_json::NC_FN,     cuio_json::NC_STR,
    cuio_json::NC_FN,   cuio_json::NC_LIST,   cuio_json::NC_VAL,    cuio_json::NC_VAL,
    cuio_json::NC_VAL,  cuio_json::NC_FN,     cuio_json::NC_STR,    cuio_json::NC_FN,
    cuio_json::NC_STR,  cuio_json::NC_FN,     cuio_json::NC_VAL,    cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_STR,    cuio_json::NC_FN,     cuio_json::NC_LIST,
    cuio_json::NC_VAL,  cuio_json::NC_STRUCT, cuio_json::NC_VAL,    cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_LIST,   cuio_json::NC_STRUCT, cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_STR,    cuio_json::NC_FN,     cuio_json::NC_STR,
    cuio_json::NC_FN,   cuio_json::NC_VAL};

  // Golden sample of node ids
  // clang-format off
  std::vector<cuio_json::NodeIndexT> golden_parent_node_ids = {
    cuio_json::parent_node_sentinel, 0, 1, 2,
    1, 4, 5, 5,
    5, 1, 9, 1,
    11, 1, 13, 0,
    15, 16, 15, 18,
    19, 19, 19, 19,
    23, 24, 25, 25,
    15, 28, 15, 30,
    15, 32};
  // clang-format on

  // Golden sample of node levels
  std::vector<cuio_json::TreeDepthT> golden_node_levels = {0, 1, 2, 3, 2, 3, 4, 4, 4, 2, 3, 2,
                                                           3, 2, 3, 1, 2, 3, 2, 3, 4, 4, 4, 4,
                                                           5, 6, 7, 7, 2, 3, 2, 3, 2, 3};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {
    2,   3,   5,   17,  29,  38,  39,  41,  44,  49,  59,  72,  81,  108, 116, 124, 126,
    138, 150, 158, 159, 161, 164, 169, 171, 174, 175, 180, 189, 199, 212, 221, 255, 263};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {
    3,   4,   13,  26,  35,  39,  40,  43,  46,  55,  69,  77,  105, 113, 120, 125, 134,
    147, 155, 159, 160, 162, 168, 170, 172, 175, 176, 181, 195, 209, 217, 252, 260, 267};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), tree_rep.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), tree_rep.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), tree_rep.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), tree_rep.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), tree_rep.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], tree_rep.node_categories[i]);
    ASSERT_EQ(golden_parent_node_ids[i], tree_rep.parent_node_ids[i]);
    ASSERT_EQ(golden_node_levels[i], tree_rep.node_levels[i]);
    ASSERT_EQ(golden_node_range_begin[i], tree_rep.node_range_begin[i]);
    ASSERT_EQ(golden_node_range_end[i], tree_rep.node_range_end[i]);
  }
}

TEST_F(JsonTest, TreeRepresentation2)
{
  auto stream = cudf::default_stream_value;
  // Test input: value end with comma, space, close-brace ", }"
  std::string input =
    //  0         1         2         3         4         5         6         7         8         9
    //  0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
    R"([ {}, { "a": { "y" : 6, "z": [] }}, { "a" : { "x" : 8, "y": 9}, "b" : {"x": 10 , "z": 11}}])";
  cudf::string_scalar d_input(input, true, stream);

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    {d_input.data(), static_cast<size_t>(d_input.size())}, stream);
  // host tree generation
  auto tree_rep = cuio_json::test::get_tree_representation_cpu(input, stream);
  cudf::io::json::test::compare_trees(tree_rep, gpu_tree);

  // Print tree representation
  if (std::getenv("CUDA_DBG_DUMP") != nullptr) { print_tree_representation(input, tree_rep); }
  // TODO compare with CPU version

  // Golden sample of node categories
  // clang-format off
  std::vector<cuio_json::node_t> golden_node_categories = {
    cuio_json::NC_LIST, cuio_json::NC_STRUCT,
    cuio_json::NC_STRUCT, cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_VAL, cuio_json::NC_FN,  cuio_json::NC_LIST,
    cuio_json::NC_STRUCT, cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_VAL, cuio_json::NC_FN,  cuio_json::NC_VAL,
                          cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_VAL, cuio_json::NC_FN,  cuio_json::NC_VAL};

  // Golden sample of node ids
  std::vector<cuio_json::NodeIndexT> golden_parent_node_ids = {
    cuio_json::parent_node_sentinel, 0,
    0, 2,  3,  4,  5,  4, 7,
    0, 9, 10, 11, 12, 11, 14,
      9, 16, 17, 18, 17, 20};
  // clang-format on

  // Golden sample of node levels
  std::vector<cuio_json::TreeDepthT> golden_node_levels = {
    0, 1, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5, 2, 3, 4, 5, 4, 5,
  };

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {0,  2,  6,  9,  13, 16, 21, 25, 29, 36, 39,
                                                      44, 47, 52, 56, 60, 65, 70, 72, 76, 82, 86};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {1,  3,  7,  10, 14, 17, 22, 26, 30, 37, 40,
                                                    45, 48, 53, 57, 61, 66, 71, 73, 78, 83, 88};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), tree_rep.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), tree_rep.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), tree_rep.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), tree_rep.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), tree_rep.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], tree_rep.node_categories[i]);
    ASSERT_EQ(golden_parent_node_ids[i], tree_rep.parent_node_ids[i]);
    ASSERT_EQ(golden_node_levels[i], tree_rep.node_levels[i]);
    ASSERT_EQ(golden_node_range_begin[i], tree_rep.node_range_begin[i]);
    ASSERT_EQ(golden_node_range_end[i], tree_rep.node_range_end[i]);
  }
}
