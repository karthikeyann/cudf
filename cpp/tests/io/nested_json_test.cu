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

#include "cudf_test/column_wrapper.hpp"

#include <io/json/nested_json.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace nested_json = cudf::io::json;

namespace {
// Forward declaration
void print_column(std::string const& input,
                  nested_json::json_column const& column,
                  uint32_t indent = 0);

/**
 * @brief Helper to generate indentation
 */
std::string pad(uint32_t indent = 0)
{
  std::string pad{};
  if (indent > 0) pad.insert(pad.begin(), indent, ' ');
  return pad;
}

/**
 * @brief Prints a string column.
 */
void print_json_string_col(std::string const& input,
                           nested_json::json_column const& column,
                           uint32_t indent = 0)
{
  for (std::size_t i = 0; i < column.string_offsets.size(); i++) {
    std::cout << pad(indent) << i << ": [" << (column.validity[i] ? "1" : "0") << "] '"
              << input.substr(column.string_offsets[i], column.string_lengths[i]) << "'\n";
  }
}

/**
 * @brief Prints a list column.
 */
void print_json_list_col(std::string const& input,
                         nested_json::json_column const& column,
                         uint32_t indent = 0)
{
  std::cout << pad(indent) << " [LIST]\n";
  std::cout << pad(indent) << " -> num. child-columns: " << column.child_columns.size() << "\n";
  std::cout << pad(indent) << " -> num. rows: " << column.current_offset << "\n";
  std::cout << pad(indent) << " -> num. valid: " << column.valid_count << "\n";
  std::cout << pad(indent) << " offsets[]: "
            << "\n";
  for (std::size_t i = 0; i < column.child_offsets.size() - 1; i++) {
    std::cout << pad(indent + 2) << i << ": [" << (column.validity[i] ? "1" : "0") << "] ["
              << column.child_offsets[i] << ", " << column.child_offsets[i + 1] << ")\n";
  }
  if (column.child_columns.size() > 0) {
    std::cout << pad(indent) << column.child_columns.begin()->first << "[]: "
              << "\n";
    print_column(input, column.child_columns.begin()->second, indent + 2);
  }
}

/**
 * @brief Prints a struct column.
 */
void print_json_struct_col(std::string const& input,
                           nested_json::json_column const& column,
                           uint32_t indent = 0)
{
  std::cout << pad(indent) << " [STRUCT]\n";
  std::cout << pad(indent) << " -> num. child-columns: " << column.child_columns.size() << "\n";
  std::cout << pad(indent) << " -> num. rows: " << column.current_offset << "\n";
  std::cout << pad(indent) << " -> num. valid: " << column.valid_count << "\n";
  std::cout << pad(indent) << " -> validity[]: "
            << "\n";
  for (std::size_t i = 0; i < column.current_offset; i++) {
    std::cout << pad(indent + 2) << i << ": [" << (column.validity[i] ? "1" : "0") << "]\n";
  }
  auto it = std::begin(column.child_columns);
  for (std::size_t i = 0; i < column.child_columns.size(); i++) {
    std::cout << pad(indent + 2) << "child #" << i << " '" << it->first << "'[] \n";
    print_column(input, it->second, indent + 2);
    it++;
  }
}

/**
 * @brief Prints the column's data and recurses through and prints all the child columns.
 */
void print_column(std::string const& input, nested_json::json_column const& column, uint32_t indent)
{
  switch (column.type) {
    case nested_json::json_col_t::StringColumn: print_json_string_col(input, column, indent); break;
    case nested_json::json_col_t::ListColumn: print_json_list_col(input, column, indent); break;
    case nested_json::json_col_t::StructColumn: print_json_struct_col(input, column, indent); break;
    case nested_json::json_col_t::Unknown: std::cout << pad(indent) << "[UNKNOWN]\n"; break;
    default: break;
  }
}
}  // namespace

// Base test fixture for tests
struct JsonTest : public cudf::test::BaseFixture {
};

TEST_F(JsonTest, StackContext)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

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
                      R"("title": "{}\\\"[], <=semantic-symbols-string\\\\",)"
                      R"("price": 8.95)"
                      R"(}] )";

  // Prepare input & output buffers
  rmm::device_uvector<SymbolT> d_input(input.size(), stream_view);
  hostdevice_vector<StackSymbolT> stack_context(input.size(), stream_view);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  // Run algorithm
  nested_json::detail::get_stack_context(d_input, stack_context.device_ptr(), stream_view);

  // Copy back the results
  stack_context.device_to_host(stream_view);

  // Make sure we copied back the stack context
  stream_view.synchronize();

  std::vector<char> golden_stack_context{
    '_', '_', '_', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '[', '[', '[', '[', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '{', '[', '[', '[', '[', '[', '[', '[', '{',
    '{', '{', '{', '{', '[', '{', '{', '[', '[', '[', '{', '[', '{', '{', '[', '[', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '[', '_'};

  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(golden_stack_context, stack_context, stack_context.size());
}

TEST_F(JsonTest, TokenStream)
{
  using nested_json::PdaTokenT;
  using nested_json::SymbolOffsetT;
  using nested_json::SymbolT;

  constexpr std::size_t single_item = 1;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

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

  // Prepare input & output buffers
  rmm::device_uvector<SymbolT> d_input(input.size(), stream_view);

  ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_input.data(),
                                        input.data(),
                                        input.size() * sizeof(SymbolT),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream_view};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream_view};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream_view};

  // Parse the JSON and get the token stream
  nested_json::detail::get_token_stream(d_input,
                                        tokens_gpu.device_ptr(),
                                        token_indices_gpu.device_ptr(),
                                        num_tokens_out.device_ptr(),
                                        stream_view);

  // Copy back the number of tokens that were written
  num_tokens_out.device_to_host(stream_view);
  tokens_gpu.device_to_host(stream_view);
  token_indices_gpu.device_to_host(stream_view);

  // Make sure we copied back all relevant data
  stream_view.synchronize();

  // Golden token stream sample
  using token_t = nested_json::token_t;
  std::vector<std::pair<std::size_t, nested_json::PdaTokenT>> golden_token_stream = {
    {2, token_t::ListBegin},        {3, token_t::StructBegin},      {4, token_t::FieldNameBegin},
    {13, token_t::FieldNameEnd},    {16, token_t::StringBegin},     {26, token_t::StringEnd},
    {28, token_t::FieldNameBegin},  {35, token_t::FieldNameEnd},    {38, token_t::ListBegin},
    {39, token_t::ValueBegin},      {40, token_t::ValueEnd},        {41, token_t::ValueBegin},
    {43, token_t::ValueEnd},        {44, token_t::ValueBegin},      {46, token_t::ValueEnd},
    {46, token_t::ListEnd},         {48, token_t::FieldNameBegin},  {55, token_t::FieldNameEnd},
    {58, token_t::StringBegin},     {69, token_t::StringEnd},       {71, token_t::FieldNameBegin},
    {77, token_t::FieldNameEnd},    {80, token_t::StringBegin},     {105, token_t::StringEnd},
    {107, token_t::FieldNameBegin}, {113, token_t::FieldNameEnd},   {116, token_t::ValueBegin},
    {120, token_t::ValueEnd},       {120, token_t::StructEnd},      {124, token_t::StructBegin},
    {125, token_t::FieldNameBegin}, {134, token_t::FieldNameEnd},   {137, token_t::StringBegin},
    {147, token_t::StringEnd},      {149, token_t::FieldNameBegin}, {155, token_t::FieldNameEnd},
    {158, token_t::ListBegin},      {159, token_t::ValueBegin},     {160, token_t::ValueEnd},
    {161, token_t::StructBegin},    {162, token_t::StructEnd},      {164, token_t::ValueBegin},
    {168, token_t::ValueEnd},       {169, token_t::StructBegin},    {170, token_t::FieldNameBegin},
    {172, token_t::FieldNameEnd},   {174, token_t::ListBegin},      {175, token_t::StructBegin},
    {177, token_t::StructEnd},      {180, token_t::StructBegin},    {181, token_t::StructEnd},
    {182, token_t::ListEnd},        {184, token_t::StructEnd},      {186, token_t::ListEnd},
    {188, token_t::FieldNameBegin}, {195, token_t::FieldNameEnd},   {198, token_t::StringBegin},
    {209, token_t::StringEnd},      {211, token_t::FieldNameBegin}, {217, token_t::FieldNameEnd},
    {220, token_t::StringBegin},    {252, token_t::StringEnd},      {254, token_t::FieldNameBegin},
    {260, token_t::FieldNameEnd},   {263, token_t::ValueBegin},     {267, token_t::ValueEnd},
    {267, token_t::StructEnd},      {268, token_t::ListEnd}};

  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), num_tokens_out[0]);

  for (std::size_t i = 0; i < num_tokens_out[0]; i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;

    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

TEST_F(JsonTest, Simple)
{
  using nested_json::PdaTokenT;
  using nested_json::SymbolOffsetT;
  using nested_json::SymbolT;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  //
  // std::string input = R"( ["foo", null, "bar"] )";
  // std::string input = R"( [{"a":0.0, "c":{"c0":"0.2.0"}}, {"b":1.1}] )";
  // std::string input = R"( [{"a":0.0}, {"b":1.1, "c":{"c0":"1.2.0"}}] )";
  std::string input = R"( [{"a":0.0}, {"b":1.1, "c":{"c0":[[1],null,[2]]}}] )";
  // std::string input =
  // R"( [{ "col0":[{"field1": 1, "field2": 2 }, null, {"field1": 3, "field2": 4 }, {"field1": 5,
  // "field2": 6 }], "col1":"foo" }] )";
  // std::string input = R"( [ {"col1": 1, "col2": 2 }, {"col1": 3, "col2": 4 }, {"col1": 5,
  // "col2": 6 }] )"; std::string input = R"( [ {"col1": 1, "col2": 2 }, null, {"col1": 3, "col2":
  // 4 },
  // {"col1": 5, "col2": 6 }] )"; std::string input = R"( [[1], [2], null, [3], [4]] )";

  // String / value
  // std::string input = R"( " Foobar" )";
  // std::string input = R"(  123.456  )";
  // std::string input = R"(  123.456)";
  // std::string input = R"(null)";
  // std::string input = R"( [null, [2], null, [3], [4]] )"; // <= will fail because col will be
  // inferred as string/val column

  // Allocate device memory for the JSON input & copy over to device
  rmm::device_uvector<SymbolT> d_input{input.size(), stream_view};
  cudaMemcpyAsync(d_input.data(),
                  input.data(),
                  input.size() * sizeof(input[0]),
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Get the JSON's tree representation
  auto json_root_col = nested_json::detail::get_json_columns(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, d_input, stream_view);

  std::cout << input << "\n";
  print_column(input, json_root_col);
}

TEST_F(JsonTest, ExtractColumn)
{
  using nested_json::PdaTokenT;
  using nested_json::SymbolOffsetT;
  using nested_json::SymbolT;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  std::string input = R"( [{"a":0.0, "b":1.0}, {"a":0.1, "b":1.1}, {"a":0.2, "b":1.2}] )";
  // Get the JSON's tree representation
  auto cudf_column = nested_json::detail::parse_json_to_columns(
    cudf::host_span<SymbolT const>{input.data(), input.size()}, stream_view);

  std::cout << std::endl << "=== PARSED COLUMN ===" << std::endl;
  cudf::test::print(*cudf_column);
  cudf::column_view cudf_struct_view =
    cudf_column->child(cudf::lists_column_view::child_column_index);

  auto expected_col1            = cudf::test::strings_column_wrapper({"0.0", "0.1", "0.2"});
  auto expected_col2            = cudf::test::strings_column_wrapper({"1.0", "1.1", "1.2"});
  cudf::column_view parsed_col1 = cudf_struct_view.child(0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col1, parsed_col1);
  std::cout << "*parsed_col1:\n";
  cudf::test::print(parsed_col1);
  cudf::column_view parsed_col2 = cudf_struct_view.child(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col2, parsed_col2);
  std::cout << "*parsed_col2:\n";
  cudf::test::print(parsed_col2);
}
