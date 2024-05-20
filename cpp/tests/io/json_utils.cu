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

#include "cudf/io/parquet.hpp"
#include "json_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/unary.hpp>

#include <sstream>
#include <stdexcept>

namespace spark_rapids_jni {

// Convert the token value into string name, for debugging purpose.
std::string token_to_string(cudf::io::json::PdaTokenT const token_type)
{
  switch (token_type) {
    case cudf::io::json::token_t::StructBegin: return "StructBegin";
    case cudf::io::json::token_t::StructEnd: return "StructEnd";
    case cudf::io::json::token_t::ListBegin: return "ListBegin";
    case cudf::io::json::token_t::ListEnd: return "ListEnd";
    case cudf::io::json::token_t::StructMemberBegin: return "StructMemberBegin";
    case cudf::io::json::token_t::StructMemberEnd: return "StructMemberEnd";
    case cudf::io::json::token_t::FieldNameBegin: return "FieldNameBegin";
    case cudf::io::json::token_t::FieldNameEnd: return "FieldNameEnd";
    case cudf::io::json::token_t::StringBegin: return "StringBegin";
    case cudf::io::json::token_t::StringEnd: return "StringEnd";
    case cudf::io::json::token_t::ValueBegin: return "ValueBegin";
    case cudf::io::json::token_t::ValueEnd: return "ValueEnd";
    case cudf::io::json::token_t::ErrorBegin: return "ErrorBegin";
    case cudf::io::json::token_t::LineEnd: return "LineEnd";
    default: return "Unknown";
  }
}

// Print the content of the input device vector.
template <typename T, typename U = int>
void print_debug(cudf::device_span<T const> input,
                 std::string const& name,
                 std::string const& separator,
                 rmm::cuda_stream_view stream)
{
  auto const h_input = cudf::detail::make_host_vector_sync(input, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << static_cast<U>(h_input[i]);
    if (separator.size() > 0 && i + 1 < h_input.size()) { ss << separator; }
  }
  std::cerr << ss.str() << std::endl;
}

// Print the content of the input device vector.
void print_debug_tokens(rmm::device_uvector<cudf::io::json::PdaTokenT> const& tokens,
                        rmm::device_uvector<uint32_t> const& offsets,
                        cudf::device_span<char const> str_data,
                        std::string const& name,
                        std::string const& separator,
                        rmm::cuda_stream_view stream)
{
  auto const h_tokens = cudf::detail::make_host_vector_sync(
    cudf::device_span<cudf::io::json::PdaTokenT const>{tokens.data(), tokens.size()}, stream);
  auto const h_offsets = cudf::detail::make_host_vector_sync(
    cudf::device_span<uint32_t const>{offsets.data(), offsets.size()}, stream);
  auto const h_str_data = cudf::detail::make_host_vector_sync(
    cudf::device_span<char const>{str_data.data(), str_data.size()}, stream);

  std::stringstream ss;
  ss << name << ":\n";
  uint32_t str_begin = 0;
  for (size_t i = 0; i < h_tokens.size(); ++i) {
    ss << token_to_string(h_tokens[i]) << " " << h_offsets[i];
    if (h_tokens[i] == cudf::io::json::token_t::FieldNameBegin ||
        h_tokens[i] == cudf::io::json::token_t::StringBegin ||
        h_tokens[i] == cudf::io::json::token_t::ValueBegin) {
      str_begin = h_offsets[i];
    }
    if (h_tokens[i] == cudf::io::json::token_t::FieldNameEnd ||
        h_tokens[i] == cudf::io::json::token_t::StringEnd) {
      uint32_t str_end = h_offsets[i];
      // strings are inclusive, but include the quotes
      std::string d(&h_str_data[str_begin + 1], str_end - str_begin - 1);
      ss << " |" << d << "|";
    }
    if (h_tokens[i] == cudf::io::json::token_t::ValueEnd) {
      uint32_t str_end = h_offsets[i];
      // value end is not inclusive
      std::string d(&h_str_data[str_begin], str_end - str_begin);
      ss << " |" << d << "|";
    }

    if (separator.size() > 0 && i + 1 < h_tokens.size()) { ss << separator; }
  }
  std::cerr << ss.str() << std::endl;
}

std::unique_ptr<cudf::column> is_empty_or_null(cudf::column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto byte_count =
    cudf::strings::count_bytes(cudf::strings_column_view{input}, mr);  // stream not exposed yet...
  using IntScalarType = cudf::scalar_type_t<int32_t>;
  auto zero = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::INT32}, stream, mr);
  reinterpret_cast<IntScalarType*>(zero.get())->set_value(0, stream);
  zero->set_valid_async(true, stream);
  auto is_empty             = cudf::binary_operation(*byte_count,
                                         *zero,
                                         cudf::binary_operator::LESS_EQUAL,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr);
  auto is_null              = cudf::is_null(input, stream, mr);
  auto mostly_empty_or_null = cudf::binary_operation(*is_empty,
                                                     *is_null,
                                                     cudf::binary_operator::NULL_LOGICAL_OR,
                                                     cudf::data_type{cudf::type_id::BOOL8},
                                                     stream,
                                                     mr);
  is_empty.reset();
  is_null.reset();
  zero.reset();
  auto null_lit    = cudf::make_string_scalar("null", stream, mr);
  auto is_lit_null = cudf::binary_operation(*null_lit,
                                            input,
                                            cudf::binary_operator::EQUAL,
                                            cudf::data_type{cudf::type_id::BOOL8},
                                            stream,
                                            mr);
  return cudf::binary_operation(*is_lit_null,
                                *mostly_empty_or_null,
                                cudf::binary_operator::NULL_LOGICAL_OR,
                                cudf::data_type{cudf::type_id::BOOL8},
                                stream,
                                mr);
}

bool contains_char(cudf::column_view const& input,
                   std::string const& needle,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  cudf::string_scalar s(needle, true, stream, mr);
  auto has_s           = cudf::strings::contains(cudf::strings_column_view(input), s);
  auto any             = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto ret             = cudf::reduce(has_s->view(),
                          *any,
                          cudf::data_type{cudf::type_id::BOOL8},
                          stream,
                          mr);  // no stream is supported for reduce yet
  using BoolScalarType = cudf::scalar_type_t<bool>;
  return ret->is_valid(stream) && reinterpret_cast<BoolScalarType*>(ret.get())->value(stream);
}

rmm::device_uvector<cudf::io::json::SymbolT> extract_character_buffer(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  // Sadly there is no good way around this. We have to make a copy of the data...
  cudf::strings_column_view scv(input);
  auto data_length = scv.chars_size(stream);
  rmm::device_uvector<cudf::io::json::SymbolT> ret(data_length, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    ret.data(), scv.chars_begin(stream), data_length, cudaMemcpyDefault, stream.value()));
  return ret;
}

std::pair<rmm::device_uvector<cudf::io::json::SymbolT>, std::unique_ptr<cudf::column>>
clean_and_concat(cudf::column_view const& input,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const input_scv = cudf::strings_column_view{input};
  auto stripped        = cudf::strings::strip(input_scv,
                                       cudf::strings::side_type::BOTH,
                                       cudf::string_scalar("", true, stream, mr),
                                       stream,
                                       mr);
  auto is_n_or_e       = is_empty_or_null(*stripped, stream, mr);
  auto empty_row       = cudf::make_string_scalar("{}", stream, mr);
  auto cleaned         = cudf::copy_if_else(*empty_row, *stripped, *is_n_or_e, stream, mr);
  stripped.reset();
  empty_row.reset();
  if (contains_char(*cleaned, "\n", stream, mr)) {
    throw std::logic_error("line separator is not currently supported in a JSON string");
  }
  if (contains_char(*cleaned, "\r", stream, mr)) {
    throw std::logic_error("carriage return is not currently supported in a JSON string");
  }
  // Eventually we want to use null, but for now...
  auto all_done = cudf::strings::join_strings(
    cudf::strings_column_view(*cleaned),
    cudf::string_scalar("\n", true, stream, mr),
    cudf::string_scalar("{}", true, stream, mr),  // This should be ignored
    stream,
    mr);
  return std::make_pair(extract_character_buffer(*all_done, stream, mr), std::move(is_n_or_e));
}

std::unique_ptr<cudf::column> tokenize_json(cudf::column_view const& input,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  if (input.is_empty()) {
    auto tok_out    = cudf::make_empty_column(cudf::type_id::INT8);
    auto offset_out = cudf::make_empty_column(cudf::type_id::UINT32);
    std::vector<std::unique_ptr<cudf::column>> tok_off_children;
    tok_off_children.push_back(std::move(tok_out));
    tok_off_children.push_back(std::move(offset_out));
    auto tok_off_out = cudf::make_structs_column(
      0, std::move(tok_off_children), 0, rmm::device_buffer{}, stream, mr);
    auto empty_offsets = cudf::make_empty_column(cudf::type_id::INT32);
    auto tokens_out    = cudf::make_lists_column(
      0, std::move(empty_offsets), std::move(tok_off_out), 0, rmm::device_buffer{}, stream, mr);
    auto buffer_out = cudf::make_empty_column(cudf::type_id::STRING);
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(buffer_out));
    children.push_back(std::move(tokens_out));
    return cudf::make_structs_column(0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
  }

  auto [cleaned, was_empty] = clean_and_concat(input, stream, mr);
  print_debug<char, char>(cleaned, "CLEANED INPUT", "", stream);
  cudf::io::datasource::owning_buffer<rmm::device_uvector<char>> cleaned_buffer(std::move(cleaned));
  cudf::io::json::detail::normalize_single_quotes(cleaned_buffer, stream, mr);
  print_debug<unsigned char, char>(
    {cleaned_buffer.data(), cleaned_buffer.size()}, "QUOTE NORMALIZED", "", stream);
  // cleaned = cudf::io::json::detail::normalize_whitespace(std::move(cleaned), stream, mr);
  // print_debug<char, char>(cleaned, "WS NORMALIZED", "", stream);
  //  We will probably do ws normalization as we write out the data. This is true for number
  //  normalization too

  auto json_opts = cudf::io::json_reader_options_builder()
                     .lines(true)
                     .mixed_types_as_string(true)
                     .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                     .build();

  auto const [tokens, token_indices] = cudf::io::json::detail::get_token_stream(
    cudf::device_span<char const>{reinterpret_cast<char const*>(cleaned_buffer.data()),
                                  cleaned_buffer.size()},
    json_opts,
    stream,
    mr);

  print_debug_tokens(tokens,
                     token_indices,
                     {reinterpret_cast<char const*>(cleaned_buffer.data()), cleaned_buffer.size()},
                     "RAW TOKES",
                     "\n",
                     stream);

  // TODO would a tree representation be better???

  // TODO we probably want a JSON options to pass in at some point. For now we are
  // just going to hard code thigns...

  // First off we need to get all of the data into a single buffer.  In the future
  // This will use \0 nul as the separator, but for now we are going to use \n
  // and check that it is not in there...

  throw std::runtime_error("NOT IMPLEMENTED YET");
}

std::unique_ptr<cudf::column> different_get_json_object(
  cudf::column_view const& input,
  std::vector<std::tuple<diff_path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");
  CUDF_FUNC_RANGE();

  auto [cleaned, was_empty] = clean_and_concat(input, stream, mr);
  cudf::io::datasource::owning_buffer<rmm::device_uvector<char>> cleaned_buffer(std::move(cleaned));
  // print_debug<char, char>(cleaned, "CLEANED INPUT", "", stream);
  cudf::io::json::detail::normalize_single_quotes(cleaned_buffer, stream, mr);
  // print_debug<char, char>(cleaned, "QUOTE NORMALIZED", "", stream);
  // cleaned = cudf::io::json::detail::normalize_whitespace(std::move(cleaned), stream, mr);
  // print_debug<char, char>(cleaned, "WS NORMALIZED", "", stream);
  //  We will probably do ws normalization as we write out the data. This is true for number
  //  normalization too

  auto json_opts = cudf::io::json_reader_options_builder()
                     .lines(true)
                     .mixed_types_as_string(true)
                     .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                     .build();

  auto const [tokens, token_indices] = cudf::io::json::detail::get_token_stream(
    cudf::device_span<char const>{reinterpret_cast<char const*>(cleaned_buffer.data()),
                                  cleaned_buffer.size()},
    json_opts,
    stream,
    mr);

  // TODO this is just for profiling for now. Lets return an empty string column...
  auto rows       = input.size();
  auto str_scalar = cudf::make_string_scalar("TODO FIX ME!!!!", stream, mr);
  return cudf::make_column_from_scalar(*str_scalar, rows, stream, mr);
}

}  // namespace spark_rapids_jni

struct JsonReaderTest2 : public cudf::test::BaseFixture {};

TEST_F(JsonReaderTest2, Spark)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = rmm::mr::get_current_device_resource();
  auto filename{
    "/home/coder/cudf/part-00000-09b34558-4309-4a7c-ad2c-a46a1ee66cab-c000.snappy.parquet"};

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filename}).columns({"columnC"});
  auto result = cudf::io::read_parquet(read_opts);

  auto columnC = result.tbl->get_column(0);
  auto json    = spark_rapids_jni::different_get_json_object(columnC, {}, stream, mr);

  // CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
  // different_get_json_object;
}
