/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

struct CsvLargeReaderTest : public cudf::test::StringsLargeTest {};

namespace {
/// Reads a CSV buffer of one string column without a header, in which no field is null
cudf::io::table_with_metadata read_string_column(std::string const& text)
{
  return cudf::io::read_csv(cudf::io::csv_reader_options::builder(
                              cudf::io::source_info{cudf::host_span<std::byte const>{
                                reinterpret_cast<std::byte const*>(text.data()), text.size()}})
                              .compression(cudf::io::compression_type::NONE)
                              .header(-1)
                              .na_filter(false)
                              .dtypes({cudf::data_type{cudf::type_id::STRING}})
                              .build());
}
}  // namespace

TEST_F(CsvLargeReaderTest, StringColumnOverTwoGigabytes)
{
  // A string column of more than 2^31 characters, which needs 64-bit offsets: strings of 700 to
  // 1023 characters, copied with the other strings of their rows, and every 1000th string of 1500
  // characters, copied separately. Each string starts with '<' and ends with '>', so that
  // misplaced characters change the column.
  constexpr std::size_t min_size = std::size_t{2200} << 20;
  std::string text;
  text.reserve(min_size + 2048);
  std::vector<int64_t> expected_offsets{0};
  for (std::size_t row = 0; text.size() < min_size; ++row) {
    auto const length = row % 1000 == 7 ? 1500 : 700 + (row * 7919) % 324;
    text += '<';
    text.append(length - 2, static_cast<char>('a' + row % 26));
    text += ">\n";
    expected_offsets.push_back(expected_offsets.back() + static_cast<int64_t>(length));
  }
  std::string expected_chars;
  expected_chars.reserve(text.size());
  for (auto const c : text) {
    if (c != '\n') { expected_chars += c; }
  }

  auto const result = read_string_column(text);
  auto const column = cudf::strings_column_view(result.tbl->get_column(0));
  auto const stream = cudf::get_default_stream();
  ASSERT_EQ(column.size() + 1, static_cast<cudf::size_type>(expected_offsets.size()));
  EXPECT_EQ(column.null_count(), 0);
  ASSERT_EQ(column.offsets().type().id(), cudf::type_id::INT64);

  auto const offsets = cudf::detail::make_std_vector(
    cudf::device_span<int64_t const>{column.offsets().data<int64_t>(), expected_offsets.size()},
    stream);
  EXPECT_EQ(offsets, expected_offsets);
  ASSERT_EQ(column.chars_size(stream), static_cast<int64_t>(expected_chars.size()));
  auto const chars = cudf::detail::make_std_vector(
    cudf::device_span<char const>{column.chars_begin(stream), expected_chars.size()}, stream);
  EXPECT_TRUE(std::equal(chars.begin(), chars.end(), expected_chars.begin()));
}
