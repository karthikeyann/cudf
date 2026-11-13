/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <src/io/parquet/decode_flat.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// How the values of a column repeat, which decides the run mix of a dictionary index stream.
enum class value_pattern {
  RANDOM,      // few distinct values in random order: bit-packed runs
  SHORT_RUNS,  // each value repeated 12 times: repeated runs
  MIXED_RUNS,  // runs of 1 to 40 equal values: short repeated and bit-packed runs
  MIXED,       // long constant stretches between random stretches
  CONSTANT,    // one distinct value: index bit width 0
  DISTINCT,    // every value different
};

// Which rows of a column are null.
enum class null_pattern {
  REQUIRED,   // the column has no validity mask
  NO_NULLS,   // nullable, every row valid
  SPARSE,     // about 5% nulls
  HALF,       // about 50% nulls
  NULL_RUNS,  // runs of 700 nulls every 3000 rows
  ENDS,       // only the first and the last row are null
  ALL_NULL,   // every row null
};

template <typename T>
std::unique_ptr<cudf::column> make_column(cudf::size_type num_rows,
                                          value_pattern values,
                                          null_pattern nulls,
                                          int seed)
{
  std::mt19937 engine(seed);
  std::uniform_int_distribution<int> few(0, 199);
  std::uniform_int_distribution<int> run_length(1, 40);
  std::vector<T> data(num_rows);
  int run_left  = 0;
  int run_value = 0;
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    if (run_left == 0) {
      run_left  = run_length(engine);
      run_value = few(engine);
    }
    --run_left;
    int value = 0;
    switch (values) {
      case value_pattern::RANDOM: value = few(engine); break;
      case value_pattern::SHORT_RUNS: value = (i / 12) % 37; break;
      case value_pattern::MIXED_RUNS: value = run_value; break;
      case value_pattern::MIXED: value = (i / 2500) % 2 == 0 ? i / 2500 : few(engine); break;
      case value_pattern::CONSTANT: value = 42; break;
      case value_pattern::DISTINCT: value = i; break;
    }
    data[i] = static_cast<T>(value * 3 - 100);
  }
  if (nulls == null_pattern::REQUIRED) {
    return cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end()).release();
  }
  std::uniform_int_distribution<int> percent(0, 99);
  std::vector<bool> valid(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    switch (nulls) {
      case null_pattern::SPARSE: valid[i] = percent(engine) >= 5; break;
      case null_pattern::HALF: valid[i] = percent(engine) >= 50; break;
      case null_pattern::NULL_RUNS: valid[i] = i % 3000 >= 700; break;
      case null_pattern::ENDS: valid[i] = i != 0 and i != num_rows - 1; break;
      case null_pattern::ALL_NULL: valid[i] = false; break;
      default: valid[i] = true; break;
    }
  }
  return cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end(), valid.begin())
    .release();
}

// One column of each physical type the tile decode kernels handle.
std::unique_ptr<cudf::table> make_table(cudf::size_type num_rows,
                                        value_pattern values,
                                        null_pattern nulls)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_column<int32_t>(num_rows, values, nulls, 1));
  columns.push_back(make_column<int64_t>(num_rows, values, nulls, 2));
  columns.push_back(make_column<float>(num_rows, values, nulls, 3));
  columns.push_back(make_column<double>(num_rows, values, nulls, 4));
  return std::make_unique<cudf::table>(std::move(columns));
}

struct write_config {
  cudf::io::dictionary_policy dictionary = cudf::io::dictionary_policy::ALWAYS;
  cudf::size_type page_rows              = 20'000;
  bool v2_headers                        = false;
  cudf::io::compression_type compression = cudf::io::compression_type::NONE;
  std::size_t max_dictionary_size        = 1024 * 1024;
};

std::vector<char> write_buffer(cudf::table_view const& table, write_config const& config)
{
  std::vector<char> buffer;
  auto const options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&buffer), table)
      .dictionary_policy(config.dictionary)
      .max_dictionary_size(config.max_dictionary_size)
      .max_page_size_rows(config.page_rows)
      .max_page_fragment_size(std::min(config.page_rows, 5000))
      .write_v2_headers(config.v2_headers)
      .compression(config.compression)
      .stats_level(cudf::io::statistics_freq::STATISTICS_NONE)
      .build();
  cudf::io::write_parquet(options);
  return buffer;
}

cudf::io::source_info source_of(std::vector<char> const& buffer)
{
  return cudf::io::source_info{cudf::host_span<std::byte const>{
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}};
}

// Values, validity masks and null counts must match exactly.
void expect_identical(cudf::table_view const& generic, cudf::table_view const& tiles)
{
  CUDF_TEST_EXPECT_TABLES_EQUAL(generic, tiles);
  ASSERT_EQ(generic.num_columns(), tiles.num_columns());
  for (cudf::size_type i = 0; i < generic.num_columns(); ++i) {
    EXPECT_EQ(generic.column(i).null_count(), tiles.column(i).null_count());
    EXPECT_EQ(generic.column(i).nullable(), tiles.column(i).nullable());
  }
}

// Returns the number of pages the tile decode kernels claimed while `read` ran.
template <typename Read>
std::size_t count_claimed_pages(Read read)
{
  auto const before = cudf::io::parquet::detail::num_tile_decode_claimed_pages();
  read();
  return cudf::io::parquet::detail::num_tile_decode_claimed_pages() - before;
}

// Reads `buffer` with the generic and with the tile decode kernels and compares the results.
// `buffer` must hold at least one column the tile decode kernels handle.
std::unique_ptr<cudf::table> read_both(std::vector<char> const& buffer,
                                       cudf::size_type skip_rows,
                                       cudf::size_type num_rows)
{
  auto const read = [&](bool use_tile_decode) {
    auto builder = cudf::io::parquet_reader_options::builder(source_of(buffer))
                     .use_tile_decode(use_tile_decode)
                     .skip_rows(skip_rows);
    if (num_rows >= 0) { builder.num_rows(num_rows); }
    return cudf::io::read_parquet(builder.build()).tbl;
  };
  auto generic = read(false);
  std::unique_ptr<cudf::table> tiles;
  auto const num_claimed = count_claimed_pages([&] { tiles = read(true); });
  // An equal result alone would also pass if no page reached the tile decode kernels.
  if (generic->num_rows() > 0) { EXPECT_GT(num_claimed, 0); }
  expect_identical(generic->view(), tiles->view());
  return generic;
}

std::unique_ptr<cudf::table> read_chunked(std::vector<char> const& buffer,
                                          bool use_tile_decode,
                                          std::size_t output_limit,
                                          std::size_t input_limit)
{
  auto const options = cudf::io::parquet_reader_options::builder(source_of(buffer))
                         .use_tile_decode(use_tile_decode)
                         .build();
  auto reader = cudf::io::chunked_parquet_reader(output_limit, input_limit, options);
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  do {
    tables.push_back(reader.read_chunk().tbl);
    views.push_back(tables.back()->view());
  } while (reader.has_next());
  return cudf::concatenate(views);
}

std::string describe(value_pattern values, null_pattern nulls, write_config const& config)
{
  return "values=" + std::to_string(static_cast<int>(values)) +
         " nulls=" + std::to_string(static_cast<int>(nulls)) +
         " dictionary=" + std::to_string(static_cast<int>(config.dictionary)) +
         " page_rows=" + std::to_string(config.page_rows) +
         " v2=" + std::to_string(config.v2_headers) +
         " compression=" + std::to_string(static_cast<int>(config.compression));
}

constexpr std::array all_values{value_pattern::RANDOM,
                                value_pattern::SHORT_RUNS,
                                value_pattern::MIXED_RUNS,
                                value_pattern::MIXED,
                                value_pattern::CONSTANT,
                                value_pattern::DISTINCT};

constexpr std::array all_nulls{null_pattern::REQUIRED,
                               null_pattern::NO_NULLS,
                               null_pattern::SPARSE,
                               null_pattern::HALF,
                               null_pattern::NULL_RUNS,
                               null_pattern::ENDS,
                               null_pattern::ALL_NULL};

}  // namespace

struct ParquetTileDecodeTest : public cudf::test::BaseFixture {};

TEST_F(ParquetTileDecodeTest, EncodingsAndNulls)
{
  constexpr cudf::size_type num_rows = 50'000;
  for (auto const values : all_values) {
    for (auto const nulls : all_nulls) {
      auto const expected = make_table(num_rows, values, nulls);
      for (auto const dictionary :
           {cudf::io::dictionary_policy::ALWAYS, cudf::io::dictionary_policy::NEVER}) {
        for (auto const v2_headers : {false, true}) {
          write_config config;
          config.dictionary = dictionary;
          config.v2_headers = v2_headers;
          SCOPED_TRACE(describe(values, nulls, config));
          auto const buffer = write_buffer(expected->view(), config);
          auto const result = read_both(buffer, 0, -1);
          CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
        }
      }
    }
  }
}

TEST_F(ParquetTileDecodeTest, PageSizes)
{
  // Page sizes around the tile size and the mask word size, so that pages and tiles start at
  // every kind of output position.
  for (auto const page_rows : {1, 31, 33, 1000, 1023, 1024, 1025, 4097}) {
    cudf::size_type const num_rows = std::min(page_rows * 40, 30'000);
    for (auto const nulls : {null_pattern::REQUIRED, null_pattern::SPARSE, null_pattern::HALF}) {
      for (auto const values : {value_pattern::RANDOM, value_pattern::SHORT_RUNS}) {
        auto const expected = make_table(num_rows, values, nulls);
        write_config config;
        config.page_rows = page_rows;
        SCOPED_TRACE(describe(values, nulls, config));
        auto const buffer = write_buffer(expected->view(), config);
        auto const result = read_both(buffer, 0, -1);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
      }
    }
  }
}

TEST_F(ParquetTileDecodeTest, LongRunStreams)
{
  // Pages whose index and level streams hold thousands of short runs, so that the checkpoints
  // of a page come from many windows of its streams.
  constexpr cudf::size_type num_rows = 400'000;
  for (auto const values : {value_pattern::SHORT_RUNS, value_pattern::MIXED_RUNS}) {
    for (auto const nulls :
         {null_pattern::REQUIRED, null_pattern::SPARSE, null_pattern::HALF, null_pattern::ENDS}) {
      auto const expected = make_table(num_rows, values, nulls);
      write_config config;
      config.page_rows = 200'000;
      SCOPED_TRACE(describe(values, nulls, config));
      auto const buffer = write_buffer(expected->view(), config);
      auto const result = read_both(buffer, 0, -1);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
      read_both(buffer, 123'457, 200'001);
    }
  }
}

TEST_F(ParquetTileDecodeTest, Compressed)
{
  constexpr cudf::size_type num_rows = 60'000;
  for (auto const nulls : {null_pattern::REQUIRED, null_pattern::SPARSE}) {
    for (auto const dictionary :
         {cudf::io::dictionary_policy::ALWAYS, cudf::io::dictionary_policy::NEVER}) {
      for (auto const v2_headers : {false, true}) {
        auto const expected = make_table(num_rows, value_pattern::MIXED, nulls);
        write_config config;
        config.dictionary  = dictionary;
        config.v2_headers  = v2_headers;
        config.compression = cudf::io::compression_type::SNAPPY;
        SCOPED_TRACE(describe(value_pattern::MIXED, nulls, config));
        auto const buffer = write_buffer(expected->view(), config);
        auto const result = read_both(buffer, 0, -1);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
      }
    }
  }
}

TEST_F(ParquetTileDecodeTest, RowBounds)
{
  constexpr cudf::size_type num_rows  = 40'000;
  constexpr cudf::size_type page_rows = 5'000;
  // Bounds that cut the first, a middle and the last page, cut inside a tile, land on tile and
  // page boundaries, and select rows of a single mask word.
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const bounds{{0, 1},
                                                                        {1, 1},
                                                                        {31, 3},
                                                                        {1024, 1024},
                                                                        {1023, 2},
                                                                        {1000, 5000},
                                                                        {4999, 2},
                                                                        {5000, 5000},
                                                                        {5001, 9998},
                                                                        {12345, 6789},
                                                                        {39'999, 1},
                                                                        {35'000, 5000},
                                                                        {17, 39'983},
                                                                        {2048, 30'000},
                                                                        {num_rows, 0},
                                                                        {7, 20}};
  for (auto const nulls : {null_pattern::REQUIRED, null_pattern::SPARSE, null_pattern::NULL_RUNS}) {
    for (auto const dictionary :
         {cudf::io::dictionary_policy::ALWAYS, cudf::io::dictionary_policy::NEVER}) {
      for (auto const values : {value_pattern::RANDOM, value_pattern::SHORT_RUNS}) {
        auto const expected = make_table(num_rows, values, nulls);
        write_config config;
        config.dictionary = dictionary;
        config.page_rows  = page_rows;
        auto const buffer = write_buffer(expected->view(), config);
        for (auto const& [skip_rows, rows_to_read] : bounds) {
          SCOPED_TRACE(describe(values, nulls, config) + " skip_rows=" + std::to_string(skip_rows) +
                       " num_rows=" + std::to_string(rows_to_read));
          auto const result = read_both(buffer, skip_rows, rows_to_read);
          auto const sliced = cudf::slice(expected->view(), {skip_rows, skip_rows + rows_to_read});
          CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sliced.front(), result->view());
        }
      }
    }
  }
}

TEST_F(ParquetTileDecodeTest, ChunkedRead)
{
  constexpr cudf::size_type num_rows = 100'000;
  for (auto const nulls : {null_pattern::REQUIRED, null_pattern::SPARSE, null_pattern::HALF}) {
    for (auto const dictionary :
         {cudf::io::dictionary_policy::ALWAYS, cudf::io::dictionary_policy::NEVER}) {
      auto const expected = make_table(num_rows, value_pattern::SHORT_RUNS, nulls);
      write_config config;
      config.dictionary  = dictionary;
      config.page_rows   = 7'000;
      config.compression = cudf::io::compression_type::SNAPPY;
      auto const buffer  = write_buffer(expected->view(), config);
      // Output limits that split pages across output chunks, and an input limit that splits the
      // row group into subpasses.
      for (auto const& [output_limit, input_limit] :
           std::vector<std::pair<std::size_t, std::size_t>>{
             {0, 0}, {100'000, 0}, {240'000, 0}, {0, 400'000}, {100'000, 400'000}}) {
        SCOPED_TRACE(describe(value_pattern::SHORT_RUNS, nulls, config) + " output_limit=" +
                     std::to_string(output_limit) + " input_limit=" + std::to_string(input_limit));
        auto const generic = read_chunked(buffer, false, output_limit, input_limit);
        auto const tiles   = read_chunked(buffer, true, output_limit, input_limit);
        expect_identical(generic->view(), tiles->view());
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), generic->view());
      }
    }
  }
}

TEST_F(ParquetTileDecodeTest, DictionaryFallback)
{
  // A dictionary limit that the distinct values outgrow: every column chunk starts with
  // dictionary pages and continues with PLAIN pages.
  constexpr cudf::size_type num_rows = 60'000;
  for (auto const nulls : {null_pattern::REQUIRED, null_pattern::SPARSE}) {
    auto const expected = make_table(num_rows, value_pattern::DISTINCT, nulls);
    write_config config;
    config.page_rows           = 3'000;
    config.max_dictionary_size = 16 * 1024;
    SCOPED_TRACE(describe(value_pattern::DISTINCT, nulls, config));
    auto const buffer = write_buffer(expected->view(), config);
    auto const result = read_both(buffer, 0, -1);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
    read_both(buffer, 2'999, 30'003);
  }
}

TEST_F(ParquetTileDecodeTest, DictionaryIndexOutOfRange)
{
  // Three dictionary entries take 2-bit indices, which leaves index 3 without an entry.
  constexpr cudf::size_type num_rows = 4'000;
  std::vector<int32_t> values(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values[i] = 10 * (i % 3 + 1);
  }
  cudf::test::fixed_width_column_wrapper<int32_t> column(values.begin(), values.end());
  auto buffer = write_buffer(cudf::table_view{{column}}, write_config{});

  // The bit-packed indices 0, 1, 2, 0, ... repeat every three bytes. Turn one index 0 into 3.
  std::array<char, 6> const pattern{
    0x24, 0x49, static_cast<char>(0x92), 0x24, 0x49, static_cast<char>(0x92)};
  auto const found = std::search(buffer.begin(), buffer.end(), pattern.begin(), pattern.end());
  ASSERT_NE(found, buffer.end());
  *found = 0x27;

  auto const result = read_both(buffer, 0, -1);
  // The index without an entry decodes as 0; every other row is unchanged.
  auto const decoded          = cudf::test::to_host<int32_t>(result->view().column(0)).first;
  cudf::size_type num_changed = 0;
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    if (decoded[i] != values[i]) {
      EXPECT_EQ(decoded[i], 0);
      ++num_changed;
    }
  }
  EXPECT_EQ(num_changed, 1);
}

TEST_F(ParquetTileDecodeTest, TimestampsAndDates)
{
  // With a timestamp type requested, a timestamp column in that unit and a date column are
  // stored as they are; a timestamp column in another unit is rescaled by the generic decoders.
  constexpr cudf::size_type num_rows = 20'000;
  std::vector<int64_t> ticks(num_rows);
  std::vector<int32_t> days(num_rows);
  std::vector<bool> valid(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    ticks[i] = 1'600'000'000'000 + (i % 500) * 1'000;
    days[i]  = 18'000 + i % 400;
    valid[i] = i % 13 != 0;
  }
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> millis(
    ticks.begin(), ticks.end(), valid.begin());
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, int64_t> micros(ticks.begin(),
                                                                             ticks.end());
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> dates(
    days.begin(), days.end(), valid.begin());
  cudf::table_view const expected{{millis, micros, dates}};
  write_config config;
  config.page_rows  = 3'000;
  auto const buffer = write_buffer(expected, config);

  for (auto const type : {cudf::type_id::EMPTY,
                          cudf::type_id::TIMESTAMP_MILLISECONDS,
                          cudf::type_id::TIMESTAMP_MICROSECONDS}) {
    auto const read = [&](bool use_tile_decode) {
      auto const options = cudf::io::parquet_reader_options::builder(source_of(buffer))
                             .use_tile_decode(use_tile_decode)
                             .timestamp_type(cudf::data_type{type})
                             .build();
      return cudf::io::read_parquet(options).tbl;
    };
    SCOPED_TRACE("timestamp_type=" + std::to_string(static_cast<int>(type)));
    auto const generic = read(false);
    auto const tiles   = read(true);
    expect_identical(generic->view(), tiles->view());
  }
}

TEST_F(ParquetTileDecodeTest, MixedWithOtherColumns)
{
  // Columns the tile decode kernels do not handle stay with the generic decoders.
  constexpr cudf::size_type num_rows = 30'000;
  auto ints   = make_column<int64_t>(num_rows, value_pattern::RANDOM, null_pattern::SPARSE, 5);
  auto narrow = make_column<int8_t>(num_rows, value_pattern::RANDOM, null_pattern::SPARSE, 6);
  auto doubles =
    make_column<double>(num_rows, value_pattern::SHORT_RUNS, null_pattern::REQUIRED, 7);
  auto bools = make_column<bool>(num_rows, value_pattern::RANDOM, null_pattern::HALF, 8);
  std::vector<std::string> strings(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings[i] = "s" + std::to_string(i % 91);
  }
  cudf::test::strings_column_wrapper string_column(strings.begin(), strings.end());
  cudf::table_view const expected{{*ints, *narrow, *doubles, *bools, string_column}};
  write_config config;
  config.page_rows  = 4'000;
  auto const buffer = write_buffer(expected, config);
  auto const result = read_both(buffer, 0, -1);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, result->view());
  read_both(buffer, 4'001, 20'000);
}

TEST_F(ParquetTileDecodeTest, ClaimsEligiblePagesOnly)
{
  constexpr cudf::size_type num_rows = 10'000;
  write_config config;
  config.dictionary = cudf::io::dictionary_policy::NEVER;
  config.page_rows  = 1'000;
  auto const read   = [](std::vector<char> const& buffer, bool use_tile_decode) {
    return count_claimed_pages([&] {
      cudf::io::read_parquet(cudf::io::parquet_reader_options::builder(source_of(buffer))
                               .use_tile_decode(use_tile_decode)
                               .build());
    });
  };
  // The environment variable claims pages for readers that do not ask for it.
  auto const forced = std::getenv("LIBCUDF_PARQUET_TILE_DECODE") != nullptr;

  {
    // Four columns of ten PLAIN pages each.
    auto const eligible = make_table(num_rows, value_pattern::RANDOM, null_pattern::SPARSE);
    auto const buffer   = write_buffer(eligible->view(), config);
    EXPECT_EQ(read(buffer, true), 40);
    if (not forced) { EXPECT_EQ(read(buffer, false), 0); }
  }
  {
    // Narrowed integers, booleans and strings stay with the generic decoders.
    auto narrow = make_column<int8_t>(num_rows, value_pattern::RANDOM, null_pattern::SPARSE, 1);
    auto bools  = make_column<bool>(num_rows, value_pattern::RANDOM, null_pattern::HALF, 2);
    std::vector<std::string> strings(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      strings[i] = "s" + std::to_string(i % 91);
    }
    cudf::test::strings_column_wrapper string_column(strings.begin(), strings.end());
    auto const buffer = write_buffer(cudf::table_view{{*narrow, *bools, string_column}}, config);
    EXPECT_EQ(read(buffer, true), 0);
  }
}

TEST_F(ParquetTileDecodeTest, ColumnWithClaimedAndGenericPages)
{
  // Two sources hold the same columns: PLAIN pages, which the tile decode kernels claim, and
  // DELTA_BINARY_PACKED and BYTE_STREAM_SPLIT pages, which they do not. The sources meet
  // inside a word of the validity mask, which both sets of kernels then write.
  constexpr cudf::size_type plain_rows = 3'001;
  constexpr cudf::size_type other_rows = 2'003;
  auto const write                     = [](cudf::table_view const& table, bool plain) {
    cudf::io::table_input_metadata metadata(table);
    metadata.column_metadata[0].set_name("ints").set_encoding(
      plain ? cudf::io::column_encoding::PLAIN : cudf::io::column_encoding::DELTA_BINARY_PACKED);
    metadata.column_metadata[1].set_name("doubles").set_encoding(
      plain ? cudf::io::column_encoding::PLAIN : cudf::io::column_encoding::BYTE_STREAM_SPLIT);
    std::vector<char> buffer;
    auto const options =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&buffer), table)
        .metadata(std::move(metadata))
        .dictionary_policy(cudf::io::dictionary_policy::NEVER)
        .max_page_size_rows(500)
        .max_page_fragment_size(500)
        .write_v2_headers(not plain)
        .stats_level(cudf::io::statistics_freq::STATISTICS_NONE)
        .build();
    cudf::io::write_parquet(options);
    return buffer;
  };
  auto const make = [](cudf::size_type num_rows, uint32_t seed) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      make_column<int64_t>(num_rows, value_pattern::RANDOM, null_pattern::SPARSE, seed));
    columns.push_back(
      make_column<double>(num_rows, value_pattern::RANDOM, null_pattern::HALF, seed + 1));
    return std::make_unique<cudf::table>(std::move(columns));
  };
  auto const plain_table  = make(plain_rows, 1);
  auto const other_table  = make(other_rows, 3);
  auto const plain_buffer = write(plain_table->view(), true);
  auto const other_buffer = write(other_table->view(), false);

  auto const span_of = [](std::vector<char> const& buffer) {
    return cudf::host_span<std::byte const>{reinterpret_cast<std::byte const*>(buffer.data()),
                                            buffer.size()};
  };
  auto const read = [&](std::vector<cudf::host_span<std::byte const>> sources,
                        bool use_tile_decode) {
    return cudf::io::read_parquet(
             cudf::io::parquet_reader_options::builder(
               cudf::io::source_info{cudf::host_span<cudf::host_span<std::byte const>>{sources}})
               .use_tile_decode(use_tile_decode)
               .build())
      .tbl;
  };

  // The pages of the second source are not claimed; those of the first are.
  EXPECT_EQ(count_claimed_pages([&] { read({span_of(other_buffer)}, true); }), 0);
  auto const num_plain_pages = count_claimed_pages([&] { read({span_of(plain_buffer)}, true); });
  EXPECT_GT(num_plain_pages, 0);

  for (auto const plain_first : {true, false}) {
    SCOPED_TRACE("plain_first=" + std::to_string(plain_first));
    std::vector<cudf::host_span<std::byte const>> const sources{
      span_of(plain_first ? plain_buffer : other_buffer),
      span_of(plain_first ? other_buffer : plain_buffer)};
    auto const expected = cudf::concatenate(
      std::vector<cudf::table_view>{plain_first ? plain_table->view() : other_table->view(),
                                    plain_first ? other_table->view() : plain_table->view()});
    auto const generic = read(sources, false);
    std::unique_ptr<cudf::table> tiles;
    EXPECT_EQ(count_claimed_pages([&] { tiles = read(sources, true); }), num_plain_pages);
    expect_identical(generic->view(), tiles->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), tiles->view());
  }
}
