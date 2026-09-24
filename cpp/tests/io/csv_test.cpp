/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using cudf::data_type;
using cudf::type_id;
using cudf::type_to_id;

template <typename T>
auto dtype()
{
  return data_type{type_to_id<T>()};
}

template <typename T, typename SourceElementT = T>
using column_wrapper =
  std::conditional_t<std::is_same_v<T, cudf::string_view>,
                     cudf::test::strings_column_wrapper,
                     cudf::test::fixed_width_column_wrapper<T, SourceElementT>>;
using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// Base test fixture for tests
struct CsvWriterTest : public cudf::test::BaseFixture {};

template <typename T>
struct CsvFixedPointWriterTest : public CsvWriterTest {};

TYPED_TEST_SUITE(CsvFixedPointWriterTest, cudf::test::FixedPointTypes);

// Base test fixture for tests
struct CsvReaderTest : public cudf::test::BaseFixture {};

// Typed test fixture for timestamp type tests
template <typename T>
struct CsvReaderNumericTypeTest : public CsvReaderTest {};

// Declare typed test cases
using SupportedNumericTypes = cudf::test::Types<int64_t, double>;
TYPED_TEST_SUITE(CsvReaderNumericTypeTest, SupportedNumericTypes);

template <typename DecimalType>
struct CsvFixedPointReaderTest : public CsvReaderTest {
  void run_tests(std::vector<std::string> const& reference_strings, numeric::scale_type scale)
  {
    cudf::test::strings_column_wrapper const strings(reference_strings.begin(),
                                                     reference_strings.end());
    auto const expected = cudf::strings::to_fixed_point(
      cudf::strings_column_view(strings), data_type{type_to_id<DecimalType>(), scale});

    auto const buffer = std::accumulate(reference_strings.begin(),
                                        reference_strings.end(),
                                        std::string{},
                                        [](std::string const& acc, std::string const& rhs) {
                                          return acc.empty() ? rhs : (acc + "\n" + rhs);
                                        });

    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .dtypes({data_type{type_to_id<DecimalType>(), scale}})
        .header(-1);

    auto const result      = cudf::io::read_csv(in_opts);
    auto const result_view = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, result_view.column(0));
    EXPECT_EQ(result_view.num_columns(), 1);
  }
};

TYPED_TEST_SUITE(CsvFixedPointReaderTest, cudf::test::FixedPointTypes);

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(std::size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<T1, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<T1>,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

MATCHER_P(FloatNearPointwise, tolerance, "Out-of-range")
{
  return (std::get<0>(arg) > std::get<1>(arg) - tolerance &&
          std::get<0>(arg) < std::get<1>(arg) + tolerance);
}

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;

// temporary method to verify the float columns until
// CUDF_TEST_EXPECT_COLUMNS_EQUAL supports floating point
template <typename T, typename valid_t>
void check_float_column(cudf::column_view const& col_lhs,
                        cudf::column_view const& col_rhs,
                        T tol,
                        valid_t const& validity)
{
  auto h_data = cudf::test::to_host<T>(col_rhs).first;

  std::vector<T> data(h_data.size());
  std::copy(h_data.begin(), h_data.end(), data.begin());

  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(col_lhs,
                                                (wrapper<T>{data.begin(), data.end(), validity}));
  EXPECT_TRUE(col_lhs.null_count() == 0 and col_rhs.null_count() == 0);
  EXPECT_THAT(cudf::test::to_host<T>(col_lhs).first,
              ::testing::Pointwise(FloatNearPointwise(tol), data));
}

// timestamp column checker within tolerance
// given by `tol_ms` (milliseconds)
void check_timestamp_column(cudf::column_view const& col_lhs,
                            cudf::column_view const& col_rhs,
                            long tol_ms = 1000l)
{
  using T = cudf::timestamp_ms;
  using namespace cuda::std::chrono;

  auto h_lhs = cudf::test::to_host<T>(col_lhs).first;
  auto h_rhs = cudf::test::to_host<T>(col_rhs).first;

  cudf::size_type nrows = h_lhs.size();
  EXPECT_TRUE(nrows == static_cast<cudf::size_type>(h_rhs.size()));

  auto begin_count = cuda::counting_iterator<cudf::size_type>{0};
  auto end_count   = cuda::counting_iterator<cudf::size_type>{nrows};

  auto* ptr_lhs = h_lhs.data();  // cannot capture host_vector in thrust,
                                 // not even in host lambda
  auto* ptr_rhs = h_rhs.data();

  auto found = thrust::find_if(
    thrust::host, begin_count, end_count, [ptr_lhs, ptr_rhs, tol_ms](auto row_index) {
      auto delta_ms = cuda::std::chrono::duration_cast<cuda::std::chrono::milliseconds>(
        ptr_lhs[row_index] - ptr_rhs[row_index]);
      return delta_ms.count() >= tol_ms;
    });

  EXPECT_TRUE(found == end_count);  // not found...
}

// helper to replace in `str`  _all_ occurrences of `from` with `to`
std::string replace_all_helper(std::string str, std::string const& from, std::string const& to)
{
  std::size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}

// compare string columns accounting for special character
// treatment: double double quotes ('\"')
// and surround whole string by double quotes if it contains:
// newline '\n', <delimiter>, and double quotes;
void check_string_column(cudf::column_view const& col_lhs,
                         cudf::column_view const& col_rhs,
                         std::string const& delimiter = ",")
{
  auto h_lhs = cudf::test::to_host<std::string>(col_lhs).first;
  auto h_rhs = cudf::test::to_host<std::string>(col_rhs).first;

  std::string newline("\n");
  std::string quotes("\"");
  std::string quotes_repl("\"\"");

  std::vector<std::string> v_lhs;
  std::transform(h_lhs.begin(),
                 h_lhs.end(),
                 std::back_inserter(v_lhs),
                 [delimiter, newline, quotes, quotes_repl](std::string const& str_row) {
                   auto found_quote = str_row.find(quotes);
                   auto found_newl  = str_row.find(newline);
                   auto found_delim = str_row.find(delimiter);

                   bool flag_found_quotes = (found_quote != std::string::npos);
                   bool need_surround = flag_found_quotes || (found_newl != std::string::npos) ||
                                        (found_delim != std::string::npos);

                   std::string str_repl;
                   if (flag_found_quotes) {
                     str_repl = replace_all_helper(str_row, quotes, quotes_repl);
                   } else {
                     str_repl = str_row;
                   }
                   return need_surround ? quotes + str_repl + quotes : str_row;
                 });
  EXPECT_TRUE(std::equal(v_lhs.begin(), v_lhs.end(), h_rhs.begin()));
}

// Helper function to compare two floating-point column contents
template <typename T>
void expect_column_data_equal(std::vector<T> const& lhs, cudf::column_view const& rhs)
  requires(std::is_floating_point_v<T>)
{
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first,
              ::testing::Pointwise(FloatNearPointwise(1e-6), lhs));
}

// Helper function to compare two column contents
template <typename T>
void expect_column_data_equal(std::vector<T> const& lhs, cudf::column_view const& rhs)
  requires(!std::is_floating_point_v<T>)
{
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first, ::testing::ElementsAreArray(lhs));
}

void write_csv_helper(std::string const& filename,
                      cudf::table_view const& table,
                      std::vector<std::string> const& names = {})
{
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(filename), table)
      .include_header(not names.empty())
      .names(names);

  cudf::io::write_csv(writer_options);
}

template <typename T>
std::string assign(T input)
{
  return std::to_string(input);
}

std::string assign(std::string input) { return input; }

template <typename T>
std::vector<std::string> prepend_zeros(std::vector<T> const& input,
                                       int zero_count         = 0,
                                       bool add_positive_sign = false)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](T const& num) {
    auto str         = assign(num);
    bool is_negative = (str[0] == '-');
    if (is_negative) {
      str.insert(1, zero_count, '0');
      return str;
    } else if (add_positive_sign) {
      return "+" + std::string(zero_count, '0') + str;
    } else {
      str.insert(0, zero_count, '0');
      return str;
    }
  });
  return output;
}

}  // namespace

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumn)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });

  auto filepath = temp_env->get_temp_filepath("SingleColumn.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<TypeParam> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(std::vector<TypeParam>(sequence, sequence + num_rows), view.column(0));
}

TYPED_TEST(CsvFixedPointReaderTest, SingleColumnNegativeScale)
{
  this->run_tests({"1.23", "876e-2", "5.43e1", "-0.12", "0.25", "-0.23", "-0.27", "0.00", "0.00"},
                  numeric::scale_type{-2});
}

TYPED_TEST(CsvFixedPointReaderTest, SingleColumnNoScale)
{
  this->run_tests({"123", "-87600e-2", "54.3e1", "-12", "25", "-23", "-27", "0", "0"},
                  numeric::scale_type{0});
}

TYPED_TEST(CsvFixedPointReaderTest, SingleColumnPositiveScale)
{
  this->run_tests(
    {"123000", "-87600000e-2", "54300e1", "-12000", "25000", "-23000", "-27000", "0000", "0000"},
    numeric::scale_type{3});
}

TYPED_TEST(CsvFixedPointWriterTest, SingleColumnNegativeScale)
{
  std::vector<std::string> reference_strings = {
    "1.23", "-8.76", "5.43", "-0.12", "0.25", "-0.23", "-0.27", "0.00", "0.00"};

  auto validity = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper strings(
    reference_strings.begin(), reference_strings.end(), validity);

  std::vector<std::string> valid_reference_strings;
  thrust::copy_if(thrust::host,
                  reference_strings.begin(),
                  reference_strings.end(),
                  cuda::counting_iterator<std::size_t>{0},
                  std::back_inserter(valid_reference_strings),
                  [](std::size_t i) { return (i % 2) == 0; });
  reference_strings = valid_reference_strings;

  using DecimalType = TypeParam;
  auto input_column =
    cudf::strings::to_fixed_point(cudf::strings_column_view(strings),
                                  data_type{type_to_id<DecimalType>(), numeric::scale_type{-2}});

  auto input_table = cudf::table_view{std::vector<cudf::column_view>{*input_column}};

  auto filepath = temp_env->get_temp_dir() + "FixedPointSingleColumnNegativeScale.csv";

  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(filepath), input_table)
      .include_header(false);

  cudf::io::write_csv(writer_options);

  std::vector<std::string> result_strings;
  result_strings.reserve(reference_strings.size());

  std::ifstream read_result_file(filepath);
  ASSERT_TRUE(read_result_file.is_open());

  std::copy(std::istream_iterator<std::string>(read_result_file),
            std::istream_iterator<std::string>(),
            std::back_inserter(result_strings));

  EXPECT_EQ(result_strings, reference_strings);
}

TYPED_TEST(CsvFixedPointWriterTest, SingleColumnPositiveScale)
{
  std::vector<std::string> reference_strings = {
    "123000", "-876000", "543000", "-12000", "25000", "-23000", "-27000", "0000", "0000"};

  auto validity = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper strings(
    reference_strings.begin(), reference_strings.end(), validity);

  std::vector<std::string> valid_reference_strings;
  thrust::copy_if(thrust::host,
                  reference_strings.begin(),
                  reference_strings.end(),
                  cuda::counting_iterator<std::size_t>{0},
                  std::back_inserter(valid_reference_strings),
                  [](std::size_t i) { return (i % 2) == 0; });
  reference_strings = valid_reference_strings;

  using DecimalType = TypeParam;
  auto input_column =
    cudf::strings::to_fixed_point(cudf::strings_column_view(strings),
                                  data_type{type_to_id<DecimalType>(), numeric::scale_type{3}});

  auto input_table = cudf::table_view{std::vector<cudf::column_view>{*input_column}};

  auto filepath = temp_env->get_temp_dir() + "FixedPointSingleColumnPositiveScale.csv";

  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(filepath), input_table)
      .include_header(false);

  cudf::io::write_csv(writer_options);

  std::vector<std::string> result_strings;
  result_strings.reserve(reference_strings.size());

  std::ifstream read_result_file(filepath);
  ASSERT_TRUE(read_result_file.is_open());

  std::copy(std::istream_iterator<std::string>(read_result_file),
            std::istream_iterator<std::string>(),
            std::back_inserter(result_strings));

  EXPECT_EQ(result_strings, reference_strings);
}

void test_quoting_disabled_with_delimiter(char delimiter_char)
{
  auto const delimiter     = std::string{delimiter_char};
  auto const input_strings = cudf::test::strings_column_wrapper{
    std::string{"All"} + delimiter + "the" + delimiter + "leaves",
    "are\"brown",
    "and\nthe\nsky\nis\ngrey"};
  auto const input_table = table_view{{input_strings}};

  auto const filepath = temp_env->get_temp_dir() + "unquoted.csv";
  auto w_options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, input_table)
                     .include_header(false)
                     .inter_column_delimiter(delimiter_char)
                     .quoting(cudf::io::quote_style::NONE);
  cudf::io::write_csv(w_options.build());

  auto r_options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
                     .header(-1)
                     .delimiter(delimiter_char)
                     .quoting(cudf::io::quote_style::NONE);
  auto r_table = cudf::io::read_csv(r_options.build());

  auto const expected =
    cudf::test::strings_column_wrapper{"All", "are\"brown", "and", "the", "sky", "is", "grey"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(r_table.tbl->view().column(0), expected);
}

TEST_F(CsvWriterTest, QuotingDisabled)
{
  test_quoting_disabled_with_delimiter(',');
  test_quoting_disabled_with_delimiter('\u0001');
}

TEST_F(CsvReaderTest, MultiColumn)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto uint8_values       = random_values<uint8_t>(num_rows);
  auto uint16_values      = random_values<uint16_t>(num_rows);
  auto uint32_values      = random_values<uint32_t>(num_rows);
  auto uint64_values      = random_values<uint64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);
  auto float64_values     = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int32_values[i]
           << "," << int64_values[i] << "," << std::to_string(uint8_values[i]) << ","
           << uint16_values[i] << "," << uint32_values[i] << "," << uint64_values[i] << ","
           << float32_values[i] << "," << float64_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .header(-1)
      .dtypes({dtype<int8_t>(),
               dtype<int16_t>(),
               dtype<int32_t>(),
               dtype<int64_t>(),
               dtype<uint8_t>(),
               dtype<uint16_t>(),
               dtype<uint32_t>(),
               dtype<uint64_t>(),
               dtype<float>(),
               dtype<double>()});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int32_values, view.column(2));
  expect_column_data_equal(int64_values, view.column(3));
  expect_column_data_equal(uint8_values, view.column(4));
  expect_column_data_equal(uint16_values, view.column(5));
  expect_column_data_equal(uint32_values, view.column(6));
  expect_column_data_equal(uint64_values, view.column(7));
  expect_column_data_equal(float32_values, view.column(8));
  expect_column_data_equal(float64_values, view.column(9));
}

TEST_F(CsvReaderTest, RepeatColumn)
{
  constexpr auto num_rows = 10;
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto uint64_values      = random_values<uint64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "RepeatColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << int16_values[i] << "," << int64_values[i] << "," << uint64_values[i] << ","
           << float32_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  // repeats column in indexes and names, misses 1 column.
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes({dtype<int16_t>(), dtype<int64_t>(), dtype<uint64_t>(), dtype<float>()})
      .names({"A", "B", "C", "D"})
      .use_cols_indexes({1, 0, 0})
      .use_cols_names({"D", "B", "B"})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(3, view.num_columns());
  expect_column_data_equal(int16_values, view.column(0));
  expect_column_data_equal(int64_values, view.column(1));
  expect_column_data_equal(float32_values, view.column(2));
}

TEST_F(CsvReaderTest, Booleans)
{
  auto filepath = temp_env->get_temp_dir() + "Booleans.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "YES,1,bar,true\nno,2,FOO,true\nBar,3,yes,false\nNo,4,NO,"
               "true\nYes,5,foo,false\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A", "B", "C", "D"})
      .dtypes({dtype<int32_t>(), dtype<int32_t>(), dtype<int16_t>(), dtype<bool>()})
      .true_values({"yes", "Yes", "YES", "foo", "FOO"})
      .false_values({"no", "No", "NO", "Bar", "bar"})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  // Booleans are the same (integer) data type, but valued at 0 or 1
  auto const view = result.tbl->view();
  EXPECT_EQ(4, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::INT32, view.column(1).type().id());
  ASSERT_EQ(type_id::INT16, view.column(2).type().id());
  ASSERT_EQ(type_id::BOOL8, view.column(3).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 0, 0, 0, 1}, view.column(0));
  expect_column_data_equal(std::vector<int16_t>{0, 1, 1, 0, 1}, view.column(2));
  expect_column_data_equal(std::vector<bool>{true, true, false, true, false}, view.column(3));
}

TEST_F(CsvReaderTest, Dates)
{
  auto filepath = temp_env->get_temp_dir() + "Dates.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{983750400000ms},
                                                           cudf::timestamp_ms{1288483200000ms},
                                                           cudf::timestamp_ms{782611200000ms},
                                                           cudf::timestamp_ms{656208000000ms},
                                                           cudf::timestamp_ms{0ms},
                                                           cudf::timestamp_ms{798163200000ms},
                                                           cudf::timestamp_ms{774144000000ms},
                                                           cudf::timestamp_ms{1149679230400ms},
                                                           cudf::timestamp_ms{1126875750400ms},
                                                           cudf::timestamp_ms{2764800000ms}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampS.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_SECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_SECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_s>{cudf::timestamp_s{983750400s},
                                                          cudf::timestamp_s{1288483200s},
                                                          cudf::timestamp_s{782611200s},
                                                          cudf::timestamp_s{656208000s},
                                                          cudf::timestamp_s{0s},
                                                          cudf::timestamp_s{798163200s},
                                                          cudf::timestamp_s{774144000s},
                                                          cudf::timestamp_s{1149679230s},
                                                          cudf::timestamp_s{1126875750s},
                                                          cudf::timestamp_s{2764800s}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMilliSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampMs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{983750400000ms},
                                                           cudf::timestamp_ms{1288483200000ms},
                                                           cudf::timestamp_ms{782611200000ms},
                                                           cudf::timestamp_ms{656208000000ms},
                                                           cudf::timestamp_ms{0ms},
                                                           cudf::timestamp_ms{798163200000ms},
                                                           cudf::timestamp_ms{774144000000ms},
                                                           cudf::timestamp_ms{1149679230400ms},
                                                           cudf::timestamp_ms{1126875750400ms},
                                                           cudf::timestamp_ms{2764800000ms}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMicroSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampUs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MICROSECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MICROSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_us>{cudf::timestamp_us{983750400000000us},
                                                           cudf::timestamp_us{1288483200000000us},
                                                           cudf::timestamp_us{782611200000000us},
                                                           cudf::timestamp_us{656208000000000us},
                                                           cudf::timestamp_us{0us},
                                                           cudf::timestamp_us{798163200000000us},
                                                           cudf::timestamp_us{774144000000000us},
                                                           cudf::timestamp_us{1149679230400000us},
                                                           cudf::timestamp_us{1126875750400000us},
                                                           cudf::timestamp_us{2764800000000us}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampNanoSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampNs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_NANOSECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_NANOSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(
    std::vector<cudf::timestamp_ns>{cudf::timestamp_ns{983750400000000000ns},
                                    cudf::timestamp_ns{1288483200000000000ns},
                                    cudf::timestamp_ns{782611200000000000ns},
                                    cudf::timestamp_ns{656208000000000000ns},
                                    cudf::timestamp_ns{0ns},
                                    cudf::timestamp_ns{798163200000000000ns},
                                    cudf::timestamp_ns{774144000000000000ns},
                                    cudf::timestamp_ns{1149679230400000000ns},
                                    cudf::timestamp_ns{1126875750400000000ns},
                                    cudf::timestamp_ns{2764800000000000ns}},
    view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampS.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column =
    column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>(input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_SECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_SECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampMilliSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampMs.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column = column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
    input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampMicroSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampUs.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column = column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>(
    input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MICROSECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MICROSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampNanoSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampNs.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column = column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>(
    input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_NANOSECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_NANOSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, FloatingPoint)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "5.6;0.5679e2;1.2e10;0.07e1;3000e-3;12.34e0;3.1e-001;-73."
               "98007199999998;";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<float>()})
      .lineterminator(';')
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::FLOAT32, view.column(0).type().id());

  auto const ref_vals =
    std::vector<float>{5.6, 56.79, 12000000000, 0.7, 3.000, 12.34, 0.31, -73.98007199999998};
  expect_column_data_equal(ref_vals, view.column(0));

  auto const bitmask = cudf::test::bitmask_to_host(view.column(0));
  ASSERT_EQ((1u << ref_vals.size()) - 1, bitmask[0]);
}

TEST_F(CsvReaderTest, Strings)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "Strings.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "10,abc def ghi" << '\n';
    outfile << "20,\"jkl mno pqr\"" << '\n';
    outfile << R"(30,stu ""vwx"" yz)" << '\n';
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"abc def ghi", "\"jkl mno pqr\"", R"(stu ""vwx"" yz)"},
    view.column(1));
}

TEST_F(CsvReaderTest, WindowsLineTerminators)
{
  std::string const buffer{"1,alpha\r\n2,beta\r\n"};
  auto options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .dtypes(std::vector<data_type>{data_type{type_id::INT64}, data_type{type_id::STRING}})
      .header(-1);

  auto const result = cudf::io::read_csv(options);

  cudf::test::fixed_width_column_wrapper<int64_t> const expected_values{1, 2};
  cudf::test::strings_column_wrapper const expected_names{"alpha", "beta"};
  table_view const expected{{expected_values, expected_names}};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, result.tbl->view());
}

TEST_F(CsvReaderTest, StringsQuotes)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotes.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "10,`abc,\ndef, ghi`" << '\n';
    outfile << "20,`jkl, ``mno``, pqr`" << '\n';
    outfile << "30,stu `vwx` yz" << '\n';
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quotechar('`');
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"abc,\ndef, ghi", "jkl, `mno`, pqr", "stu `vwx` yz"}, view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotesIgnored)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotesIgnored.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "10,\"abcdef ghi\"" << '\n';
    outfile << R"(20,"jkl ""mno"" pqr")" << '\n';
    outfile << "30,stu \"vwx\" yz" << '\n';
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE)
      .doublequote(false);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"\"abcdef ghi\"", R"("jkl ""mno"" pqr")", "stu \"vwx\" yz"},
    view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotesWhitespace)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotesIgnored.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "A,a" << '\n';              // unquoted no whitespace
    outfile << "    B,b" << '\n';          // unquoted leading whitespace
    outfile << "C    ,c" << '\n';          // unquoted trailing whitespace
    outfile << "    D    ,d" << '\n';      // unquoted leading and trailing whitespace
    outfile << "\"E\",e" << '\n';          // quoted no whitespace
    outfile << "\"F\"    ,f" << '\n';      // quoted trailing whitespace
    outfile << "    \"G\",g" << '\n';      // quoted leading whitespace
    outfile << "    \"H\"    ,h" << '\n';  // quoted leading and trailing whitespace
    outfile << "    \"    I    \"    ,i"
            << '\n';  // quoted leading and trailing whitespace with spaces inside quotes
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<cudf::string_view>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::ALL)
      .doublequote(false)
      .detect_whitespace_around_quotes(true);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  ASSERT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::STRING, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"A", "    B", "C    ", "    D    ", "E", "F", "G", "H", "    I    "},
    view.column(0));
  expect_column_data_equal(std::vector<std::string>{"a", "b", "c", "d", "e", "f", "g", "h", "i"},
                           view.column(1));
}

TEST_F(CsvReaderTest, SkiprowsNrows)
{
  auto filepath = temp_env->get_temp_dir() + "SkiprowsNrows.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n2\n3\n4\n5\n6\n7\n8\n9\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<int32_t>()})
      .header(1)
      .skiprows(2)
      .nrows(2);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{5, 6}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRange)
{
  auto filepath = temp_env->get_temp_dir() + "ByteRange.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1000\n2000\n3000\n4000\n5000\n6000\n7000\n8000\n9000\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<int32_t>()})
      .header(-1)
      .byte_range_offset(11)
      .byte_range_size(15);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{4000, 5000, 6000}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRangeStrings)
{
  std::string input = "\"a\"\n\"b\"\n\"c\"";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(input.c_str()), input.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A"})
      .dtypes({dtype<cudf::string_view>()})
      .header(-1)
      .byte_range_offset(4);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::STRING, view.column(0).type().id());

  expect_column_data_equal(std::vector<std::string>{"c"}, view.column(0));
}

TEST_F(CsvReaderTest, BlanksAndComments)
{
  auto filepath = temp_env->get_temp_dir() + "BlanksAndComments.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n#blank\n3\n4\n5\n#blank\n\n\n8\n9\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<int32_t>()})
      .header(-1)
      .comment('#');
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 3, 4, 5, 8, 9}, view.column(0));
}

TEST_F(CsvReaderTest, EmptyFile)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, NoDataFile)
{
  auto filepath = temp_env->get_temp_dir() + "NoDataFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\n\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, HeaderOnlyFile)
{
  auto filepath = temp_env->get_temp_dir() + "HeaderOnlyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\"a\",\"b\",\"c\"\n\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_rows());
  EXPECT_EQ(3, view.num_columns());
}

TEST_F(CsvReaderTest, InvalidFloatingPoint)
{
  auto const filepath = temp_env->get_temp_dir() + "InvalidFloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1.2e1+\n3.4e2-\n5.6e3e\n7.8e3A\n9.0Be1\n1C.2";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<float>()})
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::FLOAT32, view.column(0).type().id());

  // ignore all data because it is all nulls.
  ASSERT_EQ(6u, result.tbl->view().column(0).null_count());
}

TEST_F(CsvReaderTest, StringInference)
{
  std::string buffer = "\"-1\"\n";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), type_id::STRING);
}

TEST_F(CsvReaderTest, DelimWhitespaceNoHeaderLeadingTrailingDelimiter)
{
  std::string buffer = "  1   2  \n  3   4  \n";
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delim_whitespace(true)
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  ASSERT_EQ(result_view.num_columns(), 2);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::INT64);
  expect_column_data_equal(std::vector<int64_t>{1, 3}, result_view.column(0));
  expect_column_data_equal(std::vector<int64_t>{2, 4}, result_view.column(1));
}

// Exercises whitespace-shape variations (no padding, leading-only, trailing-only,
// internal-runs-only, leading+trailing+internal, and quoted header names with and
// without surrounding whitespace) that should all yield the same `(col_a, col_b)`
// schema and `[(1,2),(3,4)]` data under `delim_whitespace=true` (pandas parity).
class CsvDelimWhitespaceShapeTest : public CsvReaderTest,
                                    public ::testing::WithParamInterface<std::string> {};

TEST_P(CsvDelimWhitespaceShapeTest, ProducesTwoColumns)
{
  auto const buffer = GetParam();
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delim_whitespace(true)
      .header(0);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "col_a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col_b");
  ASSERT_EQ(result_view.num_columns(), 2);
  expect_column_data_equal(std::vector<int64_t>{1, 3}, result_view.column(0));
  expect_column_data_equal(std::vector<int64_t>{2, 4}, result_view.column(1));
}

INSTANTIATE_TEST_SUITE_P(CsvReaderTest,
                         CsvDelimWhitespaceShapeTest,
                         ::testing::Values(std::string{"col_a col_b\n1 2\n3 4\n"},
                                           std::string{"  col_a col_b\n  1 2\n  3 4\n"},
                                           std::string{"col_a col_b  \n1 2  \n3 4  \n"},
                                           std::string{"col_a   col_b\n1   2\n3   4\n"},
                                           std::string{"  col_a   col_b  \n  1   2  \n  3   4  \n"},
                                           std::string{"\"col_a\" \"col_b\"\n1 2\n3 4\n"},
                                           std::string{"  \"col_a\"   \"col_b\"  \n1 2\n3 4\n"}));

TEST_F(CsvReaderTest, TypeInferenceEmptyDelimitedFields)
{
  std::string const buffer = "1,,3\n4,,6\n";
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .na_filter(false)
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::STRING);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::INT64);

  expect_column_data_equal(std::vector<int64_t>{1, 4}, result_view.column(0));
  expect_column_data_equal(std::vector<std::string>{"", ""}, result_view.column(1));
  expect_column_data_equal(std::vector<int64_t>{3, 6}, result_view.column(2));
}

TEST_F(CsvReaderTest, MultiChunkRowCount)
{
  // TODO: add reader option to set chunk size and use it here
  constexpr size_t chunk_threshold = 64ull * 1024 * 1024;
  std::string const row            = "123,456,789\n";
  size_t const num_rows            = (chunk_threshold / row.size()) + 1024;

  std::string buffer;
  buffer.reserve(num_rows * row.size());
  for (size_t i = 0; i < num_rows; ++i) {
    buffer.append(row);
  }

  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(static_cast<size_t>(result_view.num_rows()), num_rows);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::INT64);

  // All rows are identical, so verifying min == max == expected
  auto const i64       = cudf::data_type{cudf::type_id::INT64};
  auto const min_agg   = cudf::make_min_aggregation<cudf::reduce_aggregation>();
  auto const max_agg   = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto const all_equal = [&](cudf::column_view const& col, int64_t expected) {
    using scalar_t = cudf::numeric_scalar<int64_t>;
    auto const min = cudf::reduce(col, *min_agg, i64);
    auto const max = cudf::reduce(col, *max_agg, i64);
    return static_cast<scalar_t const&>(*min).value() == expected &&
           static_cast<scalar_t const&>(*max).value() == expected;
  };
  EXPECT_TRUE(all_equal(result_view.column(0), 123));
  EXPECT_TRUE(all_equal(result_view.column(1), 456));
  EXPECT_TRUE(all_equal(result_view.column(2), 789));

  // The whole file is parsed as a single chunk; selecting rows parses it in chunks
  auto chunked_opts = in_opts;
  chunked_opts.set_nrows(static_cast<cudf::size_type>(num_rows));
  CUDF_TEST_EXPECT_TABLES_EQUAL(result_view, cudf::io::read_csv(chunked_opts).tbl->view());
}

TEST_F(CsvReaderTest, TypeInferenceThousands)
{
  std::string buffer = "1`400,123,1`234.56\n123`456,123456,12.34";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .thousands('`');
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  EXPECT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::FLOAT64);

  auto tsnd_sep_col = std::vector<int64_t>{1400L, 123456L};
  auto int_col      = std::vector<int64_t>{123L, 123456L};
  auto dbl_col      = std::vector<double>{1234.56, 12.34};
  expect_column_data_equal(tsnd_sep_col, result_view.column(0));
  expect_column_data_equal(int_col, result_view.column(1));
  expect_column_data_equal(dbl_col, result_view.column(2));
}

TEST_F(CsvReaderTest, TypeInferenceWithDecimal)
{
  // Given that thousands:'`' and decimal(';'), we expect:
  // col#0 => INT64 (column contains only digits & thousands sep)
  // col#1 => STRING (contains digits and period character, which is NOT the decimal point here)
  // col#2 => FLOAT64 (column contains digits and decimal point (i.e., ';'))
  std::string buffer = "1`400,1.23,1`234;56\n123`456,123.456,12;34";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .thousands('`')
      .decimal(';');
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  EXPECT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::STRING);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::FLOAT64);

  auto int_col = std::vector<int64_t>{1400L, 123456L};
  auto str_col = std::vector<std::string>{"1.23", "123.456"};
  auto dbl_col = std::vector<double>{1234.56, 12.34};
  expect_column_data_equal(int_col, result_view.column(0));
  expect_column_data_equal(str_col, result_view.column(1));
  expect_column_data_equal(dbl_col, result_view.column(2));
}

TEST_F(CsvReaderTest, SkipRowsXorSkipFooter)
{
  std::string buffer = "1,2,3";

  cudf::io::csv_reader_options skiprows_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .skiprows(1);
  EXPECT_NO_THROW(cudf::io::read_csv(skiprows_options));

  cudf::io::csv_reader_options skipfooter_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .skipfooter(1);
  EXPECT_NO_THROW(cudf::io::read_csv(skipfooter_options));
}

TEST_F(CsvReaderTest, nullHandling)
{
  auto const filepath = temp_env->get_temp_dir() + "NullValues.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "NULL\n\nnull\nn/a\nNull\nNA\nnan";
  }

  // Test disabling na_filter
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .na_filter(false)
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Test enabling na_filter
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {false, false, false, false, true, false, false});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Setting na_values with default values
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .na_values({"Null"})
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {false, false, false, false, false, false, false});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Setting na_values without default values
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .keep_default_na(false)
        .na_values({"Null"})
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {true, true, true, true, false, true, true, true});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Filter enabled, but no NA values
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .keep_default_na(false)
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }
}

TEST_F(CsvReaderTest, FailCases)
{
  std::string buffer = "1,2,3";
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_offset(4)
                   .skiprows(1),
                 std::invalid_argument);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_offset(4)
                   .skipfooter(1),
                 std::invalid_argument);
  }

  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_offset(4)
                   .nrows(1),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_size(4)
                   .skiprows(1),
                 std::invalid_argument);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_size(4)
                   .skipfooter(1),
                 std::invalid_argument);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_size(4)
                   .nrows(1),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skiprows(1)
                   .byte_range_offset(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skipfooter(1)
                   .byte_range_offset(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .nrows(1)
                   .byte_range_offset(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skiprows(1)
                   .byte_range_size(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skipfooter(1)
                   .byte_range_size(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .nrows(1)
                   .byte_range_size(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .nrows(1)
                   .skipfooter(1),
                 std::invalid_argument);
    ;
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skipfooter(1)
                   .nrows(1),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .na_filter(false)
                   .na_values({"Null"}),
                 cudf::logic_error);
  }
}

TEST_F(CsvReaderTest, HexTest)
{
  auto filepath = temp_env->get_temp_filepath("Hexadecimal.csv");
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "0x0\n-0x1000\n0xfedcba\n0xABCDEF\n0xaBcDeF\n9512c20b\n";
  }
  // specify hex columns by name
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .dtypes({dtype<int64_t>()})
        .header(-1)
        .parse_hex({"A"});
    auto result = cudf::io::read_csv(in_opts);

    expect_column_data_equal(
      std::vector<int64_t>{0, -4096, 16702650, 11259375, 11259375, 2501034507},
      result.tbl->view().column(0));
  }

  // specify hex columns by index
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .dtypes({dtype<int64_t>()})
        .header(-1)
        .parse_hex(std::vector<int>{0});
    auto result = cudf::io::read_csv(in_opts);

    expect_column_data_equal(
      std::vector<int64_t>{0, -4096, 16702650, 11259375, 11259375, 2501034507},
      result.tbl->view().column(0));
  }
}

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumnWithWriter)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });
  auto input_column = column_wrapper<TypeParam>(sequence, sequence + num_rows);
  auto input_table  = cudf::table_view{std::vector<cudf::column_view>{input_column}};

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithWriter.csv");

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, MultiColumnWithWriter)
{
  constexpr auto num_rows = 10;
  auto int8_column        = []() {
    auto values = random_values<int8_t>(num_rows);
    return column_wrapper<int8_t>(values.begin(), values.end());
  }();
  auto int16_column = []() {
    auto values = random_values<int16_t>(num_rows);
    return column_wrapper<int16_t>(values.begin(), values.end());
  }();
  auto int32_column = []() {
    auto values = random_values<int32_t>(num_rows);
    return column_wrapper<int32_t>(values.begin(), values.end());
  }();
  auto int64_column = []() {
    auto values = random_values<int64_t>(num_rows);
    return column_wrapper<int64_t>(values.begin(), values.end());
  }();
  auto uint8_column = []() {
    auto values = random_values<uint8_t>(num_rows);
    return column_wrapper<uint8_t>(values.begin(), values.end());
  }();
  auto uint16_column = []() {
    auto values = random_values<uint16_t>(num_rows);
    return column_wrapper<uint16_t>(values.begin(), values.end());
  }();
  auto uint32_column = []() {
    auto values = random_values<uint32_t>(num_rows);
    return column_wrapper<uint32_t>(values.begin(), values.end());
  }();
  auto uint64_column = []() {
    auto values = random_values<uint64_t>(num_rows);
    return column_wrapper<uint64_t>(values.begin(), values.end());
  }();
  auto float32_column = []() {
    auto values = random_values<float>(num_rows);
    return column_wrapper<float>(values.begin(), values.end());
  }();
  auto float64_column = []() {
    auto values = random_values<double>(num_rows);
    return column_wrapper<double>(values.begin(), values.end());
  }();

  std::vector<cudf::column_view> input_columns{int8_column,
                                               int16_column,
                                               int32_column,
                                               int64_column,
                                               uint8_column,
                                               uint16_column,
                                               uint32_column,
                                               uint64_column,
                                               float32_column,
                                               float64_column};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_dir() + "MultiColumnWithWriter.csv";

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .header(-1)
      .dtypes({dtype<int8_t>(),
               dtype<int16_t>(),
               dtype<int32_t>(),
               dtype<int64_t>(),
               dtype<uint8_t>(),
               dtype<uint16_t>(),
               dtype<uint32_t>(),
               dtype<uint64_t>(),
               dtype<float>(),
               dtype<double>()});
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();

  std::vector<cudf::size_type> non_float64s{0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto const input_sliced_view  = input_table.select(non_float64s);
  auto const result_sliced_view = result_table.select(non_float64s);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_sliced_view, result_sliced_view);

  auto validity = cudf::test::iterators::no_nulls();
  double tol{1.0e-6};
  auto float64_col_idx = non_float64s.size();
  check_float_column(
    input_table.column(float64_col_idx), result_table.column(float64_col_idx), tol, validity);
}

TEST_F(CsvReaderTest, DatesWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "DatesWithWriter.csv";

  auto input_column = column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{983750400000,
                                                                                  1288483200000,
                                                                                  782611200000,
                                                                                  656208000000,
                                                                                  0L,
                                                                                  798163200000,
                                                                                  774144000000,
                                                                                  1149679230400,
                                                                                  1126875750400,
                                                                                  2764800000};
  cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

  // TODO need to add a dayfirst flag?
  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();

  check_timestamp_column(input_table.column(0), result_table.column(0));
}

TEST_F(CsvReaderTest, DatesStringWithWriter)
{
  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_D.csv";

    auto input_column = column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{-106751, 106751};
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-22", "2262-04-11"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_s.csv";

    auto input_column =
      column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{-9223372036, 9223372036};
    auto expected_column =
      column_wrapper<cudf::string_view>{"1677-09-21T00:12:44Z", "2262-04-11T23:47:16Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_ms.csv";

    auto input_column =
      column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{-9223372036854, 9223372036854};
    auto expected_column =
      column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.146Z", "2262-04-11T23:47:16.854Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_us.csv";

    auto input_column = column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>{
      -9223372036854775, 9223372036854775};
    auto cast_column     = cudf::strings::from_timestamps(input_column, "%Y-%m-%dT%H:%M:%S.%fZ");
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.145225Z",
                                                             "2262-04-11T23:47:16.854775Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_ns.csv";

    auto input_column = column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>{
      -9223372036854775807, 9223372036854775807};
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.145224193Z",
                                                             "2262-04-11T23:47:16.854775807Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }
}

TEST_F(CsvReaderTest, FloatingPointWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPointWithWriter.csv";

  auto input_column =
    column_wrapper<double>{5.6, 56.79, 12000000000., 0.7, 3.000, 12.34, 0.31, -73.98007199999998};
  cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

  // TODO add lineterminator=";"
  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<double>()})
      .header(-1);
  // in_opts.lineterminator = ';';
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, StringsWithWriter)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriter.csv";

  auto int_column = column_wrapper<int32_t>{10, 20, 30};
  auto string_column =
    column_wrapper<cudf::string_view>{"abc def ghi", "\"jkl mno pqr\"", R"(stu ""vwx"" yz)"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  // TODO add quoting style flag?
  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_table.column(0), result_table.column(0));
  check_string_column(input_table.column(1), result_table.column(1));
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, StringsWithWriterSimple)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriterSimple.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc def ghi", "jkl mno pq", "stu vwx y"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  // TODO add quoting style flag?
  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_table.column(0), result_table.column(0));
  check_string_column(input_table.column(1), result_table.column(1));
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, StringsEmbeddedDelimiter)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriterSimple.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc def ghi", "jkl,mno,pq", "stu vwx y"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto result = cudf::io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, HeaderEmbeddedDelimiter)
{
  std::vector<std::string> names{
    "header1", "header,2", "quote\"embedded", "new\nline", "\"quoted\""};

  auto filepath = temp_env->get_temp_dir() + "HeaderEmbeddedDelimiter.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc", "jkl,mno", "xyz"};
  cudf::table_view input_table(
    std::vector<cudf::column_view>{int_column, string_column, int_column, int_column, int_column});

  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes({dtype<int32_t>(),
               dtype<cudf::string_view>(),
               dtype<int32_t>(),
               dtype<int32_t>(),
               dtype<int32_t>()});
  auto result = cudf::io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, EmptyFileWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFileWithWriter.csv";

  cudf::table_view empty_table;
  write_csv_helper(filepath, empty_table);
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty_table, result.tbl->view());
}

class TestSource : public cudf::io::datasource {
 public:
  std::string const str;

  TestSource(std::string s) : str(std::move(s)) {}
  std::unique_ptr<buffer> host_read(std::size_t offset, std::size_t size) override
  {
    size = std::min(size, str.size() - offset);
    return std::make_unique<non_owning_buffer>((uint8_t*)str.data() + offset, size);
  }

  std::size_t host_read(std::size_t offset, std::size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, str.size() - offset);
    memcpy(dst, str.data() + offset, size);
    return read_size;
  }

  [[nodiscard]] std::size_t size() const override { return str.size(); }
};

TEST_F(CsvReaderTest, UserImplementedSource)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);

  std::ostringstream csv_data;
  for (int i = 0; i < num_rows; ++i) {
    csv_data << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int32_values[i]
             << "\n";
  }
  TestSource source{csv_data.str()};
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{&source})
      .compression(cudf::io::compression_type::NONE)
      .dtypes({dtype<int8_t>(), dtype<int16_t>(), dtype<int32_t>()})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int32_values, view.column(2));
}

TEST_F(CsvReaderTest, DurationsWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "DurationsWithWriter.csv";

  constexpr long max_value_d  = std::numeric_limits<cudf::duration_D::rep>::max();
  constexpr long min_value_d  = std::numeric_limits<cudf::duration_D::rep>::min();
  constexpr long max_value_ns = std::numeric_limits<cudf::duration_s::rep>::max();
  constexpr long min_value_ns = std::numeric_limits<cudf::duration_s::rep>::min();
  column_wrapper<cudf::duration_D, cudf::duration_D::rep> durations_D{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_d, max_value_d}};
  column_wrapper<cudf::duration_s, int64_t> durations_s{{-86400L,
                                                         -3600L,
                                                         -2L,
                                                         -1L,
                                                         0L,
                                                         1L,
                                                         2L,
                                                         min_value_ns / 1000000000 + 1,
                                                         max_value_ns / 1000000000}};
  column_wrapper<cudf::duration_ms, int64_t> durations_ms{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns / 1000000 + 1, max_value_ns / 1000000}};
  column_wrapper<cudf::duration_us, int64_t> durations_us{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns / 1000 + 1, max_value_ns / 1000}};
  column_wrapper<cudf::duration_ns, int64_t> durations_ns{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns, max_value_ns}};

  cudf::table_view input_table(std::vector<cudf::column_view>{
    durations_D, durations_s, durations_ms, durations_us, durations_ns});
  std::vector<std::string> names{"D", "s", "ms", "us", "ns"};

  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes({data_type{type_id::DURATION_DAYS},
               data_type{type_id::DURATION_SECONDS},
               data_type{type_id::DURATION_MILLISECONDS},
               data_type{type_id::DURATION_MICROSECONDS},
               data_type{type_id::DURATION_NANOSECONDS}});
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, ParseInRangeIntegers)
{
  std::vector<int64_t> small_int               = {0, -10, 20, -30};
  std::vector<int64_t> less_equal_int64_max    = {std::numeric_limits<int64_t>::max() - 3,
                                                  std::numeric_limits<int64_t>::max() - 2,
                                                  std::numeric_limits<int64_t>::max() - 1,
                                                  std::numeric_limits<int64_t>::max()};
  std::vector<int64_t> greater_equal_int64_min = {std::numeric_limits<int64_t>::min() + 3,
                                                  std::numeric_limits<int64_t>::min() + 2,
                                                  std::numeric_limits<int64_t>::min() + 1,
                                                  std::numeric_limits<int64_t>::min()};
  std::vector<uint64_t> greater_int64_max      = {uint64_t{std::numeric_limits<int64_t>::max()} - 1,
                                                  uint64_t{std::numeric_limits<int64_t>::max()},
                                                  uint64_t{std::numeric_limits<int64_t>::max()} + 1,
                                                  uint64_t{std::numeric_limits<int64_t>::max()} + 2};
  std::vector<uint64_t> less_equal_uint64_max  = {std::numeric_limits<uint64_t>::max() - 3,
                                                  std::numeric_limits<uint64_t>::max() - 2,
                                                  std::numeric_limits<uint64_t>::max() - 1,
                                                  std::numeric_limits<uint64_t>::max()};
  auto input_small_int = column_wrapper<int64_t>(small_int.begin(), small_int.end());
  auto input_less_equal_int64_max =
    column_wrapper<int64_t>(less_equal_int64_max.begin(), less_equal_int64_max.end());
  auto input_greater_equal_int64_min =
    column_wrapper<int64_t>(greater_equal_int64_min.begin(), greater_equal_int64_min.end());
  auto input_greater_int64_max =
    column_wrapper<uint64_t>(greater_int64_max.begin(), greater_int64_max.end());
  auto input_less_equal_uint64_max =
    column_wrapper<uint64_t>(less_equal_uint64_max.begin(), less_equal_uint64_max.end());

  auto small_int_append_zeros               = prepend_zeros(small_int, 32, true);
  auto less_equal_int64_max_append_zeros    = prepend_zeros(less_equal_int64_max, 32, true);
  auto greater_equal_int64_min_append_zeros = prepend_zeros(greater_equal_int64_min, 17);
  auto greater_int64_max_append_zeros       = prepend_zeros(greater_int64_max, 5);
  auto less_equal_uint64_max_append_zeros   = prepend_zeros(less_equal_uint64_max, 8, true);

  auto input_small_int_append =
    column_wrapper<cudf::string_view>(small_int_append_zeros.begin(), small_int_append_zeros.end());
  auto input_less_equal_int64_max_append = column_wrapper<cudf::string_view>(
    less_equal_int64_max_append_zeros.begin(), less_equal_int64_max_append_zeros.end());
  auto input_greater_equal_int64_min_append = column_wrapper<cudf::string_view>(
    greater_equal_int64_min_append_zeros.begin(), greater_equal_int64_min_append_zeros.end());
  auto input_greater_int64_max_append = column_wrapper<cudf::string_view>(
    greater_int64_max_append_zeros.begin(), greater_int64_max_append_zeros.end());
  auto input_less_equal_uint64_max_append = column_wrapper<cudf::string_view>(
    less_equal_uint64_max_append_zeros.begin(), less_equal_uint64_max_append_zeros.end());

  std::vector<cudf::column_view> input_columns{input_small_int,
                                               input_less_equal_int64_max,
                                               input_greater_equal_int64_min,
                                               input_greater_int64_max,
                                               input_less_equal_uint64_max,
                                               input_small_int_append,
                                               input_less_equal_int64_max_append,
                                               input_greater_equal_int64_min_append,
                                               input_greater_int64_max_append,
                                               input_less_equal_uint64_max_append};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_filepath("ParseInRangeIntegers.csv");

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_small_int, view.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_int64_max, view.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_equal_int64_min, view.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_int64_max, view.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_uint64_max, view.column(4));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_small_int, view.column(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_int64_max, view.column(6));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_equal_int64_min, view.column(7));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_int64_max, view.column(8));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_uint64_max, view.column(9));
}

TEST_F(CsvReaderTest, ParseOutOfRangeIntegers)
{
  std::vector<std::string> out_of_range_positive = {"111111111111111111111",
                                                    "2222222222222222222222",
                                                    "33333333333333333333333",
                                                    "444444444444444444444444"};
  std::vector<std::string> out_of_range_negative = {"-111111111111111111111",
                                                    "-2222222222222222222222",
                                                    "-33333333333333333333333",
                                                    "-444444444444444444444444"};
  std::vector<std::string> greater_uint64_max    = {
    "18446744073709551615", "18446744073709551616", "18446744073709551617", "18446744073709551618"};
  std::vector<std::string> less_int64_min = {
    "-9223372036854775807", "-9223372036854775808", "-9223372036854775809", "-9223372036854775810"};
  std::vector<std::string> mixed_range = {
    "18446744073709551613", "18446744073709551614", "18446744073709551615", "-5"};
  auto input_out_of_range_positive =
    column_wrapper<cudf::string_view>(out_of_range_positive.begin(), out_of_range_positive.end());
  auto input_out_of_range_negative =
    column_wrapper<cudf::string_view>(out_of_range_negative.begin(), out_of_range_negative.end());
  auto input_greater_uint64_max =
    column_wrapper<cudf::string_view>(greater_uint64_max.begin(), greater_uint64_max.end());
  auto input_less_int64_min =
    column_wrapper<cudf::string_view>(less_int64_min.begin(), less_int64_min.end());
  auto input_mixed_range =
    column_wrapper<cudf::string_view>(mixed_range.begin(), mixed_range.end());

  auto out_of_range_positive_append_zeros = prepend_zeros(out_of_range_positive, 32, true);
  auto out_of_range_negative_append_zeros = prepend_zeros(out_of_range_negative, 5);
  auto greater_uint64_max_append_zeros    = prepend_zeros(greater_uint64_max, 8, true);
  auto less_int64_min_append_zeros        = prepend_zeros(less_int64_min, 17);
  auto mixed_range_append_zeros           = prepend_zeros(mixed_range, 2, true);

  auto input_out_of_range_positive_append = column_wrapper<cudf::string_view>(
    out_of_range_positive_append_zeros.begin(), out_of_range_positive_append_zeros.end());
  auto input_out_of_range_negative_append = column_wrapper<cudf::string_view>(
    out_of_range_negative_append_zeros.begin(), out_of_range_negative_append_zeros.end());
  auto input_greater_uint64_max_append = column_wrapper<cudf::string_view>(
    greater_uint64_max_append_zeros.begin(), greater_uint64_max_append_zeros.end());
  auto input_less_int64_min_append = column_wrapper<cudf::string_view>(
    less_int64_min_append_zeros.begin(), less_int64_min_append_zeros.end());
  auto input_mixed_range_append = column_wrapper<cudf::string_view>(
    mixed_range_append_zeros.begin(), mixed_range_append_zeros.end());

  std::vector<cudf::column_view> input_columns{input_out_of_range_positive,
                                               input_out_of_range_negative,
                                               input_greater_uint64_max,
                                               input_less_int64_min,
                                               input_mixed_range,
                                               input_out_of_range_positive_append,
                                               input_out_of_range_negative_append,
                                               input_greater_uint64_max_append,
                                               input_less_int64_min_append,
                                               input_mixed_range_append};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_filepath("ParseOutOfRangeIntegers.csv");

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_positive, view.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_negative, view.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_uint64_max, view.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_int64_min, view.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_mixed_range, view.column(4));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_positive_append, view.column(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_negative_append, view.column(6));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_uint64_max_append, view.column(7));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_int64_min_append, view.column(8));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_mixed_range_append, view.column(9));
}

TEST_F(CsvReaderTest, ReadMaxNumericValue)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::numeric_limits<uint64_t>::max() - i; });

  auto filepath = temp_env->get_temp_filepath("ReadMaxNumericValue.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<uint64_t> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(std::vector<uint64_t>(sequence, sequence + num_rows), view.column(0));
}

TEST_F(CsvReaderTest, DefaultWriteChunkSize)
{
  for (auto num_rows : {1, 20, 100, 1000}) {
    auto sequence = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<int32_t>(i + 1000.50f); });
    auto input_column = column_wrapper<int32_t>(sequence, sequence + num_rows);
    auto input_table  = cudf::table_view{std::vector<cudf::column_view>{input_column}};

    cudf::io::csv_writer_options opts =
      cudf::io::csv_writer_options::builder(cudf::io::sink_info{"unused.path"}, input_table);
    ASSERT_EQ(num_rows, opts.get_rows_per_chunk());
  }
}

TEST_F(CsvReaderTest, DtypesMap)
{
  std::string csv_in{"12,9\n34,8\n56,7"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A", "B"})
      .dtypes({{"B", dtype<int16_t>()}, {"A", dtype<int32_t>()}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  ASSERT_EQ(result_table.num_columns(), 2);
  ASSERT_EQ(result_table.column(0).type(), data_type{type_id::INT32});
  ASSERT_EQ(result_table.column(1).type(), data_type{type_id::INT16});
  expect_column_data_equal(std::vector<int32_t>{12, 34, 56}, result_table.column(0));
  expect_column_data_equal(std::vector<int16_t>{9, 8, 7}, result_table.column(1));
}

TEST_F(CsvReaderTest, DtypesMapPartial)
{
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{nullptr, 0}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A", "B"})
      .dtypes({{"A", dtype<int16_t>()}});
  {
    auto result = cudf::io::read_csv(in_opts);

    auto const view = result.tbl->view();
    ASSERT_EQ(type_id::INT16, view.column(0).type().id());
    // Default to String if there's no data
    ASSERT_EQ(type_id::STRING, view.column(1).type().id());
  }

  in_opts.set_dtypes({{"B", dtype<uint32_t>()}});
  {
    auto result = cudf::io::read_csv(in_opts);

    auto const view = result.tbl->view();
    ASSERT_EQ(type_id::STRING, view.column(0).type().id());
    ASSERT_EQ(type_id::UINT32, view.column(1).type().id());
  }
}

TEST_F(CsvReaderTest, DtypesArrayInvalid)
{
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{nullptr, 0}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A", "B", "C"})
      .dtypes(std::vector<cudf::data_type>{dtype<int16_t>(), dtype<int8_t>()});

  EXPECT_THROW(cudf::io::read_csv(in_opts), cudf::logic_error);
}

TEST_F(CsvReaderTest, CsvDefaultOptionsWriteReadMatch)
{
  auto const filepath = temp_env->get_temp_dir() + "issue.csv";

  // make up some kind of dataframe
  auto int_column = column_wrapper<int32_t>{10, 20, 30};
  auto str_column = column_wrapper<cudf::string_view>{"abc", "mno", "xyz"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, str_column});

  // write that dataframe to a csv using default options to some temporary file
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, input_table);
  cudf::io::write_csv(writer_options);

  // read the temp csv file using default options
  cudf::io::csv_reader_options read_options =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});

  cudf::io::table_with_metadata new_table_and_metadata = cudf::io::read_csv(read_options);

  // verify that the tables are identical, or as identical as expected.
  auto const new_table_view = new_table_and_metadata.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, new_table_view);
  EXPECT_EQ(new_table_and_metadata.metadata.schema_info[0].name, "0");
  EXPECT_EQ(new_table_and_metadata.metadata.schema_info[1].name, "1");
}

TEST_F(CsvReaderTest, UseColsValidation)
{
  std::string const buffer = "1,2,3";

  cudf::io::csv_reader_options const idx_cnt_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b"})
      .use_cols_indexes({0});
  EXPECT_THROW(cudf::io::read_csv(idx_cnt_options), cudf::logic_error);

  cudf::io::csv_reader_options unique_idx_cnt_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b"})
      .use_cols_indexes({0, 0});
  EXPECT_THROW(cudf::io::read_csv(unique_idx_cnt_options), cudf::logic_error);

  cudf::io::csv_reader_options bad_name_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b", "c"})
      .use_cols_names({"nonexistent_name"});
  EXPECT_THROW(cudf::io::read_csv(bad_name_options), cudf::logic_error);
}

TEST_F(CsvReaderTest, CropColumns)
{
  std::string const csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<float>()})
      .names({"a", "b"})
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  ASSERT_EQ(result_table.num_columns(), 2);
  ASSERT_EQ(result_table.column(0).type(), data_type{type_id::INT32});
  ASSERT_EQ(result_table.column(1).type(), data_type{type_id::FLOAT32});
  expect_column_data_equal(std::vector<int32_t>{12, 34, 56}, result_table.column(0));
  expect_column_data_equal(std::vector<float>{9., 8., 7.}, result_table.column(1));
}

TEST_F(CsvReaderTest, CropColumnsUseColsNames)
{
  std::string csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<float>()})
      .names({"a", "b"})
      .use_cols_names({"b"})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  ASSERT_EQ(result_table.num_columns(), 1);
  ASSERT_EQ(result_table.column(0).type(), data_type{type_id::FLOAT32});
  expect_column_data_equal(std::vector<float>{9., 8., 7.}, result_table.column(0));
}

TEST_F(CsvReaderTest, ExtraColumns)
{
  std::string csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};
  {
    cudf::io::csv_reader_options opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .header(-1);
    auto result = cudf::io::read_csv(opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 4);
    ASSERT_EQ(result_table.column(3).type(), data_type{type_id::INT8});
    ASSERT_EQ(result_table.column(3).null_count(), 3);
  }
  {
    cudf::io::csv_reader_options with_dtypes_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .dtypes({dtype<int32_t>(), dtype<int32_t>(), dtype<int32_t>(), dtype<float>()})
        .header(-1);
    auto result = cudf::io::read_csv(with_dtypes_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 4);
    ASSERT_EQ(result_table.column(3).type(), data_type{type_id::FLOAT32});
    ASSERT_EQ(result_table.column(3).null_count(), 3);
  }
}

TEST_F(CsvReaderTest, ExtraColumnsUseCols)
{
  std::string csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};

  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .use_cols_names({"b", "d"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 2);
    ASSERT_EQ(result_table.column(1).type(), data_type{type_id::INT8});
    ASSERT_EQ(result_table.column(1).null_count(), 3);
  }
  {
    cudf::io::csv_reader_options with_dtypes_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .use_cols_names({"b", "d"})
        .dtypes({dtype<int32_t>(), dtype<int32_t>(), dtype<int32_t>(), dtype<cudf::string_view>()})
        .header(-1);
    auto result = cudf::io::read_csv(with_dtypes_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 2);
    ASSERT_EQ(result_table.column(1).type(), data_type{type_id::STRING});
    ASSERT_EQ(result_table.column(1).null_count(), 3);
  }
}

TEST_F(CsvReaderTest, EmptyColumns)
{
  // First column only has empty fields. second column contains only "null" literals
  std::string csv_in{",null\n,null"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b", "c", "d"})
      .header(-1);
  // More elements in `names` than in the file; additional columns are filled with nulls
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  EXPECT_EQ(result_table.num_columns(), 4);
  // All columns should contain only nulls; expect INT8 type to use as little memory as possible
  for (auto& column : result_table) {
    EXPECT_EQ(column.type(), data_type{type_id::INT8});
    EXPECT_EQ(column.null_count(), 2);
  }
}

TEST_F(CsvReaderTest, BlankLineAfterFirstRow)
{
  std::string csv_in{"12,9., 10\n\n"};

  {
    cudf::io::csv_reader_options no_header_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .header(-1);
    // No header, getting column names/count from first row
    auto result = cudf::io::read_csv(no_header_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 3);
  }
  {
    cudf::io::csv_reader_options header_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE);
    // Getting column names/count from header
    auto result = cudf::io::read_csv(header_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 3);
  }
}

TEST_F(CsvReaderTest, NullCount)
{
  std::string buffer = "0,,\n1,1.,\n2,,\n3,,\n4,4.,\n5,5.,\n6,6.,\n7,7.,\n";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  EXPECT_EQ(result_view.num_rows(), 8);
  EXPECT_EQ(result_view.column(0).null_count(), 0);
  EXPECT_EQ(result_view.column(1).null_count(), 3);
  EXPECT_EQ(result_view.column(2).null_count(), 8);
}

TEST_F(CsvReaderTest, UTF8BOM)
{
  std::string buffer = "\xEF\xBB\xBFMonth,Day,Year\nJune,6,2023\nAugust,25,1990\nMay,1,2000\n";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();
  EXPECT_EQ(result_view.num_rows(), 3);
  EXPECT_EQ(result.metadata.schema_info.front().name, "Month");

  auto col1     = cudf::test::strings_column_wrapper({"June", "August", "May"});
  auto col2     = cudf::test::fixed_width_column_wrapper<int64_t>({6, 25, 1});
  auto col3     = cudf::test::fixed_width_column_wrapper<int64_t>({2023, 1990, 2000});
  auto expected = cudf::table_view({col1, col2, col3});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result_view, expected);
}

void expect_buffers_equal(cudf::io::datasource::buffer* lhs, cudf::io::datasource::buffer* rhs)
{
  ASSERT_EQ(lhs->size(), rhs->size());
  EXPECT_EQ(0, std::memcmp(lhs->data(), rhs->data(), lhs->size()));
}

TEST_F(CsvReaderTest, OutOfMapBoundsReads)
{
  // write a lot of data into a file
  auto filepath        = temp_env->get_temp_dir() + "OutOfMapBoundsReads.csv";
  auto const num_rows  = 1 << 20;
  auto const row       = std::string{"0,1,2,3,4,5,6,7,8,9\n"};
  auto const file_size = num_rows * row.size();
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (std::size_t i = 0; i < num_rows; ++i) {
      outfile << row;
    }
  }

  // Only memory map the middle of the file
  auto source         = cudf::io::datasource::create(filepath, file_size / 2, file_size / 4);
  auto full_source    = cudf::io::datasource::create(filepath);
  auto const all_data = source->host_read(0, file_size);
  auto ref_data       = full_source->host_read(0, file_size);
  expect_buffers_equal(ref_data.get(), all_data.get());

  auto const start_data = source->host_read(file_size / 2, file_size / 2);
  expect_buffers_equal(full_source->host_read(file_size / 2, file_size / 2).get(),
                       start_data.get());

  auto const end_data = source->host_read(0, file_size / 2 + 512);
  expect_buffers_equal(full_source->host_read(0, file_size / 2 + 512).get(), end_data.get());
}

struct CsvWriterTypeSupportTest : public cudf::test::BaseFixture {};

TEST(CsvWriterTypeSupportTest, SupportedTypes)
{
  using cudf::io::is_supported_write_csv;

  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::INT32}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::FLOAT64}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::STRING}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::DECIMAL64}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::DURATION_SECONDS}));
}

TEST(CsvWriterTypeSupportTest, UnsupportedTypes)
{
  using cudf::io::is_supported_write_csv;

  EXPECT_FALSE(is_supported_write_csv(cudf::data_type{cudf::type_id::LIST}));
  EXPECT_FALSE(is_supported_write_csv(cudf::data_type{cudf::type_id::STRUCT}));
  EXPECT_FALSE(is_supported_write_csv(cudf::data_type{cudf::type_id::DICTIONARY32}));
}

TEST_F(CsvReaderTest, DoubleQuotesOddCount)
{
  std::string const content{R"(Monday" "Tuesday" Wednesday)"};
  std::string const buffer = content + "\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .dtypes({dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);
  EXPECT_EQ(result.tbl->view().num_columns(), 1);

  auto const expected = cudf::test::strings_column_wrapper({content});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), expected);
}

TEST_F(CsvReaderTest, UnquotedStringDoubledQuotes)
{
  // Tab delimiter makes the entire line a single unquoted field containing commas and quotes
  std::string const buffer = R"(hello,"",world
foo
bar
end
)";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .dtypes({dtype<cudf::string_view>()})
      .delimiter('\t');
  auto const result = cudf::io::read_csv(in_opts);
  EXPECT_EQ(result.tbl->view().num_columns(), 1);
  EXPECT_EQ(result.tbl->view().num_rows(), 4);

  auto const expected =
    cudf::test::strings_column_wrapper({R"(hello,"",world)", R"(foo)", R"(bar)", R"(end)"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), expected);
}

TEST_F(CsvReaderTest, QuotedFieldWithTrailingDelimiter)
{
  std::string const buffer = R"(a,b
1,"trailing,"
2,"normal"
3,end
)";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 3);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto const expected_col1 = cudf::test::strings_column_wrapper({"trailing,", "normal", "end"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
}

TEST_F(CsvReaderTest, EscapedQuotesWithSemicolonDelimiter)
{
  std::string const buffer = R"(a;b
1;"hello""world"
2;"test"
3;"end"
)";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delimiter(';')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 3);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto const expected_col1 = cudf::test::strings_column_wrapper({"hello\"world", "test", "end"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
}

TEST_F(CsvReaderTest, EscapedQuoteBeforeDelimiterInQuotedField)
{
  std::string const buffer =
    "a;b\n"
    "1;\"hello\"\";\"\n"  // hello"; escaped as "hello"";"
    "2;\"test\"\n"
    "3;\"end\"\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delimiter(';')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 3);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);
}

TEST_F(CsvReaderTest, CommentLines)
{
  std::string const buffer =
    "# This is a comment\n"
    "a,b,c\n"
    "1,2,3\n"
    "# Another comment\n"
    "4,5,6\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .comment('#')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<int32_t>(), dtype<int32_t>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 2);
  EXPECT_EQ(result.tbl->view().num_columns(), 3);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 4});
  auto const expected_col1 = cudf::test::fixed_width_column_wrapper<int32_t>({2, 5});
  auto const expected_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({3, 6});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(2), expected_col2);
}

TEST_F(CsvReaderTest, CommentLinesWithQuotedStrings)
{
  std::string const buffer =
    "# comment\n"
    "a,b\n"
    "1,\"hello\"\"world\"\n"
    "# another comment\n"
    "2,\"test\"\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .comment('#')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 2);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  auto const expected_col1 = cudf::test::strings_column_wrapper({"hello\"world", "test"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
}

TEST_F(CsvReaderTest, TimestampsWithIncompleteTimeOfDay)
{
  // Each time of day but the last lacks separators or digits. The last row is well-formed and
  // holds the separators that a parser reading past the end of the preceding fields would find.
  std::string const buffer =
    "2024-01-01T10\n"
    "2024-01-01T10PM\n"
    "2024-01-01T\n"
    "2024-01-01 M\n"
    "2024-01-01T11:30:00.500\n";

  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  using namespace cuda::std::chrono_literals;
  auto constexpr midnight = 1704067200000ms;  // 2024-01-01T00:00:00
  expect_column_data_equal(
    std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{midnight + 10h},
                                    cudf::timestamp_ms{midnight + 22h},
                                    cudf::timestamp_ms{midnight},
                                    cudf::timestamp_ms{midnight},
                                    cudf::timestamp_ms{midnight + 11h + 30min + 500ms}},
    result.tbl->view().column(0));
}

TEST_F(CsvReaderTest, TimestampsWithIncompleteDate)
{
  // Each date but the last lacks a separator between its components; the missing components parse
  // as zero. The last row holds the separators that a parser reading past the end of the preceding
  // fields would find.
  std::string const buffer = "2024T10:00\n12T10:00\n12/5T10:00\n2001-02-03\n";
  auto const read_dates    = [&](bool dayfirst) {
    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
        .dayfirst(dayfirst)
        .header(-1);
    return cudf::io::read_csv(in_opts);
  };

  // Converts the components as the reader does, including components that are out of range
  auto const date = [](int y, unsigned m, unsigned d) {
    using namespace cuda::std::chrono;
    return cudf::timestamp_ms{sys_days{year_month_day{year{y}, month{m}, day{d}}}};
  };
  using namespace cuda::std::chrono_literals;

  // A year of four digits comes first; otherwise the month (the day with dayfirst) comes first
  // and the year comes last
  expect_column_data_equal(
    std::vector<cudf::timestamp_ms>{
      date(2024, 0, 1) + 10h, date(0, 12, 1) + 10h, date(5, 12, 1) + 10h, date(2001, 2, 3)},
    read_dates(false).tbl->view().column(0));
  expect_column_data_equal(
    std::vector<cudf::timestamp_ms>{
      date(2024, 0, 1) + 10h, date(0, 0, 12) + 10h, date(0, 5, 12) + 10h, date(2001, 2, 3)},
    read_dates(true).tbl->view().column(0));
}

TEST_F(CsvReaderTest, DurationsAtEndOfInput)
{
  // Each duration ends where the input ends, so the parser must not look at the character after
  // the last component. Run under compute-sanitizer with exact allocations (--rmm_mode=cuda) to
  // detect such reads.
  auto const read_last_duration = [](std::string const& buffer) {
    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .dtypes({data_type{type_id::DURATION_MILLISECONDS}})
        .header(-1);
    return cudf::io::read_csv(in_opts);
  };

  using namespace cuda::std::chrono_literals;
  auto const expect_duration = [&](std::string const& buffer, cudf::duration_ms expected) {
    SCOPED_TRACE(buffer);
    expect_column_data_equal(std::vector<cudf::duration_ms>{expected},
                             read_last_duration(buffer).tbl->view().column(0));
  };
  expect_duration("1 days", cudf::duration_ms{24h});
  expect_duration("1 days +", cudf::duration_ms{24h});
  expect_duration("00:00:01", cudf::duration_ms{1s});
  expect_duration("1 days 00:00:01", cudf::duration_ms{24h + 1s});
  expect_duration("1 days 00:00:01.", cudf::duration_ms{24h + 1s});
}

TEST_F(CsvReaderTest, NaValuesDoNotMatchLongerFields)
{
  // The fields that are not null extend an NA value with characters that occur in the NA values
  std::string const buffer = "NA\nNAA\nNA/A\nNAULL\n#NA\n#NAA\nN/A\n";

  cudf::io::csv_reader_options const default_na_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes({dtype<cudf::string_view>()})
      .header(-1);
  auto const default_na_result   = cudf::io::read_csv(default_na_opts);
  auto const default_na_expected = cudf::test::strings_column_wrapper(
    {"", "NAA", "NA/A", "NAULL", "", "#NAA", ""}, {false, true, true, true, false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(default_na_expected, default_na_result.tbl->view().column(0));

  std::string const custom_buffer = "a\nb\naa\nab\nba\nbb\n";
  cudf::io::csv_reader_options const custom_na_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(custom_buffer.data()), custom_buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes({dtype<cudf::string_view>()})
      .header(-1)
      .keep_default_na(false)
      .na_values({"a", "b"});
  auto const custom_na_result   = cudf::io::read_csv(custom_na_opts);
  auto const custom_na_expected = cudf::test::strings_column_wrapper(
    {"", "", "aa", "ab", "ba", "bb"}, {false, false, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(custom_na_expected, custom_na_result.tbl->view().column(0));
}

TEST_F(CsvReaderTest, NonAsciiNaAndBooleanValues)
{
  // "é" and "í" are two-byte UTF-8 sequences whose bytes are negative as signed char
  std::string const buffer = "a,sí\né,no\nb,no\néé,sí\naé,no\n";

  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<cudf::string_view>(), dtype<bool>()})
      .header(-1)
      .keep_default_na(false)
      .na_values({"a", "é"})
      .true_values({"sí"})
      .false_values({"no"});
  auto const result = cudf::io::read_csv(in_opts);

  auto const expected_strings =
    cudf::test::strings_column_wrapper({"", "", "b", "éé", "aé"}, {false, false, true, true, true});
  auto const expected_bools =
    cudf::test::fixed_width_column_wrapper<bool>({true, false, false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strings, result.tbl->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_bools, result.tbl->view().column(1));
}

TEST_F(CsvReaderTest, LoneQuoteField)
{
  // A quote character that opens a quoted field at the end of the input is not a quoted string;
  // the field keeps the quote character
  auto const read_last_field = [](std::string const& buffer, bool detect_whitespace_around_quotes) {
    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
        .detect_whitespace_around_quotes(detect_whitespace_around_quotes);
    return cudf::io::read_csv(in_opts);
  };

  auto const expected_ints = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  {
    auto const result = read_last_field("a,b\n1,x\n2,\"", false);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_ints, result.tbl->view().column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::strings_column_wrapper({"x", "\""}),
                                   result.tbl->view().column(1));
  }
  {
    // Whitespace is only removed around quoted strings
    auto const result = read_last_field("a,b\n1,x\n2, \"", true);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_ints, result.tbl->view().column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::strings_column_wrapper({"x", " \""}),
                                   result.tbl->view().column(1));
  }
}

TEST_F(CsvReaderTest, LoneQuoteNonStringField)
{
  // With the enclosing quote stripped, a lone quote at the end of the input is an empty field, like
  // an empty quoted field. Parsing it must not read the byte after the input; run under
  // compute-sanitizer with exact allocations (--rmm_mode=cuda) to detect such reads.
  std::string const buffer = "\"\"\n\"";
  for (auto const type :
       {type_id::INT32, type_id::FLOAT64, type_id::BOOL8, type_id::DURATION_SECONDS}) {
    SCOPED_TRACE(static_cast<int>(type));
    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .dtypes({data_type{type}})
        .header(-1)
        .na_filter(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const column = result.tbl->view().column(0);
    ASSERT_EQ(column.size(), 2);
    EXPECT_EQ(column.null_count(), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::slice(column, {0, 1}).front(),
                                   cudf::slice(column, {1, 2}).front());
  }
}

TEST_F(CsvReaderTest, EmptyFirstField)
{
  // The first field of the input is empty, so the byte before it is outside the input. Parsing it
  // must not read that byte; run under compute-sanitizer with exact allocations (--rmm_mode=cuda)
  // to detect such reads.
  std::string const buffer = ",1\n2,3\n";
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<int32_t>()})
      .header(-1)
      .na_filter(false);
  auto const result = cudf::io::read_csv(in_opts);

  // Without NA filtering, an empty numeric field parses as zero
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::test::fixed_width_column_wrapper<int32_t>({0, 2}),
                                      result.tbl->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::test::fixed_width_column_wrapper<int32_t>({1, 3}),
                                      result.tbl->view().column(1));
}

TEST_F(CsvReaderTest, UseColsIndexesOutOfRange)
{
  auto const read_use_cols = [](std::string const& buffer,
                                std::vector<int> indexes,
                                std::vector<std::string> names = {},
                                std::size_t range_offset       = 0,
                                std::size_t range_size         = 0,
                                int header                     = -1) {
    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .header(header)
        .names(std::move(names))
        .use_cols_indexes(std::move(indexes))
        .byte_range_offset(range_offset)
        .byte_range_size(range_size);
    return cudf::io::read_csv(in_opts);
  };
  auto const column_names = [](cudf::io::table_with_metadata const& result) {
    std::vector<std::string> names;
    for (auto const& info : result.metadata.schema_info) {
      names.push_back(info.name);
    }
    return names;
  };

  // Selecting a column that is not in the input is an error
  std::string const buffer = "1,2,3,4,5,6\n7,8,9,10,11,12\n";
  EXPECT_THROW(read_use_cols(buffer, {0, 6}), std::out_of_range);
  EXPECT_THROW(read_use_cols(buffer, {-1}), std::out_of_range);
  EXPECT_THROW(read_use_cols("", {-1}), std::out_of_range);
  // The header names the columns of the input, even without data rows
  EXPECT_THROW(read_use_cols("a,b,c\n", {5}, {}, 0, 0, 0), std::out_of_range);

  // Without rows or names, the columns are unknown and none is selected
  EXPECT_EQ(read_use_cols("", {0}).tbl->num_columns(), 0);
  // The byte range [14, 22) contains no row start
  auto const range_offset = 14;
  auto const range_size   = 8;
  EXPECT_EQ(read_use_cols(buffer, {2, 5}, {}, range_offset, range_size).tbl->num_columns(), 0);

  // Without rows, as many names as selected columns name the selected columns
  for (auto const& indexes : std::vector<std::vector<int>>{{2, 5}, {0, 5}, {1, 5}, {0, 1}}) {
    auto const result = read_use_cols(buffer, indexes, {"x", "y"}, range_offset, range_size);
    EXPECT_EQ(result.tbl->num_rows(), 0);
    EXPECT_EQ(column_names(result), (std::vector<std::string>{"x", "y"}));
  }
  EXPECT_EQ(column_names(read_use_cols("", {1, 5}, {"x", "y"})),
            (std::vector<std::string>{"x", "y"}));
  // Other names name all columns, and an index must match one of them
  std::vector<std::string> const six_names{"a", "b", "c", "d", "e", "f"};
  auto const all_names = read_use_cols(buffer, {2, 5}, six_names, range_offset, range_size);
  EXPECT_EQ(all_names.tbl->num_rows(), 0);
  EXPECT_EQ(column_names(all_names), (std::vector<std::string>{"c", "f"}));
  EXPECT_THROW(read_use_cols(buffer, {2, 6}, six_names, range_offset, range_size),
               std::out_of_range);
  EXPECT_THROW(read_use_cols(buffer, {2, 5}, {"a", "b", "c"}, range_offset, range_size),
               std::out_of_range);
}

TEST_F(CsvReaderTest, ParseDatesAndHexIndexesOutOfRange)
{
  // Like column names that match no column, parse_dates and parse_hex indexes that match no column
  // are ignored; the other indexes still apply
  std::string const buffer = "ff,1\n10,2\n";
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .dtypes(std::vector<data_type>{dtype<int64_t>(), dtype<int64_t>()})
      .parse_dates(std::vector<int>{-1, 3})
      .parse_hex(std::vector<int>{-2, 0, 7});
  auto const result = cudf::io::read_csv(in_opts);

  ASSERT_EQ(result.tbl->num_columns(), 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::test::fixed_width_column_wrapper<int64_t>({255, 16}),
                                      result.tbl->view().column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::test::fixed_width_column_wrapper<int64_t>({1, 2}),
                                      result.tbl->view().column(1));
}

namespace {
// Options builder for reading an uncompressed host buffer
cudf::io::csv_reader_options_builder host_buffer_options(std::string const& buffer)
{
  return cudf::io::csv_reader_options::builder(
           cudf::io::source_info{cudf::host_span<std::byte const>{
             reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
    .compression(cudf::io::compression_type::NONE);
}
}  // namespace

TEST_F(CsvReaderTest, EscapedQuotePairs)
{
  // Two string columns per row, so a field is unescaped next to other fields of the same row
  std::string const buffer =
    "0,\"a\"\"b\",\"c\"\"d\"\n"         // pair in the middle
    "1,\"\"\"a\",\"b\"\"\"\n"           // pair at the start / at the end
    "2,\"\"\"\",\"\"\"\"\"\"\n"         // content "" and """"
    "3,\"\"\"a\"\"\",\"a\"\"\"\"b\"\n"  // quoted content, two adjacent pairs
    "4,\"a\"\",b\",\"x\"\"\n\"\"y\"\n"  // pair before a delimiter / around a newline
    "5,a\"\"b,\"a\"\"b\"\n"             // pairs in an unquoted field are kept
    "6,\"\",\"plain\"\n";               // empty quoted field and a field without pairs
  std::vector<std::string> const expected_a{"a\"b", "\"a", "\"", "\"a\"", "a\",b", "a\"\"b", ""};
  std::vector<std::string> const expected_b{
    "c\"d", "b\"", "\"\"", "a\"\"b", "x\"\n\"y", "a\"b", "plain"};

  auto in_opts =
    host_buffer_options(buffer).names({"id", "a", "b"}).header(-1).na_filter(false).build();
  auto const inferred = cudf::io::read_csv(in_opts);

  in_opts.set_dtypes(std::vector<data_type>{
    dtype<int32_t>(), dtype<cudf::string_view>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  ASSERT_EQ(view.num_columns(), 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(view.column(0), column_wrapper<int32_t>{0, 1, 2, 3, 4, 5, 6});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    view.column(1), cudf::test::strings_column_wrapper(expected_a.begin(), expected_a.end()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    view.column(2), cudf::test::strings_column_wrapper(expected_b.begin(), expected_b.end()));

  // Inferred string columns are unescaped the same way
  CUDF_TEST_EXPECT_TABLES_EQUAL(view.select({1, 2}), inferred.tbl->view().select({1, 2}));
}

TEST_F(CsvReaderTest, EscapedQuotePairsOptions)
{
  auto const read_strings = [](std::string const& buffer, auto&& configure) {
    auto in_opts =
      host_buffer_options(buffer).header(-1).dtypes({dtype<cudf::string_view>()}).build();
    configure(in_opts);
    return cudf::io::read_csv(in_opts);
  };
  auto const strings = [](std::vector<std::string> const& values) {
    return cudf::test::strings_column_wrapper(values.begin(), values.end());
  };

  {
    // Custom quote character: only its pairs are escapes
    auto const result = read_strings("'it''s'\n'\"a\"\"b\"'\n\"x\"\"y\"\n",
                                     [](auto& opts) { opts.set_quotechar('\''); });
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0),
                                   strings({"it's", "\"a\"\"b\"", "\"x\"\"y\""}));
  }
  {
    // Whitespace around the quotes is removed before unescaping
    auto const result = read_strings("  \"a\"\"b\"  \n\"\"\"c\" \n", [](auto& opts) {
      opts.enable_detect_whitespace_around_quotes(true);
    });
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), strings({"a\"b", "\"c"}));
  }
  {
    // CRLF line ends, with a pair right before the closing quote
    auto const result = read_strings("\"a\"\"\"\r\n\"b\"\"c\"\r\n\"\"\"\"\r\n", [](auto&) {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), strings({"a\"", "b\"c", "\""}));
  }
  {
    // Without doublequote, pairs are kept as they are
    auto const result =
      read_strings("\"a\"\"b\"\n\"c\"\n", [](auto& opts) { opts.enable_doublequote(false); });
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), strings({"a\"\"b", "c"}));
  }
  {
    // With quoting disabled, quotes are ordinary characters
    auto const result = read_strings(
      "\"a\"\"b\"\n", [](auto& opts) { opts.set_quoting(cudf::io::quote_style::NONE); });
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), strings({"\"a\"\"b\""}));
  }
}

TEST_F(CsvReaderTest, EscapedQuotePairsManyRows)
{
  // Enough rows for many thread blocks; every other row is quoted and only those are unescaped
  constexpr int num_rows = 10'000;
  std::string buffer;
  std::vector<int32_t> expected_ids;
  std::vector<std::string> expected_strings;
  std::vector<double> expected_values;
  for (int i = 0; i < num_rows; ++i) {
    auto const text = std::to_string(i);
    if (i % 2 == 0) {
      buffer += text + ",\"\"\"" + text + "\"\",\"\"\"," + text + ".5\n";
      expected_strings.push_back("\"" + text + "\",\"");
    } else {
      buffer += text + "," + text + "\"\"," + text + ".5\n";
      expected_strings.push_back(text + "\"\"");
    }
    expected_ids.push_back(i);
    expected_values.push_back(i + 0.5);
  }

  auto const result = cudf::io::read_csv(host_buffer_options(buffer).header(-1).dtypes(
    std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>(), dtype<double>()}));

  auto const view = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    view.column(0), column_wrapper<int32_t>(expected_ids.begin(), expected_ids.end()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    view.column(1),
    cudf::test::strings_column_wrapper(expected_strings.begin(), expected_strings.end()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    view.column(2), column_wrapper<double>(expected_values.begin(), expected_values.end()));
}

TEST_F(CsvReaderTest, EscapedQuotePairsByteRanges)
{
  // Rows are decoded in place, so the rows selected by a byte range must be unescaped exactly as in
  // a full read: reading consecutive byte ranges and concatenating the results gives the full read
  auto const read_range = [](std::string const& buffer, std::size_t offset, std::size_t size) {
    return cudf::io::read_csv(
      host_buffer_options(buffer)
        .names({"a", "b", "c"})
        .header(-1)
        .dtypes(std::vector<data_type>{
          dtype<int32_t>(), dtype<cudf::string_view>(), dtype<cudf::string_view>()})
        .byte_range_offset(offset)
        .byte_range_size(size));
  };
  auto const expect_concatenated_ranges_equal = [&](std::string const& buffer,
                                                    std::vector<std::size_t> const& offsets) {
    auto const full = read_range(buffer, 0, 0);
    std::vector<std::unique_ptr<cudf::table>> parts;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      auto const end = i + 1 < offsets.size() ? offsets[i + 1] : buffer.size();
      parts.push_back(read_range(buffer, offsets[i], end - offsets[i]).tbl);
    }
    std::vector<cudf::table_view> views;
    std::transform(parts.begin(), parts.end(), std::back_inserter(views), [](auto const& part) {
      return part->view();
    });
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(full.tbl->view(), cudf::concatenate(views)->view());
  };

  constexpr int num_rows = 50;
  {
    // Ranges of any size, with pairs next to delimiters inside and outside quoted fields
    std::string buffer;
    for (int i = 0; i < num_rows; ++i) {
      auto const text = std::to_string(i);
      buffer += text + ",\"q\"\"" + text + ",\"\"z\"," + (i % 3 == 0 ? "u\"\"" : "\"\"\"\"") + "\n";
    }
    for (std::size_t const range_size : {7, 16, 33, 100}) {
      std::vector<std::size_t> offsets;
      for (std::size_t offset = 0; offset < buffer.size(); offset += range_size) {
        offsets.push_back(offset);
      }
      expect_concatenated_ranges_equal(buffer, offsets);
    }
  }
  {
    // Quoted newlines. A range that starts after the beginning of the input can't tell them from
    // row ends, so these ranges start at the beginning. A range selects the rows whose preceding
    // row end is in the range, i.e. the rows that start at or before its end.
    std::string buffer;
    std::vector<std::size_t> row_starts;
    for (int i = 0; i < num_rows; ++i) {
      auto const text = std::to_string(i);
      row_starts.push_back(buffer.size());
      buffer += text + ",\"q\"\"" + text + "\n\"\"z\",\"a\r\n\"\"\"\n";
    }
    auto const full = read_range(buffer, 0, 0);
    for (std::size_t const range_size : {1, 20, 21, 22, 100, 500}) {
      auto const num_selected = std::count_if(
        row_starts.begin(), row_starts.end(), [&](auto start) { return start <= range_size; });
      auto const expected =
        cudf::slice(full.tbl->view(), {0, static_cast<cudf::size_type>(num_selected)}).front();
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, read_range(buffer, 0, range_size).tbl->view());
    }
  }
}

TEST_F(CsvReaderTest, EscapedQuotePairsDeviceSourceUnchanged)
{
  // The reader unescapes its own copy of the data, never the caller's buffer. The reader copies
  // device inputs today; this guards against future zero-copy paths writing the caller's buffer.
  std::string const buffer = "a,b\n1,\"x\"\"y\"\n2,\"\"\"\"\"\"\n";
  auto const stream        = cudf::get_default_stream();
  auto const d_buffer =
    cudf::detail::make_device_uvector(cudf::host_span<char const>{buffer.data(), buffer.size()},
                                      stream,
                                      cudf::get_current_device_resource_ref());

  auto const result = cudf::io::read_csv(
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::device_span<std::byte const>{
        reinterpret_cast<std::byte const*>(d_buffer.data()), d_buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1),
                                 cudf::test::strings_column_wrapper({"x\"y", "\"\""}));

  auto const h_after = cudf::detail::make_std_vector(d_buffer, stream);
  EXPECT_EQ(std::string(h_after.begin(), h_after.end()), buffer);
}

TEST_F(CsvReaderTest, ShortRowsMissingFieldsAreNull)
{
  // Row i has 5 - i % 5 of the 5 fields (the first row has all of them, so that the number of
  // columns is detected as 5); the missing fields and the NA fields are null. The row counts cover
  // a partial warp, exact and partial multiples of 32 rows, and multiple blocks.
  auto const dtypes = std::vector<data_type>{dtype<int32_t>(),
                                             dtype<cudf::string_view>(),
                                             dtype<double>(),
                                             dtype<cudf::string_view>(),
                                             dtype<bool>()};
  for (int const num_rows : {1, 31, 32, 33, 129, 1000}) {
    std::string buffer;
    std::vector<int32_t> ints;
    std::vector<std::string> strings_a;
    std::vector<double> doubles;
    std::vector<std::string> strings_b;
    std::vector<bool> bools;
    std::vector<std::vector<bool>> valid(dtypes.size());
    for (int i = 0; i < num_rows; ++i) {
      auto const num_fields = 5 - i % 5;
      auto const text       = std::to_string(i);
      auto const is_na      = [&](int col) { return (i + col) % 7 == 3; };
      std::vector<std::string> fields;
      fields.push_back(is_na(0) ? "NA" : text);
      fields.push_back(is_na(1) ? "NA" : (i % 3 == 0 ? "\"q\"\"" + text + "\"" : "s" + text));
      fields.push_back(is_na(2) ? "NA" : text + ".25");
      fields.push_back(is_na(3) ? "NA" : "t" + text);
      fields.push_back(is_na(4) ? "NA" : (i % 2 == 0 ? "true" : "false"));
      for (int col = 0; col < num_fields; ++col) {
        buffer += (col == 0 ? "" : ",") + fields[col];
      }
      buffer += '\n';

      ints.push_back(i);
      strings_a.push_back(i % 3 == 0 ? "q\"" + text : "s" + text);
      doubles.push_back(i + 0.25);
      strings_b.push_back("t" + text);
      bools.push_back(i % 2 == 0);
      for (int col = 0; col < static_cast<int>(dtypes.size()); ++col) {
        valid[col].push_back(col < num_fields && not is_na(col));
      }
    }

    auto in_opts = host_buffer_options(buffer)
                     .names({"a", "b", "c", "d", "e"})
                     .header(-1)
                     .dtypes(dtypes)
                     .build();
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    ASSERT_EQ(view.num_rows(), num_rows);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      view.column(0), column_wrapper<int32_t>(ints.begin(), ints.end(), valid[0].begin()));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      view.column(1),
      cudf::test::strings_column_wrapper(strings_a.begin(), strings_a.end(), valid[1].begin()));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      view.column(2), column_wrapper<double>(doubles.begin(), doubles.end(), valid[2].begin()));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      view.column(3),
      cudf::test::strings_column_wrapper(strings_b.begin(), strings_b.end(), valid[3].begin()));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      view.column(4), column_wrapper<bool>(bools.begin(), bools.end(), valid[4].begin()));

    // Missing fields of unselected columns don't affect the selected ones
    in_opts.set_use_cols_indexes({1, 2, 4});
    auto const selected = cudf::io::read_csv(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(view.select({1, 2, 4}), selected.tbl->view());

    // Whitespace-delimited rows take the same path
    std::replace(buffer.begin(), buffer.end(), ',', ' ');
    auto ws_opts = host_buffer_options(buffer)
                     .names({"a", "b", "c", "d", "e"})
                     .header(-1)
                     .dtypes(dtypes)
                     .delim_whitespace(true)
                     .build();
    CUDF_TEST_EXPECT_TABLES_EQUAL(view, cudf::io::read_csv(ws_opts).tbl->view());
  }
}

namespace {
// Host source that copies the requested range on every read. Records the largest read, and can
// fail a given read, by throwing or by returning fewer bytes than requested.
class copying_host_source : public cudf::io::datasource {
 public:
  enum class failure { NONE, THROW, SHORT_READ };

  explicit copying_host_source(std::string const& data,
                               failure fail          = failure::NONE,
                               size_t fail_read_size = 0)
    : _data{data}, _fail{fail}, _fail_read_size{fail_read_size}
  {
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    size = read_size(offset, size);
    return buffer::create(std::vector<char>(_data.begin() + offset, _data.begin() + offset + size));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    size = read_size(offset, size);
    std::memcpy(dst, _data.data() + offset, size);
    return size;
  }

  [[nodiscard]] size_t size() const override { return _data.size(); }

  [[nodiscard]] size_t max_read_size() const { return _max_read_size; }

 private:
  size_t read_size(size_t offset, size_t size)
  {
    size = std::min(size, _data.size() - offset);
    std::lock_guard lock{_mutex};
    _max_read_size = std::max(_max_read_size, size);
    if (size == _fail_read_size) {
      if (_fail == failure::THROW) { throw std::runtime_error("read failure"); }
      if (_fail == failure::SHORT_READ) { return size / 2; }
    }
    return size;
  }

  std::string const& _data;
  failure _fail;
  size_t _fail_read_size;
  std::mutex _mutex;
  size_t _max_read_size = 0;
};

// Rows with an INT32, a quoted multi-line STRING and a FLOAT64 column, and the expected columns
struct multiline_rows {
  std::string text = "id,text,value\n";
  std::vector<int32_t> ids;
  std::vector<std::string> texts;
  std::vector<double> values;

  explicit multiline_rows(int num_rows)
  {
    for (int i = 0; i < num_rows; ++i) {
      auto const id  = std::to_string(i);
      auto const str = i % 3 == 0 ? "line\n" + id : "text" + id;
      text += id + ",\"" + str + "\"," + id + ".5\n";
      ids.push_back(i);
      texts.push_back(str);
      values.push_back(i + 0.5);
    }
  }

  [[nodiscard]] cudf::io::table_with_metadata read(cudf::io::source_info const& source) const
  {
    return cudf::io::read_csv(cudf::io::csv_reader_options::builder(source)
                                .compression(cudf::io::compression_type::NONE)
                                .dtypes(std::vector<data_type>{
                                  dtype<int32_t>(), dtype<cudf::string_view>(), dtype<double>()})
                                .build());
  }

  void expect_equal(cudf::table_view const& table) const
  {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table.column(0),
                                        column_wrapper<int32_t>(ids.begin(), ids.end()));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      table.column(1), cudf::test::strings_column_wrapper(texts.begin(), texts.end()));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table.column(2),
                                        column_wrapper<double>(values.begin(), values.end()));
  }
};

// Size of the reads with which the reader loads sources that do not prefer device reads
constexpr size_t reader_host_read_bytes = 32 * 1024 * 1024;
}  // namespace

TEST_F(CsvReaderTest, HostSourcesReadInBoundedChunks)
{
  // Larger than the chunks in which the reader reads host sources
  multiline_rows const rows{1'500'000};
  ASSERT_GT(rows.text.size(), reader_host_read_bytes);

  // Sources that don't prefer device reads are never read into host memory all at once
  copying_host_source source{rows.text};
  rows.expect_equal(rows.read(cudf::io::source_info{&source}).tbl->view());
  EXPECT_EQ(source.max_read_size(), reader_host_read_bytes);

  // Host buffers in pageable and in pinned memory
  auto const as_bytes = [](char const* data, size_t size) {
    return cudf::host_span<std::byte const>{reinterpret_cast<std::byte const*>(data), size};
  };
  rows.expect_equal(
    rows.read(cudf::io::source_info{as_bytes(rows.text.data(), rows.text.size())}).tbl->view());
  auto const stream = cudf::get_default_stream();
  auto pinned       = cudf::detail::make_pinned_vector<char>(rows.text.size(), stream);
  std::copy(rows.text.begin(), rows.text.end(), pinned.begin());
  rows.expect_equal(
    rows.read(cudf::io::source_info{as_bytes(pinned.data(), pinned.size())}).tbl->view());
}

TEST_F(CsvReaderTest, HostSourceReadFailures)
{
  multiline_rows const rows{1'500'000};
  auto const last_read_size = rows.text.size() - reader_host_read_bytes;
  ASSERT_LT(last_read_size, reader_host_read_bytes);

  // The first and the last of the reads fail
  for (auto const fail_read_size : {reader_host_read_bytes, last_read_size}) {
    copying_host_source throwing{rows.text, copying_host_source::failure::THROW, fail_read_size};
    EXPECT_THROW(std::ignore = rows.read(cudf::io::source_info{&throwing}), std::runtime_error);

    copying_host_source short_reads{
      rows.text, copying_host_source::failure::SHORT_READ, fail_read_size};
    EXPECT_THROW(std::ignore = rows.read(cudf::io::source_info{&short_reads}), cudf::logic_error);
  }
}

namespace {
// CSV document with an INT32, a STRING and a FLOAT64 column, and the values it holds
class csv_document {
 public:
  csv_document(char delimiter, char terminator) : _delimiter{delimiter}, _terminator{terminator} {}

  // Appends a row whose string field is written as `field` and reads as `value`; the last field is
  // quoted if `quote_last`
  void add_row(std::string const& field, std::string const& value, bool quote_last = false)
  {
    auto const id = static_cast<int32_t>(_ids.size());
    _text += row_text(id, field, quote_last);
    _ids.push_back(id);
    _strings.push_back(value);
    _values.push_back(id + 0.5);
  }

  void add_row(std::string const& field) { add_row(field, field); }

  // Appends a line that is not a row (e.g. a comment), followed by the terminator
  void add_line(std::string const& line) { _text += line + _terminator; }

  // Appends rows until the document is exactly `length` characters long
  void pad_to(size_t length)
  {
    constexpr size_t max_row_length = 64;
    while (_text.size() + max_row_length < length) {
      add_row("filler");
    }
    auto const padding = length - _text.size() - row_text(_ids.size(), "").size();
    add_row(std::string(padding, 'x'));
  }

  void append_text(std::string const& text) { _text += text; }

  [[nodiscard]] std::string const& text() const { return _text; }
  [[nodiscard]] size_t num_rows() const { return _ids.size(); }

  [[nodiscard]] cudf::io::csv_reader_options_builder options() const
  {
    return cudf::io::csv_reader_options::builder(
             cudf::io::source_info{cudf::host_span<std::byte const>{
               reinterpret_cast<std::byte const*>(_text.data()), _text.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .names({"id", "text", "value"})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>(), dtype<double>()})
      .delimiter(_delimiter)
      .lineterminator(_terminator);
  }

  // Expects `table` to hold the rows [first, first + table.num_rows())
  void expect_rows(cudf::table_view const& table, size_t first = 0) const
  {
    auto const last = first + table.num_rows();
    ASSERT_LE(last, num_rows());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      table.column(0), column_wrapper<int32_t>(_ids.begin() + first, _ids.begin() + last));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      table.column(1),
      cudf::test::strings_column_wrapper(_strings.begin() + first, _strings.begin() + last));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      table.column(2), column_wrapper<double>(_values.begin() + first, _values.begin() + last));
  }

 private:
  [[nodiscard]] std::string row_text(size_t id,
                                     std::string const& field,
                                     bool quote_last = false) const
  {
    auto const text  = std::to_string(id);
    auto const value = quote_last ? "\"" + text + ".5\"" : text + ".5";
    return text + _delimiter + field + _delimiter + value + _terminator;
  }

  char _delimiter;
  char _terminator;
  std::string _text;
  std::vector<int32_t> _ids;
  std::vector<std::string> _strings;
  std::vector<double> _values;
};

// Reads all rows of the document as a whole file, which gathers its rows in a single pass, and
// also through the chunked row selection path (nrows), and expects both to hold the document's rows
void expect_document_rows(csv_document const& doc, char comment = '\0')
{
  auto opts         = doc.options().comment(comment).build();
  auto const result = cudf::io::read_csv(opts);
  ASSERT_EQ(static_cast<size_t>(result.tbl->num_rows()), doc.num_rows());
  doc.expect_rows(result.tbl->view());

  opts.set_nrows(static_cast<cudf::size_type>(doc.num_rows()));
  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), cudf::io::read_csv(opts).tbl->view());
}
}  // namespace

TEST_F(CsvReaderTest, CommentCharacterEndingAlignedSlice)
{
  // A comment character that starts a row at the last character of a 32-character slice leaves the
  // slice in the COMMENT state, which changes how the quote at the start of the next slice is
  // parsed; the data starts aligned, so the first slice ends at character 31. With the '#' comment
  // character, `#""` enters a quoted field that extends to the end of the data. With the comment
  // character equal to the delimiter, the quote after it opens no field.
  std::string const tail = std::string(30, 'a') + "\n";
  std::string rows;
  for (int i = 0; i < 20; ++i) {
    rows += "4\n";
  }
  auto const read_rows = [](std::string const& text, char comment) {
    auto opts = cudf::io::csv_reader_options::builder(
                  cudf::io::source_info{cudf::host_span<std::byte const>{
                    reinterpret_cast<std::byte const*>(text.data()), text.size()}})
                  .compression(cudf::io::compression_type::NONE)
                  .header(-1)
                  .comment(comment)
                  .dtypes(std::vector<data_type>{dtype<cudf::string_view>()})
                  .build();
    auto const whole = cudf::io::read_csv(opts).tbl;
    opts.set_nrows(std::numeric_limits<cudf::size_type>::max());
    CUDF_TEST_EXPECT_TABLES_EQUAL(whole->view(), cudf::io::read_csv(opts).tbl->view());
    return whole->num_rows();
  };
  EXPECT_EQ(read_rows(tail + "#\"\"x\n1\n2\n3\n" + rows, '#'), 1);
  EXPECT_EQ(read_rows(tail + ",\"x\n1\n2\n3\n" + rows, ','), 24);
}

TEST_F(CsvReaderTest, RowBoundariesAcrossBlocks)
{
  // A probe with quoted fields that hold terminators (also after '\r'), delimiters and escaped
  // quotes, and a comment line is repeated so that it crosses the first four boundaries of the
  // 16KB blocks of the chunked row gathering, two of which are boundaries of the 32KB tiles of the
  // single-pass row gathering. Each character of the probe is placed in turn at the boundaries, so
  // that every parser state occurs at the start of a block and every character at each position of
  // a 32-character slice.
  constexpr size_t block_size = 16 * 1024;
  constexpr size_t num_blocks = 4;
  for (auto const [delimiter, terminator] :
       {std::pair{',', '\n'}, std::pair{',', '\r'}, std::pair{'\x01', '\xFE'}}) {
    std::string const t{terminator};
    std::string const d{delimiter};
    auto const add_probe = [&](csv_document& doc) {
      doc.add_row("\"a\r" + t + "b\"\"c\"\"" + t + "d" + d + "e\"",
                  "a\r" + t + "b\"c\"" + t + "d" + d + "e");
      doc.add_line("#" + d);
      doc.add_row("\"\"\"\"", "\"");
      doc.add_row("\"a" + t + "\"", "a" + t);
    };
    csv_document probe_only{delimiter, terminator};
    add_probe(probe_only);
    auto const probe_size = probe_only.text().size();

    for (size_t shift = 0; shift <= probe_size; ++shift) {
      SCOPED_TRACE("terminator " + std::to_string(static_cast<int>(terminator)) + ", shift " +
                   std::to_string(shift));
      csv_document doc{delimiter, terminator};
      for (size_t block = 1; block <= num_blocks; ++block) {
        doc.pad_to(block * block_size - shift);
        add_probe(doc);
      }
      doc.add_row("last");
      expect_document_rows(doc, '#');
    }
  }
}

TEST_F(CsvReaderTest, QuotedFieldsSpanningTiles)
{
  // Every 32KB tile of the single-pass row gathering but the first starts inside a quoted field,
  // so the parser state assumed for these tiles is wrong. There are more tiles than the device runs
  // blocks at once.
  constexpr size_t tile_size = 32 * 1024;
  constexpr size_t num_tiles = 1100;
  auto const text            = std::string(tile_size - 64, 'q') + "\n\"";
  csv_document doc{',', '\n'};
  doc.pad_to(100);
  for (size_t tile = 1; tile < num_tiles; ++tile) {
    // The row starts 100 characters before the tile and ends after its start
    doc.add_row("\"" + std::string(tile_size - 64, 'q') + "\n\"\"\"", text);
    doc.pad_to(tile * tile_size + 100);
  }
  expect_document_rows(doc);

  // A single quoted field spans many tiles
  constexpr size_t num_spanned_tiles = 40;
  csv_document single{',', '\n'};
  auto const long_text = std::string(num_spanned_tiles * tile_size, 'q') + "\n";
  single.add_row("\"" + long_text + "\"", long_text);
  expect_document_rows(single);
}

TEST_F(CsvReaderTest, TileStartStates)
{
  // Places each character of short probes in turn at the start of a 32KB tile of the single-pass
  // row gathering, one probe position per tile, so that tiles start in each parser state: inside a
  // quoted field, right after a closing quote (before a delimiter, a terminator, or the second
  // quote of an escaped quote), and in a comment line. Terminators after the escaped quotes make
  // the rows depend on the start state.
  constexpr size_t tile_size = 32 * 1024;
  csv_document doc{',', '\n'};
  auto const add_probe = [&](int probe) {
    switch (probe) {
      case 0: doc.add_row("\"q\"", "q"); break;
      case 1: doc.add_row("\"a\"\"\nb\"", "a\"\nb"); break;
      case 2: doc.add_row("\"\"\"a\nb\"", "\"a\nb"); break;
      case 3: doc.add_row("\"a\nb\"", "a\nb", true); break;
      default: doc.add_line("#comment"); break;
    }
  };
  constexpr int num_probes        = 5;
  constexpr size_t max_probe_size = 32;
  size_t tile                     = 1;
  for (int probe = 0; probe < num_probes; ++probe) {
    for (size_t shift = 1; shift <= max_probe_size; ++shift, ++tile) {
      doc.pad_to(tile * tile_size - shift);
      add_probe(probe);
    }
  }
  doc.add_row("last");
  expect_document_rows(doc, '#');
}

namespace {
// Random document of quoted fields, escaped quotes, comment and blank lines, over several tiles
std::string random_multi_tile_document(
  std::mt19937& rng, char delimiter, char terminator, char quotechar, char comment)
{
  constexpr size_t tile_size = 32 * 1024;
  auto const rnd             = [&](size_t n) { return static_cast<size_t>(rng() % n); };
  auto const size            = tile_size / 2 + rnd(5 * tile_size);
  std::string const specials{
    delimiter, terminator, quotechar, '\r', comment != '\0' ? comment : '#'};
  std::string text;
  while (text.size() < size) {
    switch (rnd(10)) {
      case 0: text += terminator; break;  // blank line
      case 1:
        text += std::string{comment != '\0' ? comment : '#', quotechar} + "x" + terminator;
        break;
      case 2: text += std::string{'\r', terminator}; break;
      default:
        for (size_t field = 0, num_fields = 1 + rnd(4); field < num_fields; ++field) {
          if (field > 0) { text += delimiter; }
          auto const quoted = rnd(3) == 0;
          if (quoted) { text += quotechar; }
          // Mostly short fields, and some that span tiles
          auto const length = rnd(20) == 0 ? rnd(2 * tile_size) : rnd(12);
          for (size_t i = 0; i < length; ++i) {
            text += rnd(5) == 0 ? specials[rnd(specials.size())] : static_cast<char>('a' + rnd(26));
          }
          if (quoted) { text += quotechar; }
        }
        text += terminator;
    }
  }
  // The data ends anywhere, and sometimes on a tile boundary
  text.resize(rnd(4) == 0 ? text.size() / tile_size * tile_size : size);
  return text;
}
}  // namespace

TEST_F(CsvReaderTest, RandomDocumentsWholeFileAndChunked)
{
  // Whole-file reads, which gather rows in a single pass, and row selection reads, which gather
  // them in chunks, find the same rows in random multi-tile documents
  std::mt19937 rng{12345};
  for (int i = 0; i < 40; ++i) {
    auto const terminator = std::string{"\n\n\r;"}[rng() % 4];
    auto const quotechar  = rng() % 4 == 0 ? '\'' : '"';
    auto const delimiter  = rng() % 4 == 0 ? '\t' : ',';
    auto const comment    = std::string{'\0', '#', quotechar, delimiter}[rng() % 4];
    auto const quoting =
      rng() % 8 == 0 ? cudf::io::quote_style::NONE : cudf::io::quote_style::MINIMAL;
    auto const doublequote      = rng() % 4 != 0;
    auto const skip_blank_lines = rng() % 4 != 0;
    auto const text = random_multi_tile_document(rng, delimiter, terminator, quotechar, comment);
    SCOPED_TRACE("document " + std::to_string(i) + ", " + std::to_string(text.size()) + " bytes");

    auto opts = cudf::io::csv_reader_options::builder(
                  cudf::io::source_info{cudf::host_span<std::byte const>{
                    reinterpret_cast<std::byte const*>(text.data()), text.size()}})
                  .compression(cudf::io::compression_type::NONE)
                  .header(-1)
                  .dtypes(std::vector<data_type>{dtype<cudf::string_view>()})
                  .delimiter(delimiter)
                  .lineterminator(terminator)
                  .quotechar(quotechar)
                  .quoting(quoting)
                  .doublequote(doublequote)
                  .comment(comment)
                  .skip_blank_lines(skip_blank_lines)
                  .build();
    auto const whole = cudf::io::read_csv(opts);
    opts.set_nrows(std::numeric_limits<cudf::size_type>::max());
    CUDF_TEST_EXPECT_TABLES_EQUAL(whole.tbl->view(), cudf::io::read_csv(opts).tbl->view());
  }
}

TEST_F(CsvReaderTest, BlankAndCommentRowsAcrossTiles)
{
  // Blank lines, CRLF blank lines and comment lines, alone and in long runs that span the 32KB
  // tiles of the single-pass row gathering, at the start and at the end of the data
  for (char const terminator : {'\n', ';'}) {
    SCOPED_TRACE("terminator " + std::string{terminator});
    csv_document doc{',', terminator};
    size_t num_blank_lines = 0;
    auto const add_blank   = [&](std::string const& line) {
      doc.add_line(line);
      ++num_blank_lines;
    };
    add_blank("");
    for (int i = 0; doc.text().size() < 200'000; ++i) {
      doc.add_row("s" + std::to_string(i));
      if (i % 7 == 0) { add_blank(""); }
      if (i % 11 == 0) { doc.add_line("#comment" + std::to_string(i)); }
      if (terminator == '\n' && i % 13 == 0) { add_blank("\r"); }
      if (i % 500 == 0) {
        for (int j = 0; j < 3000; ++j) {
          if (j % 2 == 0) {
            add_blank("");
          } else {
            doc.add_line("#");
          }
        }
      }
    }
    add_blank("");
    doc.add_line("#comment");
    add_blank("");
    expect_document_rows(doc, '#');

    // Without skipping blank lines, blank lines are rows of nulls; comment lines are still skipped
    auto opts         = doc.options().comment('#').skip_blank_lines(false).build();
    auto const result = cudf::io::read_csv(opts);
    ASSERT_EQ(static_cast<size_t>(result.tbl->num_rows()), doc.num_rows() + num_blank_lines);
    EXPECT_EQ(static_cast<size_t>(result.tbl->view().column(2).null_count()), num_blank_lines);
    opts.set_nrows(result.tbl->num_rows());
    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), cudf::io::read_csv(opts).tbl->view());
  }
}

TEST_F(CsvReaderTest, UnterminatedQuoteAtEndOfData)
{
  // The last field extends from the unterminated quote, which is kept, to the end of the data
  for (size_t const length : {100ul, 64ul * 1024, 64ul * 1024 + 100}) {
    SCOPED_TRACE("length " + std::to_string(length));
    csv_document doc{',', '\n'};
    doc.pad_to(length);
    doc.append_text(std::to_string(doc.num_rows()) + ",\"unterminated\n1,2\n");
    auto opts         = doc.options().build();
    auto const result = cudf::io::read_csv(opts);
    ASSERT_EQ(static_cast<size_t>(result.tbl->num_rows()), doc.num_rows() + 1);
    auto const last_row = static_cast<cudf::size_type>(doc.num_rows());
    doc.expect_rows(cudf::slice(result.tbl->view(), {0, last_row})[0]);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      cudf::slice(result.tbl->view().column(1), {last_row, last_row + 1})[0],
      cudf::test::strings_column_wrapper({"\"unterminated\n1,2\n"}));

    opts.set_nrows(static_cast<cudf::size_type>(doc.num_rows() + 1));
    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), cudf::io::read_csv(opts).tbl->view());
  }
}

TEST_F(CsvReaderTest, ByteRangesCoverAllRows)
{
  // Quoted fields with delimiters and escaped quotes (byte ranges are located by searching for
  // terminators, so quoted fields must not hold any)
  csv_document doc{',', '\n'};
  for (int i = 0; i < 6000; ++i) {
    auto const text = std::to_string(i);
    if (i % 3 == 0) {
      doc.add_row("\"a,\"\"" + text + "\"\"\"", "a,\"" + text + "\"");
    } else {
      doc.add_row("s" + text);
    }
  }
  auto const full = cudf::io::read_csv(doc.options().build());
  doc.expect_rows(full.tbl->view());

  auto const data_size = doc.text().size();
  for (size_t const range_size : {1000ul, 4099ul, 16384ul, 70000ul}) {
    SCOPED_TRACE("range size " + std::to_string(range_size));
    std::vector<std::unique_ptr<cudf::table>> ranges;
    for (size_t offset = 0; offset < data_size; offset += range_size) {
      ranges.push_back(
        cudf::io::read_csv(
          doc.options().byte_range_offset(offset).byte_range_size(range_size).build())
          .tbl);
    }
    std::vector<cudf::table_view> views;
    std::transform(ranges.begin(), ranges.end(), std::back_inserter(views), [](auto const& t) {
      return t->view();
    });
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(full.tbl->view(), cudf::concatenate(views)->view());
  }
}

TEST_F(CsvReaderTest, RowSelectionAcrossChunks)
{
  // Row selection reads the input in 64MB chunks; a quoted field with terminators spans the end of
  // the first chunk
  constexpr size_t chunk_size = 64 * 1024 * 1024;
  csv_document doc{',', '\n'};
  doc.pad_to(chunk_size - 12);
  auto const special_row = doc.num_rows();
  doc.add_row("\"a\nb\nc\nd\ne\nacross the\nchunk\"\"boundary\"",
              "a\nb\nc\nd\ne\nacross the\nchunk\"boundary");
  for (int i = 0; i < 1000; ++i) {
    doc.add_row("tail" + std::to_string(i));
  }

  auto const whole = cudf::io::read_csv(doc.options().build());
  ASSERT_EQ(static_cast<size_t>(whole.tbl->num_rows()), doc.num_rows());
  auto const tail_start = static_cast<cudf::size_type>(special_row - 100);
  doc.expect_rows(cudf::slice(whole.tbl->view(), {tail_start, whole.tbl->num_rows()})[0],
                  tail_start);

  // The first chunk is kept in part, or discarded
  for (auto const skip_rows : {special_row - 5, special_row + 5}) {
    SCOPED_TRACE("skiprows " + std::to_string(skip_rows));
    auto const selected = cudf::io::read_csv(
      doc.options().skiprows(static_cast<cudf::size_type>(skip_rows)).nrows(100).build());
    ASSERT_EQ(selected.tbl->num_rows(), 100);
    doc.expect_rows(selected.tbl->view(), skip_rows);
  }
}

TEST_F(CsvReaderTest, ShortInputsWithoutFinalTerminator)
{
  // Inputs of 1 to 17 characters whose last field is numeric and ends at the end of the data, so
  // that the data ends at every position of an 8-byte word. They are also read from device buffers
  // at every alignment, each allocated to the exact size of the input. The reader currently copies
  // device sources into its own aligned buffer, so these reads only check the results; they guard
  // the bounds of the word loads for sources that would be parsed in place.
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  for (size_t const field_length : {1ul, 3ul, 8ul, 17ul}) {
    std::string fields;
    for (int i = 0; fields.size() < 17; ++i) {
      fields += std::string(field_length, static_cast<char>('1' + i % 9)) + ',';
    }
    for (size_t length = 1; length <= 17; ++length) {
      auto buffer = fields.substr(0, length);
      if (buffer.back() == ',') { buffer.back() = '0'; }
      SCOPED_TRACE("input \"" + buffer + "\"");

      std::vector<std::unique_ptr<cudf::column>> expected_columns;
      std::istringstream field_stream(buffer);
      for (std::string field; std::getline(field_stream, field, ',');) {
        expected_columns.push_back(column_wrapper<int64_t>{std::stoll(field)}.release());
      }
      auto const expected        = cudf::table{std::move(expected_columns)};
      auto const explicit_dtypes = std::vector<data_type>(expected.num_columns(), dtype<int64_t>());

      auto const read = [&](cudf::io::source_info const& source, bool infer_types) {
        auto opts = cudf::io::csv_reader_options::builder(source)
                      .compression(cudf::io::compression_type::NONE)
                      .header(-1)
                      .build();
        if (not infer_types) { opts.set_dtypes(explicit_dtypes); }
        return cudf::io::read_csv(opts).tbl;
      };
      auto const host_source = cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}};
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.view(), read(host_source, true)->view());
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.view(), read(host_source, false)->view());

      for (size_t offset = 0; offset < 8; ++offset) {
        rmm::device_uvector<char> d_buffer(offset + buffer.size(), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_buffer.data() + offset, buffer.data(), buffer.size(), cudaMemcpyDefault, stream.get()));
        auto const device_source = cudf::io::source_info{cudf::device_span<std::byte const>{
          reinterpret_cast<std::byte const*>(d_buffer.data() + offset), buffer.size()}};
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.view(), read(device_source, true)->view());
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.view(), read(device_source, false)->view());
      }
    }
  }
}

TEST_F(CsvReaderTest, FieldEndsAtEveryWordPosition)
{
  // Fields of 1 to 20 characters, so that the characters that end them fall at every position of
  // an 8-byte word. Quoted fields hold delimiters, terminators, '\r' and escaped quotes, and
  // unquoted fields hold '\r' characters that are not followed by a line feed.
  for (auto const [delimiter, terminator] : {std::pair{',', '\n'},
                                             std::pair{'\x01', '\n'},
                                             std::pair{'\xFE', '\r'},
                                             std::pair{'\t', '\xFE'}}) {
    SCOPED_TRACE("delimiter " + std::to_string(static_cast<int>(delimiter)) + ", terminator " +
                 std::to_string(static_cast<int>(terminator)));
    std::string const d{delimiter};
    std::string const t{terminator};
    csv_document doc{delimiter, terminator};
    for (size_t length = 1; length <= 20; ++length) {
      std::string const text(length, 'x');
      doc.add_row(text);
      doc.add_row("\"" + text + "\"", text);
      doc.add_row("\"" + text + d + t + "\r\n" + d + "\"\"\"", text + d + t + "\r\n" + d + "\"");
      if (terminator != '\r') {
        doc.add_row(text + "\r" + text);
        doc.add_row("\r" + text + "\r");
      }
    }
    expect_document_rows(doc);
  }
}

TEST_F(CsvReaderTest, CrLfLineEnds)
{
  // Rows that end with "\r\n" read the same as rows that end with '\n', including with inferred
  // types and with whitespace delimiters; "\r\n" inside of quotes is kept
  std::string lf_buffer;
  std::vector<int64_t> ints;
  std::vector<std::string> strings;
  std::vector<double> doubles;
  for (int i = 0; i < 300; ++i) {
    auto const text         = std::string(i % 19, 'a' + i % 26);
    auto const is_multiline = i % 4 == 0;
    ints.push_back(i * 1001);
    strings.push_back(is_multiline ? text + "\r\n" + text : text + "_");
    doubles.push_back(i + 0.25);
    lf_buffer +=
      std::to_string(ints.back()) + ",\"" + strings.back() + "\"," + std::to_string(i) + ".25\n";
  }
  std::string crlf_buffer;
  for (size_t pos = 0; pos < lf_buffer.size(); ++pos) {
    // Only the line feeds that end rows are preceded by a '\r'
    if (lf_buffer[pos] == '\n' && lf_buffer[pos - 1] != '\r') { crlf_buffer += '\r'; }
    crlf_buffer += lf_buffer[pos];
  }

  auto const expected = [&] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(column_wrapper<int64_t>(ints.begin(), ints.end()).release());
    columns.push_back(cudf::test::strings_column_wrapper(strings.begin(), strings.end()).release());
    columns.push_back(column_wrapper<double>(doubles.begin(), doubles.end()).release());
    return cudf::table{std::move(columns)};
  }();
  for (auto const& buffer : {lf_buffer, crlf_buffer}) {
    auto const result = cudf::io::read_csv(host_buffer_options(buffer).header(-1).build());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.view(), result.tbl->view());
  }

  // Whitespace delimiters (fields without whitespace)
  std::vector<std::unique_ptr<cudf::column>> ws_columns;
  ws_columns.push_back(column_wrapper<int64_t>(ints.begin(), ints.end()).release());
  ws_columns.push_back(column_wrapper<double>(doubles.begin(), doubles.end()).release());
  auto const ws_expected = cudf::table{std::move(ws_columns)};
  for (auto const& line_end : {std::string{"\n"}, std::string{"\r\n"}}) {
    std::string buffer;
    for (size_t i = 0; i < ints.size(); ++i) {
      buffer += "  " + std::to_string(ints[i]) + "   " + std::to_string(i) + ".25" + line_end;
    }
    auto const result =
      cudf::io::read_csv(host_buffer_options(buffer).header(-1).delim_whitespace(true).build());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(ws_expected.view(), result.tbl->view());
  }
}

namespace {
// Writer settings a round trip varies; the reader side follows from them
struct csv_roundtrip_settings {
  cudf::io::compression_type compression        = cudf::io::compression_type::NONE;
  std::optional<cudf::size_type> rows_per_chunk = std::nullopt;
  size_t compression_block_size                 = cudf::io::default_csv_compression_block_size;
  bool include_header                           = true;
};

// Writes `table` with the given settings and reads the output back
std::unique_ptr<cudf::table> roundtrip_csv(cudf::table_view const& table,
                                           std::vector<std::string> const& names,
                                           csv_roundtrip_settings const& settings)
{
  std::vector<char> buffer;
  auto write_builder = cudf::io::csv_writer_options::builder(cudf::io::sink_info(&buffer), table)
                         .include_header(settings.include_header)
                         .names(names)
                         .compression(settings.compression)
                         .compression_block_size(settings.compression_block_size);
  if (settings.rows_per_chunk.has_value()) {
    write_builder.rows_per_chunk(settings.rows_per_chunk.value());
  }
  cudf::io::csv_writer_options write_opts = write_builder;
  cudf::io::write_csv(write_opts);
  EXPECT_GT(buffer.size(), 0);

  auto read_builder = cudf::io::csv_reader_options::builder(
                        cudf::io::source_info(cudf::host_span<char>(buffer.data(), buffer.size())))
                        .compression(settings.compression);
  if (not settings.include_header) { read_builder.header(-1); }
  return cudf::io::read_csv(read_builder.build()).tbl;
}
}  // namespace

TEST_F(CsvWriterTest, ZstdCompression)
{
  auto int_col = column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto str_col = column_wrapper<cudf::string_view>{"a", "b", "c", "d", "e"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_col, str_col});
  auto const names = std::vector<std::string>{"int_col", "str_col"};

  auto const expected = roundtrip_csv(input_table, names, {});
  auto const result =
    roundtrip_csv(input_table, names, {.compression = cudf::io::compression_type::ZSTD});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(CsvWriterTest, ZstdCompressionChunked)
{
  // ZSTD supports concatenated frames, so each chunk can be compressed independently
  auto const num_rows = 100;
  auto sequence       = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto int_col        = column_wrapper<int32_t>(sequence, sequence + num_rows);

  std::vector<std::string> strings(num_rows);
  std::generate(
    strings.begin(), strings.end(), [i = 0]() mutable { return "row_" + std::to_string(i++); });
  cudf::test::strings_column_wrapper str_col(strings.begin(), strings.end());

  cudf::table_view input_table(std::vector<cudf::column_view>{int_col, str_col});
  auto const names = std::vector<std::string>{"value", "name"};

  // the uncompressed output is the reference; chunking must not corrupt row separators
  auto const expected = roundtrip_csv(input_table, names, {.rows_per_chunk = 10});
  auto const result   = roundtrip_csv(
    input_table, names, {.compression = cudf::io::compression_type::ZSTD, .rows_per_chunk = 10});

  EXPECT_EQ(result->num_rows(), num_rows);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(CsvWriterTest, ZstdCompressionNoRows)
{
  // only the header is written, so the output is a single ZSTD frame
  auto int_col = column_wrapper<int32_t>{};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_col});
  auto const names = std::vector<std::string>{"value"};

  auto const expected = roundtrip_csv(input_table, names, {.rows_per_chunk = 8});
  auto const result   = roundtrip_csv(
    input_table, names, {.compression = cudf::io::compression_type::ZSTD, .rows_per_chunk = 8});

  EXPECT_EQ(result->num_rows(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(CsvWriterTest, ZstdCompressionNoHeader)
{
  auto const num_rows = 20;
  auto sequence       = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto int_col        = column_wrapper<int32_t>(sequence, sequence + num_rows);
  cudf::table_view input_table(std::vector<cudf::column_view>{int_col});
  auto const names = std::vector<std::string>{"value"};

  // the header is compressed into its own frame ahead of the data, so this is
  // the only case where the first frame in the output is a data chunk
  auto const expected =
    roundtrip_csv(input_table, names, {.rows_per_chunk = 8, .include_header = false});
  auto const result = roundtrip_csv(input_table,
                                    names,
                                    {.compression    = cudf::io::compression_type::ZSTD,
                                     .rows_per_chunk = 8,
                                     .include_header = false});

  EXPECT_EQ(result->num_rows(), num_rows);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(CsvWriterTest, ZstdCompressionNullsAndUnicode)
{
  auto const nulls = cudf::test::iterators::nulls_at({1, 3});
  auto int_col     = column_wrapper<int32_t>{{1, 2, 3, 4, 5}, nulls};
  auto null_col    = column_wrapper<int32_t>{{0, 0, 0, 0, 0}, cudf::test::iterators::all_nulls()};
  auto str_col =
    cudf::test::strings_column_wrapper{{"ascii é", "nulled", "", "nulled", "mixed 混合 🙂"}, nulls};

  cudf::table_view input_table(std::vector<cudf::column_view>{int_col, null_col, str_col});
  auto const names = std::vector<std::string>{"int_col", "null_col", "str_col"};

  // a block size below the row width splits the output inside multi-byte characters
  auto const expected = roundtrip_csv(input_table, names, {});
  auto const result =
    roundtrip_csv(input_table,
                  names,
                  {.compression = cudf::io::compression_type::ZSTD, .compression_block_size = 16});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(CsvWriterTest, ZstdCompressionSlicedInput)
{
  auto const num_rows = 500;
  auto sequence       = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto int_col        = column_wrapper<int32_t>(
    sequence, sequence + num_rows, cudf::test::iterators::nulls_at({7, 300}));

  std::vector<std::string> strings(num_rows);
  std::generate(
    strings.begin(), strings.end(), [i = 0]() mutable { return "rów_" + std::to_string(i++); });
  auto str_col = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  cudf::table_view input_table(std::vector<cudf::column_view>{int_col, str_col});
  auto const sliced = cudf::slice(input_table, {37, 452}).front();
  auto const names  = std::vector<std::string>{"int_col", "str_col"};

  auto const expected = roundtrip_csv(sliced, names, {.rows_per_chunk = 64});
  auto const result   = roundtrip_csv(sliced,
                                    names,
                                      {.compression            = cudf::io::compression_type::ZSTD,
                                       .rows_per_chunk         = 64,
                                       .compression_block_size = 512});

  EXPECT_EQ(result->num_rows(), 452 - 37);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(CsvWriterTest, UnsupportedCompression)
{
  std::vector<char> buffer;
  auto int_col = column_wrapper<int32_t>{1, 2, 3};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_col});

  EXPECT_THROW(cudf::io::csv_writer_options::builder(cudf::io::sink_info(&buffer), input_table)
                 .compression(cudf::io::compression_type::SNAPPY),
               cudf::logic_error);
}

TEST_F(CsvWriterTest, ZstdCompressionBlockSize)
{
  // enough rows that a small block size splits the output into many blocks
  auto const num_rows = 20'000;
  auto sequence       = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto int_col        = column_wrapper<int32_t>(sequence, sequence + num_rows);

  std::vector<std::string> strings(num_rows);
  std::generate(strings.begin(), strings.end(), [i = 0]() mutable {
    return "row_value_" + std::to_string(i++);
  });
  auto str_col = column_wrapper<cudf::string_view>(strings.begin(), strings.end());
  cudf::table_view input_table(std::vector<cudf::column_view>{int_col, str_col});
  auto const names = std::vector<std::string>{"int_col", "str_col"};

  auto const expected = roundtrip_csv(input_table, names, {});

  // the block size is independent of the chunk size, so all combinations must produce the same
  // table, whether a chunk spans many blocks or a block spans many chunks
  for (auto const block_size :
       {size_t{1}, size_t{4095}, size_t{4096}, size_t{64} * 1024, size_t{1} << 30}) {
    for (auto const rows_per_chunk :
         {std::optional<cudf::size_type>{}, std::optional<cudf::size_type>{1'000}}) {
      auto const result = roundtrip_csv(input_table,
                                        names,
                                        {.compression            = cudf::io::compression_type::ZSTD,
                                         .rows_per_chunk         = rows_per_chunk,
                                         .compression_block_size = block_size});
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
    }
  }
}

TEST_F(CsvWriterTest, InvalidCompressionBlockSize)
{
  std::vector<char> buffer;
  auto int_col = column_wrapper<int32_t>{1, 2, 3};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_col});

  EXPECT_THROW(cudf::io::csv_writer_options::builder(cudf::io::sink_info(&buffer), input_table)
                 .compression_block_size(0),
               cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
