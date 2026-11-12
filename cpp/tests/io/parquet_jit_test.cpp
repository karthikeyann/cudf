/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>

#include <gmock/gmock.h>
#include <io/parquet/jit/fused_filter.hpp>
#include <io/parquet/jit/kernel_provider.hpp>

#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace jit = cudf::io::parquet::jit;

namespace {

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// What a leaf is expected to do with a value.
enum class leaf_effect {
  STORE,           // Stores the value
  DICT_KEEP,       // Stores the value and copies the keep flag of its dictionary entry
  TEST,            // Writes the keep flag of the predicate `lower <= value < upper`
  STORE_AND_TEST,  // Stores the value and writes the keep flag of the predicate
};

// Value never produced by a leaf, to detect rows a kernel should not have written.
template <typename T>
constexpr T untouched = static_cast<T>(0x5A5A'5A5A'5A5A'5A5AULL);

template <typename T>
bool in_range(T value, uint64_t lower, uint64_t upper)
{
  // Leaves compare the value as the signed integer of its width.
  auto const signed_value =
    sizeof(T) == 4 ? int64_t{static_cast<int32_t>(value)} : static_cast<int64_t>(value);
  return signed_value >= static_cast<int64_t>(lower) and signed_value < static_cast<int64_t>(upper);
}

// Returns the leaves of the predicate `lower <= column < upper` on a column of signed type T.
template <typename T>
jit::column_predicate make_range_predicate(T lower, T upper)
{
  auto const stream        = cudf::get_default_stream();
  auto const column        = cudf::test::fixed_width_column_wrapper<T>({T{0}});
  auto lower_scalar        = cudf::numeric_scalar<T>(lower);
  auto upper_scalar        = cudf::numeric_scalar<T>(upper);
  auto const lower_literal = cudf::ast::literal(lower_scalar);
  auto const upper_literal = cudf::ast::literal(upper_scalar);
  auto const column_ref    = cudf::ast::column_reference(0);
  auto const at_least =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, column_ref, lower_literal);
  auto const below = cudf::ast::operation(cudf::ast::ast_operator::LESS, column_ref, upper_literal);
  auto const range = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, at_least, below);
  auto predicate   = jit::make_column_predicate(
    range, cudf::table_view{{column}}, stream, cudf::get_current_device_resource_ref());
  CUDF_EXPECTS(predicate.has_value(), "A range predicate must qualify for a leaf");
  return *predicate;
}

// Runs one kernel over three buffers of synthetic rows and compares everything it wrote with a
// host evaluation.
template <typename T>
void run_kernel(jit::flat_buffer_shape shape, jit::leaf const& leaf, leaf_effect effect)
{
  SCOPED_TRACE(std::string{"is_dict="} + std::to_string(shape.is_dict) + " num_bytes=" +
               std::to_string(shape.num_bytes) + " nullable=" + std::to_string(shape.nullable) +
               " effect=" + std::to_string(static_cast<int>(effect)));
  ASSERT_EQ(sizeof(T), static_cast<std::size_t>(shape.num_bytes));

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const kernel = jit::get_flat_buffer_kernel(shape, leaf);
  ASSERT_TRUE(kernel.has_value());

  constexpr uint32_t num_dict_entries = 1'000;
  constexpr uint32_t dict_first       = 7;
  // Rows of each buffer; the middle one spans several tiles of the launch below.
  auto const buffer_num_rows = std::vector<uint32_t>{1, 70'001, 4'096};
  // Bit of row zero in each buffer's validity bitmap.
  auto const validity_offsets = std::vector<uint32_t>{0, 3, 37};
  constexpr uint64_t lower    = 1'000;
  constexpr uint64_t upper    = 600'000;

  auto dictionary = std::vector<T>(num_dict_entries);
  auto dict_keep  = std::vector<uint8_t>(dict_first + num_dict_entries + 1);
  for (uint32_t i = 0; i < num_dict_entries; ++i) {
    dictionary[i]             = static_cast<T>(i * 7'919u + 13u);
    dict_keep[dict_first + i] = (i % 3) == 0;
  }
  // The entry past the dictionary answers for out-of-range indices.
  dict_keep.back() = 1;

  auto num_rows = uint32_t{0};
  for (auto const rows : buffer_num_rows) {
    num_rows += rows;
  }
  auto values          = std::vector<T>(num_rows);
  auto indices         = std::vector<uint32_t>(num_rows);
  auto validity        = std::vector<uint32_t>(buffer_num_rows.size() * 4'096, 0);
  auto expected_values = std::vector<T>(num_rows, untouched<T>);
  auto expected_keep   = std::vector<uint8_t>(num_rows, 0);

  auto buffers  = std::vector<jit::flat_buffer_ref>{};
  auto tiles    = std::vector<jit::flat_buffer_tile>{};
  auto row_base = uint32_t{0};
  for (std::size_t b = 0; b < buffer_num_rows.size(); ++b) {
    for (uint32_t r = 0; r < buffer_num_rows[b]; ++r) {
      auto const row = row_base + r;
      values[row]    = static_cast<T>(row * 40'503u);
      // A few indices are past the end of the dictionary.
      indices[row]     = (row * 2'654'435'761u) % (num_dict_entries + 5);
      auto const valid = not shape.nullable or ((row * 2'246'822'519u) >> 13) % 7 != 0;
      if (valid and shape.nullable) {
        auto const bit = validity_offsets[b] + r;
        validity[b * 4'096 + (bit >> 5)] |= 1u << (bit & 31);
      }
      if (not valid) { continue; }

      auto const in_dictionary = indices[row] < num_dict_entries;
      auto const value =
        shape.is_dict ? (in_dictionary ? dictionary[indices[row]] : T{0}) : values[row];
      if (effect != leaf_effect::TEST) { expected_values[row] = value; }
      if (effect == leaf_effect::DICT_KEEP) {
        expected_keep[row] =
          dict_keep[dict_first + (in_dictionary ? indices[row] : num_dict_entries)];
      }
      if (effect == leaf_effect::TEST or effect == leaf_effect::STORE_AND_TEST) {
        expected_keep[row] = in_range(value, lower, upper);
      }
    }
    buffers.push_back(jit::flat_buffer_ref{.validity_offset  = validity_offsets[b],
                                           .row_base         = row_base,
                                           .num_dict_entries = num_dict_entries,
                                           .dict_first       = dict_first,
                                           .ctx_idx          = 1});
    constexpr uint32_t tile_num_rows = 10'000;
    for (uint32_t first = 0; first < buffer_num_rows[b]; first += tile_num_rows) {
      tiles.push_back(
        {static_cast<uint32_t>(b), first, std::min(tile_num_rows, buffer_num_rows[b] - first)});
    }
    row_base += buffer_num_rows[b];
  }

  auto const d_dictionary = cudf::detail::make_device_uvector(dictionary, stream, mr);
  auto const d_dict_keep  = cudf::detail::make_device_uvector(dict_keep, stream, mr);
  auto const d_values     = cudf::detail::make_device_uvector(values, stream, mr);
  auto const d_indices    = cudf::detail::make_device_uvector(indices, stream, mr);
  auto const d_validity   = cudf::detail::make_device_uvector(validity, stream, mr);
  auto d_out =
    cudf::detail::make_device_uvector(std::vector<T>(num_rows, untouched<T>), stream, mr);
  auto d_keep = cudf::detail::make_device_uvector(std::vector<uint8_t>(num_rows, 0), stream, mr);

  // The buffers hold the rows of the whole launch back to back; the kernel must read each
  // buffer from its own start, so every buffer points at its first row.
  row_base = 0;
  for (std::size_t b = 0; b < buffers.size(); ++b) {
    buffers[b].values = shape.is_dict
                          ? reinterpret_cast<uint8_t const*>(d_indices.data() + row_base)
                          : reinterpret_cast<uint8_t const*>(d_values.data() + row_base);
    buffers[b].dict   = reinterpret_cast<uint8_t const*>(d_dictionary.data());
    if (shape.nullable) { buffers[b].validity = d_validity.data() + b * 4'096; }
    row_base += buffer_num_rows[b];
  }

  // Context zero is never referenced; it would make the kernel write through null pointers.
  auto const ctxs = std::vector<jit::leaf_ctx>{jit::leaf_ctx{},
                                               jit::leaf_ctx{.out       = d_out.data(),
                                                             .keep      = d_keep.data(),
                                                             .dict_keep = d_dict_keep.data(),
                                                             .scale     = 0,
                                                             .lit       = {lower, upper, 0, 0}}};

  jit::launch_flat_buffer_kernel(*kernel, buffers, tiles, ctxs, stream, mr);

  EXPECT_EQ(cudf::detail::make_std_vector(d_out, stream), expected_values);
  EXPECT_EQ(cudf::detail::make_std_vector(d_keep, stream), expected_keep);
}

template <typename T>
void run_kernel_of_width(bool is_dict, bool nullable, jit::leaf const& leaf, leaf_effect effect)
{
  run_kernel<T>({.is_dict = is_dict, .num_bytes = sizeof(T), .nullable = nullable}, leaf, effect);
}

// Returns leaf source that no other test and no earlier process has compiled.
jit::leaf_source make_unique_leaf_source(std::string const& tag)
{
  auto leaf = make_range_predicate<int32_t>(1'000, 600'000).test_leaf;
  leaf.code += "\n// " + tag + " " +
               std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "\n";
  return leaf;
}

}  // namespace

struct ParquetJitTest : public cudf::test::BaseFixture {};

TEST_F(ParquetJitTest, StoreLeafEveryShape)
{
  for (auto const is_dict : {false, true}) {
    for (auto const nullable : {false, true}) {
      run_kernel_of_width<uint32_t>(
        is_dict, nullable, jit::builtin_leaf::STORE, leaf_effect::STORE);
      run_kernel_of_width<uint64_t>(
        is_dict, nullable, jit::builtin_leaf::STORE, leaf_effect::STORE);
    }
  }
}

TEST_F(ParquetJitTest, DictKeepLeaf)
{
  for (auto const nullable : {false, true}) {
    run_kernel_of_width<uint32_t>(
      true, nullable, jit::builtin_leaf::DICT_KEEP, leaf_effect::DICT_KEEP);
    run_kernel_of_width<uint64_t>(
      true, nullable, jit::builtin_leaf::DICT_KEEP, leaf_effect::DICT_KEEP);
  }
}

TEST_F(ParquetJitTest, PredicateLeafEveryShape)
{
  auto const predicate32 = make_range_predicate<int32_t>(1'000, 600'000);
  auto const predicate64 = make_range_predicate<int64_t>(1'000, 600'000);
  for (auto const is_dict : {false, true}) {
    for (auto const nullable : {false, true}) {
      run_kernel_of_width<uint32_t>(is_dict, nullable, predicate32.test_leaf, leaf_effect::TEST);
      run_kernel_of_width<uint32_t>(
        is_dict, nullable, predicate32.store_leaf, leaf_effect::STORE_AND_TEST);
      run_kernel_of_width<uint64_t>(is_dict, nullable, predicate64.test_leaf, leaf_effect::TEST);
      run_kernel_of_width<uint64_t>(
        is_dict, nullable, predicate64.store_leaf, leaf_effect::STORE_AND_TEST);
    }
  }
}

TEST_F(ParquetJitTest, LiteralsAreNotPartOfTheLeaf)
{
  auto const first  = make_range_predicate<int32_t>(1'000, 600'000);
  auto const second = make_range_predicate<int32_t>(-5, 17);
  EXPECT_EQ(first.test_leaf.code, second.test_leaf.code);
  EXPECT_EQ(first.store_leaf.code, second.store_leaf.code);
  EXPECT_NE(first.test_leaf.code, first.store_leaf.code);
  EXPECT_THAT(first.literals, testing::ElementsAre(1'000, 600'000, 0, 0));
  EXPECT_THAT(second.literals, testing::ElementsAre(static_cast<uint32_t>(-5), 17, 0, 0));
}

TEST_F(ParquetJitTest, KernelIdentityFollowsContent)
{
  auto const plain32 = jit::flat_buffer_shape{.is_dict = false, .num_bytes = 4, .nullable = false};
  auto const plain64 = jit::flat_buffer_shape{.is_dict = false, .num_bytes = 8, .nullable = false};
  auto const predicate = make_range_predicate<int32_t>(1'000, 600'000);

  auto const store = jit::get_kernel_identity(plain32, jit::builtin_leaf::STORE);
  // The fragments compiled ahead of time are identified by the hash of their bytes.
  EXPECT_THAT(store.consumer, testing::HasSubstr("/frag"));
  EXPECT_THAT(store.leaf, testing::HasSubstr("/frag"));
  EXPECT_EQ(store.consumer, jit::get_kernel_identity(plain32, jit::builtin_leaf::STORE).consumer);

  // Another shape is another kernel fragment linked with the same leaf.
  auto const store64 = jit::get_kernel_identity(plain64, jit::builtin_leaf::STORE);
  EXPECT_NE(store.consumer, store64.consumer);
  EXPECT_EQ(store.leaf, store64.leaf);

  // Another leaf is another leaf name under the same kernel fragment.
  auto const dict_keep = jit::get_kernel_identity(plain32, jit::builtin_leaf::DICT_KEEP);
  auto const test      = jit::get_kernel_identity(plain32, predicate.test_leaf);
  auto const stored    = jit::get_kernel_identity(plain32, predicate.store_leaf);
  EXPECT_EQ(store.consumer, test.consumer);
  EXPECT_NE(store.leaf, dict_keep.leaf);
  EXPECT_NE(store.leaf, test.leaf);
  EXPECT_NE(test.leaf, stored.leaf);

  // Any change of the leaf source changes its name, and the same source keeps it.
  auto changed = predicate.test_leaf;
  changed.code += "\n";
  EXPECT_NE(test.leaf, jit::get_kernel_identity(plain32, changed).leaf);
  EXPECT_EQ(test.leaf,
            jit::get_kernel_identity(plain32, make_range_predicate<int32_t>(0, 1).test_leaf).leaf);
}

TEST_F(ParquetJitTest, FailedBuildIsRetriedOnlyAfterTheDelay)
{
  auto const shape = jit::flat_buffer_shape{.is_dict = false, .num_bytes = 4, .nullable = false};
  auto const broken =
    jit::leaf_source{"#error leaf that cannot be compiled " +
                     std::to_string(std::chrono::system_clock::now().time_since_epoch().count())};

  jit::set_kernel_retry_policy({.delay = std::chrono::hours{1}, .max_attempts = 3});
  auto const before = jit::get_kernel_provider_stats();
  EXPECT_FALSE(jit::get_flat_buffer_kernel(shape, broken).has_value());
  auto const failed = jit::get_kernel_provider_stats();
  EXPECT_EQ(failed.attempts, before.attempts + 1);
  EXPECT_EQ(failed.failures, before.failures + 1);

  // Within the delay the kernel is refused without building it again.
  for (auto i = 0; i < 100; ++i) {
    EXPECT_FALSE(jit::get_flat_buffer_kernel(shape, broken).has_value());
  }
  auto const refused = jit::get_kernel_provider_stats();
  EXPECT_EQ(refused.attempts, failed.attempts);
  EXPECT_EQ(refused.failures, failed.failures);
  EXPECT_EQ(refused.rejected, failed.rejected + 100);

  // After the delay it is built again, until its attempts are used up.
  jit::set_kernel_retry_policy({.delay = std::chrono::milliseconds{0}, .max_attempts = 3});
  for (auto i = 0; i < 10; ++i) {
    EXPECT_FALSE(jit::get_flat_buffer_kernel(shape, broken).has_value());
  }
  auto const exhausted = jit::get_kernel_provider_stats();
  EXPECT_EQ(exhausted.attempts, failed.attempts + 2);
  EXPECT_EQ(exhausted.failures, failed.failures + 2);

  // A failure leaves other kernels of the same shape usable.
  EXPECT_TRUE(
    jit::get_flat_buffer_kernel(shape, make_unique_leaf_source("after failure")).has_value());
  jit::set_kernel_retry_policy({});
}

TEST_F(ParquetJitTest, ConcurrentRequestsForOneKernel)
{
  constexpr int num_threads = 8;
  auto const shape  = jit::flat_buffer_shape{.is_dict = false, .num_bytes = 4, .nullable = true};
  auto const leaf   = make_unique_leaf_source("concurrent");
  auto const before = jit::get_kernel_provider_stats();

  auto succeeded = std::vector<int>(num_threads, 0);
  auto threads   = std::vector<std::thread>{};
  for (auto t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      // Make the device's context current on this thread before the kernel is loaded.
      auto const warmup = rmm::device_uvector<int>(
        1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
      succeeded[t] = jit::get_flat_buffer_kernel(shape, leaf).has_value();
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_THAT(succeeded, testing::Each(1));
  auto const after = jit::get_kernel_provider_stats();
  EXPECT_EQ(after.failures, before.failures);
  // The kernel the threads raced for is correct.
  run_kernel<uint32_t>(shape, leaf, leaf_effect::TEST);
}

TEST_F(ParquetJitTest, PredicateQualification)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const ints   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto const longs  = cudf::test::fixed_width_column_wrapper<int64_t>({1, 2, 3});
  auto const shorts = cudf::test::fixed_width_column_wrapper<int16_t>({1, 2, 3});
  auto const text   = cudf::test::strings_column_wrapper({"a", "b", "c"});
  auto const table  = cudf::table_view{{ints, longs, shorts, text}};

  auto int_scalar           = cudf::numeric_scalar<int32_t>(2);
  auto long_scalar          = cudf::numeric_scalar<int64_t>(2);
  auto short_scalar         = cudf::numeric_scalar<int16_t>(2);
  auto null_scalar          = cudf::numeric_scalar<int32_t>(2, false);
  auto string_scalar        = cudf::string_scalar("b");
  auto const int_literal    = cudf::ast::literal(int_scalar);
  auto const long_literal   = cudf::ast::literal(long_scalar);
  auto const short_literal  = cudf::ast::literal(short_scalar);
  auto const null_literal   = cudf::ast::literal(null_scalar);
  auto const string_literal = cudf::ast::literal(string_scalar);
  auto const int_ref        = cudf::ast::column_reference(0);
  auto const long_ref       = cudf::ast::column_reference(1);
  auto const short_ref      = cudf::ast::column_reference(2);
  auto const text_ref       = cudf::ast::column_reference(3);

  using op             = cudf::ast::ast_operator;
  auto const qualifies = [&](cudf::ast::expression const& expr) {
    return jit::make_column_predicate(expr, table, stream, mr).has_value();
  };

  auto const int_less   = cudf::ast::operation(op::LESS, int_ref, int_literal);
  auto const long_equal = cudf::ast::operation(op::EQUAL, long_literal, long_ref);
  EXPECT_TRUE(qualifies(int_less));
  EXPECT_TRUE(qualifies(long_equal));
  EXPECT_TRUE(qualifies(cudf::ast::operation(op::NOT, int_less)));
  auto const int_greater = cudf::ast::operation(op::GREATER, int_ref, int_literal);
  EXPECT_TRUE(qualifies(cudf::ast::operation(op::LOGICAL_OR, int_less, int_greater)));

  {
    auto const predicate = jit::make_column_predicate(long_equal, table, stream, mr);
    ASSERT_TRUE(predicate.has_value());
    EXPECT_EQ(predicate->column_index, 1);
    EXPECT_EQ(predicate->num_bytes, 8);
    EXPECT_THAT(predicate->literals, testing::ElementsAre(2, 0, 0, 0));
  }

  // Two columns, a column that is not 4 or 8 bytes wide, and a string column.
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::LOGICAL_AND, int_less, long_equal)));
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::LESS, short_ref, short_literal)));
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::EQUAL, text_ref, string_literal)));
  // Operators that are null-aware or can fail, and a null literal.
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::IS_NULL, int_ref)));
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::NULL_EQUAL, int_ref, int_literal)));
  auto const sum = cudf::ast::operation(op::ADD, int_ref, int_literal);
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::LESS, sum, int_literal)));
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::LESS, int_ref, null_literal)));
  // A literal of another type than the column, and no column at all.
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::LESS, int_ref, long_literal)));
  EXPECT_FALSE(qualifies(cudf::ast::operation(op::LESS, int_literal, int_literal)));
  // More literals than a leaf can read.
  auto const two   = cudf::ast::operation(op::LOGICAL_OR, int_less, int_greater);
  auto const four  = cudf::ast::operation(op::LOGICAL_OR, two, two);
  auto const eight = cudf::ast::operation(op::LOGICAL_OR, four, four);
  EXPECT_TRUE(qualifies(four));
  EXPECT_FALSE(qualifies(eight));
}

TEST_F(ParquetJitTest, DictionaryEvaluation)
{
  auto const stream    = cudf::get_default_stream();
  auto const mr        = cudf::get_current_device_resource_ref();
  auto const predicate = make_range_predicate<int64_t>(0, 500);

  constexpr uint32_t num_entries = 40'000;
  auto dictionary                = std::vector<int64_t>(num_entries);
  auto expected                  = std::vector<uint8_t>(num_entries + 1);
  for (uint32_t i = 0; i < num_entries; ++i) {
    dictionary[i] = static_cast<int64_t>(i % 1'000) - 250;
    expected[i]   = dictionary[i] >= 0 and dictionary[i] < 500;
  }
  // Value zero, which is what an out-of-range index decodes to, is inside the range.
  expected.back() = 1;

  auto const d_dictionary = cudf::detail::make_device_uvector(dictionary, stream, mr);
  auto const keep         = jit::evaluate_dictionary(
    predicate, reinterpret_cast<uint8_t const*>(d_dictionary.data()), num_entries, stream, mr);
  ASSERT_TRUE(keep.has_value());
  EXPECT_EQ(cudf::detail::make_std_vector(*keep, stream), expected);

  // An empty dictionary still answers for value zero.
  auto const empty = jit::evaluate_dictionary(predicate, nullptr, 0, stream, mr);
  ASSERT_TRUE(empty.has_value());
  EXPECT_THAT(cudf::detail::make_std_vector(*empty, stream), testing::ElementsAre(1));
}

namespace {

// The table the filter tests read: columns `id` (INT32, unique), `key` (INT32, 50 distinct
// values, nullable), `amount` (FLOAT64, nullable), `total` (INT64), `day` (TIMESTAMP_DAYS) and
// `name` (STRING).
std::unique_ptr<cudf::table> make_filter_table(cudf::size_type num_rows)
{
  auto const sequence = cuda::counting_iterator<int32_t>{0};
  auto const key_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 11 != 0; });
  auto const amount_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 5 != 3; });
  auto const keys =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i * 37) % 50; });
  auto const amounts = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<double>((i * 7) % 1'000) / 8.0; });
  auto const totals = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return int64_t{5'000'000'000} + (i * 13) % 300; });
  auto const days = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::timestamp_D{cudf::duration_D{8'000 + (i * 3) % 2'500}}; });
  auto const names = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::string{"name"} + std::to_string(i % 17); });

  auto columns = std::vector<std::unique_ptr<cudf::column>>{};
  columns.push_back(
    cudf::test::fixed_width_column_wrapper<int32_t>(sequence, sequence + num_rows).release());
  columns.push_back(
    cudf::test::fixed_width_column_wrapper<int32_t>(keys, keys + num_rows, key_validity).release());
  columns.push_back(
    cudf::test::fixed_width_column_wrapper<double>(amounts, amounts + num_rows, amount_validity)
      .release());
  columns.push_back(
    cudf::test::fixed_width_column_wrapper<int64_t>(totals, totals + num_rows).release());
  columns.push_back(
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_D>(days, days + num_rows).release());
  columns.push_back(cudf::test::strings_column_wrapper(names, names + num_rows).release());
  return std::make_unique<cudf::table>(std::move(columns));
}

// Owns the scalars and expressions of the filters the differential tests run.
struct filter_set {
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  cudf::ast::tree tree;
  std::vector<std::reference_wrapper<cudf::ast::expression const>> filters;
  std::vector<std::string> descriptions;

  template <typename T>
  cudf::ast::expression const& literal(T value)
  {
    if constexpr (cudf::is_timestamp<T>()) {
      scalars.push_back(std::make_unique<cudf::timestamp_scalar<T>>(value, true));
      return tree.push(
        cudf::ast::literal{static_cast<cudf::timestamp_scalar<T>&>(*scalars.back())});
    } else if constexpr (std::is_same_v<T, std::string>) {
      scalars.push_back(std::make_unique<cudf::string_scalar>(value));
      return tree.push(cudf::ast::literal{static_cast<cudf::string_scalar&>(*scalars.back())});
    } else {
      scalars.push_back(std::make_unique<cudf::numeric_scalar<T>>(value, true));
      return tree.push(cudf::ast::literal{static_cast<cudf::numeric_scalar<T>&>(*scalars.back())});
    }
  }

  cudf::ast::expression const& column(cudf::size_type index)
  {
    return tree.push(cudf::ast::column_reference{index});
  }

  cudf::ast::expression const& apply(cudf::ast::ast_operator op,
                                     cudf::ast::expression const& left,
                                     cudf::ast::expression const& right)
  {
    return tree.push(cudf::ast::operation{op, left, right});
  }

  void add(std::string description, cudf::ast::expression const& filter)
  {
    descriptions.push_back(std::move(description));
    filters.emplace_back(filter);
  }
};

filter_set make_filters()
{
  using op      = cudf::ast::ast_operator;
  auto set      = filter_set{};
  auto const id = 0, key = 1, amount = 2, total = 3, day = 4, name = 5;

  // Literals below, at and beyond the bounds of the low-cardinality key [0, 49].
  for (auto const bound : {-1, 0, 1, 25, 49, 50, std::numeric_limits<int32_t>::max()}) {
    auto const text = std::to_string(bound);
    set.add("key < " + text, set.apply(op::LESS, set.column(key), set.literal<int32_t>(bound)));
    set.add("key >= " + text,
            set.apply(op::GREATER_EQUAL, set.column(key), set.literal<int32_t>(bound)));
    set.add("key == " + text, set.apply(op::EQUAL, set.column(key), set.literal<int32_t>(bound)));
    set.add("key != " + text,
            set.apply(op::NOT_EQUAL, set.literal<int32_t>(bound), set.column(key)));
  }
  // A range on one column, written as two conjuncts.
  set.add("10 <= key < 20",
          set.apply(op::LOGICAL_AND,
                    set.apply(op::GREATER_EQUAL, set.column(key), set.literal<int32_t>(10)),
                    set.apply(op::LESS, set.column(key), set.literal<int32_t>(20))));
  // A disjunction on one column.
  set.add("key < 5 or key > 45",
          set.apply(op::LOGICAL_OR,
                    set.apply(op::LESS, set.column(key), set.literal<int32_t>(5)),
                    set.apply(op::GREATER, set.column(key), set.literal<int32_t>(45))));
  // Other types: nullable FLOAT64, INT64 and TIMESTAMP_DAYS.
  set.add("amount < 60.5", set.apply(op::LESS, set.column(amount), set.literal<double>(60.5)));
  set.add("amount >= 1e9",
          set.apply(op::GREATER_EQUAL, set.column(amount), set.literal<double>(1e9)));
  set.add("total == 5000000007",
          set.apply(op::EQUAL, set.column(total), set.literal<int64_t>(5'000'000'007)));
  set.add(
    "day <= 9000",
    set.apply(
      op::LESS_EQUAL, set.column(day), set.literal(cudf::timestamp_D{cudf::duration_D{9'000}})));
  // Conjuncts on three columns, as in a scan with several range filters.
  set.add("day range and amount range and total bound",
          set.apply(
            op::LOGICAL_AND,
            set.apply(
              op::LOGICAL_AND,
              set.apply(op::LOGICAL_AND,
                        set.apply(op::GREATER_EQUAL,
                                  set.column(day),
                                  set.literal(cudf::timestamp_D{cudf::duration_D{8'500}})),
                        set.apply(op::LESS,
                                  set.column(day),
                                  set.literal(cudf::timestamp_D{cudf::duration_D{9'500}}))),
              set.apply(op::LOGICAL_AND,
                        set.apply(op::GREATER_EQUAL, set.column(amount), set.literal<double>(20.0)),
                        set.apply(op::LESS_EQUAL, set.column(amount), set.literal<double>(90.0)))),
            set.apply(op::LESS, set.column(total), set.literal<int64_t>(5'000'000'200))));
  // Conjuncts that stay an ordinary expression: two columns, and a string column.
  set.add("key < 30 and key > id",
          set.apply(op::LOGICAL_AND,
                    set.apply(op::LESS, set.column(key), set.literal<int32_t>(30)),
                    set.apply(op::GREATER, set.column(key), set.column(id))));
  set.add(
    "name == name3 and key >= 10 and amount < 100",
    set.apply(op::LOGICAL_AND,
              set.apply(op::LOGICAL_AND,
                        set.apply(op::EQUAL, set.column(name), set.literal<std::string>("name3")),
                        set.apply(op::GREATER_EQUAL, set.column(key), set.literal<int32_t>(10))),
              set.apply(op::LESS, set.column(amount), set.literal<double>(100.0))));
  // Nothing a leaf can evaluate.
  set.add("name == name3",
          set.apply(op::EQUAL, set.column(name), set.literal<std::string>("name3")));
  return set;
}

}  // namespace

TEST_F(ParquetJitTest, FusedFilterMatchesExpression)
{
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const table   = make_filter_table(40'000);
  auto const filters = make_filters();

  auto num_fused = 0;
  for (std::size_t i = 0; i < filters.filters.size(); ++i) {
    SCOPED_TRACE(filters.descriptions[i]);
    auto const& filter  = filters.filters[i].get();
    auto const expected = cudf::compute_column(table->view(), filter);
    auto const fused    = jit::fused_filter(filter, table->view(), stream, mr);
    if (fused.predicates().empty()) { continue; }
    ++num_fused;

    auto const result = fused.evaluate(table->view(), false, stream, mr);
    ASSERT_TRUE(result.has_value());
    // The two masks may differ in where they are null, never in which rows they keep.
    auto const kept          = cudf::apply_retention_mask(table->view(), (*result)->view());
    auto const expected_kept = cudf::apply_retention_mask(table->view(), expected->view());
    CUDF_TEST_EXPECT_TABLES_EQUAL(kept->view(), expected_kept->view());
  }
  // Every filter but the last has a conjunct for a leaf.
  EXPECT_EQ(num_fused, static_cast<int>(filters.filters.size()) - 1);
}

TEST_F(ParquetJitTest, ReaderResultsMatch)
{
  auto const table    = make_filter_table(60'000);
  auto const filepath = temp_env->get_temp_filepath("ReaderResultsMatch.parquet");
  // Small row groups and pages give every read several of each.
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table->view())
      .row_group_size_rows(20'000)
      .max_page_size_rows(5'000));

  auto const filters  = make_filters();
  auto max_num_chunks = std::size_t{0};
  for (std::size_t i = 0; i < filters.filters.size(); ++i) {
    SCOPED_TRACE(filters.descriptions[i]);
    auto const read = [&](bool use_jit_decode_filter) {
      return cudf::io::read_parquet(
        cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
          .filter(filters.filters[i].get())
          .use_jit_decode_filter(use_jit_decode_filter));
    };
    auto const expected = read(false);
    auto const result   = read(true);
    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected.tbl->view());

    // The same rows, in the same order, when the file is read in chunks.
    auto const options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .filter(filters.filters[i].get())
                           .use_jit_decode_filter(true)
                           .build();
    auto const reader = cudf::io::chunked_parquet_reader(200'000, 400'000, options);
    auto chunks       = std::vector<std::unique_ptr<cudf::table>>{};
    auto views        = std::vector<cudf::table_view>{};
    while (reader.has_next()) {
      chunks.push_back(reader.read_chunk().tbl);
      views.push_back(chunks.back()->view());
    }
    max_num_chunks = std::max(max_num_chunks, views.size());
    // Concatenation may drop a null mask that has no nulls.
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::concatenate(views)->view(), expected.tbl->view());
  }
  // A filter that prunes every row group reads one empty chunk; the others read several.
  EXPECT_GT(max_num_chunks, std::size_t{1});
}

TEST_F(ParquetJitTest, ReaderWithFilterOnlyAndPrependedColumns)
{
  auto const table    = make_filter_table(30'000);
  auto const filepath = temp_env->get_temp_filepath("ReaderWithFilterOnlyColumns.parquet");
  auto metadata       = cudf::io::table_input_metadata{table->view()};
  auto const names    = std::vector<std::string>{"id", "key", "amount", "total", "day", "name"};
  for (std::size_t i = 0; i < names.size(); ++i) {
    metadata.column_metadata[i].set_name(names[i]);
  }
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table->view())
      .metadata(std::move(metadata)));

  auto lower               = cudf::numeric_scalar<int32_t>(10);
  auto upper               = cudf::numeric_scalar<double>(60.5);
  auto const lower_literal = cudf::ast::literal(lower);
  auto const upper_literal = cudf::ast::literal(upper);
  auto const key_ref       = cudf::ast::column_name_reference("key");
  auto const amount_ref    = cudf::ast::column_name_reference("amount");
  auto const key_filter =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, key_ref, lower_literal);
  auto const amount_filter =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, amount_ref, upper_literal);
  auto const filter =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, key_filter, amount_filter);

  // The filter columns are not read into the output, and two columns are put before it.
  auto const read = [&](bool use_jit_decode_filter) {
    return cudf::io::read_parquet(
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_names({"id", "name"})
        .filter(filter)
        .prepend_source_index_column(true)
        .prepend_row_index_column(true)
        .use_jit_decode_filter(use_jit_decode_filter));
  };
  auto const expected = read(false);
  auto const result   = read(true);
  EXPECT_EQ(result.tbl->num_columns(), 4);
  EXPECT_GT(result.tbl->num_rows(), 0);
  EXPECT_LT(result.tbl->num_rows(), 30'000);
  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected.tbl->view());
}

CUDF_TEST_PROGRAM_MAIN()
