/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <io/parquet/jit/kernel_provider.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::parquet::jit {

/**
 * @brief What a predicate leaf does with a value besides testing it.
 */
enum class leaf_action : uint8_t {
  TEST,            ///< Only writes the keep flag
  STORE_AND_TEST,  ///< Stores the value in the output, then writes the keep flag
};

/**
 * @brief A filter predicate on one fixed-width column, in the form the leaves evaluate it.
 *
 * The literal values are not part of the leaf sources: they travel in the leaf context, so every
 * predicate with the same structure and types shares its compiled leaves.
 */
struct column_predicate {
  size_type column_index = 0;                          ///< Index of the column the predicate reads
  data_type type{};                                    ///< Type of the column the predicate reads
  int num_bytes = 4;                                   ///< Width of the column's values: 4 or 8
  leaf_source test_leaf;                               ///< Leaf of action `TEST`
  leaf_source store_leaf;                              ///< Leaf of action `STORE_AND_TEST`
  std::array<uint64_t, max_leaf_literals> literals{};  ///< Values for `leaf_ctx::lit`

  /**
   * @brief Returns the leaf source of an action.
   *
   * @param action What the leaf does with a value
   * @return Source of the leaf
   */
  [[nodiscard]] leaf_source const& get_leaf(leaf_action action) const
  {
    return action == leaf_action::TEST ? test_leaf : store_leaf;
  }
};

/**
 * @brief Converts a predicate on one column to leaves.
 *
 * A predicate qualifies if it reads exactly one column of a 4- or 8-byte numeric, timestamp or
 * duration type, combines only comparisons with `LOGICAL_AND`, `LOGICAL_OR` and `NOT`, and
 * compares against at most `max_leaf_literals` valid literals. Such a predicate is null when
 * its column is null, so a null row is never kept.
 *
 * @param predicate The predicate, referencing its column by index in @p table
 * @param table Table providing the type of the referenced column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations
 * @return The predicate's leaves and literals, or `std::nullopt` if it does not qualify
 */
[[nodiscard]] std::optional<column_predicate> make_column_predicate(
  ast::expression const& predicate,
  table_view const& table,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

/**
 * @brief Evaluates a predicate on every entry of a dictionary of fixed-width values.
 *
 * @param predicate The predicate
 * @param dictionary The dictionary's values, `predicate.num_bytes` wide each; needs no alignment
 * @param num_entries Number of dictionary entries
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned table
 * @return One keep flag per entry followed by the flag of value zero, which is what an
 * out-of-range dictionary index decodes to; `std::nullopt` if the kernel is not available
 */
[[nodiscard]] std::optional<rmm::device_uvector<uint8_t>> evaluate_dictionary(
  column_predicate const& predicate,
  uint8_t const* dictionary,
  uint32_t num_entries,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

/**
 * @brief A row filter whose single-column conjuncts are evaluated by leaves.
 *
 * The filter is split at its top-level `LOGICAL_AND` operators. Conjuncts that qualify as a
 * `column_predicate` are grouped by column and evaluated by one kernel per column; the remaining
 * conjuncts stay an ordinary expression that is evaluated together with the kernels' results.
 */
class fused_filter {
 public:
  /**
   * @brief Plans the evaluation of a filter over tables with the columns of a table.
   *
   * @param filter The filter, referencing columns by index in @p table
   * @param table Table providing the types of the columns the filter reads
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Memory resources used for temporary allocations
   */
  fused_filter(ast::expression const& filter,
               table_view const& table,
               cuda::stream_ref stream,
               cudf::memory_resources mr);

  /**
   * @brief Returns the predicates evaluated by leaves.
   *
   * @return One predicate per group of fused conjuncts
   */
  [[nodiscard]] std::vector<column_predicate> const& predicates() const { return _predicates; }

  /**
   * @brief Evaluates the filter over a table.
   *
   * @param table The table to filter; its columns must have the types the filter was planned for
   * @param use_jit Whether the conjuncts not evaluated by leaves use the JIT expression path
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Memory resources used for temporary allocations and the returned column
   * @return A BOOL8 column that is valid and true for the rows the filter keeps, or
   * `std::nullopt` if a kernel is not available or @p table does not have the planned columns,
   * in which case the caller evaluates the original filter
   */
  [[nodiscard]] std::optional<std::unique_ptr<column>> evaluate(table_view const& table,
                                                                bool use_jit,
                                                                cuda::stream_ref stream,
                                                                cudf::memory_resources mr) const;

 private:
  ast::tree _tree;                            // Owns the expressions built for the plan
  std::vector<column_predicate> _predicates;  // Fused conjuncts, one group per entry
  // Kernel of each predicate for a column without and with a validity bitmap, once linked.
  mutable std::vector<std::array<std::optional<flat_buffer_kernel>, 2>> _kernels;
  // Conjunction of the conjuncts that are not fused and of one mask column per predicate,
  // referenced as the columns following those of the filtered table. Unset if a single
  // predicate is the whole filter.
  std::optional<std::reference_wrapper<ast::expression const>> _combined;
  size_type _num_columns = 0;  // Number of columns of the filtered table
};

}  // namespace io::parquet::jit
}  // namespace CUDF_EXPORT cudf
