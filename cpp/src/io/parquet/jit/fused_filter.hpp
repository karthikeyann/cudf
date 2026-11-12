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
#include <cudf/utilities/span.hpp>

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
 * The literal values are not part of the leaves: they travel in the leaf context, so every
 * predicate with the same structure and types shares its leaves. A predicate made of one or two
 * comparisons joined by `LOGICAL_AND` uses a comparison leaf compiled ahead of time; any other
 * predicate uses leaves generated and compiled at runtime.
 */
struct column_predicate {
  size_type column_index = 0;                          ///< Index of the column the predicate reads
  data_type type{};                                    ///< Type of the column the predicate reads
  int num_bytes = 4;                                   ///< Width of the column's values: 4 or 8
  leaf_stage test_leaf;                                ///< Leaf of action `TEST`
  leaf_stage store_leaf;                               ///< Leaf of action `STORE_AND_TEST`
  std::array<uint64_t, max_leaf_literals> literals{};  ///< Values for `leaf_ctx::lit`

  /**
   * @brief Returns the leaf source of an action.
   *
   * @param action What the leaf does with a value
   * @return Source of the leaf
   */
  [[nodiscard]] leaf_stage const& get_leaf(leaf_action action) const
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
 * @param predicate The predicate, referencing its column by index in @p column_types
 * @param column_types Types of the columns the predicate may reference
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations
 * @return The predicate's leaves and literals, or `std::nullopt` if it does not qualify
 */
[[nodiscard]] std::optional<column_predicate> make_column_predicate(
  ast::expression const& predicate,
  host_span<data_type const> column_types,
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
 *
 * A predicate's leaf can run in two places. While its column is decoded, a tile decode kernel
 * linked with the leaf stores each value and writes the row's keep flag. For rows that were not
 * decoded that way, a flat-buffer kernel runs the leaf over the decoded column.
 */
class fused_filter {
 public:
  /**
   * @brief Plans the evaluation of a filter over tables with columns of the given types.
   *
   * @param filter The filter, referencing columns by index in @p column_types
   * @param column_types Types of the columns of the tables to filter
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Memory resources used for temporary allocations
   */
  fused_filter(ast::expression const& filter,
               host_span<data_type const> column_types,
               cuda::stream_ref stream,
               cudf::memory_resources mr);

  /**
   * @brief The kernels that decode a column while evaluating one predicate on it.
   */
  struct decode_kernels {
    decode_flat_kernel const* plain = nullptr;  ///< Kernel for PLAIN pages, or null
    decode_flat_kernel const* dict  = nullptr;  ///< Kernel for dictionary pages, or null
  };

  /**
   * @brief Returns the tile decode kernels linked with a predicate's leaf.
   *
   * The kernels are linked when first requested. A kernel that cannot be built is null, and the
   * pages it would decode are decoded without the predicate.
   *
   * @param predicate_index Index of the predicate in `predicates()`
   * @param nullable Whether the predicate's column has definition levels
   * @return The kernels, owned by this object
   */
  [[nodiscard]] decode_kernels get_decode_kernels(std::size_t predicate_index, bool nullable) const;

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
   * @param decode_masks For each predicate, the BOOL8 keep flags its leaf wrote for every row of
   * @p table while the column was decoded, or null if the leaf did not see every row; may be
   * empty. The masks are consumed.
   * @param use_jit Whether the conjuncts not evaluated by leaves use the JIT expression path
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Memory resources used for temporary allocations and the returned column
   * @return A BOOL8 column that is valid and true for the rows the filter keeps, or
   * `std::nullopt` if a kernel is not available or @p table does not have the planned columns,
   * in which case the caller evaluates the original filter
   */
  [[nodiscard]] std::optional<std::unique_ptr<column>> evaluate(
    table_view const& table,
    host_span<std::unique_ptr<column>> decode_masks,
    bool use_jit,
    cuda::stream_ref stream,
    cudf::memory_resources mr) const;

 private:
  ast::tree _tree;                            // Owns the expressions built for the plan
  std::vector<column_predicate> _predicates;  // Fused conjuncts, one group per entry
  // Kernel of each predicate for a column without and with a validity bitmap, once linked.
  mutable std::vector<std::array<std::optional<flat_buffer_kernel>, 2>> _kernels;
  // Tile decode kernels of a predicate for one nullability, requested at most once.
  struct linked_decode_kernels {
    bool requested = false;                   // Whether the kernels were requested
    std::optional<decode_flat_kernel> plain;  // Kernel for PLAIN pages
    std::optional<decode_flat_kernel> dict;   // Kernel for dictionary pages
  };
  // Tile decode kernels of each predicate for a column without and with definition levels.
  mutable std::vector<std::array<linked_decode_kernels, 2>> _decode_kernels;
  // Conjunction of the conjuncts that are not fused and of one mask column per predicate,
  // referenced as the columns following those of the filtered table. Unset if a single
  // predicate is the whole filter.
  std::optional<std::reference_wrapper<ast::expression const>> _combined;
  size_type _num_columns = 0;  // Number of columns of the filtered table
};

/**
 * @brief Counts of how predicates were evaluated since the process started.
 */
struct fused_filter_stats {
  uint64_t decoded   = 0;  ///< Predicate evaluations taken from flags written during decode
  uint64_t evaluated = 0;  ///< Predicate evaluations run over a decoded column
};

/**
 * @brief Returns how the predicates of all fused filters were evaluated.
 *
 * @return Counts since the process started
 */
[[nodiscard]] fused_filter_stats get_fused_filter_stats();

}  // namespace io::parquet::jit
}  // namespace CUDF_EXPORT cudf
