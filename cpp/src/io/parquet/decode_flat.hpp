/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "error.hpp"
#include "io/utilities/hostdevice_vector.hpp"
#include "jit/kernel_abi.hpp"
#include "parquet_gpu.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <cstdint>
#include <memory>

namespace cudf::io::parquet::detail {

/**
 * @brief Identifies one page claimed by the tile decode kernels.
 */
struct flat_page_claim {
  uint32_t page_idx;          ///< Index of the page in the subpass page array
  uint32_t first_checkpoint;  ///< Index of the page's first checkpoint entry
  uint32_t shape;             ///< Combination of `flat_shape_*`
};

/// Set in `flat_page_claim::shape` for a dictionary encoded page.
inline constexpr uint32_t flat_shape_dict = 1;
/// Set in `flat_page_claim::shape` for a page of 8-byte values.
inline constexpr uint32_t flat_shape_wide = 2;
/// Set in `flat_page_claim::shape` for a page of a nullable column.
inline constexpr uint32_t flat_shape_nullable = 4;
/// Number of distinct `flat_page_claim::shape` values.
inline constexpr uint32_t num_flat_shapes = 8;

/**
 * @brief State the tile decode kernels keep for the claimed pages of one subpass.
 */
struct flat_decode_plan {
  /**
   * @brief Constructs an empty plan.
   *
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the plan's device memory
   */
  flat_decode_plan(cuda::stream_ref stream, cudf::memory_resources mr);

  /// Claimed pages, in subpass page order.
  cudf::detail::hostdevice_vector<flat_page_claim> claims;
  /// Checkpoints of every claimed page, filled once per subpass.
  cuda::device_buffer<jit::checkpoint> checkpoints;
  /// One descriptor per claimed page, rewritten for every decode call.
  cuda::device_buffer<jit::page_ref> page_refs;
  /// One null counter per claimed page, rewritten for every decode call.
  cuda::device_buffer<int32_t> null_counts;
};

/**
 * @brief Returns whether the tile decode kernels can decode a page.
 *
 * A page qualifies if it is a PLAIN or dictionary encoded data page of a flat INT32, INT64,
 * FLOAT or DOUBLE column with a maximum definition level of at most 1, and its values are
 * stored in the output column without conversion.
 *
 * @param page The page
 * @param chunk Column chunk the page belongs to
 * @param output_type Type of the output column
 * @return Whether the page can be claimed
 */
[[nodiscard]] bool is_flat_decode_eligible(PageInfo const& page,
                                           ColumnChunkDesc const& chunk,
                                           data_type output_type);

/**
 * @brief Builds the plan for the claimed pages of a subpass and records their checkpoints.
 *
 * Must be called after the definition levels of the subpass have been decoded.
 *
 * @param pages Pages of the subpass; claimed pages have the `FLAT_TILES` kernel mask
 * @param chunks Column chunks of the pass
 * @param level_type_size Size in bytes of a pre-decoded level
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used to allocate the plan's device memory
 * @return The plan
 */
[[nodiscard]] std::unique_ptr<flat_decode_plan> prepare_flat_decode(
  cudf::detail::hostdevice_span<PageInfo const> pages,
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  int level_type_size,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

/**
 * @brief Decodes the claimed pages of a subpass for one decode call.
 *
 * Writes the values and validity of the rows in `[min_row, min_row + num_rows)` and sets the
 * null count of every claimed page, including pages outside of that range.
 *
 * @param plan Plan of the subpass
 * @param pages Pages of the subpass
 * @param chunks Column chunks of the pass, with their output pointers set for this decode call
 * @param column_data Output data of each input column for this decode call
 * @param min_row First row to decode
 * @param num_rows Number of rows to decode
 * @param level_type_size Size in bytes of a pre-decoded level
 * @param error_code Error code to set if an error is encountered
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used to allocate temporary device memory
 */
void decode_flat_pages(flat_decode_plan& plan,
                       cudf::detail::hostdevice_span<PageInfo> pages,
                       cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                       cudf::host_span<void* const> column_data,
                       size_t min_row,
                       size_t num_rows,
                       int level_type_size,
                       kernel_error::pointer error_code,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr);

}  // namespace cudf::io::parquet::detail
