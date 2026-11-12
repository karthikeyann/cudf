/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "error.hpp"
#include "io/utilities/hostdevice_vector.hpp"
#include "jit/kernel_provider.hpp"
#include "parquet_gpu.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>

namespace cudf::io::parquet::detail {

/**
 * @brief State the tile decode kernels keep for the claimed pages of one subpass.
 */
struct flat_decode_plan;

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
 * The definition levels of claimed pages are read from the page data; they are not
 * pre-decoded.
 *
 * @param pages Pages of the subpass; claimed pages have the `FLAT_TILES` kernel mask
 * @param chunks Column chunks of the pass
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used to allocate the plan's device memory
 * @return The plan
 */
[[nodiscard]] std::shared_ptr<flat_decode_plan> prepare_flat_decode(
  cudf::detail::hostdevice_span<PageInfo const> pages,
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

/**
 * @brief The leaf of one input column for one decode call.
 *
 * A column without kernels is decoded by the kernels compiled into libcudf, whose leaf stores
 * each value unchanged. A column with kernels is decoded by them, with `ctx` as their leaf's
 * context, for the pages whose encoding has a kernel.
 */
struct flat_column_leaf {
  jit::leaf_ctx ctx{};  ///< Context of the column's leaf; `out` is set by `decode_flat_pages`
  jit::decode_flat_kernel const* plain_kernel = nullptr;  ///< Kernel for PLAIN pages, or null
  jit::decode_flat_kernel const* dict_kernel  = nullptr;  ///< Kernel for dictionary pages, or null
  size_t num_rows_decoded = 0;  ///< Rows of the decode call that the two kernels decoded
};

/**
 * @brief Decodes the claimed pages of a subpass for one decode call.
 *
 * Writes the values and validity of the rows in `[min_row, min_row + num_rows)` and sets the
 * null count of every claimed page, including pages outside of that range. The kernels are
 * queued on `stream`; `plan` must stay untouched until they have finished.
 *
 * Sets `flat_column_leaf::num_rows_decoded` of every entry of `column_leaves`. A caller that
 * relies on what a column's leaf wrote for every row must check that this is `num_rows`: rows
 * in pages that are not claimed, or whose encoding has no kernel, never reach the leaf.
 *
 * @param plan Plan of the subpass
 * @param pages Pages of the subpass
 * @param chunks Column chunks of the pass, with their output pointers set for this decode call
 * @param column_data Output data of each input column for this decode call
 * @param column_leaves Leaf of each input column, or empty to store the values of every column
 * @param min_row First row to decode
 * @param num_rows Number of rows to decode
 * @param error_code Error code to set if an error is encountered
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used to allocate temporary device memory
 */
void decode_flat_pages(flat_decode_plan& plan,
                       cudf::detail::hostdevice_span<PageInfo> pages,
                       cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                       cudf::host_span<void* const> column_data,
                       cudf::host_span<flat_column_leaf> column_leaves,
                       size_t min_row,
                       size_t num_rows,
                       kernel_error::pointer error_code,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr);

}  // namespace cudf::io::parquet::detail
