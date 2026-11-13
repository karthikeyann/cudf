/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "decode_flat_types.hpp"
#include "error.hpp"
#include "io/utilities/hostdevice_vector.hpp"
#include "parquet_gpu.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <cstdint>
#include <memory>

namespace cudf::io::parquet::detail {

/**
 * @brief State the tile decode kernels keep for the claimed pages of one subpass.
 */
struct flat_decode_plan;

/**
 * @brief Returns the number of pages claimed for tile decode in this process so far.
 *
 * Lets a test tell that pages went through the tile decode kernels, which the output of a
 * read does not show.
 *
 * @return Number of pages claimed, summed over the subpasses of all readers
 */
[[nodiscard]] CUDF_EXPORT size_t num_tile_decode_claimed_pages();

/**
 * @brief Returns the number of column outputs that received selected rows only, in this process
 * so far.
 *
 * Counts every column of every decode call that was given a row selection. Lets a test tell
 * that columns were stored compacted, which the output of a read does not show.
 *
 * @return Number of compacted column outputs, summed over the decode calls of all readers
 */
[[nodiscard]] CUDF_EXPORT size_t num_tile_decode_compacted_columns();

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
 * @brief Computes, per word of a row selection mask, the number of set bits before it.
 *
 * @param mask One bit per row; set bits are selected rows
 * @param num_rows Number of rows in `mask`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the result
 * @return One entry per word of `mask` and a final entry with the number of selected rows
 */
[[nodiscard]] cuda::device_buffer<uint32_t> compute_selection_rank(
  cudf::device_span<bitmask_type const> mask,
  size_type num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Decodes the claimed pages of a subpass for one decode call.
 *
 * Writes the values and validity of the rows in `[min_row, min_row + num_rows)` and sets the
 * null count of every claimed page, including pages outside of that range. The kernels are
 * queued on `stream`; `plan` must stay untouched until they have finished.
 *
 * A column marked in `selected_columns` receives only the rows set in `selection`, stored next
 * to each other in row order; its output must be sized for those rows and its null counts
 * cover them only. Every page of such a column must be a claimed page.
 *
 * @param plan Plan of the subpass
 * @param pages Pages of the subpass
 * @param chunks Column chunks of the pass, with their output pointers set for this decode call
 * @param column_data Output data of each input column for this decode call
 * @param selected_columns Per input column, whether it stores selected rows only; empty if none
 * does
 * @param selection Rows of `[min_row, min_row + num_rows)` to store, bit 0 being `min_row`; read
 * only for the columns marked in `selected_columns`
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
                       cudf::host_span<uint8_t const> selected_columns,
                       row_selection selection,
                       size_t min_row,
                       size_t num_rows,
                       kernel_error::pointer error_code,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr);

}  // namespace cudf::io::parquet::detail
