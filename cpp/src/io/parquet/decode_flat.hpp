/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "error.hpp"
#include "io/utilities/hostdevice_vector.hpp"
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
 * Must be called after the definition levels of the subpass have been decoded.
 *
 * @param pages Pages of the subpass; claimed pages have the `FLAT_TILES` kernel mask
 * @param chunks Column chunks of the pass
 * @param level_type_size Size in bytes of a pre-decoded level
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used to allocate the plan's device memory
 * @return The plan
 */
[[nodiscard]] std::shared_ptr<flat_decode_plan> prepare_flat_decode(
  cudf::detail::hostdevice_span<PageInfo const> pages,
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  int level_type_size,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

/**
 * @brief Decodes the claimed pages of a subpass for one decode call.
 *
 * Writes the values and validity of the rows in `[min_row, min_row + num_rows)` and sets the
 * null count of every claimed page, including pages outside of that range. The kernels are
 * queued on `stream`; `plan` must stay untouched until they have finished.
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
