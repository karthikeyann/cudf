/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/cstdint>

#include <cstdint>

/**
 * @file decode_flat_types.hpp
 * @brief Types shared by the host and device sides of tile decode of flat fixed-width pages.
 */

namespace cudf::io::parquet::detail {

/// Number of page rows covered by one tile. A tile is decoded by one warp.
inline constexpr uint32_t tile_rows = 1024;

/// log2 of `tile_rows`.
inline constexpr uint32_t tile_rows_log2 = 10;

/// Thread block size of every tile decode kernel. Warps of a block do not cooperate.
inline constexpr int decode_flat_block_size = 128;

/// Set in `page_ref::flags` when a page of a nullable column holds no nulls. Its levels are not
/// read and its rows are not written to the validity mask.
inline constexpr uint8_t page_flag_no_nulls = 1;

/**
 * @brief Describes one flat fixed-width data page for one decode call.
 *
 * Page row `r` is stored at output row `r + row_base`. Only the page rows in
 * `[first_row, first_row + num_rows)` belong to the decode call.
 */
struct page_ref {
  uint8_t const* values;      ///< PLAIN: the values. Dictionary: the index stream after its bit
                              ///< width byte
  uint8_t const* dict;        ///< Dictionary values; unused for PLAIN pages
  uint8_t const* levels;      ///< Definition level stream; nullable only
  uint32_t* valid;            ///< Validity mask of the output column; nullable only
  uint32_t values_size;       ///< Size of `values` in bytes
  uint32_t dict_entries;      ///< Number of entries in `dict`
  uint32_t levels_size;       ///< Size of `levels` in bytes
  int32_t row_base;           ///< Output row of page row 0; negative if the page starts earlier
  uint32_t first_row;         ///< First page row of the decode call
  uint32_t num_rows;          ///< Number of page rows of the decode call; 0 skips the page
  uint32_t first_checkpoint;  ///< Index of the page's first entry in the checkpoint array
  uint16_t output_idx;        ///< Index of the page's column in the array of output pointers
  uint8_t bit_width;          ///< Dictionary index bit width, at most 32
  uint8_t flags;              ///< Combination of `page_flag_*`
};
static_assert(sizeof(page_ref) == 64);

/**
 * @brief Rows of a decode call that are stored, for kernels that store selected rows only.
 *
 * Selected rows are stored next to each other in row order: row `r` is stored at the number
 * of selected rows before it.
 */
struct row_selection {
  uint32_t const* mask;  ///< One bit per row of the decode call; set if the row is stored
  uint32_t const* rank;  ///< Per word of `mask`, the number of set bits in the words before it,
                         ///< with one more entry for the total
};

/**
 * @brief Names one tile of one page: page rows `[tile * tile_rows, (tile + 1) * tile_rows)`.
 */
struct tile_ref {
  uint32_t page;  ///< Index into the `page_ref` array
  uint32_t tile;  ///< Tile number within the page
};

/**
 * @brief Resume state of one sequential stream of a page, recorded every `tile_rows` values.
 *
 * A stream that can only be read front to back (the run headers of a dictionary index stream,
 * the non-null count of a level stream) is walked once per subpass. The walk stores what a tile
 * needs to start in the middle of the stream. Each stage of a kernel declares how many entries
 * per tile it uses and gives the two fields the meaning its stream needs:
 * - dictionary indices: entry `k` locates the run that holds value `k * tile_rows`;
 * - definition levels: two entries; the first locates the run that holds the level of page
 *   row `k * tile_rows`, the second holds, in `first_value`, the number of non-null rows before
 *   that row.
 *
 * Tile `k` of a page owns entries `[first_checkpoint + k * stride, + stride)`, the value
 * stream's entry first and the level stream's after it.
 */
struct checkpoint {
  uint32_t byte_offset;  ///< Offset from `page_ref::values` at which to resume
  uint32_t first_value;  ///< Stream position at `byte_offset`, or a count of earlier values
};

/**
 * @brief Returns the number of checkpoint entries a flat fixed-width page has per tile.
 *
 * @param is_dict Whether the page is dictionary encoded
 * @param nullable Whether the page's column is nullable
 * @return Number of checkpoint entries per tile
 */
[[nodiscard]] CUDF_HOST_DEVICE constexpr uint32_t checkpoint_stride(bool is_dict, bool nullable)
{
  return static_cast<uint32_t>(is_dict) + 2 * static_cast<uint32_t>(nullable);
}

}  // namespace cudf::io::parquet::detail
