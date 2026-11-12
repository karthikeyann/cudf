/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

#include <cstdint>

namespace cudf::io::parquet::jit {

/// Number of page rows covered by one tile. A tile is decoded by one warp.
inline constexpr uint32_t tile_rows = 1024;

/// log2 of `tile_rows`.
inline constexpr uint32_t tile_rows_log2 = 10;

/// Thread block size of every tile decode kernel. Warps of a block do not cooperate.
inline constexpr int decode_flat_block_size = 128;

/// Set in `page_ref::flags` when a page of a nullable column holds no nulls. Such a page has no
/// level buffer and its rows are not written to the validity mask.
inline constexpr uint8_t page_flag_no_nulls = 1;

/// Set in `page_ref::flags` when the pre-decoded definition levels are two bytes wide.
inline constexpr uint8_t page_flag_wide_levels = 2;

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
  uint8_t const* levels;      ///< One pre-decoded definition level per page row; nullable only
  uint32_t* valid;            ///< Validity mask of the output column; nullable only
  uint32_t values_size;       ///< Size of `values` in bytes
  uint32_t dict_entries;      ///< Number of entries in `dict`
  uint32_t dict_first;        ///< Added to the dictionary index passed to the leaf
  int32_t row_base;           ///< Output row of page row 0; negative if the page starts earlier
  uint32_t first_row;         ///< First page row of the decode call
  uint32_t num_rows;          ///< Number of page rows of the decode call; 0 skips the page
  uint32_t first_checkpoint;  ///< Index of the page's first entry in the checkpoint array
  uint16_t ctx_idx;           ///< Index of the page's column in the leaf context array
  uint8_t bit_width;          ///< Dictionary index bit width, at most 32
  uint8_t flags;              ///< Combination of `page_flag_*`
};
static_assert(sizeof(page_ref) == 64);

/**
 * @brief Names one tile of one page: page rows `[tile * tile_rows, (tile + 1) * tile_rows)`.
 */
struct tile_ref {
  uint32_t page;  ///< Index into the `page_ref` array
  uint32_t tile;  ///< Tile number within the page
};

/**
 * @brief Entry point into a page's serial streams, recorded once per `tile_rows` values.
 *
 * A page has `checkpoint_stride` entries per tile. A dictionary entry `k` locates the run that
 * holds value `k * tile_rows` of the index stream. A rank entry `k` uses only `first_value`,
 * which holds the number of non-null rows before page row `k * tile_rows`.
 */
struct checkpoint {
  uint32_t byte_offset;  ///< Offset of a run header from `page_ref::values`
  uint32_t first_value;  ///< Index of the first value of that run, or a rank
};

/**
 * @brief Returns the number of checkpoint entries a page has per tile.
 *
 * Tile `k` of a page has its dictionary entry at `first_checkpoint + k * stride` and its rank
 * entry directly after the dictionary entry, or in its place for a PLAIN page.
 *
 * @param is_dict Whether the page is dictionary encoded
 * @param nullable Whether the page's column is nullable
 * @return Number of checkpoint entries per tile
 */
#ifdef __CUDACC__
__host__ __device__
#endif
  [[nodiscard]] constexpr uint32_t
  checkpoint_stride(bool is_dict, bool nullable)
{
  return static_cast<uint32_t>(is_dict) + static_cast<uint32_t>(nullable);
}

}  // namespace cudf::io::parquet::jit
