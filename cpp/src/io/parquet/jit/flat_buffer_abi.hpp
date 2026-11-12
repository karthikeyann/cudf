/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

#include <io/parquet/jit/abi/leaf_abi.cuh>

namespace cudf::io::parquet::jit {

/**
 * @brief A run of fixed-width values read by the flat-buffer kernels.
 *
 * Row `r` of a plain buffer holds its value at `values + r * width`. Row `r` of a dictionary
 * buffer holds a 32-bit index at `values + r * 4` into the `num_dict_entries` values at `dict`.
 * Addresses need no alignment.
 */
struct flat_buffer_ref {
  uint8_t const* values;      ///< Values of a plain buffer, 32-bit indices of a dictionary buffer
  uint8_t const* dict;        ///< Dictionary values; null for a plain buffer
  uint32_t const* validity;   ///< One bit per row, set if the row is valid; null if all rows are
  uint32_t validity_offset;   ///< Bit index of row zero in `validity`
  uint32_t row_base;          ///< Output row of row zero
  uint32_t num_dict_entries;  ///< Number of dictionary entries
  uint32_t dict_first;        ///< Added to a dictionary index before it reaches the leaf
  uint32_t ctx_idx;           ///< Index of the buffer's leaf context
  uint32_t padding;           ///< Keeps the size a multiple of eight bytes
};

static_assert(sizeof(flat_buffer_ref) == 48);

/**
 * @brief A range of rows of one buffer handled by one thread block.
 */
struct flat_buffer_tile {
  uint32_t buffer;     ///< Index of the buffer
  uint32_t first_row;  ///< First row of the range
  uint32_t num_rows;   ///< Number of rows in the range
};

/// Signature of the flat-buffer kernel entry point: one thread block per tile.
using flat_buffer_entry_t = void(flat_buffer_ref const* buffers,
                                 flat_buffer_tile const* tiles,
                                 uint32_t num_tiles,
                                 leaf_ctx const* ctxs);

}  // namespace cudf::io::parquet::jit
