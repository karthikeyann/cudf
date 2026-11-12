/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

#include <io/parquet/jit/abi/leaf_abi.cuh>

// The flat-buffer kernel and the stages it is composed from.
//
// The kernel is one loop over the rows of a tile, compiled once. It leaves three stages
// unresolved, and each stage is supplied by a fragment of its own when the kernel is linked:
//
// - the level source says whether a row is valid and which value it reads,
// - the value source reads that value and passes it to the leaf of its width,
// - the leaf (`abi/leaf_abi.cuh`) converts, tests and stores the value.
//
// A stage keeps its per-buffer state in a small struct that its `init` function returns by value
// and its other functions take by value. After linking, the stages are inlined into the loop and
// the state lives in registers.

namespace cudf::io::parquet::jit {

/**
 * @brief A run of fixed-width values read by the flat-buffer kernel.
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

/**
 * @brief State of a level source for one buffer.
 */
struct flat_level_state {
  uint32_t const* validity;  ///< Validity bitmap of the buffer
  uint32_t validity_offset;  ///< Bit index of row zero in `validity`
};

/**
 * @brief What a level source says about one row.
 */
struct flat_level {
  uint32_t valid;        ///< One if the row holds a value, zero if it is null
  uint32_t value_index;  ///< Index of the row's value in the value source
};

/**
 * @brief State of a value source for one buffer.
 */
struct flat_value_state {
  uint8_t const* values;      ///< Values, or dictionary indices
  uint8_t const* dict;        ///< Dictionary values
  uint32_t num_dict_entries;  ///< Number of dictionary entries
  uint32_t dict_first;        ///< Added to a dictionary index before it reaches the leaf
};

/**
 * @brief A value read by a value source.
 */
struct flat_value {
  uint64_t bits;      ///< The value, zero-extended to 64 bits
  uint32_t dict_idx;  ///< Index of the value's dictionary entry for the leaf; zero for plain values
};

/// Signature of a level source's `init` function.
using flat_level_init_t = flat_level_state(flat_buffer_ref buffer);
/// Signature of a level source's `get` function.
using flat_level_get_t = flat_level(flat_level_state state, uint32_t row);
/// Signature of a value source's `init` function.
using flat_value_init_t = flat_value_state(flat_buffer_ref buffer);
/// Signature of a value source's `get` function.
using flat_value_get_t = flat_value(flat_value_state state, uint32_t value_index);
/// Signature of a value source's `emit` function, which passes a value to the leaf of its width
/// and returns the leaf's result.
using flat_value_emit_t = uint32_t(leaf_ctx ctx, uint32_t row, flat_value value);

}  // namespace cudf::io::parquet::jit

#ifdef __CUDACC__
extern "C" __device__ cudf::io::parquet::jit::flat_level_init_t cudf_pq_flat_level_v1_init;
extern "C" __device__ cudf::io::parquet::jit::flat_level_get_t cudf_pq_flat_level_v1_get;
extern "C" __device__ cudf::io::parquet::jit::flat_value_init_t cudf_pq_flat_value_v1_init;
extern "C" __device__ cudf::io::parquet::jit::flat_value_get_t cudf_pq_flat_value_v1_get;
extern "C" __device__ cudf::io::parquet::jit::flat_value_emit_t cudf_pq_flat_value_v1_emit;
#endif
