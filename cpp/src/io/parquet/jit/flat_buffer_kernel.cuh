/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <io/parquet/jit/flat_buffer_abi.hpp>

namespace cudf::io::parquet::jit {

/// Reads a possibly unaligned 4- or 8-byte value.
template <int num_bytes>
__device__ inline auto load_unaligned(uint8_t const* address)
{
  if constexpr (num_bytes == 4) {
    uint32_t value;
    __builtin_memcpy(&value, address, sizeof(value));
    return value;
  } else {
    uint64_t value;
    __builtin_memcpy(&value, address, sizeof(value));
    return value;
  }
}

/// Passes one value to the leaf of its width.
template <int num_bytes, typename Value>
__device__ inline uint32_t call_leaf(leaf_ctx ctx, uint32_t row, Value value, uint32_t dict_idx)
{
  if constexpr (num_bytes == 4) {
    return cudf_pq_leaf_v1_b32(ctx, row, value, dict_idx);
  } else {
    return cudf_pq_leaf_v1_b64(ctx, row, value, dict_idx);
  }
}

/**
 * @brief Passes every valid row of the block's tile to the leaf.
 *
 * An out-of-range dictionary index reads as value zero and reaches the leaf as index
 * `num_dict_entries`, so tables indexed by `dict_idx` need one entry more than the dictionary.
 *
 * @tparam is_dict Whether buffers hold dictionary indices instead of values
 * @tparam num_bytes Width of a value, 4 or 8
 * @tparam nullable Whether buffers may have a validity bitmap
 */
template <bool is_dict, int num_bytes, bool nullable>
__device__ void flat_buffer_kernel(flat_buffer_ref const* __restrict__ buffers,
                                   flat_buffer_tile const* __restrict__ tiles,
                                   uint32_t num_tiles,
                                   leaf_ctx const* __restrict__ ctxs)
{
  if (blockIdx.x >= num_tiles) { return; }
  auto const tile   = tiles[blockIdx.x];
  auto const buffer = buffers[tile.buffer];
  auto const ctx    = ctxs[buffer.ctx_idx];

  for (auto i = threadIdx.x; i < tile.num_rows; i += blockDim.x) {
    auto const row = tile.first_row + i;
    if constexpr (nullable) {
      if (buffer.validity != nullptr) {
        auto const bit = buffer.validity_offset + row;
        if (((buffer.validity[bit >> 5] >> (bit & 31)) & 1) == 0) { continue; }
      }
    }
    if constexpr (is_dict) {
      auto const idx      = load_unaligned<4>(buffer.values + size_t{row} * 4);
      auto const in_range = idx < buffer.num_dict_entries;
      auto const value =
        in_range ? load_unaligned<num_bytes>(buffer.dict + size_t{idx} * num_bytes) : 0;
      call_leaf<num_bytes>(ctx,
                           buffer.row_base + row,
                           value,
                           buffer.dict_first + (in_range ? idx : buffer.num_dict_entries));
    } else {
      auto const value = load_unaligned<num_bytes>(buffer.values + size_t{row} * num_bytes);
      call_leaf<num_bytes>(ctx, buffer.row_base + row, value, 0);
    }
  }
}

}  // namespace cudf::io::parquet::jit
