/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/type_traits>

#include <io/parquet/jit/flat_buffer_abi.hpp>

// The flat-buffer kernel: passes every valid row of the block's tile through the level source,
// the value source and the leaf it is linked with.
//
// The entry point has a fixed C name so that the host finds it in every linked kernel.
extern "C" __global__ void cudf_kernel_entry(
  cudf::io::parquet::jit::flat_buffer_ref const* __restrict__ buffers,
  cudf::io::parquet::jit::flat_buffer_tile const* __restrict__ tiles,
  uint32_t num_tiles,
  cudf::io::parquet::jit::leaf_ctx const* __restrict__ ctxs)
{
  if (blockIdx.x >= num_tiles) { return; }
  auto const tile   = tiles[blockIdx.x];
  auto const buffer = buffers[tile.buffer];
  auto const ctx    = ctxs[buffer.ctx_idx];
  auto const levels = cudf_pq_flat_level_v1_init(buffer);
  auto const values = cudf_pq_flat_value_v1_init(buffer);

  for (auto i = threadIdx.x; i < tile.num_rows; i += blockDim.x) {
    auto const row   = tile.first_row + i;
    auto const level = cudf_pq_flat_level_v1_get(levels, row);
    if (level.valid == 0) { continue; }
    cudf_pq_flat_value_v1_emit(
      ctx, buffer.row_base + row, cudf_pq_flat_value_v1_get(values, level.value_index));
  }
}

static_assert(
  cuda::std::is_same_v<decltype(cudf_kernel_entry), cudf::io::parquet::jit::flat_buffer_entry_t>);
