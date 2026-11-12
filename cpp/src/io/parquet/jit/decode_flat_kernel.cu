/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/type_traits>

#include <io/parquet/jit/decode_flat.cuh>
#include <io/parquet/jit/decode_flat_abi.hpp>

// Defines CUDF_KERNEL_INSTANCE, the stages of this fragment's tile decode kernel as the template
// arguments `is_dict, in_bytes, nullable` of `decode_flat_tiles`.
#include <cudf/detail/kernel_instance.cuh>

// A tile decode kernel whose leaf is linked at runtime. The level source and the value source are
// compiled in: they are warp-cooperative stages that keep per-tile state, written as class
// templates, so one fragment is built for each of their combinations.

namespace {

// Passes a value to the unresolved leaf of its width.
struct linked_leaf {
  template <typename Value>
  __device__ uint32_t operator()(cudf::io::parquet::jit::leaf_ctx const& ctx,
                                 uint32_t row,
                                 Value value,
                                 uint32_t dict_idx) const
  {
    if constexpr (sizeof(Value) == sizeof(uint32_t)) {
      return cudf_pq_leaf_v1_b32(ctx, row, value, dict_idx);
    } else {
      return cudf_pq_leaf_v1_b64(ctx, row, value, dict_idx);
    }
  }
};

}  // namespace

// The entry point has a fixed C name so that the host finds it in every linked kernel.
extern "C" __global__ void __launch_bounds__(cudf::io::parquet::jit::decode_flat_block_size)
  cudf_kernel_entry(cudf::io::parquet::jit::page_ref const* pages,
                    cudf::io::parquet::jit::tile_ref const* tiles,
                    uint32_t num_tiles,
                    cudf::io::parquet::jit::checkpoint const* checkpoints,
                    cudf::io::parquet::jit::leaf_ctx const* contexts,
                    int32_t* null_counts)
{
  cudf::io::parquet::jit::decode_flat_tiles<CUDF_KERNEL_INSTANCE>(
    pages, tiles, num_tiles, checkpoints, contexts, null_counts, linked_leaf{});
}

static_assert(
  cuda::std::is_same_v<decltype(cudf_kernel_entry), cudf::io::parquet::jit::decode_flat_entry_t>);
