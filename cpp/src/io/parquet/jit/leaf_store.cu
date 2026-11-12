/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/type_traits>

#include <io/parquet/jit/abi/leaf_abi.cuh>

// Leaves compiled ahead of time. CUDF_PQ_LEAF_DICT_KEEP selects the leaf that also answers the
// predicate from the table of per-dictionary-entry results; without it the leaf only stores.

namespace {

template <typename Value>
__device__ inline uint32_t store(cudf::io::parquet::jit::leaf_ctx ctx,
                                 uint32_t row,
                                 Value value,
                                 [[maybe_unused]] uint32_t dict_idx)
{
  static_cast<Value*>(ctx.out)[row] = value;
#ifdef CUDF_PQ_LEAF_DICT_KEEP
  auto const keep = ctx.dict_keep[dict_idx];
  ctx.keep[row]   = keep;
  return keep;
#else
  return 1;
#endif
}

}  // namespace

extern "C" __device__ uint32_t cudf_pq_leaf_v1_b32(cudf::io::parquet::jit::leaf_ctx ctx,
                                                   uint32_t row,
                                                   uint32_t value,
                                                   uint32_t dict_idx)
{
  return store(ctx, row, value, dict_idx);
}

extern "C" __device__ uint32_t cudf_pq_leaf_v1_b64(cudf::io::parquet::jit::leaf_ctx ctx,
                                                   uint32_t row,
                                                   uint64_t value,
                                                   uint32_t dict_idx)
{
  return store(ctx, row, value, dict_idx);
}

static_assert(
  cuda::std::is_same_v<decltype(cudf_pq_leaf_v1_b32), cudf::io::parquet::jit::leaf_b32_t>);
static_assert(
  cuda::std::is_same_v<decltype(cudf_pq_leaf_v1_b64), cudf::io::parquet::jit::leaf_b64_t>);
