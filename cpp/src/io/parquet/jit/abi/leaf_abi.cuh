/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda/std/cstdint>
namespace cudf::io::parquet::jit {
inline constexpr int leaf_abi_version = 1;
struct leaf_ctx {
  void* out;
  uint8_t* keep;
  uint8_t const* dict_keep;
  int64_t scale;
  uint64_t lit[4];
};
static_assert(sizeof(leaf_ctx) == 64);
using leaf_b32_t = uint32_t(leaf_ctx, uint32_t row, uint32_t value, uint32_t dict_idx);
using leaf_b64_t = uint32_t(leaf_ctx, uint32_t row, uint64_t value, uint32_t dict_idx);
}  // namespace cudf::io::parquet::jit
extern "C" __device__ cudf::io::parquet::jit::leaf_b32_t cudf_pq_leaf_v1_b32;
extern "C" __device__ cudf::io::parquet::jit::leaf_b64_t cudf_pq_leaf_v1_b64;
