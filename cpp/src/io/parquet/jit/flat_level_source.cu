/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/type_traits>

#include <io/parquet/jit/flat_buffer_abi.hpp>

// Level sources of the flat-buffer kernel. CUDF_PQ_LEVEL_OPTIONAL selects the source that reads a
// validity bitmap; without it every row is valid. Every row has a value slot either way.

using cudf::io::parquet::jit::flat_buffer_ref;
using cudf::io::parquet::jit::flat_level;
using cudf::io::parquet::jit::flat_level_state;

extern "C" __device__ flat_level_state cudf_pq_flat_level_v1_init(flat_buffer_ref buffer)
{
  return {buffer.validity, buffer.validity_offset};
}

extern "C" __device__ flat_level cudf_pq_flat_level_v1_get([[maybe_unused]] flat_level_state state,
                                                           uint32_t row)
{
#ifdef CUDF_PQ_LEVEL_OPTIONAL
  auto const bit = state.validity_offset + row;
  return {(state.validity[bit >> 5] >> (bit & 31)) & 1, row};
#else
  return {1, row};
#endif
}

static_assert(cuda::std::is_same_v<decltype(cudf_pq_flat_level_v1_init),
                                   cudf::io::parquet::jit::flat_level_init_t>);
static_assert(cuda::std::is_same_v<decltype(cudf_pq_flat_level_v1_get),
                                   cudf::io::parquet::jit::flat_level_get_t>);
