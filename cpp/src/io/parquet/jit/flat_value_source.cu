/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include <io/parquet/jit/flat_buffer_abi.hpp>

// Value sources of the flat-buffer kernel. CUDF_PQ_VALUE_BYTES is the width of a value, 4 or 8.
// CUDF_PQ_VALUE_DICT selects the source that reads 32-bit dictionary indices; without it the
// buffer holds the values themselves.
//
// An out-of-range dictionary index reads as value zero and reaches the leaf as index
// `num_dict_entries`, so tables indexed by `dict_idx` need one entry more than the dictionary.

using cudf::io::parquet::jit::flat_buffer_ref;
using cudf::io::parquet::jit::flat_value;
using cudf::io::parquet::jit::flat_value_state;
using cudf::io::parquet::jit::leaf_ctx;

namespace {

#if CUDF_PQ_VALUE_BYTES == 4
using value_type = uint32_t;
#elif CUDF_PQ_VALUE_BYTES == 8
using value_type = uint64_t;
#else
#error "CUDF_PQ_VALUE_BYTES must be 4 or 8"
#endif

// Reads a possibly unaligned value.
template <typename T>
__device__ inline T load_unaligned(uint8_t const* address)
{
  T value;
  __builtin_memcpy(&value, address, sizeof(T));
  return value;
}

}  // namespace

extern "C" __device__ flat_value_state cudf_pq_flat_value_v1_init(flat_buffer_ref buffer)
{
  return {buffer.values, buffer.dict, buffer.num_dict_entries, buffer.dict_first};
}

extern "C" __device__ flat_value cudf_pq_flat_value_v1_get(flat_value_state state,
                                                           uint32_t value_index)
{
#ifdef CUDF_PQ_VALUE_DICT
  auto const idx      = load_unaligned<uint32_t>(state.values + size_t{value_index} * 4);
  auto const in_range = idx < state.num_dict_entries;
  auto const value =
    in_range ? load_unaligned<value_type>(state.dict + size_t{idx} * sizeof(value_type)) : 0;
  return {value, state.dict_first + (in_range ? idx : state.num_dict_entries)};
#else
  return {load_unaligned<value_type>(state.values + size_t{value_index} * sizeof(value_type)), 0};
#endif
}

extern "C" __device__ uint32_t cudf_pq_flat_value_v1_emit(leaf_ctx ctx,
                                                          uint32_t row,
                                                          flat_value value)
{
#if CUDF_PQ_VALUE_BYTES == 4
  return cudf_pq_leaf_v1_b32(ctx, row, static_cast<uint32_t>(value.bits), value.dict_idx);
#else
  return cudf_pq_leaf_v1_b64(ctx, row, value.bits, value.dict_idx);
#endif
}

static_assert(cuda::std::is_same_v<decltype(cudf_pq_flat_value_v1_init),
                                   cudf::io::parquet::jit::flat_value_init_t>);
static_assert(cuda::std::is_same_v<decltype(cudf_pq_flat_value_v1_get),
                                   cudf::io::parquet::jit::flat_value_get_t>);
static_assert(cuda::std::is_same_v<decltype(cudf_pq_flat_value_v1_emit),
                                   cudf::io::parquet::jit::flat_value_emit_t>);
