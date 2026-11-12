/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <io/parquet/jit/abi/leaf_abi.cuh>
#include <io/parquet/jit/leaf_compare.hpp>

// Comparison leaves compiled ahead of time: one or two comparisons of the value with a literal,
// joined by a logical AND. They cover the range and equality filters that are pushed into most
// scans, so that such a scan links its kernels without compiling a leaf.
//
// CUDF_PQ_LEAF_VALUE_TYPE is the type the value is compared as. CUDF_PQ_LEAF_STORE selects the
// leaf that also stores the value.
//
// The operators are not known when the leaf is compiled. They arrive in the context, in the slot
// `compare_operators_slot`, and are the same for every row of a launch.

namespace {

using cudf::io::parquet::jit::compare_operator;
using value_type = CUDF_PQ_LEAF_VALUE_TYPE;
using raw_type   = cuda::std::conditional_t<sizeof(value_type) == 4, uint32_t, uint64_t>;

__device__ inline bool compare(compare_operator op, value_type lhs, value_type rhs)
{
  switch (op) {
    case compare_operator::EQUAL: return lhs == rhs;
    case compare_operator::NOT_EQUAL: return lhs != rhs;
    case compare_operator::LESS: return lhs < rhs;
    case compare_operator::GREATER: return lhs > rhs;
    case compare_operator::LESS_EQUAL: return lhs <= rhs;
    case compare_operator::GREATER_EQUAL: return lhs >= rhs;
    default: return true;
  }
}

__device__ inline uint32_t compare_leaf(cudf::io::parquet::jit::leaf_ctx ctx,
                                        uint32_t row,
                                        raw_type raw)
{
#ifdef CUDF_PQ_LEAF_STORE
  static_cast<raw_type*>(ctx.out)[row] = raw;
#endif
  auto const value     = cuda::std::bit_cast<value_type>(raw);
  auto const operators = ctx.lit[cudf::io::parquet::jit::compare_operators_slot];
  auto const first     = static_cast<compare_operator>(operators & 0xff);
  auto const second    = static_cast<compare_operator>((operators >> 8) & 0xff);
  auto const keep =
    compare(first, value, cuda::std::bit_cast<value_type>(static_cast<raw_type>(ctx.lit[0]))) and
    compare(second, value, cuda::std::bit_cast<value_type>(static_cast<raw_type>(ctx.lit[1])));
  ctx.keep[row] = keep;
  return keep;
}

}  // namespace

// Only the leaf of the value's width is defined: the other width has no meaning for this type.
#if CUDF_PQ_LEAF_VALUE_BYTES == 4
extern "C" __device__ uint32_t cudf_pq_leaf_v1_b32(cudf::io::parquet::jit::leaf_ctx ctx,
                                                   uint32_t row,
                                                   uint32_t value,
                                                   uint32_t)
{
  return compare_leaf(ctx, row, value);
}
#else
extern "C" __device__ uint32_t cudf_pq_leaf_v1_b64(cudf::io::parquet::jit::leaf_ctx ctx,
                                                   uint32_t row,
                                                   uint64_t value,
                                                   uint32_t)
{
  return compare_leaf(ctx, row, value);
}
#endif

static_assert(sizeof(value_type) == CUDF_PQ_LEAF_VALUE_BYTES);
