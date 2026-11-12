/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/type_traits>

#include <io/parquet/jit/flat_buffer_kernel.cuh>

// Defines CUDF_KERNEL_INSTANCE, the kernel instantiation of this fragment.
#include <cudf/detail/kernel_instance.cuh>

// The entry point has a fixed C name so that the host finds it without knowing the instantiation.
extern "C" __global__ void cudf_kernel_entry(cudf::io::parquet::jit::flat_buffer_ref const* buffers,
                                             cudf::io::parquet::jit::flat_buffer_tile const* tiles,
                                             uint32_t num_tiles,
                                             cudf::io::parquet::jit::leaf_ctx const* ctxs)
{
  CUDF_KERNEL_INSTANCE(buffers, tiles, num_tiles, ctxs);
}

static_assert(
  cuda::std::is_same_v<decltype(cudf_kernel_entry), cudf::io::parquet::jit::flat_buffer_entry_t>);
