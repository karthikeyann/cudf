/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

#include <io/parquet/jit/abi/leaf_abi.cuh>
#include <io/parquet/jit/kernel_abi.hpp>

namespace cudf::io::parquet::jit {

/// Signature of the entry point of a tile decode kernel: one warp per tile.
using decode_flat_entry_t = void(page_ref const* pages,
                                 tile_ref const* tiles,
                                 uint32_t num_tiles,
                                 checkpoint const* checkpoints,
                                 leaf_ctx const* contexts,
                                 int32_t* null_counts);

}  // namespace cudf::io::parquet::jit
