/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "csv_gpu.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

namespace cudf::io::csv::gpu {

/**
 * @brief Creates strings columns from the (pointer, size) pairs of their rows.
 *
 * A row whose pointer is null is a null row; any other row is valid, and its string is the
 * `size` characters at the pointer, which are copied into the column. All columns have the same
 * number of rows.
 *
 * The columns are built together, with a fixed number of kernels and a single host
 * synchronization: one kernel computes the validity masks and the character counts of all
 * columns, and after the offsets and characters are allocated, one kernel writes the offsets and
 * copies the characters of all columns. A column whose characters reach
 * `cudf::strings::get_offset64_threshold()` has 64-bit offsets.
 *
 * @throw std::overflow_error if a column's characters exceed the size limit of a strings column
 * @throw std::overflow_error if a field is too long for a string (see `oversized_string_size`)
 *
 * @param columns The (pointer, size) pairs of the rows of each column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return The strings columns, in the order of `columns`
 */
std::vector<std::unique_ptr<column>> make_strings_columns(
  host_span<device_span<decoded_string const> const> columns,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::io::csv::gpu
