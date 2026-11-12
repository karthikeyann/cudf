/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <io/parquet/jit/flat_buffer_abi.hpp>
#include <jit/cache.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace CUDF_EXPORT cudf {
namespace io::parquet::jit {

/**
 * @brief A kernel that can only be launched with the argument types of its entry point.
 *
 * A linked kernel takes its arguments as untyped pointers. Naming the entry point's signature
 * here makes the compiler convert every argument to the parameter type the kernel reads.
 *
 * @tparam Signature Function type of the kernel's entry point
 */
template <typename Signature>
class typed_kernel;

/// @copydoc typed_kernel
template <typename... Params>
class typed_kernel<void(Params...)> {
  static_assert((std::is_trivially_copyable_v<Params> and ...));

 public:
  /**
   * @brief Wraps a kernel whose entry point has the signature `void(Params...)`.
   *
   * @param kernel The linked kernel
   */
  explicit typed_kernel(cudf::kernel kernel) : _kernel{std::move(kernel)} {}

  /**
   * @brief Launches the kernel.
   *
   * @param grid Number of thread blocks
   * @param block Number of threads in a block
   * @param stream CUDA stream to launch the kernel on
   * @param args Kernel arguments
   */
  void launch(rtcx::cuda_dim3 grid,
              rtcx::cuda_dim3 block,
              cuda::stream_ref stream,
              std::type_identity_t<Params>... args) const
  {
    _kernel.launch_with(grid, block, 0, stream, args...);
  }

 private:
  cudf::kernel _kernel;
};

/// A linked flat-buffer kernel.
using flat_buffer_kernel = typed_kernel<flat_buffer_entry_t>;

/**
 * @brief Identifies one flat-buffer kernel compiled ahead of time.
 */
struct flat_buffer_shape {
  bool is_dict  = false;  ///< Whether buffers hold dictionary indices instead of values
  int num_bytes = 4;      ///< Width of a value: 4 or 8
  bool nullable = false;  ///< Whether buffers may have a validity bitmap
};

/**
 * @brief Leaves compiled ahead of time.
 */
enum class builtin_leaf : uint8_t {
  STORE,      ///< Stores the value and keeps every row
  DICT_KEEP,  ///< Stores the value and copies the keep flag of its dictionary entry
};

/**
 * @brief CUDA source of a leaf compiled at runtime.
 */
struct leaf_source {
  std::string code;  ///< Source defining the leaf symbol of the kernel's value width
};

/// The leaf to link into a kernel.
using leaf = std::variant<builtin_leaf, leaf_source>;

/**
 * @brief Names under which a kernel and its inputs are cached.
 *
 * Each name identifies the content it stands for, so a kernel is never reused after the
 * ahead-of-time fragments, the leaf source or the compiler change.
 */
struct kernel_identity {
  std::string consumer;  ///< Name of the kernel fragment
  std::string leaf;      ///< Name of the leaf fragment
  std::string link;      ///< Name of the linked kernel
};

/**
 * @brief Returns the cache names of the kernel linked from a shape and a leaf.
 *
 * @param shape The kernel compiled ahead of time
 * @param leaf The leaf to link into it
 * @return Names of the kernel and of its two fragments
 */
[[nodiscard]] kernel_identity get_kernel_identity(flat_buffer_shape shape, leaf const& leaf);

/**
 * @brief Returns the flat-buffer kernel of a shape linked with a leaf.
 *
 * The first request for a kernel compiles its leaf if needed and links it, blocking the caller
 * and every concurrent caller that requests the same kernel. Later requests return the cached
 * kernel.
 *
 * Returns `std::nullopt` if the kernel cannot be compiled, linked or loaded. The failure is
 * logged once, and requests for the same kernel keep returning `std::nullopt` without another
 * attempt until the retry delay has passed and while attempts remain.
 *
 * @param shape The kernel compiled ahead of time
 * @param leaf The leaf to link into it
 * @return The linked kernel, or `std::nullopt` if it is not available
 */
[[nodiscard]] std::optional<flat_buffer_kernel> get_flat_buffer_kernel(flat_buffer_shape shape,
                                                                       leaf const& leaf);

/**
 * @brief Launches a flat-buffer kernel with one thread block per tile.
 *
 * @param kernel The kernel
 * @param buffers Buffers the tiles refer to
 * @param tiles Row ranges to process
 * @param ctxs Leaf contexts the buffers refer to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations
 */
void launch_flat_buffer_kernel(flat_buffer_kernel const& kernel,
                               host_span<flat_buffer_ref const> buffers,
                               host_span<flat_buffer_tile const> tiles,
                               host_span<leaf_ctx const> ctxs,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr);

/**
 * @brief Controls when a kernel that failed to build is attempted again.
 */
struct retry_policy {
  std::chrono::milliseconds delay{30'000};  ///< Delay after a failure; doubles with each failure
  int max_attempts = 3;                     ///< Attempts per kernel in the life of the process
};

/**
 * @brief Sets the retry policy of kernels that failed to build.
 *
 * @param policy The new policy
 */
void set_kernel_retry_policy(retry_policy policy);

/**
 * @brief Counts of kernel provider events since the process started.
 */
struct kernel_provider_stats {
  uint64_t requests = 0;  ///< Kernels requested
  uint64_t attempts = 0;  ///< Requests that reached the compiler or the kernel cache
  uint64_t failures = 0;  ///< Distinct failed attempts
  uint64_t rejected = 0;  ///< Requests answered `std::nullopt` without an attempt
};

/**
 * @brief Returns the kernel provider's event counts.
 *
 * @return Counts since the process started
 */
[[nodiscard]] kernel_provider_stats get_kernel_provider_stats();

}  // namespace io::parquet::jit
}  // namespace CUDF_EXPORT cudf
