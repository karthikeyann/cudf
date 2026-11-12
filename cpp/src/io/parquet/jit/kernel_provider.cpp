/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_provider.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_fragments.hpp>
#include <runtime/context.hpp>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <exception>
#include <format>
#include <mutex>
#include <span>
#include <unordered_map>

namespace cudf::io::parquet::jit {

namespace {

std::string to_hex(std::span<uint8_t const> bytes)
{
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (auto const byte : bytes) {
    hex += std::format("{:02x}", byte);
  }
  return hex;
}

// Content hash of every fragment compiled ahead of time; it changes whenever any of them does.
std::string const& fragments_hash()
{
  static std::string const hash = to_hex(cudf_fragments::hash);
  return hash;
}

std::span<uint8_t const> fragment_bytes(std::size_t file_index)
{
  auto const range = cudf_fragments::file_ranges[file_index];
  return cudf_fragments::files.subspan(range[0], range[1]);
}

// Position of a shape in tables ordered by (is_dict, num_bytes, nullable).
std::size_t shape_index(flat_buffer_shape shape)
{
  CUDF_EXPECTS(shape.num_bytes == 4 or shape.num_bytes == 8,
               "A flat-buffer kernel reads 4- or 8-byte values",
               std::invalid_argument);
  return (shape.is_dict ? 4 : 0) + (shape.num_bytes == 8 ? 2 : 0) + (shape.nullable ? 1 : 0);
}

constexpr std::array<std::size_t, 8> shape_fragments{
  cudf_fragments::pq_flat_plain_b32,
  cudf_fragments::pq_flat_plain_b32_nullable,
  cudf_fragments::pq_flat_plain_b64,
  cudf_fragments::pq_flat_plain_b64_nullable,
  cudf_fragments::pq_flat_dict_b32,
  cudf_fragments::pq_flat_dict_b32_nullable,
  cudf_fragments::pq_flat_dict_b64,
  cudf_fragments::pq_flat_dict_b64_nullable,
};

constexpr std::array<char const*, 8> shape_names{
  "plain_b32",
  "plain_b32_nullable",
  "plain_b64",
  "plain_b64_nullable",
  "dict_b32",
  "dict_b32_nullable",
  "dict_b64",
  "dict_b64_nullable",
};

std::size_t builtin_leaf_fragment(builtin_leaf leaf)
{
  switch (leaf) {
    case builtin_leaf::STORE: return cudf_fragments::pq_leaf_store;
    case builtin_leaf::DICT_KEEP: return cudf_fragments::pq_leaf_dict_keep;
  }
  CUDF_FAIL("Unknown leaf");
}

char const* builtin_leaf_name(builtin_leaf leaf)
{
  switch (leaf) {
    case builtin_leaf::STORE: return "store";
    case builtin_leaf::DICT_KEEP: return "dict_keep";
  }
  CUDF_FAIL("Unknown leaf");
}

// Failure history of one kernel.
struct kernel_state {
  int attempt = 0;  // Attempts that have failed; also the number of the next attempt
  std::chrono::steady_clock::time_point failed_at{};  // Time of the last failed attempt
};

struct provider_state {
  std::mutex mutex;
  std::unordered_map<std::string, kernel_state> failed;  // Keyed by the first attempt's link name
  retry_policy policy;
  kernel_provider_stats stats;
};

provider_state& state()
{
  static provider_state instance;
  return instance;
}

// A failed compile or link leaves its cache entry unusable, so a later attempt needs names the
// cache has not seen.
kernel_identity identity_of_attempt(kernel_identity identity, int attempt)
{
  if (attempt > 0) {
    identity.leaf += std::format("/retry{}", attempt);
    identity.link += std::format("/retry{}", attempt);
  }
  return identity;
}

cudf::kernel link_kernel(flat_buffer_shape shape,
                         leaf const& leaf,
                         kernel_identity const& names,
                         int attempt)
{
  CUDF_FUNC_RANGE();

  auto const consumer = fragment_bytes(shape_fragments[shape_index(shape)]);

  rtcx::blob compiled_leaf;
  auto leaf_fragment = rtcx::memory_fragment{};
  if (auto const* builtin = std::get_if<builtin_leaf>(&leaf)) {
    leaf_fragment = {.data = fragment_bytes(builtin_leaf_fragment(*builtin)),
                     .type = rtcx::binary_type::FATBIN,
                     .name = names.leaf.c_str()};
  } else {
    auto const& source = std::get<leaf_source>(leaf);
    // The translation-unit name is part of the fragment's cache key.
    auto const unit_name = attempt == 0 ? std::string{"cudf_pq_leaf.cu"}
                                        : std::format("cudf_pq_leaf_retry{}.cu", attempt);
    compiled_leaf =
      cudf::get_user_fragment(unit_name.c_str(), source.code.c_str(), {}, {}, {}, true);
    leaf_fragment = {
      .data = compiled_leaf->view(), .type = rtcx::binary_type::LTO_IR, .name = names.leaf.c_str()};
  }

  std::array<rtcx::memory_fragment, 2> const fragments{
    rtcx::memory_fragment{
      .data = consumer, .type = rtcx::binary_type::FATBIN, .name = names.consumer.c_str()},
    leaf_fragment};

  return cudf::get_lto_linked_kernel(names.link, {}, fragments);
}

}  // namespace

kernel_identity get_kernel_identity(flat_buffer_shape shape, leaf const& leaf)
{
  auto const& ctx      = cudf::get_context();
  auto const shape_str = shape_names[shape_index(shape)];

  auto consumer = std::format("pq/flat/{}/abi{}/frag{}/lto{}",
                              shape_str,
                              leaf_abi_version,
                              fragments_hash(),
                              CUDF_LTO_ARCHITECTURE);

  auto leaf_name = std::string{};
  if (auto const* builtin = std::get_if<builtin_leaf>(&leaf)) {
    leaf_name = std::format("pqleaf/{}/frag{}", builtin_leaf_name(*builtin), fragments_hash());
  } else {
    auto const& code   = std::get<leaf_source>(leaf).code;
    auto const digest  = XXH3_128bits(code.data(), code.size());
    auto const version = ctx.nvrtc_version();
    leaf_name          = std::format(
      "pqleaf/src/{:016x}{:016x}/nvrtc{}", digest.high64, digest.low64, version.value_or(0));
  }

  // The linked kernel is fully determined by its two fragments and the linker. The kernel cache
  // adds the device architecture and the runtime and driver versions.
  auto link = std::format("cudf/pq/flat/nvjitlink{}", ctx.nvjitlink_version().value_or(0));

  return {std::move(consumer), std::move(leaf_name), std::move(link)};
}

std::optional<flat_buffer_kernel> get_flat_buffer_kernel(flat_buffer_shape shape, leaf const& leaf)
{
  auto const identity = get_kernel_identity(shape, leaf);
  auto const key      = std::format("{}|{}|{}", identity.link, identity.consumer, identity.leaf);
  auto& provider      = state();

  auto attempt = 0;
  {
    std::lock_guard const lock{provider.mutex};
    ++provider.stats.requests;
    if (auto const it = provider.failed.find(key); it != provider.failed.end()) {
      // The delay doubles with every failed attempt.
      auto const delay = provider.policy.delay * (1 << (it->second.attempt - 1));
      if (it->second.attempt >= provider.policy.max_attempts or
          std::chrono::steady_clock::now() < it->second.failed_at + delay) {
        ++provider.stats.rejected;
        return std::nullopt;
      }
      attempt = it->second.attempt;
    }
    ++provider.stats.attempts;
  }

  try {
    return flat_buffer_kernel{
      link_kernel(shape, leaf, identity_of_attempt(identity, attempt), attempt)};
  } catch (std::exception const& e) {
    std::lock_guard const lock{provider.mutex};
    auto& failed = provider.failed[key];
    // Callers that waited on another caller's failed attempt report the same attempt.
    if (failed.attempt == attempt) {
      ++failed.attempt;
      ++provider.stats.failures;
      failed.failed_at = std::chrono::steady_clock::now();
      CUDF_LOG_WARN(std::format(
        "Parquet reader: cannot build kernel {} with leaf {} (attempt {} of {}); using the "
        "generic path instead: {}",
        identity.consumer,
        identity.leaf,
        attempt + 1,
        provider.policy.max_attempts,
        e.what()));
    }
    return std::nullopt;
  }
}

void launch_flat_buffer_kernel(flat_buffer_kernel const& kernel,
                               host_span<flat_buffer_ref const> buffers,
                               host_span<flat_buffer_tile const> tiles,
                               host_span<leaf_ctx const> ctxs,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  if (tiles.empty()) { return; }
  // Threads of one thread block; a tile's rows are spread evenly over them.
  constexpr uint32_t block_num_threads = 256;

  auto const temporary_mr = mr.get_temporary_mr();
  auto const d_buffers    = cudf::detail::make_device_uvector_async(buffers, stream, temporary_mr);
  auto const d_tiles      = cudf::detail::make_device_uvector_async(tiles, stream, temporary_mr);
  auto const d_ctxs       = cudf::detail::make_device_uvector_async(ctxs, stream, temporary_mr);
  auto const num_tiles    = static_cast<uint32_t>(tiles.size());
  // The arrays are freed in stream order, after the kernel has read them.
  kernel.launch({num_tiles},
                {block_num_threads},
                stream,
                d_buffers.data(),
                d_tiles.data(),
                num_tiles,
                d_ctxs.data());
}

void set_kernel_retry_policy(retry_policy policy)
{
  auto& provider = state();
  std::lock_guard const lock{provider.mutex};
  provider.policy = policy;
}

kernel_provider_stats get_kernel_provider_stats()
{
  auto& provider = state();
  std::lock_guard const lock{provider.mutex};
  return provider.stats;
}

}  // namespace cudf::io::parquet::jit
