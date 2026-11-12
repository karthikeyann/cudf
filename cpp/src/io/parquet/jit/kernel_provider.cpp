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
#include <string_view>
#include <unordered_map>
#include <vector>

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

// An ahead-of-time fragment and the name its content is cached under.
struct fragment_ref {
  std::size_t file_index;  // Index of the fragment among the embedded fragments
  char const* name;        // Name of the fragment, unique among the fragments of its kind
};

fragment_ref level_fragment(level_source levels)
{
  switch (levels) {
    case level_source::REQUIRED: return {cudf_fragments::pq_flat_level_required, "required"};
    case level_source::OPTIONAL: return {cudf_fragments::pq_flat_level_optional, "optional"};
  }
  CUDF_FAIL("Unknown level source");
}

fragment_ref value_fragment(value_source values)
{
  CUDF_EXPECTS(values.num_bytes == 4 or values.num_bytes == 8,
               "A flat-buffer kernel reads 4- or 8-byte values",
               std::invalid_argument);
  auto const wide = values.num_bytes == 8;
  switch (values.encoding) {
    case value_encoding::PLAIN:
      return wide ? fragment_ref{cudf_fragments::pq_flat_value_plain_b64, "plain_b64"}
                  : fragment_ref{cudf_fragments::pq_flat_value_plain_b32, "plain_b32"};
    case value_encoding::DICTIONARY:
      return wide ? fragment_ref{cudf_fragments::pq_flat_value_dict_b64, "dict_b64"}
                  : fragment_ref{cudf_fragments::pq_flat_value_dict_b32, "dict_b32"};
  }
  CUDF_FAIL("Unknown value encoding");
}

fragment_ref builtin_leaf_fragment(builtin_leaf leaf)
{
  switch (leaf) {
    case builtin_leaf::STORE: return {cudf_fragments::pq_leaf_store, "store"};
    case builtin_leaf::DICT_KEEP: return {cudf_fragments::pq_leaf_dict_keep, "dict_keep"};
  }
  CUDF_FAIL("Unknown leaf");
}

fragment_ref compare_leaf_fragment(compare_leaf leaf)
{
  static constexpr std::array<fragment_ref, 12> fragments{{
    {cudf_fragments::pq_leaf_compare_int32, "compare_int32"},
    {cudf_fragments::pq_leaf_compare_int32_store, "compare_int32_store"},
    {cudf_fragments::pq_leaf_compare_int64, "compare_int64"},
    {cudf_fragments::pq_leaf_compare_int64_store, "compare_int64_store"},
    {cudf_fragments::pq_leaf_compare_uint32, "compare_uint32"},
    {cudf_fragments::pq_leaf_compare_uint32_store, "compare_uint32_store"},
    {cudf_fragments::pq_leaf_compare_uint64, "compare_uint64"},
    {cudf_fragments::pq_leaf_compare_uint64_store, "compare_uint64_store"},
    {cudf_fragments::pq_leaf_compare_float32, "compare_float32"},
    {cudf_fragments::pq_leaf_compare_float32_store, "compare_float32_store"},
    {cudf_fragments::pq_leaf_compare_float64, "compare_float64"},
    {cudf_fragments::pq_leaf_compare_float64_store, "compare_float64_store"},
  }};
  auto const index = static_cast<std::size_t>(leaf.value) * 2 + (leaf.store ? 1 : 0);
  CUDF_EXPECTS(index < fragments.size(), "Unknown comparison leaf", std::invalid_argument);
  return fragments[index];
}

// The ahead-of-time fragment of a leaf, if the leaf is not compiled at runtime.
std::optional<fragment_ref> builtin_fragment(leaf_stage const& leaf)
{
  if (auto const* builtin = std::get_if<builtin_leaf>(&leaf)) {
    return builtin_leaf_fragment(*builtin);
  }
  if (auto const* compare = std::get_if<compare_leaf>(&leaf)) {
    return compare_leaf_fragment(*compare);
  }
  return std::nullopt;
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

// An ahead-of-time fragment as it is passed to the linker.
struct linked_fragment {
  std::size_t file_index;   // Index of the fragment among the embedded fragments
  std::string const* name;  // Name the fragment's content is cached under
};

// Links ahead-of-time fragments and a leaf into a kernel through the kernel cache.
cudf::kernel link_kernel(std::span<linked_fragment const> builtin_fragments,
                         leaf_stage const& leaf,
                         kernel_identity const& names,
                         int attempt)
{
  CUDF_FUNC_RANGE();

  auto fragments = std::vector<rtcx::memory_fragment>{};
  for (auto const& fragment : builtin_fragments) {
    fragments.push_back({.data = fragment_bytes(fragment.file_index),
                         .type = rtcx::binary_type::FATBIN,
                         .name = fragment.name->c_str()});
  }

  rtcx::blob compiled_leaf;
  if (auto const builtin = builtin_fragment(leaf); builtin.has_value()) {
    fragments.push_back({.data = fragment_bytes(builtin->file_index),
                         .type = rtcx::binary_type::FATBIN,
                         .name = names.leaf.c_str()});
  } else {
    auto const& source = std::get<leaf_source>(leaf);
    // The translation-unit name is part of the fragment's cache key.
    auto const unit_name = attempt == 0 ? std::string{"cudf_pq_leaf.cu"}
                                        : std::format("cudf_pq_leaf_retry{}.cu", attempt);
    compiled_leaf =
      cudf::get_user_fragment(unit_name.c_str(), source.code.c_str(), {}, {}, {}, true);
    fragments.push_back({.data = compiled_leaf->view(),
                         .type = rtcx::binary_type::LTO_IR,
                         .name = names.leaf.c_str()});
  }

  return cudf::get_lto_linked_kernel(names.link, {}, fragments);
}

// Returns the kernel `link` builds for an identity, or nothing if it fails or has failed too
// recently or too often. `link` is called with the names of the attempt and its number.
template <typename Link>
std::optional<cudf::kernel> acquire_kernel(kernel_identity const& identity, Link const& link)
{
  auto const key = identity.key();
  auto& provider = state();

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
    return link(identity_of_attempt(identity, attempt), attempt);
  } catch (std::exception const& e) {
    std::lock_guard const lock{provider.mutex};
    auto& failed = provider.failed[key];
    // Callers that waited on another caller's failed attempt report the same attempt.
    if (failed.attempt == attempt) {
      ++failed.attempt;
      ++provider.stats.failures;
      failed.failed_at = std::chrono::steady_clock::now();
      CUDF_LOG_WARN(std::format(
        "Parquet reader: cannot build kernel {} (attempt {} of {}); using the generic path "
        "instead: {}",
        key,
        attempt + 1,
        provider.policy.max_attempts,
        e.what()));
    }
    return std::nullopt;
  }
}

// A fragment compiled ahead of time is identified by the hash of all such fragments, which
// changes whenever any of them does.
std::string builtin_name(std::string_view family, std::string_view kind, std::string_view name)
{
  return std::format("pq/{}/{}/{}/abi{}/frag{}/lto{}",
                     family,
                     kind,
                     name,
                     leaf_abi_version,
                     fragments_hash(),
                     CUDF_LTO_ARCHITECTURE);
}

std::string leaf_name(leaf_stage const& leaf)
{
  if (auto const builtin = builtin_fragment(leaf); builtin.has_value()) {
    return builtin_name("leaf", "builtin", builtin->name);
  }
  auto const& code  = std::get<leaf_source>(leaf).code;
  auto const digest = XXH3_128bits(code.data(), code.size());
  return std::format("pq/leaf/src/{:016x}{:016x}/nvrtc{}",
                     digest.high64,
                     digest.low64,
                     cudf::get_context().nvrtc_version().value_or(0));
}

// The linked kernel is fully determined by its fragments and the linker. The kernel cache adds
// the device architecture and the runtime and driver versions.
std::string link_name(std::string_view family)
{
  return std::format(
    "cudf/pq/{}/nvjitlink{}", family, cudf::get_context().nvjitlink_version().value_or(0));
}

// The tile decode kernel fragment of a pair of level source and value source.
fragment_ref decode_fragment(level_source levels, value_source values)
{
  CUDF_EXPECTS(values.num_bytes == 4 or values.num_bytes == 8,
               "A tile decode kernel reads 4- or 8-byte values",
               std::invalid_argument);
  auto const optional = levels == level_source::OPTIONAL;
  auto const dict     = values.encoding == value_encoding::DICTIONARY;
  auto const wide     = values.num_bytes == 8;
  static constexpr std::array<fragment_ref, 8> fragments{{
    {cudf_fragments::pq_decode_required_plain_b32, "required_plain_b32"},
    {cudf_fragments::pq_decode_required_plain_b64, "required_plain_b64"},
    {cudf_fragments::pq_decode_required_dict_b32, "required_dict_b32"},
    {cudf_fragments::pq_decode_required_dict_b64, "required_dict_b64"},
    {cudf_fragments::pq_decode_optional_plain_b32, "optional_plain_b32"},
    {cudf_fragments::pq_decode_optional_plain_b64, "optional_plain_b64"},
    {cudf_fragments::pq_decode_optional_dict_b32, "optional_dict_b32"},
    {cudf_fragments::pq_decode_optional_dict_b64, "optional_dict_b64"},
  }};
  return fragments[(optional ? 4 : 0) + (dict ? 2 : 0) + (wide ? 1 : 0)];
}

}  // namespace

std::string kernel_identity::key() const
{
  return std::format("{}|{}|{}|{}|{}", link, kernel, levels, values, leaf);
}

kernel_identity get_kernel_identity(kernel_stages const& stages)
{
  return {.kernel = builtin_name("flat", "kernel", "tile_loop"),
          .levels = builtin_name("flat", "level", level_fragment(stages.levels).name),
          .values = builtin_name("flat", "value", value_fragment(stages.values).name),
          .leaf   = leaf_name(stages.leaf),
          .link   = link_name("flat")};
}

std::optional<flat_buffer_kernel> get_flat_buffer_kernel(kernel_stages const& stages)
{
  auto kernel =
    acquire_kernel(get_kernel_identity(stages), [&](kernel_identity const& names, int attempt) {
      auto const fragments = std::array<linked_fragment, 3>{
        linked_fragment{cudf_fragments::pq_flat_kernel, &names.kernel},
        linked_fragment{level_fragment(stages.levels).file_index, &names.levels},
        linked_fragment{value_fragment(stages.values).file_index, &names.values}};
      return link_kernel(fragments, stages.leaf, names, attempt);
    });
  if (not kernel.has_value()) { return std::nullopt; }
  return flat_buffer_kernel{std::move(*kernel)};
}

kernel_identity get_decode_kernel_identity(kernel_stages const& stages)
{
  // The level source and the value source are compiled into the kernel fragment.
  return {
    .kernel = builtin_name("decode", "kernel", decode_fragment(stages.levels, stages.values).name),
    .levels = {},
    .values = {},
    .leaf   = leaf_name(stages.leaf),
    .link   = link_name("decode")};
}

std::optional<decode_flat_kernel> get_decode_flat_kernel(kernel_stages const& stages)
{
  auto kernel = acquire_kernel(
    get_decode_kernel_identity(stages), [&](kernel_identity const& names, int attempt) {
      auto const fragments = std::array<linked_fragment, 1>{
        linked_fragment{decode_fragment(stages.levels, stages.values).file_index, &names.kernel}};
      return link_kernel(fragments, stages.leaf, names, attempt);
    });
  if (not kernel.has_value()) { return std::nullopt; }
  return decode_flat_kernel{std::move(*kernel)};
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
