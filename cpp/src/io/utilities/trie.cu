/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Serialized trie implementation for C++/CUDA
 * @file trie.cu
 */

#include "trie.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Serializes the trie of a non-empty set of keys on the host.
 *
 * The distinct keys are sorted, and the trie is built breadth-first from the sorted keys: the keys
 * below a node (the keys that start with its prefix) are consecutive, the key equal to the prefix,
 * if any, comes first, and the others are grouped by their next character in ascending order. So
 * each group is a child of the node, listed in the order that `serialized_trie_contains` relies on,
 * and a child is the end of a key if the first key of its group ends with it.
 *
 * `std::string_view` compares characters with `std::char_traits<char>::lt`, which compares them as
 * `unsigned char`, the order of the children in the serialized trie.
 */
std::vector<serial_trie_node> serialize_trie(std::vector<std::string> const& keys)
{
  std::vector<std::string_view> sorted_keys(keys.begin(), keys.end());
  std::sort(sorted_keys.begin(), sorted_keys.end());
  sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()), sorted_keys.end());

  // The root matches the empty key, which sorts first. Its children follow it, where lookups start,
  // so its `children_offset` holds the length of the longest key instead.
  auto const max_key_length =
    std::max_element(sorted_keys.begin(), sorted_keys.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.size() < rhs.size();
    })->size();
  CUDF_EXPECTS(max_key_length <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "Trie keys are too long",
               std::overflow_error);
  std::vector<serial_trie_node> nodes;
  nodes.emplace_back(trie_terminating_character, sorted_keys.front().empty());
  nodes.front().children_offset = static_cast<int32_t>(max_key_length);

  // A node whose children are yet to be serialized: the keys [first_key, last_key) are the keys
  // below it, whose first `depth` characters are its prefix
  struct pending_node {
    size_t first_key;
    size_t last_key;
    size_t depth;
    size_t index;
  };
  std::queue<pending_node> to_visit;
  to_visit.push({0, sorted_keys.size(), 0, 0});
  while (not to_visit.empty()) {
    auto const node = to_visit.front();
    to_visit.pop();

    auto key = node.first_key;
    if (sorted_keys[key].size() == node.depth) { ++key; }
    // A node without children has no children list, and keeps a negative `children_offset`
    if (key == node.last_key) { continue; }

    if (node.index != 0) {
      auto const offset = nodes.size() - node.index;
      CUDF_EXPECTS(offset <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                   "Too many keys to build a serialized trie",
                   std::overflow_error);
      nodes[node.index].children_offset = static_cast<int32_t>(offset);
    }
    while (key < node.last_key) {
      auto const character = sorted_keys[key][node.depth];
      auto const group_end =
        std::find_if(sorted_keys.begin() + key + 1,
                     sorted_keys.begin() + node.last_key,
                     [&](auto const& other) { return other[node.depth] != character; }) -
        sorted_keys.begin();
      nodes.emplace_back(character, sorted_keys[key].size() == node.depth + 1);
      to_visit.push({key, static_cast<size_t>(group_end), node.depth + 1, nodes.size() - 1});
      key = group_end;
    }
    nodes.emplace_back(trie_terminating_character);
  }
  return nodes;
}

}  // namespace

rmm::device_uvector<serial_trie_node> create_serialized_trie(std::vector<std::string> const& keys,
                                                             cuda::stream_ref stream)
{
  if (keys.empty()) { return rmm::device_uvector<serial_trie_node>{0, stream}; }
  return cudf::detail::make_device_uvector(
    serialize_trie(keys), stream, cudf::get_current_device_resource_ref());
}

}  // namespace detail
}  // namespace cudf
