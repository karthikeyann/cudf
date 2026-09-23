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
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace cudf {
namespace detail {

std::vector<serial_trie_node> serialize_trie(std::vector<std::string> const& keys)
{
  if (keys.empty()) { return {}; }

  // Distinct keys in byte order: the keys below a node are then a contiguous range, and its
  // children's first characters appear in ascending order
  std::vector<std::string_view> sorted(keys.begin(), keys.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

  // If the trie matches empty strings, the root node is marked as 'end of word'. The first node in
  // the serialized trie is also used to match empty strings.
  std::vector<serial_trie_node> nodes;
  nodes.emplace_back(trie_terminating_character, sorted.front().empty());
  // The root's children offset is never followed (lookups start at index 1); it stores the length
  // of the longest key so that longer keys can be rejected without walking the trie
  auto const max_key_length =
    std::max_element(sorted.cbegin(), sorted.cend(), [](auto const& a, auto const& b) {
      return a.size() < b.size();
    })->size();
  if (max_key_length <= static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    nodes.front().children_offset = static_cast<int16_t>(max_key_length);
  }

  // Breadth-first over the nodes: the keys [begin, end) share the node's `depth` characters. The
  // root (index -1) is not part of the serialized children lists.
  struct pending_node {
    size_t begin;
    size_t end;
    size_t depth;
    int64_t index;
  };
  std::deque<pending_node> to_visit{{0, sorted.size(), 0, -1}};
  while (!to_visit.empty()) {
    auto const node = to_visit.front();
    to_visit.pop_front();
    // Keys that end at this node come first
    auto first = node.begin;
    while (first < node.end && sorted[first].size() == node.depth) {
      ++first;
    }
    bool has_children = false;
    while (first < node.end) {
      auto const character = sorted[first][node.depth];
      auto last            = first;
      while (last < node.end && sorted[last][node.depth] == character) {
        ++last;
      }
      // Update the children offset of the parent node, unless at the root
      if (node.index >= 0 && nodes[node.index].children_offset < 0) {
        nodes[node.index].children_offset =
          static_cast<int16_t>(static_cast<uint16_t>(nodes.size() - node.index));
      }
      nodes.emplace_back(character, sorted[first].size() == node.depth + 1);
      to_visit.push_back({first, last, node.depth + 1, static_cast<int64_t>(nodes.size()) - 1});
      has_children = true;
      first        = last;
    }
    // Only add the terminating character if any nodes were added
    if (has_children) { nodes.emplace_back(trie_terminating_character); }
  }
  return nodes;
}

rmm::device_uvector<serial_trie_node> create_serialized_trie(std::vector<std::string> const& keys,
                                                             cuda::stream_ref stream)
{
  if (keys.empty()) { return rmm::device_uvector<serial_trie_node>{0, stream}; }
  return cudf::detail::make_device_uvector(
    serialize_trie(keys), stream, cudf::get_current_device_resource_ref());
}

}  // namespace detail
}  // namespace cudf
