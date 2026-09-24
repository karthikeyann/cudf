/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Serialized trie implementation for C++/CUDA
 * @file trie.cuh
 */

#pragma once

#include "trie.hpp"

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <optional>

namespace cudf {
namespace detail {

/*
 * @brief Searches for a string in a serialized trie.
 *
 * @param trie Pointer to the array of nodes that make up the trie
 * @param key Pointer to the start of the string to find
 * @param key_len Length of the string to find
 *
 * @return Boolean value; true if string is found, false otherwise
 */
__device__ inline bool serialized_trie_contains(device_span<serial_trie_node const> trie,
                                                device_span<char const> key)
{
  if (trie.empty()) { return false; }
  if (key.empty()) { return trie.front().is_leaf; }
  auto curr_node = trie.begin() + 1;
  for (auto curr_key = key.begin(); curr_key < key.end(); ++curr_key) {
    // Don't jump away from root node
    if (curr_key != key.begin()) {
      // A node without children has a negative offset: no key continues past it. Following the
      // offset would resume the search at an unrelated node and could report a false match.
      if (curr_node->children_offset < 0) { return false; }
      curr_node += curr_node->children_offset;
    }
    // Search for the next character in the array of children nodes
    // Nodes are sorted as unsigned bytes - terminate search if the node is larger or equal
    auto const key_char = static_cast<unsigned char>(*curr_key);
    while (curr_node->character != trie_terminating_character &&
           static_cast<unsigned char>(curr_node->character) < key_char) {
      ++curr_node;
    }
    // Could not find the next character, done with the search
    if (curr_node->character != *curr_key) { return false; }
  }
  // Even if the node is present, return true only if that node is at the end of a word
  return curr_node->is_leaf;
}
}  // namespace detail
}  // namespace cudf
