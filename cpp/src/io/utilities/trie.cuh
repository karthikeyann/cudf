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
  // The root's children offset is the length of the longest key, if not negative (see
  // `create_serialized_trie`)
  if (auto const max_key_length = trie.front().children_offset;
      max_key_length >= 0 && key.size() > static_cast<size_t>(max_key_length)) {
    return false;
  }
  if (key.empty()) { return trie.front().is_leaf; }
  // A single loop, which compiles into faster parsing kernels than a loop over the key characters
  // with a nested search of the children
  auto curr_node = trie.begin() + 1;
  size_t index   = 0;
  while (true) {
    // Children are sorted as unsigned bytes: the search skips the ones before the key character
    if (curr_node->character != trie_terminating_character &&
        static_cast<unsigned char>(curr_node->character) < static_cast<unsigned char>(key[index])) {
      ++curr_node;
      continue;
    }
    // Could not find the character, done with the search
    if (curr_node->character != key[index]) { return false; }
    // Even if the node is present, return true only if that node is at the end of a word
    if (++index == key.size()) { return curr_node->is_leaf; }
    // A node without children has a negative offset: no key continues past it. Following the
    // offset would resume the search at an unrelated node and could report a false match.
    if (curr_node->children_offset < 0) { return false; }
    curr_node += curr_node->children_offset;
  }
}
}  // namespace detail
}  // namespace cudf
