/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/trie.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

#include <cstdint>
#include <string>
#include <vector>

class TrieTest : public cudf::test::BaseFixture {};

namespace {
/**
 * @brief Looks up each of `queries` in `trie` on the device, and returns whether each is found.
 */
std::vector<bool> contains(cudf::detail::trie const& trie, std::vector<std::string> const& queries)
{
  auto const stream = cudf::test::get_default_stream();
  std::string chars;
  std::vector<int64_t> offsets{0};
  for (auto const& query : queries) {
    chars += query;
    offsets.push_back(chars.size());
  }
  auto const d_chars =
    cudf::detail::make_device_uvector(cudf::host_span<char const>{chars.data(), chars.size()},
                                      stream,
                                      cudf::get_current_device_resource_ref());
  auto const d_offsets =
    cudf::detail::make_device_uvector(offsets, stream, cudf::get_current_device_resource_ref());
  rmm::device_uvector<uint8_t> found(queries.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<std::size_t>{0},
                    cuda::counting_iterator<std::size_t>{queries.size()},
                    found.begin(),
                    [trie    = cudf::detail::trie_view{trie.data(), trie.size()},
                     chars   = d_chars.data(),
                     offsets = d_offsets.data()] __device__(std::size_t i) -> uint8_t {
                      auto const length = static_cast<std::size_t>(offsets[i + 1] - offsets[i]);
                      return cudf::detail::serialized_trie_contains(trie,
                                                                    {chars + offsets[i], length});
                    });
  auto const h_found = cudf::detail::make_std_vector(found, stream);
  return {h_found.begin(), h_found.end()};
}
}  // namespace

TEST_F(TrieTest, OnlyTheEmptyKey)
{
  // The trie is its root node alone: looking up other keys must not search children past it
  auto const trie = cudf::detail::create_serialized_trie({""}, cudf::test::get_default_stream());
  EXPECT_EQ(contains(trie, {"", "a", "ab", "\n"}), std::vector<bool>({true, false, false, false}));
}

TEST_F(TrieTest, KeysThatArePrefixesOfOtherKeys)
{
  // "é" is the two bytes C3 A9, which sort after the ASCII keys as unsigned bytes
  auto const trie = cudf::detail::create_serialized_trie({"abc", "a", "ab", "b", "é", "ab"},
                                                         cudf::test::get_default_stream());
  std::vector<std::string> const queries{
    "", "a", "ab", "abc", "b", "é", "abcd", "abd", "ac", "ba", "c", "\xC3", "éa", "è", "abcde"};
  std::vector<bool> const expected{false,
                                   true,
                                   true,
                                   true,
                                   true,
                                   true,
                                   false,
                                   false,
                                   false,
                                   false,
                                   false,
                                   false,
                                   false,
                                   false,
                                   false};
  EXPECT_EQ(contains(trie, queries), expected);
}

TEST_F(TrieTest, KeysLongerThan32767Characters)
{
  // Each character of the key is a node, so the nodes of its end are more than 32767 nodes after
  // the root
  std::string const key(40'000, 'x');
  auto const trie =
    cudf::detail::create_serialized_trie({key, "y"}, cudf::test::get_default_stream());
  auto const prefix = key.substr(0, key.size() - 1);
  EXPECT_EQ(contains(trie, {key, prefix, key + "x", prefix + "y", "y"}),
            std::vector<bool>({true, false, false, false, true}));
}

TEST_F(TrieTest, ChildrenMoreThan32767NodesAfterTheirParent)
{
  // 40000 keys of six characters. A node of the fifth level is followed by the rest of its level
  // and by the children of the nodes before it, so the last ones are more than 32767 nodes before
  // their children.
  std::vector<std::string> keys;
  for (int i = 0; i < 40'000; ++i) {
    auto const digits = std::to_string(i);
    keys.push_back("k" + std::string(5 - digits.size(), '0') + digits);
  }
  auto const trie = cudf::detail::create_serialized_trie(keys, cudf::test::get_default_stream());
  auto queries    = keys;
  queries.insert(queries.end(), {"k40000", "k3999", "k399990", "k0000", "k"});
  auto expected = std::vector<bool>(keys.size(), true);
  expected.insert(expected.end(), 5, false);
  EXPECT_EQ(contains(trie, queries), expected);
}

TEST_F(TrieTest, NoKeys)
{
  auto const trie = cudf::detail::create_serialized_trie({}, cudf::test::get_default_stream());
  EXPECT_TRUE(trie.is_empty());
  EXPECT_EQ(contains(trie, {"", "a"}), std::vector<bool>({false, false}));
}

CUDF_TEST_PROGRAM_MAIN()
