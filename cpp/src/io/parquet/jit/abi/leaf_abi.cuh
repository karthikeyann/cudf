/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

// The leaf ABI of the Parquet decode kernels.
//
// A consumer kernel is compiled ahead of time with the leaf left unresolved. A leaf is compiled
// either ahead of time or at runtime from generated source, and exactly one leaf is linked into a
// consumer when its kernel is first needed.
//
// This is the only Parquet header visible to runtime-compiled leaves. It must stay
// self-contained, and any change to the leaf signature must change `leaf_abi_version` and the
// version in the leaf symbol names: the symbols have C linkage, so only their names keep a leaf
// built against one signature from linking into a consumer built against another.

//
// Extension points. None of the following exists yet. They are recorded here so that the changes
// which add them keep one shape.
//
// - Wider values. A DECIMAL128, FIXED_LEN_BYTE_ARRAY or INT96 value does not fit `leaf_b64_t`.
//   It gets a leaf symbol of its own (`_b128` taking two 64-bit words, or `_bytes` taking the
//   address and length of the value in the page) instead of a wider `value` in the existing
//   signatures, so that the 4- and 8-byte kernels keep their register use.
// - Variable-width sink. The output position of a string depends on the lengths of all earlier
//   rows, so a string column is decoded in two passes over the same sources: a size pass whose
//   leaf writes each row's length, then, once the host has turned lengths into offsets, a copy
//   pass whose leaf writes the bytes at the row's offset. It needs the `_bytes` signature above
//   and context fields for the offsets and the character buffer.
// - Compacting sink. The row a leaf writes to is chosen by the consumer, so a consumer that has
//   the row mask before it decodes a column can skip the rows that are dropped and pass each kept
//   row its position among the kept rows. This removes the gather after the filter and needs no
//   change here: the consumer needs the mask as bits and, for each of its tiles, the number of
//   kept rows before it.
// - Decompression. Pages are decompressed before any kernel using this ABI runs. A stage that
//   reads compressed bytes would sit in front of the value source and is invisible to a leaf.

namespace cudf::io::parquet::jit {

/// Version of the leaf signature, repeated in the leaf symbol names.
inline constexpr int leaf_abi_version = 1;

/**
 * @brief State of one leaf for one input column and one launch, passed to the leaf by value.
 */
struct leaf_ctx {
  void* out;                 ///< Output values; element `row` belongs to output row `row`
  uint8_t* keep;             ///< BOOL8 per output row, zeroed by the host; null without a predicate
  uint8_t const* dict_keep;  ///< BOOL8 per dictionary entry of the column; null if unused
  int64_t scale;             ///< Conversion parameter; zero if unused
  uint64_t lit[4];           ///< Predicate literals, each in the low bytes of its slot
};

static_assert(sizeof(leaf_ctx) == 64);

/// Maximum number of predicate literals a leaf can read.
inline constexpr int max_leaf_literals = 4;

/**
 * @brief Signature of a leaf.
 *
 * The consumer calls the leaf once for each valid row and never for a null row. `value` is the
 * decoded physical value and `dict_idx` the index of its dictionary entry, or zero when the value
 * does not come from a dictionary. A leaf may store the value at `out[row]` and, when it has a
 * predicate, writes `keep[row]`. It returns one if the row is kept and zero otherwise, writes
 * nothing else and reports no errors.
 */
using leaf_b32_t = uint32_t(leaf_ctx, uint32_t row, uint32_t value, uint32_t dict_idx);
/// @copydoc leaf_b32_t
using leaf_b64_t = uint32_t(leaf_ctx, uint32_t row, uint64_t value, uint32_t dict_idx);

}  // namespace cudf::io::parquet::jit

#ifdef __CUDACC__
extern "C" __device__ cudf::io::parquet::jit::leaf_b32_t cudf_pq_leaf_v1_b32;
extern "C" __device__ cudf::io::parquet::jit::leaf_b64_t cudf_pq_leaf_v1_b64;
#endif
