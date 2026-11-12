/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

namespace cudf::io::parquet::jit {

/**
 * @brief A comparison a comparison leaf applies to a value and a literal.
 */
enum class compare_operator : uint8_t {
  NONE = 0,       ///< No comparison; every value passes
  EQUAL,          ///< value == literal
  NOT_EQUAL,      ///< value != literal
  LESS,           ///< value < literal
  GREATER,        ///< value > literal
  LESS_EQUAL,     ///< value <= literal
  GREATER_EQUAL,  ///< value >= literal
};

/// Slot of `leaf_ctx::lit` in which a comparison leaf finds its operators: the operator applied
/// to literal slot zero in the low byte, the one applied to literal slot one in the next byte.
inline constexpr int compare_operators_slot = 3;

}  // namespace cudf::io::parquet::jit
