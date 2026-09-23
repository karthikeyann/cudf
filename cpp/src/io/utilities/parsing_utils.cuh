/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "column_type_histogram.hpp"
#include "io/csv/datetime.cuh"
#include "string_parsing.hpp"
#include "trie.cuh"

#include <cudf/io/types.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/detail/convert/fixed_point.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <nv/target>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/mismatch.h>

using cudf::device_span;

namespace cudf {
namespace io {

/**
 * @brief Returns the escaped characters for a given character.
 *
 * @param escaped_char The character to escape.
 * @return The escaped characters for a given character.
 */
__device__ __forceinline__ cuda::std::pair<char, char> get_escaped_char(char escaped_char)
{
  switch (escaped_char) {
    case '"': return {'\\', '"'};
    case '\\': return {'\\', '\\'};
    case '/': return {'\\', '/'};
    case '\b': return {'\\', 'b'};
    case '\f': return {'\\', 'f'};
    case '\n': return {'\\', 'n'};
    case '\r': return {'\\', 'r'};
    case '\t': return {'\\', 't'};
    // case 'u': return UNICODE_SEQ;
    default: return {'\0', escaped_char};
  }
}

/**
 * @brief Returns the numeric value of an ASCII/UTF-8 character.
 * Handles hexadecimal digits, both uppercase and lowercase
 * for integral types and only decimal digits for floating point types.
 * If the character is not a valid numeric digit then `0` is returned and
 * valid_flag is set to false.
 *
 * @param c ASCII or UTF-8 character
 * @param valid_flag Set to false if input is not valid. Unchanged otherwise.
 *
 * @return uint8_t Numeric value of the character, or `0`
 */
template <typename T, bool as_hex = false>
__device__ constexpr uint8_t decode_digit(char c, bool* valid_flag)
{
  if (c >= '0' && c <= '9') return c - '0';
  if constexpr (as_hex and std::is_integral_v<T>) {
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  }

  *valid_flag = false;
  return 0;
}

// Converts character to lowercase.
CUDF_HOST_DEVICE constexpr char to_lower(char const c)
{
  return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
}

/**
 * @brief Checks if string is infinity, case insensitive with/without sign
 * Valid infinity strings are inf, +inf, -inf, infinity, +infinity, -infinity
 * String comparison is case insensitive.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return true if string is valid infinity, else false.
 */
CUDF_HOST_DEVICE constexpr bool is_infinity(char const* begin, char const* end)
{
  if (*begin == '-' || *begin == '+') begin++;
  char const* cinf = "infinity";
  auto index       = begin;
  while (index < end) {
    if (*cinf != to_lower(*index)) break;
    index++;
    cinf++;
  }
  return ((index == begin + 3 || index == begin + 8) && index >= end);
}

namespace detail {

/// Number of successive `divisor /= 10` steps tabulated; the last entry is the underflow to zero
constexpr int num_decimal_divisors = 324;

/**
 * @brief The divisors 1/10, (1/10)/10, ... exactly as produced by repeated IEEE double division
 * (generated offline, round-to-nearest with subnormals), so parsing can look them up instead of
 * dividing: double division is very slow on GPUs with reduced FP64 throughput.
 */
static __device__ double const decimal_divisors[num_decimal_divisors] = {
  0x1.999999999999ap-4, 0x1.47ae147ae147bp-7, 0x1.0624dd2f1a9fcp-10,
  0x1.a36e2eb1c432dp-14, 0x1.4f8b588e368f1p-17, 0x1.0c6f7a0b5ed8ep-20,
  0x1.ad7f29abcaf4ap-24, 0x1.5798ee2308c3bp-27, 0x1.12e0be826d696p-30,
  0x1.b7cdfd9d7bdbdp-34, 0x1.5fd7fe1796497p-37, 0x1.19799812dea12p-40,
  0x1.c25c268497683p-44, 0x1.6849b86a12b9cp-47, 0x1.203af9ee75616p-50,
  0x1.cd2b297d889bdp-54, 0x1.70ef54646d497p-57, 0x1.2725dd1d243acp-60,
  0x1.d83c94fb6d2adp-64, 0x1.79ca10c924224p-67, 0x1.2e3b40a0e9b50p-70,
  0x1.e392010175ee6p-74, 0x1.82db34012b252p-77, 0x1.357c299a88ea8p-80,
  0x1.ef2d0f5da7ddap-84, 0x1.8c240c4aecb15p-87, 0x1.3ce9a36f23c11p-90,
  0x1.fb0f6be50601bp-94, 0x1.95a5efea6b349p-97, 0x1.4484bfeebc2a1p-100,
  0x1.039d665896881p-103, 0x1.9f623d5a8a735p-107, 0x1.4c4e977ba1f5ep-110,
  0x1.09d8792fb4c4bp-113, 0x1.a95a5b7f87a12p-117, 0x1.54484932d2e75p-120,
  0x1.1039d428a8b91p-123, 0x1.b38fb9daa78e8p-127, 0x1.5c72fb1552d86p-130,
  0x1.16c262777579ep-133, 0x1.be03d0bf225cap-137, 0x1.64cfda3281e3bp-140,
  0x1.1d7314f534b62p-143, 0x1.c8b821885456ap-147, 0x1.6d601ad376abbp-150,
  0x1.244ce242c5562p-153, 0x1.d3ae36d13bbd0p-157, 0x1.7624f8a762fdap-160,
  0x1.2b50c6ec4f315p-163, 0x1.dee7a4ad4b822p-167, 0x1.7f1fb6f10934ep-170,
  0x1.327fc58da0f72p-173, 0x1.ea6608e29b250p-177, 0x1.8851a0b548ea6p-180,
  0x1.39dae6f76d885p-183, 0x1.f62b0b257c0d5p-187, 0x1.91bc08eac9a44p-190,
  0x1.41633a556e1d0p-193, 0x1.011c2eaabe7dap-196, 0x1.9b604aaaca62ap-200,
  0x1.4919d5556eb55p-203, 0x1.0747ddddf22aap-206, 0x1.a53fc9631d110p-210,
  0x1.50ffd44f4a740p-213, 0x1.0d9976a5d529ap-216, 0x1.af5bf109550f6p-220,
  0x1.59165a6ddda5ep-223, 0x1.1411e1f17e1e5p-226, 0x1.b9b6364f30308p-230,
  0x1.615e91d8f35a0p-233, 0x1.1ab20e472914dp-236, 0x1.c45016d841baep-240,
  0x1.69d9abe034958p-243, 0x1.217aefe69077ap-246, 0x1.cf2b1970e725dp-250,
  0x1.7288e1271f517p-253, 0x1.286d80ec190dfp-256, 0x1.da48ce468e7cbp-260,
  0x1.7b6d71d20b96fp-263, 0x1.2f8ac174d6126p-266, 0x1.e5aacf215683dp-270,
  0x1.8488a5b445364p-273, 0x1.36d3b7c36a91dp-276, 0x1.f152bf9f10e95p-280,
  0x1.8ddbcc7f40baap-283, 0x1.3e497065cd622p-286, 0x1.fd424d6faf036p-290,
  0x1.97683df2f2692p-293, 0x1.45ecfe5bf520ep-296, 0x1.04bd984990e72p-299,
  0x1.a12f5a0f4e3eap-303, 0x1.4dbf7b3f71cbbp-306, 0x1.0aff95cc5b096p-309,
  0x1.ab328946f80f0p-313, 0x1.55c2076bf9a5ap-316, 0x1.116805effaeaep-319,
  0x1.b5733cb32b116p-323, 0x1.5df5ca28ef412p-326, 0x1.17f7d4ed8c342p-329,
  0x1.bff2ee48e0536p-333, 0x1.665bf1d3e6a92p-336, 0x1.1eaff4a985542p-339,
  0x1.cab3210f3bb9dp-343, 0x1.6ef5b40c2fc7ep-346, 0x1.25915cd68c9fep-349,
  0x1.d5b5615747663p-353, 0x1.77c44ddf6c51cp-356, 0x1.2c9d0b192374ap-359,
  0x1.e0fb44f505876p-363, 0x1.80c903f7379f8p-366, 0x1.33d4032c2c7fap-369,
  0x1.ec866b79e0cc3p-373, 0x1.8a0522c7e709cp-376, 0x1.3b374f06526e3p-379,
  0x1.f8587e7083e38p-383, 0x1.9379fec06982dp-386, 0x1.42c7ff005468ap-389,
  0x1.023998cd1053bp-392, 0x1.9d28f47b4d52bp-396, 0x1.4a8729fc3ddbcp-399,
  0x1.086c219697e30p-402, 0x1.a71368f0f304dp-406, 0x1.5275ed8d8f371p-409,
  0x1.0ec4be0ad8f8ep-412, 0x1.b13ac9aaf4c16p-416, 0x1.5a956e225d678p-419,
  0x1.1544581b7dec6p-422, 0x1.bba08cf8c97a3p-426, 0x1.62e6d72d6dfb6p-429,
  0x1.1bebdf578b2f8p-432, 0x1.c6463225ab7f3p-436, 0x1.6b6b5b5155ff6p-439,
  0x1.22bc490dde65ep-442, 0x1.d12d41afca3cap-446, 0x1.7424348ca1ca2p-449,
  0x1.29b69070816e8p-452, 0x1.dc574d80cf173p-456, 0x1.7d12a4670c129p-459,
  0x1.30dbb6b8d6754p-462, 0x1.e7c5f127bd886p-466, 0x1.8637f41fcad38p-469,
  0x1.382cc34ca242dp-472, 0x1.f37ad21436d15p-476, 0x1.8f9574dcf8a77p-479,
  0x1.3faac3e3fa1f9p-482, 0x1.ff779fd329cc2p-486, 0x1.992c7fdc21702p-489,
  0x1.4756ccb01ac02p-492, 0x1.05df0a267bccep-495, 0x1.a2fe76a3f947dp-499,
  0x1.4f31f8832dd31p-502, 0x1.0c27fa028b0f4p-505, 0x1.ad0cc33744e53p-509,
  0x1.573d68f903ea9p-512, 0x1.1297872d9cbbap-515, 0x1.b758d848fac5dp-519,
  0x1.5f7a46a0c89e4p-522, 0x1.192e9ee706e50p-525, 0x1.c1e43171a4a1ap-529,
  0x1.67e9c127b6e7bp-532, 0x1.1fee341fc5862p-535, 0x1.ccb0536608d6ap-539,
  0x1.708d0f84d3deep-542, 0x1.26d73f9d764bep-545, 0x1.d7becc2f23acap-549,
  0x1.79657025b623bp-552, 0x1.2deac01e2b4fcp-555, 0x1.e3113363787fap-559,
  0x1.8274291c60662p-562, 0x1.3529ba7d19eb5p-565, 0x1.eea92a61c3122p-569,
  0x1.8bba884e35a82p-572, 0x1.3c9539d82aecep-575, 0x1.fa885c8d117b0p-579,
  0x1.9539e3a40dfc0p-582, 0x1.442e4fb671966p-585, 0x1.03583fc527ab8p-588,
  0x1.9ef3993b72ac0p-592, 0x1.4bf6142f8ef00p-595, 0x1.0991a9bfa58cdp-598,
  0x1.a8e90f9908e15p-602, 0x1.53eda614071aap-605, 0x1.0ff151a99f488p-608,
  0x1.b31bb5dc320dap-612, 0x1.5c162b168e715p-615, 0x1.1678227871f44p-618,
  0x1.bd8d03f3e986dp-622, 0x1.6470cff6546bep-625, 0x1.1d270cc510565p-628,
  0x1.c83e7ad4e6f08p-632, 0x1.6cfec8aa525a0p-635, 0x1.23ff06eea8480p-638,
  0x1.d331a4b10d400p-642, 0x1.75c1508da4333p-645, 0x1.2b010d3e1cf5cp-648,
  0x1.de6815302e560p-652, 0x1.7eb9aa8cf1de6p-655, 0x1.322e220a5b185p-658,
  0x1.e9e369aa2b5a2p-662, 0x1.87e92154ef7b5p-665, 0x1.39874ddd8c62ap-668,
  0x1.f5a549627a376p-672, 0x1.91510781fb5f8p-675, 0x1.410d9f9b2f7fap-678,
  0x1.00d7b2e28c662p-681, 0x1.9af2b7d0e0a36p-685, 0x1.48c22ca71a1c5p-688,
  0x1.0701bd527b49ep-691, 0x1.a4cf9550c5430p-695, 0x1.50a6110d6a9c0p-698,
  0x1.0d51a73deee33p-701, 0x1.aee90b964b052p-705, 0x1.58ba6fab6f375p-708,
  0x1.13c85955f292ap-711, 0x1.b9408eefea843p-715, 0x1.610072598869cp-718,
  0x1.1a66c1e139ee3p-721, 0x1.c3d79c9b8fe38p-725, 0x1.69794a160cb60p-728,
  0x1.212dd4de7091ap-731, 0x1.ceafbafd80e90p-735, 0x1.72262f3133edap-738,
  0x1.281e8c275cbe2p-741, 0x1.d9ca79d894636p-745, 0x1.7b08617a104f8p-748,
  0x1.2f39e794d9d93p-751, 0x1.e5297287c2f52p-755, 0x1.8421286c9bf75p-758,
  0x1.3680ed23aff91p-761, 0x1.f0ce4839198e8p-765, 0x1.8d71d360e13edp-768,
  0x1.3df4a91a4dcbep-771, 0x1.fcbaa82a16130p-775, 0x1.96fbb9bb44dc0p-778,
  0x1.45962e2f6a49ap-781, 0x1.047824f2bb6e2p-784, 0x1.a0c03b1df8b03p-788,
  0x1.4d6695b193c02p-791, 0x1.0ab877c143002p-794, 0x1.aac0bf9b9e66ap-798,
  0x1.5566ffafb1ebbp-801, 0x1.111f32f2f4bc9p-804, 0x1.b4feb7eb212dbp-808,
  0x1.5d98932280f16p-811, 0x1.17ad428200c12p-814, 0x1.bf7b9d9cce01dp-818,
  0x1.65fc7e170b34ap-821, 0x1.1e6398126f5d5p-824, 0x1.ca38f350b22eep-828,
  0x1.6e93f5da28258p-831, 0x1.25432b14eceadp-834, 0x1.d53844ee47de2p-838,
  0x1.77603725064b5p-841, 0x1.2c4cf8ea6b6f7p-844, 0x1.e07b27dd78b25p-848,
  0x1.8062864ac6f51p-851, 0x1.338205089f2a7p-854, 0x1.ec033b40feaa5p-858,
  0x1.899c2f673221ep-861, 0x1.3ae3591f5b4e5p-864, 0x1.f7d228322bb08p-868,
  0x1.930e868e895a0p-871, 0x1.4272053ed4480p-874, 0x1.01f4d0ff1039ap-877,
  0x1.9cbae7fe805c3p-881, 0x1.4a2f1ffecd169p-884, 0x1.0825b3323dabap-887,
  0x1.a6a2b85062ac3p-891, 0x1.521bc6a6b5569p-894, 0x1.0e7c9eebc4454p-897,
  0x1.b0c764ac6d3bap-901, 0x1.5a391d56bdc95p-904, 0x1.14fa7ddefe3aap-907,
  0x1.bb2a62fe63910p-911, 0x1.62884f31e940dp-914, 0x1.1ba03f5b2100ap-917,
  0x1.c5cd322b68010p-921, 0x1.6b0a8e892000dp-924, 0x1.226ed86db333ep-927,
  0x1.d0b15a491eb96p-931, 0x1.73c115074bc78p-934, 0x1.29674405d6393p-937,
  0x1.dbd86cd6238ebp-941, 0x1.7cad23de82d89p-944, 0x1.308a831868ad4p-947,
  0x1.e74404f3daaedp-951, 0x1.85d003f6488bep-954, 0x1.37d99cc506d65p-957,
  0x1.f2f5c7a1a48a2p-961, 0x1.8f2b061aea082p-964, 0x1.3f559e7bee6cep-967,
  0x1.feef63f97d7b0p-971, 0x1.98bf832dfdfc0p-974, 0x1.46ff9c24cb300p-977,
  0x1.059949b708f33p-980, 0x1.a28edc580e51ep-984, 0x1.4ed8b04671db2p-987,
  0x1.0be08d0527e28p-990, 0x1.ac9a7b3b73040p-994, 0x1.56e1fc2f8f366p-997,
  0x1.124e63593f5ebp-1000, 0x1.b6e3d22865645p-1004, 0x1.5f1ca820511d1p-1007,
  0x1.18e3b9b374174p-1010, 0x1.c16c5c5253586p-1014, 0x1.6789e3750f79ep-1017,
  0x1.1fa182c40c618p-1020, 0x0.730d67819e8d6p-1022, 0x0.0b8157268fdafp-1022,
  0x0.012688b70e62bp-1022, 0x0.001d74124e3d1p-1022, 0x0.0002f201d49fbp-1022,
  0x0.00004b6695433p-1022, 0x0.0000078a42205p-1022, 0x0.000000c1069cdp-1022,
  0x0.000000134d761p-1022, 0x0.00000001ee256p-1022, 0x0.00000000316a2p-1022,
  0x0.0000000004f10p-1022, 0x0.00000000007e8p-1022, 0x0.00000000000cap-1022,
  0x0.0000000000014p-1022, 0x0.0000000000002p-1022, 0x0p+0,
};

/**
 * @brief Returns the divisor after `index + 1` successive divisions of 1 by 10.
 */
CUDF_HOST_DEVICE inline double decimal_divisor(int index)
{
#ifdef __CUDA_ARCH__
  return index < num_decimal_divisors ? decimal_divisors[index] : 0.0;
#else
  double divisor = 1;
  for (int i = 0; i <= index && divisor != 0; ++i) {
    divisor /= 10;
  }
  return divisor;
#endif
}

}  // namespace detail

/**
 * @brief Parses a character string and returns its numeric value.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param opts The global parsing behavior options
 * @tparam base Base (radix) to use for conversion
 *
 * @return The parsed and converted value
 */
template <typename T, int base = 10>
CUDF_HOST_DEVICE cuda::std::optional<T> parse_numeric(char const* begin,
                                                      char const* end,
                                                      parse_options_view const& opts)
{
  T value{};
  bool all_digits_valid = true;
  constexpr bool as_hex = (base == 16);

  // Handle negative values if necessary
  int32_t sign = (*begin == '-') ? -1 : 1;

  // Handle infinity
  if (cuda::std::is_floating_point_v<T> && is_infinity(begin, end)) {
    return sign * cuda::std::numeric_limits<T>::infinity();
  }
  if (*begin == '-' || *begin == '+') begin++;

  // Skip over the "0x" prefix for hex notation
  if (base == 16 && begin + 2 < end && *begin == '0' && *(begin + 1) == 'x') { begin += 2; }

  // Handle the whole part of the number
  if constexpr (cuda::std::is_same_v<T, double> && base == 10) {
    // Accumulate leading digits as an integer: while below 10^15 (< 2^53) every step of the
    // double computation below is exact, so the result is identical
    uint64_t whole          = 0;
    int num_digits          = 0;
    constexpr int max_exact = 15;
    while (begin < end && num_digits < max_exact) {
      if (*begin == opts.decimal || *begin == 'e' || *begin == 'E') { break; }
      if (*begin != opts.thousands && *begin != '+') {
        whole = whole * 10 + decode_digit<T, as_hex>(*begin, &all_digits_valid);
        ++num_digits;
      }
      ++begin;
    }
    value = static_cast<T>(whole);
  }
  while (begin < end) {
    if (*begin == opts.decimal) {
      ++begin;
      break;
    } else if (base == 10 && (*begin == 'e' || *begin == 'E')) {
      break;
    } else if (*begin != opts.thousands && *begin != '+') {
      value = (value * base) + decode_digit<T, as_hex>(*begin, &all_digits_valid);
    }
    ++begin;
  }

  if (cuda::std::is_floating_point_v<T>) {
    // Handle fractional part of the number if necessary
    double divisor      = 1;
    int divisor_index   = 0;
    while (begin < end) {
      if (*begin == 'e' || *begin == 'E') {
        ++begin;
        break;
      } else if (*begin != opts.thousands && *begin != '+') {
        if constexpr (base == 10) {
          divisor = detail::decimal_divisor(divisor_index++);
        } else {
          divisor /= base;
        }
        value += decode_digit<T, as_hex>(*begin, &all_digits_valid) * divisor;
      }
      ++begin;
    }

    // Handle exponential part of the number if necessary
    if (begin < end) {
      int32_t const exponent_sign = *begin == '-' ? -1 : 1;
      if (*begin == '-' || *begin == '+') { ++begin; }
      int32_t exponent = 0;
      while (begin < end) {
        exponent = (exponent * 10) + decode_digit<T, as_hex>(*(begin++), &all_digits_valid);
      }
      if (exponent != 0) {
        auto const power = exponent * exponent_sign;
        // The table holds the device's own exp10 results, so the lookup is bit-identical
        NV_IF_ELSE_TARGET(
          NV_IS_DEVICE,
          (value *= (opts.exp10_table != nullptr && power >= -exp10_table_bias &&
                     power <= exp10_table_bias)
                      ? opts.exp10_table[power + exp10_table_bias]
                      : exp10(double(power));),
          (value *= exp10(double(power));))
      }
    }
  }
  if (!all_digits_valid) { return cuda::std::optional<T>{}; }

  return value * sign;
}

namespace gpu {

/**
 * @brief Advances past a run of delimiter characters at the start of a field.
 */
__device__ __inline__ char const* skip_leading_delimiter_run(char const* begin,
                                                             char const* end,
                                                             parse_options_view const& opts)
{
  if (!opts.multi_delimiter) { return begin; }
  while (begin < end && *begin == opts.delimiter) {
    ++begin;
  }
  return begin;
}

/**
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param opts A set of parsing options
 * @param escape_char A boolean value to signify whether to consider `\` as escape character or
 * just a character.
 *
 * @return Pointer to the last character in the field, including the
 *  delimiter(s) following the field data
 */
__device__ __inline__ char const* seek_field_end(char const* begin,
                                                 char const* end,
                                                 parse_options_view const& opts,
                                                 bool escape_char = false)
{
  bool quotation   = false;
  auto current     = begin;
  bool escape_next = false;

  auto const field_starts_with_quote = (begin < end && *begin == opts.quotechar);
  while (current < end) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields.
    // Check for instances such as "a2\"bc" and "\\" if `escape_char` is true.

    // Only process quotes if field started with a quote
    if (field_starts_with_quote && *current == opts.quotechar && !escape_next) {
      quotation = !quotation;
    } else if (!quotation) {
      if (*current == opts.delimiter) {
        while (opts.multi_delimiter && (current + 1 < end) && *(current + 1) == opts.delimiter) {
          ++current;
        }
        break;
      } else if (*current == opts.terminator) {
        break;
      } else if (*current == '\r' && (current + 1 < end && *(current + 1) == '\n')) {
        --end;
        break;
      }
    }

    if (escape_char) {
      // If a escape character is encountered, escape next character in next loop.
      if (not escape_next and *current == '\\') {
        escape_next = true;
      } else {
        escape_next = false;
      }
    }

    if (current < end) { current++; }
  }
  return current;
}

/**
 * @brief Lexicographically compare digits in input against string
 * representing an integer
 *
 * @param data The pointer to beginning of character string
 * @param golden The pointer to beginning of character string representing
 * the value to be compared against
 * @return bool True if integer represented by character string is less
 * than or equal to golden data
 */
template <int N>
__device__ __inline__ bool less_equal_than(char const* data, char const (&golden)[N])
{
  auto mismatch_pair = thrust::mismatch(thrust::seq, data, data + N - 1, golden);
  if (mismatch_pair.first != data + N - 1) {
    return *mismatch_pair.first <= *mismatch_pair.second;
  } else {
    // Exact match
    return true;
  }
}

/**
 * @brief Determine which counter to increment when a sequence of digits
 * and a parity sign is encountered.
 *
 * @param data_begin The pointer to beginning of character string
 * @param data_end The pointer to end of character string
 * @param is_negative Whether the number is negative
 * @param stats Reference to structure with counters
 * @return Pointer to appropriate counter that belong to
 * the interpreted data type
 */
__device__ __inline__ cudf::size_type* infer_integral_field_counter(char const* data_begin,
                                                                    char const* data_end,
                                                                    bool is_negative,
                                                                    column_type_histogram& stats)
{
  static constexpr char uint64_max_abs[] = "18446744073709551615";
  static constexpr char int64_min_abs[]  = "9223372036854775808";
  static constexpr char int64_max_abs[]  = "9223372036854775807";

  auto digit_count = data_end - data_begin;

  // Remove preceding zeros
  if (digit_count >= (sizeof(int64_max_abs) - 1)) {
    // Trim zeros at the beginning of raw_data
    while (*data_begin == '0' && (data_begin < data_end)) {
      data_begin++;
    }
  }
  digit_count = data_end - data_begin;

  // After trimming the number of digits could be less than maximum
  // int64 digit count
  if (digit_count < (sizeof(int64_max_abs) - 1)) {  // CASE 0 : Accept validity
    // If the length of the string representing the integer is smaller
    // than string length of Int64Max then count this as an integer
    // representable by int64
    // If digit_count is 0 then ignore - sign, i.e. -000..00 should
    // be treated as a positive small integer
    return is_negative && (digit_count != 0) ? &stats.negative_small_int_count
                                             : &stats.positive_small_int_count;
  } else if (digit_count > (sizeof(uint64_max_abs) - 1)) {  // CASE 1 : Reject validity
    // If the length of the string representing the integer is greater
    // than string length of UInt64Max then count this as a string
    // since it cannot be represented as an int64 or uint64
    return &stats.string_count;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1) && is_negative) {
    // A negative integer of length UInt64Max digit count cannot be represented
    // as a 64 bit integer
    return &stats.string_count;
  }

  if (digit_count == (sizeof(int64_max_abs) - 1) && is_negative) {
    return less_equal_than(data_begin, int64_min_abs) ? &stats.negative_small_int_count
                                                      : &stats.string_count;
  } else if (digit_count == (sizeof(int64_max_abs) - 1) && !is_negative) {
    return less_equal_than(data_begin, int64_max_abs) ? &stats.positive_small_int_count
                                                      : &stats.big_int_count;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1)) {
    return less_equal_than(data_begin, uint64_max_abs) ? &stats.big_int_count : &stats.string_count;
  }

  return &stats.string_count;
}

}  // namespace gpu

/**
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 */
__inline__ __device__ bool is_whitespace(char ch) { return ch == '\t' || ch == ' '; }

/**
 * @brief Skips past the current character if it matches the given value.
 */
template <typename It>
__inline__ __device__ It skip_character(It const& it, char ch)
{
  return it + (*it == ch);
}

/**
 * @brief Adjusts the range to ignore starting/trailing whitespace and quotation characters.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end Pointer to the first character after the parsing range
 * @param quotechar The character used to denote quotes; '\0' if none
 *
 * @return Trimmed range
 */
__inline__ __device__ cuda::std::pair<char const*, char const*> trim_whitespaces_quotes(
  char const* begin, char const* end, char quotechar = '\0')
{
  auto not_whitespace = [] __device__(auto c) { return !is_whitespace(c); };

  auto const trim_begin = thrust::find_if(thrust::seq, begin, end, not_whitespace);
  auto const trim_end   = thrust::find_if(thrust::seq,
                                        cuda::std::make_reverse_iterator(end),
                                        cuda::std::make_reverse_iterator(trim_begin),
                                        not_whitespace);

  auto const trimmed_begin = skip_character(trim_begin, quotechar);
  auto const trimmed_end   = skip_character(trim_end, quotechar).base();
  // A lone quote character would otherwise be skipped from both ends, leaving end < begin
  return {trimmed_begin, cuda::std::max(trimmed_begin, trimmed_end)};
}

/**
 * @brief Adjusts the range to ignore starting/trailing whitespace characters.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end Pointer to the first character after the parsing range
 *
 * @return Trimmed range
 */
__inline__ __device__ cuda::std::pair<char const*, char const*> trim_whitespaces(char const* begin,
                                                                                 char const* end)
{
  auto not_whitespace = [] __device__(auto c) { return !is_whitespace(c); };

  auto const trim_begin = thrust::find_if(thrust::seq, begin, end, not_whitespace);
  auto const trim_end   = thrust::find_if(thrust::seq,
                                        cuda::std::make_reverse_iterator(end),
                                        cuda::std::make_reverse_iterator(trim_begin),
                                        not_whitespace);

  return {trim_begin, trim_end.base()};
}

/**
 * @brief Adjusts the range to ignore starting/trailing quotation characters.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end Pointer to the first character after the parsing range
 * @param quotechar The character used to denote quotes. Provide '\0' if no quotes should be
 * trimmed.
 *
 * @return Trimmed range
 */
__inline__ __device__ cuda::std::pair<char const*, char const*> trim_quotes(char const* begin,
                                                                            char const* end,
                                                                            char quotechar)
{
  if ((cuda::std::distance(begin, end) >= 2 && *begin == quotechar &&
       *cuda::std::prev(end) == quotechar)) {
    cuda::std::advance(begin, 1);
    cuda::std::advance(end, -1);
  }
  return {begin, end};
}

struct ConvertFunctor {
  /**
   * @brief Dispatch for numeric types whose values can be convertible to
   * 0 or 1 to represent boolean false/true, based upon checking against a
   * true/false values list.
   *
   * @return bool Whether the parsed value is valid.
   */
  template <typename T,
            CUDF_ENABLE_IF(std::is_integral_v<T> and !std::is_same_v<T, bool> and
                           !cudf::is_fixed_point<T>())>
  __device__ __forceinline__ bool operator()(char const* begin,
                                             char const* end,
                                             void* out_buffer,
                                             size_t row,
                                             data_type const output_type,
                                             parse_options_view const& opts,
                                             bool as_hex = false)
  {
    auto const value = [as_hex, &opts, begin, end]() -> cuda::std::optional<T> {
      // Check for user-specified true/false values
      auto const field_len = static_cast<size_t>(end - begin);
      if (serialized_trie_contains(opts.trie_true, {begin, field_len})) { return 1; }
      if (serialized_trie_contains(opts.trie_false, {begin, field_len})) { return 0; }
      return as_hex ? cudf::io::parse_numeric<T, 16>(begin, end, opts)
                    : cudf::io::parse_numeric<T>(begin, end, opts);
    }();
    if (value.has_value()) { static_cast<T*>(out_buffer)[row] = *value; }

    return value.has_value();
  }

  /**
   * @brief Dispatch for fixed point types.
   *
   * @return bool Whether the parsed value is valid.
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                      char const* end,
                                                      void* out_buffer,
                                                      size_t row,
                                                      data_type const output_type,
                                                      parse_options_view const& opts,
                                                      bool as_hex)
  {
    // TODO decide what's invalid input and update parsing functions
    static_cast<device_storage_type_t<T>*>(out_buffer)[row] =
      [&opts, output_type, begin, end]() -> device_storage_type_t<T> {
      return strings::detail::parse_decimal<device_storage_type_t<T>>(
        begin, end, output_type.scale());
    }();

    return true;
  }

  /**
   * @brief Dispatch for boolean type types.
   */
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, bool>)>
  __device__ __forceinline__ bool operator()(char const* begin,
                                             char const* end,
                                             void* out_buffer,
                                             size_t row,
                                             data_type const output_type,
                                             parse_options_view const& opts,
                                             bool as_hex)
  {
    auto const value = [&opts, begin, end]() -> cuda::std::optional<T> {
      // Check for user-specified true/false values
      auto const field_len = static_cast<size_t>(end - begin);
      if (serialized_trie_contains(opts.trie_true, {begin, field_len})) {
        return static_cast<T>(true);
      }
      if (serialized_trie_contains(opts.trie_false, {begin, field_len})) {
        return static_cast<T>(false);
      }
      return cudf::io::parse_numeric<T>(begin, end, opts);
    }();
    if (value.has_value()) { static_cast<T*>(out_buffer)[row] = *value; }

    return value.has_value();
  }

  /**
   * @brief Dispatch for floating points, which are set to NaN if the input
   * is not valid. In such case, the validity mask is set to zero too.
   */
  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  __device__ __forceinline__ bool operator()(char const* begin,
                                             char const* end,
                                             void* out_buffer,
                                             size_t row,
                                             data_type const output_type,
                                             parse_options_view const& opts,
                                             bool as_hex)
  {
    auto const value = [&opts, begin, end]() -> cuda::std::optional<T> {
      // Check for user-specified true/false values
      auto const field_len = static_cast<size_t>(end - begin);
      if (serialized_trie_contains(opts.trie_true, {begin, field_len})) {
        return static_cast<T>(true);
      }
      if (serialized_trie_contains(opts.trie_false, {begin, field_len})) {
        return static_cast<T>(false);
      }
      return cudf::io::parse_numeric<T>(begin, end, opts);
    }();
    if (value.has_value()) { static_cast<T*>(out_buffer)[row] = *value; }

    return value.has_value() and !std::isnan(*value);
  }

  /**
   * @brief Dispatch for remaining supported types, i.e., timestamp and duration types.
   */
  template <typename T,
            CUDF_ENABLE_IF(!std::is_integral_v<T> and !std::is_floating_point_v<T> and
                           !cudf::is_fixed_point<T>())>
  __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                      char const* end,
                                                      void* out_buffer,
                                                      size_t row,
                                                      data_type const output_type,
                                                      parse_options_view const& opts,
                                                      bool as_hex)
  {
    // TODO decide what's invalid input and update parsing functions
    if constexpr (cudf::is_timestamp<T>()) {
      static_cast<T*>(out_buffer)[row] = to_timestamp<T>(begin, end, opts.dayfirst);
    } else if constexpr (cudf::is_duration<T>()) {
      static_cast<T*>(out_buffer)[row] = to_duration<T>(begin, end);
    } else {
      return false;
    }
    return true;
  }
};

}  // namespace io
}  // namespace cudf
