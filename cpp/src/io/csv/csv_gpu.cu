/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "csv_common.hpp"
#include "csv_gpu.hpp"
#include "io/utilities/block_utils.cuh"
#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/trie.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/convert/fixed_point.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/algorithm>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/count.h>
#include <thrust/detail/copy.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cstddef>
#include <type_traits>

using namespace ::cudf::io;

using cudf::device_span;
using cudf::detail::grid_1d;

namespace cudf {
namespace io {
namespace csv {
namespace gpu {

/// Block dimension for dtype detection and conversion kernels
constexpr uint32_t csvparse_block_dim = 128;

/*
 * @brief Returns true is the input character is a valid digit.
 * Supports both decimal and hexadecimal digits (uppercase and lowercase).
 *
 * @param c Character to check
 * @param is_hex Whether to check as a hexadecimal
 *
 * @return `true` if it is digit-like, `false` otherwise
 */
__device__ __inline__ bool is_digit(char c, bool is_hex = false)
{
  if (c >= '0' && c <= '9') return true;

  if (is_hex) {
    if (c >= 'A' && c <= 'F') return true;
    if (c >= 'a' && c <= 'f') return true;
  }

  return false;
}

/*
 * @brief Checks whether the given character counters indicate a potentially
 * valid date and/or time field.
 *
 * For performance and simplicity, we detect only the most common date
 * formats. Example formats that are detectable:
 *
 *    `2001/02/30`
 *    `2001-02-30 00:00:00`
 *    `2/30/2001 T04:05:60.7`
 *    `2 / 1 / 2011`
 *    `02/January`
 *
 * @param len Number of non special-symbol or numeric characters
 * @param decimal_count Number of '.' characters
 * @param colon_count Number of ':' characters
 * @param dash_count Number of '-' characters
 * @param slash_count Number of '/' characters
 *
 * @return `true` if it is date-like, `false` otherwise
 */
__device__ __inline__ bool is_datetime(
  long len, long decimal_count, long colon_count, long dash_count, long slash_count)
{
  // Must not exceed count of longest month (September) plus `T` time indicator
  if (len > 10) { return false; }
  // Must not exceed more than one decimals or more than two time separators
  if (decimal_count > 1 || colon_count > 2) { return false; }
  // Must have one or two '-' or '/' but not both as date separators
  if ((dash_count > 0 && dash_count < 3 && slash_count == 0) ||
      (dash_count == 0 && slash_count > 0 && slash_count < 3)) {
    return true;
  }

  return false;
}

/*
 * @brief Returns true if the counters indicate a potentially valid float.
 * False positives are possible because positions are not taken into account.
 * For example, field "e.123-" would match the pattern.
 *
 * @param len Number of non special-symbol or numeric characters
 * @param digit_count Number of digits characters
 * @param decimal_count Number of occurrences of the decimal point character
 * @param thousands_count Number of occurrences of the thousands separator character
 * @param dash_count Number of '-' characters
 * @param exponent_count Number of 'e or E' characters
 *
 * @return `true` if it is floating point-like, `false` otherwise
 */
__device__ __inline__ bool is_floatingpoint(long len,
                                            long digit_count,
                                            long decimal_count,
                                            long thousands_count,
                                            long dash_count,
                                            long exponent_count)
{
  // Can't have more than one exponent and one decimal point
  if (decimal_count > 1) return false;
  if (exponent_count > 1) return false;

  // Without the exponent or a decimal point, this is an integer, not a float
  if (decimal_count == 0 && exponent_count == 0) return false;

  // Can only have one '-' per component
  if (dash_count > 1 + exponent_count) return false;

  // If anything other than these characters is present, it's not a float
  if (digit_count + decimal_count + dash_count + exponent_count + thousands_count != len) {
    return false;
  }

  // Needs at least 1 digit, 2 if exponent is present
  if (digit_count < 1 + exponent_count) return false;

  return true;
}

namespace {
/// Number of counters in a column_type_histogram
constexpr int histogram_slots = sizeof(column_type_histogram) / sizeof(cudf::size_type);
static_assert(histogram_slots * sizeof(cudf::size_type) == sizeof(column_type_histogram));

/// Index of each counter within column_type_histogram
constexpr int slot_null     = offsetof(column_type_histogram, null_count) / sizeof(cudf::size_type);
constexpr int slot_float    = offsetof(column_type_histogram, float_count) / sizeof(cudf::size_type);
constexpr int slot_datetime = offsetof(column_type_histogram, datetime_count) / sizeof(cudf::size_type);
constexpr int slot_string   = offsetof(column_type_histogram, string_count) / sizeof(cudf::size_type);
constexpr int slot_bool     = offsetof(column_type_histogram, bool_count) / sizeof(cudf::size_type);
constexpr int slot_negative_small_int =
  offsetof(column_type_histogram, negative_small_int_count) / sizeof(cudf::size_type);
constexpr int slot_positive_small_int =
  offsetof(column_type_histogram, positive_small_int_count) / sizeof(cudf::size_type);
constexpr int slot_big_int =
  offsetof(column_type_histogram, big_int_count) / sizeof(cudf::size_type);

/**
 * @brief Adds one to counter `idx` of a counter array, aggregating across the lanes of the warp
 * that target the same counter so that only one atomic is issued per distinct counter.
 */
template <typename T>
__device__ __forceinline__ void warp_aggregated_increment(T* counters, int idx)
{
  auto const peers  = __match_any_sync(__activemask(), idx);
  auto const leader = __ffs(peers) - 1;
  if (static_cast<int>(threadIdx.x % cudf::detail::warp_size) == leader) {
    atomicAdd(&counters[idx], static_cast<T>(__popc(peers)));
  }
}

/**
 * @brief Marks row `rec_id` of a column as valid and counts it, aggregated across the warp.
 *
 * A warp covers 32 consecutive, word-aligned rows (the grid is 1D with a multiple-of-32 block
 * size), so all lanes that mark the same column target the same bitmask word, with bit == lane.
 * One lane per column sets all of their bits and adds their count with a single atomic each.
 */
__device__ __forceinline__ void set_valid_warp_aggregated(cudf::bitmask_type* valid_mask,
                                                          size_type* valid_count,
                                                          int column,
                                                          size_type rec_id)
{
  static_assert(csvparse_block_dim % cudf::detail::warp_size == 0);
  auto const peers  = __match_any_sync(__activemask(), column);
  auto const leader = __ffs(peers) - 1;
  if (static_cast<int>(threadIdx.x % cudf::detail::warp_size) == leader) {
    atomicOr(&valid_mask[cudf::word_index(rec_id)], peers);
    atomicAdd(valid_count, static_cast<size_type>(__popc(peers)));
  }
}

/**
 * @brief Equivalent of `cudf::io::gpu::seek_field_end` (without escape characters) that reads the
 * row in aligned 8-byte words instead of one character at a time.
 *
 * `data_begin` is the start of the buffer that contains the row; no memory before it is read.
 *
 * In thread-per-row kernels each thread streams through its own row, so byte-wise loads issue
 * one memory transaction per character per thread; word loads cut that by up to 8x.
 *
 * If `may_have_quote_pair` is given, it is set to false only when the field starts with a quote
 * and no two adjacent characters of `[begin, field end)` are both quote characters (so the
 * field has no escaped quotes to unescape); otherwise it is set to true.
 */
/// Word source of `seek_field_end_by_words` that loads every word
struct direct_word_reader {
  __device__ __forceinline__ void start_field(char const*) {}
  __device__ __forceinline__ char first_char(char const* begin, char const*, char const*)
  {
    return *begin;
  }
  __device__ __forceinline__ char at(char const* pos) const { return *pos; }
  __device__ __forceinline__ uint64_t load(char const* word_begin)
  {
    return *reinterpret_cast<uint64_t const*>(word_begin);
  }
};

/**
 * @brief Word source of `seek_field_end_by_words` for one thread's row that reuses the last
 * loaded word (the word holding a field's end usually also holds the next field's start) and
 * keeps the first words of the current field for parsing (`window`).
 */
struct caching_word_reader {
  char const* cached_begin = nullptr;
  uint64_t cached          = 0;
  cudf::io::field_window window{};

  __device__ __forceinline__ void start_field(char const* begin)
  {
    window.base =
      reinterpret_cast<char const*>(reinterpret_cast<uintptr_t>(begin) & ~uintptr_t{7});
    window.count = 0;
  }
  /// `*pos`, from the current field's words when they hold it
  __device__ __forceinline__ char at(char const* pos) const
  {
    auto const offset = pos - window.base;
    if (window.base != nullptr && offset >= 0 && offset < 8 * window.count) {
      auto const word = offset < 8    ? window.w0
                        : offset < 16 ? window.w1
                        : offset < 24 ? window.w2
                                      : window.w3;
      return static_cast<char>(word >> (8 * (offset & 7)));
    }
    return *pos;
  }
  /// `*begin`, read through the word cache when the word lies within `[data_begin, end)`
  __device__ __forceinline__ char first_char(char const* begin,
                                             char const* data_begin,
                                             char const* end)
  {
    auto const word_begin =
      reinterpret_cast<char const*>(reinterpret_cast<uintptr_t>(begin) & ~uintptr_t{7});
    if (word_begin >= data_begin && word_begin + 8 <= end) {
      return static_cast<char>(load(word_begin) >> (8 * (begin - word_begin)));
    }
    return *begin;
  }
  __device__ __forceinline__ uint64_t load(char const* word_begin)
  {
    if (word_begin != cached_begin) {
      cached       = *reinterpret_cast<uint64_t const*>(word_begin);
      cached_begin = word_begin;
    }
    // Only words that continue the window without a gap are kept
    if (window.count < 4 && word_begin == window.base + 8 * window.count) {
      if (window.count == 0) { window.w0 = cached; }
      if (window.count == 1) { window.w1 = cached; }
      if (window.count == 2) { window.w2 = cached; }
      if (window.count == 3) { window.w3 = cached; }
      ++window.count;
    }
    return cached;
  }
};

template <typename WordReader = direct_word_reader>
__device__ __forceinline__ char const* seek_field_end_by_words(
  char const* begin,
  char const* end,
  char const* data_begin,
  parse_options_view const& opts,
  bool* may_have_quote_pair = nullptr,
  WordReader&& words        = WordReader{})
{
  words.start_field(begin);
  if (may_have_quote_pair != nullptr) { *may_have_quote_pair = true; }
  if (opts.multi_delimiter) { return cudf::io::gpu::seek_field_end(begin, end, opts); }

  bool const field_starts_with_quote =
    (begin < end && words.first_char(begin, data_begin, end) == opts.quotechar);
  bool quotation         = false;
  bool previous_is_quote = false;
  bool quote_pair        = false;
  // Returns true if the field ends at character `c`, located at `pos`
  auto const ends_field = [&](char c, char const* pos) {
    if (field_starts_with_quote) {
      if (c == opts.quotechar) {
        quote_pair        = quote_pair || previous_is_quote;
        previous_is_quote = true;
        quotation         = !quotation;
        return false;
      }
      previous_is_quote = false;
    }
    if (quotation) { return false; }
    return c == opts.delimiter || c == opts.terminator ||
           (c == '\r' && pos + 1 < end && pos[1] == '\n');
  };

  auto const field_end = [&](char const* pos) {
    if (may_have_quote_pair != nullptr) {
      *may_have_quote_pair = !field_starts_with_quote || quote_pair;
    }
    return pos;
  };

  auto current = begin;
  while (current < end) {
    auto const word_begin =
      reinterpret_cast<char const*>(reinterpret_cast<uintptr_t>(current) & ~uintptr_t{7});
    // Words must lie within [data_begin, end); the data buffer itself need not be aligned
    if (word_begin >= data_begin && word_begin + 8 <= end) {
      auto const word  = words.load(word_begin);
      auto const first = static_cast<int>(current - word_begin);
      if (field_starts_with_quote) {
        for (auto i = first; i < 8; ++i) {
          if (ends_field(static_cast<char>(word >> (8 * i)), word_begin + i)) {
            return field_end(word_begin + i);
          }
        }
      } else {
        // Unquoted field: only a delimiter, a terminator or '\r' can end it. Flag candidate bytes
        // with a zero-byte test; the lowest flag is exact, and every flag is verified below
        constexpr uint64_t ones  = 0x0101'0101'0101'0101ULL;
        constexpr uint64_t highs = 0x8080'8080'8080'8080ULL;
        auto const zero_bytes    = [](uint64_t v) { return (v - ones) & ~v & highs; };
        auto const repeat_byte   = [](char c) { return ones * static_cast<uint8_t>(c); };
        auto candidates          = (zero_bytes(word ^ repeat_byte(opts.delimiter)) |
                           zero_bytes(word ^ repeat_byte(opts.terminator)) |
                           zero_bytes(word ^ repeat_byte('\r'))) &
                          (~uint64_t{0} << (8 * first));
        while (candidates != 0) {
          auto const i = (__ffsll(static_cast<long long>(candidates)) - 1) / 8;
          if (ends_field(static_cast<char>(word >> (8 * i)), word_begin + i)) {
            return field_end(word_begin + i);
          }
          candidates &= candidates - 1;
        }
      }
      current = word_begin + 8;
    } else {
      if (ends_field(*current, current)) { return field_end(current); }
      ++current;
    }
  }
  return field_end(current);
}
}  // namespace

/*
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param opts A set of parsing options
 * @param csv_text The entire CSV data to read
 * @param column_flags Per-column parsing behavior flags
 * @param row_offsets The start the CSV data of interest
 * @param d_column_data The count for each column data type
 */
/**
 * @brief Returns the index of the column_type_histogram counter that a field falls into
 *
 * @param opts A set of parsing options
 * @param field_start Pointer to the first character of the field
 * @param next_delimiter Pointer to the character that ends the field (see `seek_field_end`)
 * @param as_datetime Whether the column is parsed as datetime
 */
/**
 * @brief Same classification as `cudf::io::gpu::infer_integral_field_counter`, returning the
 * histogram slot directly (no counter pointer, so no histogram object in local memory)
 */
__device__ __forceinline__ int integral_field_slot(char const* data_begin,
                                                   char const* data_end,
                                                   bool is_negative)
{
  static constexpr char uint64_max_abs[] = "18446744073709551615";
  static constexpr char int64_min_abs[]  = "9223372036854775808";
  static constexpr char int64_max_abs[]  = "9223372036854775807";
  using cudf::io::gpu::less_equal_than;

  auto digit_count = data_end - data_begin;
  // Remove preceding zeros
  if (digit_count >= (sizeof(int64_max_abs) - 1)) {
    while (*data_begin == '0' && (data_begin < data_end)) {
      data_begin++;
    }
  }
  digit_count = data_end - data_begin;

  if (digit_count < (sizeof(int64_max_abs) - 1)) {
    return is_negative && (digit_count != 0) ? slot_negative_small_int : slot_positive_small_int;
  } else if (digit_count > (sizeof(uint64_max_abs) - 1)) {
    return slot_string;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1) && is_negative) {
    return slot_string;
  }
  if (digit_count == (sizeof(int64_max_abs) - 1) && is_negative) {
    return less_equal_than(data_begin, int64_min_abs) ? slot_negative_small_int : slot_string;
  } else if (digit_count == (sizeof(int64_max_abs) - 1) && !is_negative) {
    return less_equal_than(data_begin, int64_max_abs) ? slot_positive_small_int : slot_big_int;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1)) {
    return less_equal_than(data_begin, uint64_max_abs) ? slot_big_int : slot_string;
  }
  return slot_string;
}

/**
 * @brief Classifies a non-empty, trimmed field from its character counts.
 *
 * @tparam Count Counter type; must be able to hold the field length
 */
template <typename Count>
__device__ int classify_trimmed_field(parse_options_view const& opts,
                                      cuda::std::pair<char const*, char const*> trimmed_field_range,
                                      bool as_datetime,
                                      device_span<char const> data,
                                      bool scan_by_words)
{
  auto const trimmed_field_len = trimmed_field_range.second - trimmed_field_range.first;
  Count count_number    = 0;
  Count count_decimal   = 0;
  Count count_thousands = 0;
  Count count_slash     = 0;
  Count count_dash      = 0;
  Count count_plus      = 0;
  Count count_colon     = 0;
  Count count_string    = 0;
  Count count_exponent  = 0;

  // Counts one character; returns true once the field can only be a string (any other
  // character; datetime columns accept up to 10 of them)
  auto const count_char = [&](char c, char const* cur) {
    if (is_digit(c)) {
      count_number++;
    } else if (c == opts.decimal) {
      count_decimal++;
    } else if (c == opts.thousands) {
      count_thousands++;
    } else {
      // Looking for unique characters that will help identify column types.
      switch (c) {
        case '-': count_dash++; break;
        case '+': count_plus++; break;
        case '/': count_slash++; break;
        case ':': count_colon++; break;
        case 'e':
        case 'E':
          if (cur > trimmed_field_range.first && cur < trimmed_field_range.second - 1)
            count_exponent++;
          break;
        default: count_string++; break;
      }
    }
    return count_string > (as_datetime ? 10 : 0);
  };
  // Fields of long rows are scanned in aligned 8-byte words (never outside `data`), bytewise
  // at the edges: with one thread per row, long rows do not stay L1-resident and bytewise
  // reads then cost one memory transaction each. Short rows are cheaper to read bytewise.
  auto const data_end = data.data() + data.size();
  auto cur            = trimmed_field_range.first;
  bool done           = false;
  if (!scan_by_words) {
    for (; !done && cur < trimmed_field_range.second; ++cur) {
      done = count_char(*cur, cur);
    }
  }
  while (!done && cur < trimmed_field_range.second) {
    auto const word_begin =
      reinterpret_cast<char const*>(reinterpret_cast<uintptr_t>(cur) & ~uintptr_t{7});
    if (word_begin >= data.data() && word_begin + 8 <= data_end) {
      auto const word = *reinterpret_cast<uint64_t const*>(word_begin);
      auto const last =
        static_cast<int>(cuda::std::min<ptrdiff_t>(8, trimmed_field_range.second - word_begin));
      for (auto i = static_cast<int>(cur - word_begin); i < last && !done; ++i) {
        done = count_char(static_cast<char>(word >> (8 * i)), word_begin + i);
      }
      cur = word_begin + last;
    } else {
      done = count_char(*cur, cur);
      ++cur;
    }
  }

  // Integers have to have the length of the string
  // Off by one if they start with a minus sign
  auto const int_req_number_cnt =
    trimmed_field_len - count_thousands -
    ((*trimmed_field_range.first == '-' || *trimmed_field_range.first == '+') &&
     trimmed_field_len > 1);

  if (as_datetime) {
    // PANDAS uses `object` dtype if the date is unparseable
    if (is_datetime(count_string, count_decimal, count_colon, count_dash, count_slash)) {
      return slot_datetime;
    } else {
      return slot_string;
    }
  } else if (count_number == int_req_number_cnt) {
    auto const is_negative = (*trimmed_field_range.first == '-');
    auto const data_begin =
      trimmed_field_range.first + (is_negative || (*trimmed_field_range.first == '+'));
    return integral_field_slot(data_begin, data_begin + count_number, is_negative);
  } else if (is_floatingpoint(trimmed_field_len,
                              count_number,
                              count_decimal,
                              count_thousands,
                              count_dash + count_plus,
                              count_exponent)) {
    return slot_float;
  } else {
    return slot_string;
  }
}

template <typename Count>
__device__ int classify_field(parse_options_view const& opts,
                              char const* field_start,
                              char const* next_delimiter,
                              bool as_datetime,
                              device_span<char const> data,
                              bool scan_by_words)
{
  auto const field_len = static_cast<size_t>(next_delimiter - field_start);
  if (serialized_trie_contains(opts.trie_na, {field_start, field_len})) {
    return slot_null;
  } else if (serialized_trie_contains(opts.trie_true, {field_start, field_len}) ||
             serialized_trie_contains(opts.trie_false, {field_start, field_len})) {
    return slot_bool;
  } else if (cudf::io::is_infinity(field_start, next_delimiter)) {
    return slot_float;
  } else {
    // Modify field_start & end to ignore whitespace and quotechars
    // This could possibly result in additional empty fields
    auto const trimmed_field_range = trim_whitespaces_quotes(field_start, next_delimiter);
    auto const trimmed_field_len   = trimmed_field_range.second - trimmed_field_range.first;

    if (trimmed_field_len == 0) { return slot_string; }
    return classify_trimmed_field<Count>(
      opts, trimmed_field_range, as_datetime, data, scan_by_words);
  }
}

/**
 * @brief Decodes an integer-classified field exactly as `convert_csv_to_cudf` would decode it into
 * an INT64 or UINT64 column, and records the value and its validity.
 *
 * Mirrors the conversion of a field that is not an N/A value: trim whitespace and quotes, map
 * user-specified true/false values to 1/0, otherwise parse as a decimal integer. Parsing as
 * uint64_t yields the same bits as parsing as int64_t for every value representable in the
 * column's final (INT64 or UINT64) type.
 */
__device__ __forceinline__ void predecode_integer_field(parse_options_view const& opts,
                                                        char const* field_start,
                                                        char const* field_end,
                                                        uint64_t* values,
                                                        cudf::bitmask_type* valid_mask,
                                                        size_type* valid_count,
                                                        int column,
                                                        size_type rec_id)
{
  auto const trimmed   = trim_whitespaces_quotes(field_start, field_end, opts.quotechar);
  auto const field_len = static_cast<size_t>(trimmed.second - trimmed.first);
  cuda::std::optional<uint64_t> value;
  if (serialized_trie_contains(opts.trie_true, {trimmed.first, field_len})) {
    value = 1;
  } else if (serialized_trie_contains(opts.trie_false, {trimmed.first, field_len})) {
    value = 0;
  } else {
    value = cudf::io::parse_numeric<uint64_t>(trimmed.first, trimmed.second, opts);
  }
  if (value.has_value()) {
    values[rec_id] = *value;
    set_valid_warp_aggregated(valid_mask, valid_count, column, rec_id);
  }
}

__device__ size_t stage_block_rows(device_span<char const> data,
                                   device_span<uint64_t const> row_offsets,
                                   char* smem,
                                   size_t smem_size);

/**
 * @tparam Count Type of the per-field character counters; must hold the length of any field
 */
template <typename Count>
CUDF_KERNEL void __launch_bounds__(csvparse_block_dim)
  data_type_detection(parse_options_view const opts,
                      device_span<char const> csv_text,
                      device_span<column_parse::flags const> const column_flags,
                      device_span<uint64_t const> const row_offsets,
                      device_span<column_type_histogram> d_column_data,
                      bool use_shared_histogram,
                      device_span<uint64_t* const> int_values,
                      device_span<cudf::bitmask_type* const> int_valids,
                      device_span<size_type> int_valid_counts,
                      size_t stage_size)
{
  // Dynamic shared memory: the per-block histogram (when it fits), flushed to global memory at the
  // end, followed by the staged copy of the block's rows (when `stage_size` != 0)
  extern __shared__ uint4 detection_smem[];
  auto* const s_counters  = reinterpret_cast<cudf::size_type*>(detection_smem);
  auto const num_counters = static_cast<int>(d_column_data.size()) * histogram_slots;
  if (use_shared_histogram) {
    for (int i = threadIdx.x; i < num_counters; i += blockDim.x) {
      s_counters[i] = 0;
    }
    __syncthreads();
  }
  auto* const counters = use_shared_histogram
                           ? s_counters
                           : reinterpret_cast<cudf::size_type*>(d_column_data.data());

  // Parse from a staged shared-memory copy of this block's rows when it fits (coalesced loads
  // instead of each thread streaming its own row), from global memory otherwise
  auto const histogram_bytes =
    use_shared_histogram
      ? util::round_up_safe<size_t>(num_counters * sizeof(cudf::size_type), sizeof(uint4))
      : 0;
  auto* const stage        = reinterpret_cast<char*>(detection_smem) + histogram_bytes;
  auto const staged_begin  = stage_block_rows(csv_text, row_offsets, stage, stage_size);
  bool const is_staged     = staged_begin != csv_text.size();
  auto const base_offset   = is_staged ? staged_begin : 0;
  auto const text          = is_staged ? device_span<char const>{stage, stage_size} : csv_text;
  char const* const raw_csv = text.data();

  // ThreadIds range per block, so also need the blockId
  // This is entry into the fields; threadId is an element within `num_records`
  auto const rec_id      = grid_1d::global_thread_id();
  auto const rec_id_next = rec_id + 1;

  // we can have more threads than data; such threads only take part in the histogram flush
  bool const has_row = rec_id_next < row_offsets.size();

  auto field_start   = raw_csv + (has_row ? row_offsets[rec_id] - base_offset : 0);
  auto const row_end = raw_csv + (has_row ? row_offsets[rec_id_next] - base_offset : 0);

  // Rows longer than this are not expected to stay L1-resident (see classify_field)
  constexpr ptrdiff_t min_word_scan_row_len = 128;
  bool const scan_by_words                  = row_end - field_start > min_word_scan_row_len;

  auto next_field = field_start;
  int col         = 0;
  int actual_col  = 0;

  // Going through all the columns of a given record
  while (has_row && col < column_flags.size() && field_start < row_end) {
    // In delim_whitespace mode, collapse leading delimiter runs so leading whitespace does
    // not produce empty fields (matches pandas behavior).
    field_start = cudf::io::gpu::skip_leading_delimiter_run(field_start, row_end, opts);
    if (field_start >= row_end) break;
    auto next_delimiter = seek_field_end_by_words(field_start, row_end, raw_csv, opts);

    // Checking if this is a column that the user wants --- user can filter columns
    if (column_flags[col] & column_parse::inferred) {
      auto const slot = classify_field<Count>(opts,
                                       field_start,
                                       next_delimiter,
                                       column_flags[col] & column_parse::as_datetime,
                                       text,
                                       scan_by_words);
      warp_aggregated_increment(counters, actual_col * histogram_slots + slot);
      // Integer fields are also decoded, in case the column is inferred as an integer column
      if (not int_values.empty() and
          (slot == slot_negative_small_int or slot == slot_positive_small_int or
           slot == slot_big_int) and
          not(column_flags[col] & column_parse::as_hexadecimal)) {
        predecode_integer_field(opts,
                                field_start,
                                next_delimiter,
                                int_values[actual_col],
                                int_valids[actual_col],
                                &int_valid_counts[actual_col],
                                actual_col,
                                static_cast<size_type>(rec_id));
      }
      actual_col++;
    }
    next_field  = next_delimiter + 1;
    field_start = next_field;
    col++;
  }

  if (use_shared_histogram) {
    __syncthreads();
    auto* const g_counters = reinterpret_cast<cudf::size_type*>(d_column_data.data());
    for (int i = threadIdx.x; i < num_counters; i += blockDim.x) {
      if (s_counters[i] != 0) { atomicAdd(&g_counters[i], s_counters[i]); }
    }
  }
}

/**
 * @brief Collapses escaped quote pairs (`""` -> `"`) of a quoted field.
 *
 * Pairs are matched left to right without overlap, matching a `""` -> `"` string replace.
 * Fields without escaped pairs are returned unchanged; otherwise the unescaped field is written to
 * `out` (which may alias `begin`, since the output never overtakes the input).
 *
 * @param begin First character of the field content (after the opening quote)
 * @param end One past the last character of the field content (before the closing quote)
 * @param out Output location for the unescaped field
 * @param quotechar Quote character
 * @return Begin and end of the unescaped field
 */
__device__ __forceinline__ cuda::std::pair<char const*, char const*> unescape_doublequotes(
  char const* begin, char const* end, char* out, char quotechar)
{
  auto in = begin;
  while (in + 1 < end && !(in[0] == quotechar && in[1] == quotechar)) {
    ++in;
  }
  if (in + 1 >= end) { return {begin, end}; }
  auto out_it = out;
  if (out != begin) {
    for (auto it = begin; it < in; ++it) {
      *out_it++ = *it;
    }
  } else {
    out_it += in - begin;
  }
  while (in < end) {
    auto const c = *in;
    *out_it++    = c;
    in += (c == quotechar && in + 1 < end && in[1] == quotechar) ? 2 : 1;
  }
  return {out, out_it};
}

/**
 * @brief `serialized_trie_contains(trie, {key, key_len})`, reading the key through `words.at`
 */
template <typename WordReader>
__device__ __forceinline__ bool trie_contains(device_span<cudf::detail::serial_trie_node const> trie,
                                             char const* key,
                                             size_t key_len,
                                             WordReader const& words)
{
  if (trie.empty()) { return false; }
  if (key_len == 0) { return trie.front().is_leaf; }
  // The root node holds the longest key length (negative if unknown)
  auto const max_key_length = trie.front().children_offset;
  if (max_key_length >= 0 && key_len > static_cast<size_t>(max_key_length)) { return false; }
  auto curr_node = trie.begin() + 1;
  for (auto curr_key = key; curr_key < key + key_len; ++curr_key) {
    // Don't jump away from root node
    if (curr_key != key) {
      // A node without children cannot match a longer key
      if (curr_node->children_offset < 0) { return false; }
      curr_node += curr_node->children_offset;
    }
    auto const c = words.at(curr_key);
    // Nodes are sorted - terminate search if the node is larger or equal
    while (curr_node->character != cudf::detail::trie_terminating_character &&
           curr_node->character < c) {
      ++curr_node;
    }
    if (curr_node->character != c) { return false; }
  }
  return curr_node->is_leaf;
}

/**
 * @brief `trim_whitespaces_quotes(begin, end, quotechar)`, reading characters through `words.at`
 * (including the characters just outside the range that the original also reads)
 */
template <typename WordReader>
__device__ __forceinline__ cuda::std::pair<char const*, char const*> trim_whitespaces_quotes_at(
  char const* begin, char const* end, char quotechar, WordReader const& words)
{
  auto trim_begin = begin;
  while (trim_begin != end && is_whitespace(words.at(trim_begin))) {
    ++trim_begin;
  }
  auto trim_end = end;
  while (trim_end != trim_begin && is_whitespace(words.at(trim_end - 1))) {
    --trim_end;
  }
  auto const trimmed_begin = trim_begin + (words.at(trim_begin) == quotechar);
  auto const trimmed_end   = trim_end - (words.at(trim_end - 1) == quotechar);
  // A lone quote character would otherwise be skipped from both ends, leaving end < begin
  return {trimmed_begin, cuda::std::max(trimmed_begin, trimmed_end)};
}

template <bool WindowedIntegers, typename WordReader>
__device__ __forceinline__ ConvertFunctor make_convert_functor(WordReader const& words)
{
  if constexpr (WindowedIntegers) {
    return ConvertFunctor{true, words.window};
  } else {
    return ConvertFunctor{false};
  }
}

/**
 * @brief Cooperatively copies the contiguous character span covering this block's rows into
 * shared memory, so per-row parsing reads shared memory instead of scattered global loads.
 *
 * All threads of the block must call this function.
 *
 * @param data The entire CSV data
 * @param row_offsets Row start offsets (one extra entry marks the end of the last row)
 * @param smem Shared memory buffer, 16-byte aligned
 * @param smem_size Size of the shared memory buffer in bytes
 * @return Offset in `data` of the first staged character, or `data.size()` if the span does not
 * fit (the caller then reads from global memory)
 */
__device__ size_t stage_block_rows(device_span<char const> data,
                                   device_span<uint64_t const> row_offsets,
                                   char* smem,
                                   size_t smem_size)
{
  auto const num_rows = row_offsets.size() - 1;
  auto const first    = static_cast<size_t>(blockIdx.x) * blockDim.x;
  auto const last     = cuda::std::min<size_t>(first + blockDim.x, num_rows);
  if (smem_size == 0 || first >= last) { return data.size(); }
  auto const begin = row_offsets[first] & ~size_t{15};
  auto const end   = row_offsets[last];
  // The last 16-byte load may extend past `end`
  if (end - begin + 16 > smem_size) { return data.size(); }

  // 16-byte loads for complete aligned chunks inside `data`, bytes for the tail
  auto const num_vec = (cuda::std::min<size_t>(end + 15, data.size()) - begin) / 16;
  auto const src_vec = reinterpret_cast<uint4 const*>(data.data() + begin);
  auto const dst_vec = reinterpret_cast<uint4*>(smem);
  for (auto i = threadIdx.x; i < num_vec; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
  for (auto i = begin + num_vec * 16 + threadIdx.x; i < end; i += blockDim.x) {
    smem[i - begin] = data[i];
  }
  __syncthreads();
  return begin;
}

/**
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed one record at a time
 *
 * @param[in] options A set of parsing options
 * @param[in] data The entire CSV data to read
 * @param[in] column_flags Per-column parsing behavior flags
 * @param[in] row_offsets The start the CSV data of interest
 * @param[in] dtypes The data type of the column
 * @param[out] columns The output column data
 * @param[out] valids The bitmaps indicating whether column fields are valid
 * @param[out] valid_counts The number of valid fields in each column
 * @tparam WindowedIntegers Parse integers from a register copy of the field (for unstaged rows)
 */
template <bool WindowedIntegers>
CUDF_KERNEL void __launch_bounds__(csvparse_block_dim)
  convert_csv_to_cudf(cudf::io::parse_options_view options,
                      device_span<char const> data,
                      char* unescape_buffer,
                      device_span<column_parse::flags const> column_flags,
                      device_span<uint64_t const> row_offsets,
                      device_span<cudf::data_type const> dtypes,
                      device_span<void* const> columns,
                      device_span<cudf::bitmask_type* const> valids,
                      device_span<size_type> valid_counts,
                      size_t smem_size)
{
  extern __shared__ uint4 staged_rows[];
  auto const smem = reinterpret_cast<char*>(staged_rows);

  // Parse from the staged copy of this block's rows if it fits, from global memory otherwise
  auto const staged_begin = stage_block_rows(data, row_offsets, smem, smem_size);
  auto const is_staged    = staged_begin != data.size();
  auto const base_offset  = is_staged ? staged_begin : 0;
  char const* const raw_csv = is_staged ? smem : data.data();
  // Maps a parsing pointer back into the global data buffer
  auto const to_global = [&](char const* ptr) {
    return data.data() + base_offset + (ptr - raw_csv);
  };

  // thread IDs range per block, so also need the block id.
  // this is entry into the field array - tid is an elements within the num_entries array
  auto const rec_id      = grid_1d::global_thread_id();
  auto const rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of the data
  if (rec_id_next >= row_offsets.size()) return;

  auto field_start   = raw_csv + (row_offsets[rec_id] - base_offset);
  auto const row_end = raw_csv + (row_offsets[rec_id_next] - base_offset);

  auto next_field = field_start;
  int col         = 0;
  int actual_col  = 0;
  // Unstaged rows: reuse loaded words across fields and pass them on to integer parsing
  cuda::std::conditional_t<WindowedIntegers, caching_word_reader, direct_word_reader> words{};

  while (col < column_flags.size() && field_start < row_end) {
    // In delim_whitespace mode, collapse leading delimiter runs so leading whitespace does
    // not produce empty fields (matches pandas behavior).
    field_start = cudf::io::gpu::skip_leading_delimiter_run(field_start, row_end, options);
    if (field_start >= row_end) break;
    next_field          = field_start;
    bool may_have_quote_pair = true;
    auto next_delimiter      = seek_field_end_by_words(
      field_start, row_end, raw_csv, options, &may_have_quote_pair, words);

    if (column_flags[col] & column_parse::predecoded) {
      // Already decoded during type inference
      ++actual_col;
    } else if (column_flags[col] & column_parse::enabled) {
      // check if the entire field is a NaN string - consistent with pandas
      // Unstaged rows read the field's characters from the words the seek already loaded
      auto const field_len = static_cast<size_t>(next_delimiter - field_start);
      bool is_valid        = false;
      if constexpr (WindowedIntegers) {
        is_valid = !trie_contains(options.trie_na, field_start, field_len, words);
      } else {
        is_valid = !serialized_trie_contains(options.trie_na, {field_start, field_len});
      }

      // Modify field_start & end to ignore whitespace and quotechars
      auto field_end = next_delimiter;
      if (is_valid && dtypes[actual_col].id() != cudf::type_id::STRING) {
        auto const trimmed_field = [&] {
          if constexpr (WindowedIntegers) {
            return trim_whitespaces_quotes_at(field_start, field_end, options.quotechar, words);
          } else {
            return trim_whitespaces_quotes(field_start, field_end, options.quotechar);
          }
        }();
        field_start = trimmed_field.first;
        field_end   = trimmed_field.second;
      }
      if (is_valid) {
        // Type dispatcher does not handle STRING
        if (dtypes[actual_col].id() == cudf::type_id::STRING) {
          auto end        = next_delimiter;
          bool was_quoted = false;
          if (not options.keepquotes) {
            if (not options.detect_whitespace_around_quotes) {
              // A lone quote character is not a quoted (empty) string: stripping it would leave
              // a negative length
              if (end - field_start >= 2 && (words.at(field_start) == options.quotechar) &&
                  (words.at(end - 1) == options.quotechar)) {
                ++field_start;
                --end;
                was_quoted = true;
              }
            } else {
              // If the string is quoted, whitespace around the quotes get removed as well
              auto const trimmed_field = trim_whitespaces(field_start, end);
              if (trimmed_field.second - trimmed_field.first >= 2 &&
                  (*trimmed_field.first == options.quotechar) &&
                  (*(trimmed_field.second - 1) == options.quotechar)) {
                field_start = trimmed_field.first + 1;
                end         = trimmed_field.second - 1;
                was_quoted  = true;
              }
            }
          }
          // Unescape doubled quotes ("" -> ") in place. The field bytes belong only to this
          // (row, column), so compacting them within [field_start, end) is race-free.
          // Strings reference the global data buffer, not the staged copy
          auto global_start = to_global(field_start);
          auto global_end   = to_global(end);
          // The seek already rules out escaped quotes in most quoted fields
          if (was_quoted && may_have_quote_pair && unescape_buffer != nullptr) {
            auto const unescaped =
              unescape_doublequotes(global_start,
                                    global_end,
                                    unescape_buffer + (global_start - data.data()),
                                    options.quotechar);
            global_start = unescaped.first;
            global_end   = unescaped.second;
          }
          auto str_list = static_cast<std::pair<char const*, size_t>*>(columns[actual_col]);
          str_list[rec_id].first  = global_start;
          str_list[rec_id].second = global_end - global_start;
        } else {
          if (cudf::type_dispatcher(dtypes[actual_col],
                                    make_convert_functor<WindowedIntegers>(words),
                                    field_start,
                                    field_end,
                                    columns[actual_col],
                                    rec_id,
                                    dtypes[actual_col],
                                    options,
                                    column_flags[col] & column_parse::as_hexadecimal)) {
            // set the valid bitmap - all bits were set to 0 to start
            set_valid_warp_aggregated(
              valids[actual_col], &valid_counts[actual_col], actual_col, rec_id);
          }
        }
      } else if (dtypes[actual_col].id() == cudf::type_id::STRING) {
        auto str_list           = static_cast<std::pair<char const*, size_t>*>(columns[actual_col]);
        str_list[rec_id].first  = nullptr;
        str_list[rec_id].second = 0;
      }
      ++actual_col;
    }
    next_field  = next_delimiter + 1;
    field_start = next_field;
    ++col;
  }

  // Columns missing from this row are null. String data is not zero-initialized, so write null
  // strings for them (other types only need their validity bit, which stays unset).
  for (; col < column_flags.size(); ++col) {
    if (not(column_flags[col] & column_parse::enabled)) { continue; }
    if (not(column_flags[col] & column_parse::predecoded) and
        dtypes[actual_col].id() == cudf::type_id::STRING) {
      auto str_list           = static_cast<std::pair<char const*, size_t>*>(columns[actual_col]);
      str_list[rec_id].first  = nullptr;
      str_list[rec_id].second = 0;
    }
    ++actual_col;
  }
}

/*
 * @brief Merge two packed row contexts (each corresponding to a block of characters)
 * and return the packed row context corresponding to the merged character block
 */
inline __device__ packed_rowctx_t merge_row_contexts(packed_rowctx_t first_ctx,
                                                     packed_rowctx_t second_ctx)
{
  uint32_t id0 = get_row_context(first_ctx, ROW_CTX_NONE) & 3;
  uint32_t id1 = get_row_context(first_ctx, ROW_CTX_QUOTE) & 3;
  uint32_t id2 = get_row_context(first_ctx, ROW_CTX_COMMENT) & 3;
  return (first_ctx & ~pack_row_contexts(3, 3, 3)) +
         pack_row_contexts(get_row_context(second_ctx, id0),
                           get_row_context(second_ctx, id1),
                           get_row_context(second_ctx, id2));
}

/*
 * @brief Per-character context:
 * 1-bit count (0 or 1) per context in the lower 4 bits
 * 2-bit output context id per input context in bits 8..15
 */
constexpr __device__ uint32_t make_char_context(uint32_t id0,
                                                uint32_t id1,
                                                uint32_t id2 = ROW_CTX_COMMENT,
                                                uint32_t c0  = 0,
                                                uint32_t c1  = 0,
                                                uint32_t c2  = 0)
{
  return (id0 << 8) | (id1 << 10) | (id2 << 12) | (ROW_CTX_EOF << 14) | (c0) | (c1 << 1) |
         (c2 << 2);
}

/*
 * @brief Merge a 1-character context to keep track of bitmasks where new rows occur
 * Merges a single-character "block" row context at position pos with the current
 * block's row context (the current block contains 32-pos characters)
 *
 * @param ctx Current block context and new rows bitmaps
 * @param char_ctx state transitions associated with new character
 * @param pos Position within the current 32-character block
 *
 * NOTE: This is probably the most performance-critical piece of the row gathering kernel.
 * The char_ctx value should be created via make_char_context, and its value should
 * have been evaluated at compile-time.
 */
inline __device__ void merge_char_context(uint4& ctx, uint32_t char_ctx, uint32_t pos)
{
  uint32_t id0 = (ctx.w >> 0) & 3;
  uint32_t id1 = (ctx.w >> 2) & 3;
  uint32_t id2 = (ctx.w >> 4) & 3;
  // Set the newrow bit in the bitmap at the corresponding position
  ctx.x |= ((char_ctx >> id0) & 1) << pos;
  ctx.y |= ((char_ctx >> id1) & 1) << pos;
  ctx.z |= ((char_ctx >> id2) & 1) << pos;
  // Update the output context ids
  ctx.w = ((char_ctx >> (8 + id0 * 2)) & 0x03) | ((char_ctx >> (6 + id1 * 2)) & 0x0c) |
          ((char_ctx >> (4 + id2 * 2)) & 0x30) | (ROW_CTX_EOF << 6);
}

/*
 * Convert the context-with-row-bitmaps version to a packed row context
 */
inline __device__ packed_rowctx_t pack_rowmaps(uint4 ctx_map)
{
  return pack_row_contexts(make_row_context(__popc(ctx_map.x), (ctx_map.w >> 0) & 3),
                           make_row_context(__popc(ctx_map.y), (ctx_map.w >> 2) & 3),
                           make_row_context(__popc(ctx_map.z), (ctx_map.w >> 4) & 3));
}

/*
 * Selects the row bitmap corresponding to the given parser state
 */
inline __device__ uint32_t select_rowmap(uint4 ctx_map, uint32_t ctxid)
{
  return (ctxid == ROW_CTX_NONE)      ? ctx_map.x
         : (ctxid == ROW_CTX_QUOTE)   ? ctx_map.y
         : (ctxid == ROW_CTX_COMMENT) ? ctx_map.z
                                      : 0;
}

/**
 * @brief Single pair-wise 512-wide row context merge transform
 *
 * Merge row context blocks and record the merge operation in a context
 * tree so that the transform is reversible.
 * The tree is organized such that the left and right children of node n
 * are located at indices n*2 and n*2+1, the root node starting at index 1
 *
 * @tparam lanemask mask to specify source of packed row context
 * @tparam tmask mask to specify principle thread for merging row context
 * @tparam base start location for writing into packed row context tree
 * @tparam level_scale level of the node in the tree
 * @param[out] ctxtree packed row context tree
 * @param[in] ctxb packed row context for the current character block
 * @param t thread id (leaf node id)
 */
template <uint32_t lanemask, uint32_t tmask, uint32_t base, uint32_t level_scale>
inline __device__ void ctx_merge(uint64_t* ctxtree, packed_rowctx_t* ctxb, uint32_t t)
{
  uint64_t tmp = shuffle_xor(*ctxb, lanemask);
  if (!(t & tmask)) {
    *ctxb                              = merge_row_contexts(*ctxb, tmp);
    ctxtree[base + (t >> level_scale)] = *ctxb;
  }
}

/**
 * @brief Single 512-wide row context inverse merge transform
 *
 * Walks the context tree starting from a root node
 *
 * @tparam rmask Mask to specify which threads write input row context
 * @param[in] base Start read location of the merge transform tree
 * @param[in] ctxtree Merge transform tree
 * @param[in] ctx Input context
 * @param[in] brow4 output row in block *4
 * @param[in] t thread id (leaf node id)
 */
template <uint32_t rmask>
inline __device__ void ctx_unmerge(
  uint32_t base, uint64_t const* ctxtree, uint32_t* ctx, uint32_t* brow4, uint32_t t)
{
  rowctx32_t ctxb_left, ctxb_right, ctxb_sum;
  ctxb_sum   = get_row_context(ctxtree[base], *ctx);
  ctxb_left  = get_row_context(ctxtree[(base) * 2 + 0], *ctx);
  ctxb_right = get_row_context(ctxtree[(base) * 2 + 1], ctxb_left & 3);
  if (t & (rmask)) {
    *brow4 += (ctxb_sum & ~3) - (ctxb_right & ~3);
    *ctx = ctxb_left & 3;
  }
}

/*
 * @brief 512-wide row context merge transform
 *
 * Repeatedly merge row context blocks, keeping track of each merge operation
 * in a context tree so that the transform is reversible
 * The tree is organized such that the left and right children of node n
 * are located at indices n*2 and n*2+1, the root node starting at index 1
 *
 * Each node contains the counts and output contexts corresponding to the
 * possible input contexts.
 * Each parent node's count is obtained by adding the corresponding counts
 * from the left child node with the right child node's count selected from
 * the left child node's output context:
 *   parent.count[k] = left.count[k] + right.count[left.outctx[k]]
 *   parent.outctx[k] = right.outctx[left.outctx[k]]
 *
 * @param[out] ctxtree packed row context tree
 * @param[in] ctxb packed row context for the current character block
 * @param t thread id (leaf node id)
 */
static inline __device__ void rowctx_merge_transform(uint64_t* ctxtree,
                                                     packed_rowctx_t ctxb,
                                                     uint32_t t)
{
  ctxtree[512 + t] = ctxb;
  ctx_merge<1, 0x1, 256, 1>(ctxtree, &ctxb, t);
  ctx_merge<2, 0x3, 128, 2>(ctxtree, &ctxb, t);
  ctx_merge<4, 0x7, 64, 3>(ctxtree, &ctxb, t);
  ctx_merge<8, 0xf, 32, 4>(ctxtree, &ctxb, t);
  __syncthreads();
  if (t < 32) {
    ctxb = ctxtree[32 + t];
    ctx_merge<1, 0x1, 16, 1>(ctxtree, &ctxb, t);
    ctx_merge<2, 0x3, 8, 2>(ctxtree, &ctxb, t);
    ctx_merge<4, 0x7, 4, 3>(ctxtree, &ctxb, t);
    ctx_merge<8, 0xf, 2, 4>(ctxtree, &ctxb, t);
    // Final stage
    uint64_t tmp = shuffle_xor(ctxb, 16);
    if (t == 0) { ctxtree[1] = merge_row_contexts(ctxb, tmp); }
  }
}

/*
 * @brief 512-wide row context inverse merge transform
 *
 * Walks the context tree starting from the root node (index 1) using
 * the starting context in node index 0.
 * The return value is the starting row and input context for the given leaf node
 *
 * @param[in] ctxtree Merge transform tree
 * @param[in] t thread id (leaf node id)
 *
 * @return Final row context and count (row_position*4 + context_id format)
 */
static inline __device__ rowctx32_t
rowctx_inverse_merge_transform(uint64_t const* ctxtree, uint32_t t)
{
  uint32_t ctx     = ctxtree[0] & 3;  // Starting input context
  rowctx32_t brow4 = 0;               // output row in block *4

  ctx_unmerge<256>(1, ctxtree, &ctx, &brow4, t);
  ctx_unmerge<128>(2 + (t >> 8), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<64>(4 + (t >> 7), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<32>(8 + (t >> 6), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<16>(16 + (t >> 5), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<8>(32 + (t >> 4), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<4>(64 + (t >> 3), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<2>(128 + (t >> 2), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<1>(256 + (t >> 1), ctxtree, &ctx, &brow4, t);

  return brow4 + ctx;
}

constexpr auto bk_ctxtree_size = rowofs_block_dim * 2;

/**
 * @brief Computes, for one thread's 32 characters, the row start bitmaps and output parser
 * states for each possible input parser state (see gather_row_offsets_gpu).
 *
 * @param char_pos Position of the 32 characters relative to `parse_pos`
 * @param[out] block_pos Position of the thread's first character in `data`
 * @return {row bitmaps for input states NONE, QUOTE and COMMENT; packed output states}
 */
/**
 * @brief Returns the row context transition of an in-range character `c` preceded by `c_prev`
 * (see compute_char_contexts for the meaning of the states).
 */
__device__ __forceinline__ uint32_t char_context(
  int c, int c_prev, int terminator, int delimiter, int quotechar, int commentchar)
{
  if (c_prev == terminator) {
    if (c == commentchar) {
      // Start of a new comment row
      return make_char_context(ROW_CTX_COMMENT, ROW_CTX_QUOTE, ROW_CTX_COMMENT, 1, 0, 1);
    } else if (c == quotechar) {
      // Quoted string on newrow, or quoted string ending in terminator
      return make_char_context(ROW_CTX_QUOTE, ROW_CTX_NONE, ROW_CTX_QUOTE, 1, 0, 1);
    }
    // Start of a new row unless within a quote
    return make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE, 1, 0, 1);
  } else if (c == quotechar) {
    // Quote handling uses ROW_CTX_COMMENT as a "pending exit" state to correctly handle
    // escaped quotes (""). When in QUOTE state and we see a quote, we can't immediately
    // exit because it might be the first quote of a "" escape sequence. We transition to
    // COMMENT (pending exit) and wait for the next character:
    //   - If next char is quote: it's a "" escape, return to QUOTE
    //   - If next char is anything else: exit confirmed, go to NONE
    // This doesn't conflict with actual comment handling because comments are only
    // detected at row boundaries (after newline), where COMMENT state is set with row
    // counting. Mid-row, COMMENT is purely used for this pending exit mechanism.
    if (c_prev == delimiter) {
      // Quote after delimiter: start field or pending exit
      return make_char_context(ROW_CTX_QUOTE, ROW_CTX_COMMENT);
    } else if (c_prev == quotechar) {
      // Quote after quote: "" escape or stay NONE (Spark compatibility)
      return make_char_context(ROW_CTX_NONE, ROW_CTX_COMMENT, ROW_CTX_QUOTE);
    }
    // Quote after regular char: pending exit or stay NONE
    return make_char_context(ROW_CTX_NONE, ROW_CTX_COMMENT);
  }
  // Non-quote char: stay in current state, or exit from pending
  return make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE);
}

__device__ uint4 compute_char_contexts(device_span<char const> const data,
                                       size_t chunk_size,
                                       size_t parse_pos,
                                       size_t start_offset,
                                       size_t data_size,
                                       size_t byte_range_start,
                                       size_t char_pos,
                                       int terminator,
                                       int delimiter,
                                       int quotechar,
                                       int commentchar,
                                       size_t& block_pos)
{
  auto start = data.data();

  // file-level end position for this scan, clamped to the file size
  size_t const end_in_file = (parse_pos >= data_size || chunk_size > data_size - parse_pos)
                               ? data_size
                               : parse_pos + chunk_size;
  // offset into the local `data` window (which begins at `start_offset`), clamped to the buffer
  size_t const end_off      = end_in_file > start_offset ? end_in_file - start_offset : 0;
  size_t const data_end_off = data_size > start_offset ? data_size - start_offset : 0;
  auto const end            = start + cuda::std::min(end_off, data.size());
  auto const data_end       = start + cuda::std::min(data_end_off, data.size());
  // Offset of `parse_pos` inside the local `data` window, clamped to avoid underflow
  auto const parse_off = parse_pos > start_offset ? parse_pos - start_offset : 0;
  block_pos            = parse_off + char_pos;
  auto cur             = start + block_pos;

  // Initial state is neutral context (no state transitions), zero rows
  uint4 ctx_map = {
    .x = 0,
    .y = 0,
    .z = 0,
    .w = (ROW_CTX_NONE << 0) | (ROW_CTX_QUOTE << 2) | (ROW_CTX_COMMENT << 4) | (ROW_CTX_EOF << 6)};
  int c, c_prev = (cur > start && cur <= end) ? cur[-1] : terminator;
  // Fast path: 32 in-range characters without quote or comment characters. Then the only
  // transitions are row starts after each terminator: rows begin at the same positions in the
  // NONE and COMMENT (pending quote exit) contexts, which both end in NONE, and not in QUOTE.
  bool fast_path = false;
  if (cur + 32 <= end && (reinterpret_cast<uintptr_t>(cur) % 16) == 0) {
    auto const v0 = reinterpret_cast<uint4 const*>(cur)[0];
    auto const v1 = reinterpret_cast<uint4 const*>(cur)[1];
    uint32_t const words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    auto const bytes4       = [](int ch) { return static_cast<uint32_t>(ch & 0xff) * 0x0101'0101u; };
    uint32_t special        = 0;
    uint32_t terminators    = 0;
    uint32_t quotes         = 0;
    // One bit per byte: gather the top bit of each 0xff byte into 4 consecutive bits
    auto const to_bits = [](uint32_t mask) {
      return ((((mask >> 7) & 0x0101'0101u) * 0x0020'4081u) >> 21) & 0xf;
    };
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (quotechar < 0x100) {
        auto const q = __vcmpeq4(words[i], bytes4(quotechar));
        special |= q;
        quotes |= to_bits(q) << (4 * i);
      }
      if (commentchar < 0x100) { special |= __vcmpeq4(words[i], bytes4(commentchar)); }
      terminators |= to_bits(__vcmpeq4(words[i], bytes4(terminator))) << (4 * i);
    }
    if (special == 0) {
      fast_path          = true;
      auto const rowmap  = (terminators << 1) | (c_prev == terminator ? 1u : 0u);
      ctx_map.x          = rowmap;
      ctx_map.y          = 0;
      ctx_map.z          = rowmap;
      ctx_map.w =
        (ROW_CTX_NONE << 0) | (ROW_CTX_QUOTE << 2) | (ROW_CTX_NONE << 4) | (ROW_CTX_EOF << 6);
    } else {
      // Event path: only row starts (after a terminator) and quote characters can change the
      // state differently from a regular character. Consecutive regular characters act like a
      // single one (NONE->NONE, QUOTE->QUOTE, COMMENT->NONE, no rows), so each run is merged once.
      fast_path              = true;
      auto const row_starts  = (terminators << 1) | (c_prev == terminator ? 1u : 0u);
      auto const regular     = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE);
      uint32_t events        = row_starts | quotes;
      uint32_t pos           = 0;
      while (events != 0) {
        auto const k = static_cast<uint32_t>(__ffs(events)) - 1;
        events &= events - 1;
        if (k > pos) { merge_char_context(ctx_map, regular, pos); }
        int const ch      = cur[k];
        int const ch_prev = k > 0 ? static_cast<int>(cur[k - 1]) : c_prev;
        merge_char_context(
          ctx_map, char_context(ch, ch_prev, terminator, delimiter, quotechar, commentchar), k);
        pos = k + 1;
      }
      if (pos < 32) { merge_char_context(ctx_map, regular, pos); }
    }
  }
  if (!fast_path) {
    // Loop through all 32 bytes and keep a bitmask of row starts for each possible input context
    for (uint32_t pos = 0; pos < 32; pos++, cur++, c_prev = c) {
      uint32_t ctx;
      if (cur < end) {
        c   = cur[0];
        ctx = char_context(c, c_prev, terminator, delimiter, quotechar, commentchar);
      } else {
        bool const is_last_chunk = data_end_off <= data.size();
        if (is_last_chunk && cur <= end && cur == data_end) {
          // Add a newline at data end (need the extra row offset to infer length of previous row)
          ctx = make_char_context(ROW_CTX_EOF, ROW_CTX_EOF, ROW_CTX_EOF, 1, 1, 1);
        } else {
          // Pass-through context (beyond chunk_size or data_end)
          ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_COMMENT);
        }
      }
      // Merge with current context, keeping track of where new rows occur
      merge_char_context(ctx_map, ctx, pos);
    }
  }

  // Eliminate rows that start before byte_range_start
  if (start_offset + block_pos < byte_range_start) {
    uint32_t dist_minus1 =
      cuda::std::min(byte_range_start - (start_offset + block_pos) - 1, UINT64_C(31));
    uint32_t mask = 0xffff'fffe << dist_minus1;
    ctx_map.x &= mask;
    ctx_map.y &= mask;
    ctx_map.z &= mask;
  }

  return ctx_map;
}

/**
 * @brief Gather row offsets from CSV character data split into 16KB chunks
 *
 * This is done in two phases: the first phase returns the possible row counts
 * per 16K character block for each possible parsing context at the start of the block,
 * along with the resulting parsing context at the end of the block.
 * The caller can then compute the actual parsing context at the beginning of each
 * individual block and total row count.
 * The second phase outputs the location of each row in the block, using the parsing
 * context and initial row counter accumulated from the results of the previous phase.
 * Row parsing context will be updated after phase 2 such that the value contains
 * the number of rows starting at byte_range_end or beyond.
 *
 * @param row_ctx Row parsing context (output of phase 1 or input to phase 2)
 * @param offsets_out Row offsets (nullptr for phase1, non-null indicates phase 2)
 * @param data Base pointer of character data (all row offsets are relative to this)
 * @param chunk_size Total number of characters to parse
 * @param parse_pos Current parsing position in the file
 * @param start_offset Position of the start of the character buffer in the file
 * @param data_size CSV file size
 * @param byte_range_start Ignore rows starting before this position in the file
 * @param byte_range_end In phase 2, store the number of rows beyond range in row_ctx
 * @param skip_rows Number of rows to skip (ignored in phase 1)
 * @param terminator Line terminator character
 * @param delimiter Column delimiter character
 * @param quotechar Quote character
 * @param escapechar Delimiter escape character
 * @param commentchar Comment line character (skip rows starting with this character)
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  gather_row_offsets_gpu(uint64_t* row_ctx,
                         device_span<uint64_t> offsets_out,
                         device_span<char const> const data,
                         size_t chunk_size,
                         size_t parse_pos,
                         size_t start_offset,
                         size_t data_size,
                         size_t byte_range_start,
                         size_t byte_range_end,
                         size_t skip_rows,
                         int terminator,
                         int delimiter,
                         int quotechar,
                         int escapechar,
                         int commentchar)
{
  // Per-block row context merge tree
  __shared__ uint64_t bk_ctxtree[bk_ctxtree_size];

  uint32_t const t = threadIdx.x;
  size_t block_pos = 0;
  auto const ctx_map = compute_char_contexts(data,
                                             chunk_size,
                                             parse_pos,
                                             start_offset,
                                             data_size,
                                             byte_range_start,
                                             blockIdx.x * static_cast<size_t>(rowofs_block_bytes) +
                                               t * size_t{32},
                                             terminator,
                                             delimiter,
                                             quotechar,
                                             commentchar,
                                             block_pos);

  // Convert the long-form {rowmap,outctx}[inctx] version into packed version
  // {rowcount,ouctx}[inctx], then merge the row contexts of the 32-character blocks into
  // a single 16K-character block context
  rowctx_merge_transform(bk_ctxtree, pack_rowmaps(ctx_map), t);

  // If this is the second phase, get the block's initial parser state and row counter
  if (offsets_out.data()) {
    if (t == 0) { bk_ctxtree[0] = row_ctx[blockIdx.x]; }
    __syncthreads();

    // Walk back the transform tree with the known initial parser state
    rowctx32_t ctx             = rowctx_inverse_merge_transform(bk_ctxtree, t);
    uint64_t row               = (bk_ctxtree[0] >> 2) + (ctx >> 2);
    uint32_t rows_out_of_range = 0;
    uint32_t rowmap            = select_rowmap(ctx_map, ctx & 3);
    // Output row positions
    while (rowmap != 0) {
      uint32_t pos = __ffs(rowmap);
      block_pos += pos;
      if (row >= skip_rows && row - skip_rows < offsets_out.size()) {
        // Output byte offsets are relative to the base of the input buffer
        offsets_out[row - skip_rows] = block_pos - 1;
        rows_out_of_range += (start_offset + block_pos - 1 >= byte_range_end);
      }
      row++;
      rowmap >>= pos;
    }
    __syncthreads();
    // Return the number of rows out of range

    using block_reduce = typename cub::BlockReduce<uint32_t, rowofs_block_dim>;
    __shared__ typename block_reduce::TempStorage bk_storage;
    rows_out_of_range = block_reduce(bk_storage).Sum(rows_out_of_range);
    if (t == 0) { row_ctx[blockIdx.x] = rows_out_of_range; }
  } else {
    // Just store the row counts and output contexts
    if (t == 0) { row_ctx[blockIdx.x] = bk_ctxtree[1]; }
  }
}

/**
 * @brief Characters that make a row blank when they start it (see remove_blank_rows).
 */
struct blank_row_chars {
  char newline;
  char comment;
  char carriage;

  static blank_row_chars from(parse_options_view const& options)
  {
    auto const newline  = options.skipblanklines ? options.terminator : options.comment;
    auto const comment  = options.comment != '\0' ? options.comment : newline;
    auto const carriage = (options.skipblanklines && options.terminator == '\n') ? '\r' : comment;
    return {newline, comment, carriage};
  }
  __device__ bool is_blank(char c) const { return c == newline || c == comment || c == carriage; }
};

/// 32-character slices per thread in single-pass row gathering (64KB tiles)
constexpr int slices_per_thread = 2;

/// Parser state transition of a tile, packed as 2 bits per input state (NONE, QUOTE, COMMENT, EOF)
using tile_transition = uint8_t;
constexpr tile_transition identity_transition =
  ROW_CTX_NONE | (ROW_CTX_QUOTE << 2) | (ROW_CTX_COMMENT << 4) | (ROW_CTX_EOF << 6);

/// The transition of `first` followed by `second`
struct compose_transitions {
  __device__ tile_transition operator()(tile_transition first, tile_transition second) const
  {
    tile_transition result = 0;
    for (int s = 0; s < 4; ++s) {
      auto const mid = (first >> (2 * s)) & 3;
      result |= ((second >> (2 * mid)) & 3) << (2 * s);
    }
    return result;
  }
};

/**
 * @brief Single-pass row gathering, phase 1: writes each thread's row start bitmap and each
 * tile's number of (non-blank) rows.
 *
 * The parser state at the start of a tile is only known once all previous tiles are processed.
 * Since it is almost always NONE (no quoted field spans the tile boundary), the first launch
 * (`start_states == nullptr`) assumes NONE for every tile and records each tile's state
 * transition. After the start states are computed from the transitions, a second launch redoes
 * only the tiles whose actual start state differs.
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  gather_row_bitmaps_gpu(device_span<char const> const data,
                         size_t chunk_size,
                         size_t parse_pos,
                         size_t start_offset,
                         size_t data_size,
                         size_t byte_range_start,
                         int terminator,
                         int delimiter,
                         int quotechar,
                         int commentchar,
                         uint32_t num_tiles,
                         tile_transition* tile_transitions,
                         uint8_t const* start_states,
                         uint32_t const* redo_tiles,
                         uint32_t const* redo_count,
                         blank_row_chars blank_chars,
                         uint64_t* tile_kept_rows,
                         uint32_t* row_bitmaps)
{
  uint32_t const t   = threadIdx.x;
  auto const is_redo = start_states != nullptr;
  using block_scan   = cub::BlockScan<tile_transition, rowofs_block_dim>;
  using block_reduce = cub::BlockReduce<uint32_t, rowofs_block_dim>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ typename block_reduce::TempStorage reduce_storage;
  // The first launch processes every tile; the redo launch strides over the listed tiles
  auto const num_items = is_redo ? *redo_count : num_tiles;
  for (uint32_t item = blockIdx.x; item < num_items; item += gridDim.x) {
  auto const tile     = is_redo ? redo_tiles[item] : item;
  auto const start_st = is_redo ? start_states[tile] : uint8_t{ROW_CTX_NONE};

  // Each thread handles slices_per_thread consecutive 32-character slices
  uint4 ctx_maps[slices_per_thread];
  size_t slice_pos[slices_per_thread];
  tile_transition thread_transition = identity_transition;
#pragma unroll
  for (int j = 0; j < slices_per_thread; ++j) {
    auto& block_pos = slice_pos[j];
    ctx_maps[j]     = compute_char_contexts(
      data,
      chunk_size,
      parse_pos,
      start_offset,
      data_size,
      byte_range_start,
      ((static_cast<size_t>(tile) * rowofs_block_dim + t) * slices_per_thread + j) * 32,
      terminator,
      delimiter,
      quotechar,
      commentchar,
      block_pos);
    // Output states of the slice (bits 0..5 of .w, EOF stays EOF)
    thread_transition = compose_transitions{}(
      thread_transition, static_cast<tile_transition>((ctx_maps[j].w & 0x3f) | (ROW_CTX_EOF << 6)));
  }
  // Each thread's input state: the tile's start state through the transitions of earlier threads
  tile_transition prefix;
  tile_transition tile_total;
  block_scan(scan_storage)
    .ExclusiveScan(thread_transition, prefix, identity_transition, compose_transitions{}, tile_total);
  if (t == 0 && !is_redo) { tile_transitions[tile] = tile_total; }
  auto state = static_cast<uint32_t>((prefix >> (2 * start_st)) & 3);
  uint32_t kept_rows = 0;
#pragma unroll
  for (int j = 0; j < slices_per_thread; ++j) {
    auto rowmap = select_rowmap(ctx_maps[j], state);
    // Drop blank rows (see remove_blank_rows); the end-of-data row is always kept
    for (auto bits = rowmap; bits != 0; bits &= bits - 1) {
      auto const pos = slice_pos[j] + __ffs(bits) - 1;
      if (pos != data.size() && blank_chars.is_blank(data[pos])) { rowmap &= ~(1u << (__ffs(bits) - 1)); }
    }
    kept_rows += __popc(rowmap);
    row_bitmaps[(static_cast<size_t>(tile) * rowofs_block_dim + t) * slices_per_thread + j] = rowmap;
    // Output state of the slice for this input state (EOF stays EOF)
    state = state < 3 ? (ctx_maps[j].w >> (2 * state)) & 3 : state;
  }
  kept_rows = block_reduce(reduce_storage).Sum(kept_rows);
  __syncthreads();  // temp storage is reused by the next tile
  if (t == 0) { tile_kept_rows[tile] = kept_rows; }
  }
}

/// Block size of the tile start state scan
constexpr int start_state_block_dim = 1024;

/**
 * @brief Computes each tile's starting parser state from the tile transitions (one block), and
 * lists the tiles that do not start in the NONE state.
 */
CUDF_KERNEL void __launch_bounds__(start_state_block_dim)
  tile_start_states_gpu(tile_transition const* tile_transitions,
                        uint32_t num_tiles,
                        uint8_t* start_states,
                        uint32_t* redo_tiles,
                        uint32_t* redo_count)
{
  using block_scan = cub::BlockScan<tile_transition, start_state_block_dim>;
  __shared__ typename block_scan::TempStorage temp_storage;
  tile_transition carry = identity_transition;  // transition of all tiles before this chunk
  for (uint32_t first = 0; first < num_tiles; first += start_state_block_dim) {
    auto const tile = first + threadIdx.x;
    auto const in   = tile < num_tiles ? tile_transitions[tile] : identity_transition;
    tile_transition prefix;
    tile_transition chunk;
    block_scan(temp_storage).ExclusiveScan(in, prefix, carry, compose_transitions{}, chunk);
    // The data starts in the NONE state
    if (tile < num_tiles) {
      auto const start_state = prefix & 3;
      start_states[tile]     = start_state;
      // Tiles that do not start in the NONE state assumed by the first launch must be redone
      if (start_state != ROW_CTX_NONE) { redo_tiles[atomicAdd(redo_count, 1u)] = tile; }
    }
    carry = compose_transitions{}(carry, chunk);
    __syncthreads();
  }
}

/**
 * @brief Single-pass row gathering, phase 2: converts the row start bitmaps into row offsets.
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  row_bitmaps_to_offsets_gpu(uint32_t const* row_bitmaps,
                             uint64_t const* tile_first_row,
                             size_t parse_off,
                             device_span<uint64_t> offsets_out)
{
  using block_scan = cub::BlockScan<uint32_t, rowofs_block_dim>;
  __shared__ typename block_scan::TempStorage temp_storage;
  uint32_t const t       = threadIdx.x;
  auto const first_word  = (static_cast<size_t>(blockIdx.x) * rowofs_block_dim + t) * slices_per_thread;
  uint32_t rowmaps[slices_per_thread];
  uint32_t thread_rows = 0;
#pragma unroll
  for (int j = 0; j < slices_per_thread; ++j) {
    rowmaps[j] = row_bitmaps[first_word + j];
    thread_rows += __popc(rowmaps[j]);
  }
  uint32_t row_in_block;
  block_scan(temp_storage).ExclusiveSum(thread_rows, row_in_block);
  auto row = tile_first_row[blockIdx.x] + row_in_block;
#pragma unroll
  for (int j = 0; j < slices_per_thread; ++j) {
    auto rowmap          = rowmaps[j];
    auto const block_pos = parse_off + (first_word + j) * 32;
    while (rowmap != 0) {
      auto const bit = __ffs(rowmap) - 1;
      if (row < offsets_out.size()) { offsets_out[row] = block_pos + bit; }
      ++row;
      rowmap &= rowmap - 1;
    }
  }
}

size_t __host__ count_blank_rows(cudf::io::parse_options_view const& opts,
                                 device_span<char const> data,
                                 device_span<uint64_t const> row_offsets,
                                 cuda::stream_ref stream)
{
  auto const newline  = opts.skipblanklines ? opts.terminator : opts.comment;
  auto const comment  = opts.comment != '\0' ? opts.comment : newline;
  auto const carriage = (opts.skipblanklines && opts.terminator == '\n') ? '\r' : comment;
  return thrust::count_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    row_offsets.begin(),
    row_offsets.end(),
    [data = data, newline, comment, carriage] __device__(uint64_t const pos) {
      return ((pos != data.size()) &&
              (data[pos] == newline || data[pos] == comment || data[pos] == carriage));
    });
}

device_span<uint64_t> __host__ remove_blank_rows(cudf::io::parse_options_view const& options,
                                                 device_span<char const> data,
                                                 device_span<uint64_t> row_offsets,
                                                 cuda::stream_ref stream)
{
  size_t d_size       = data.size();
  auto const newline  = options.skipblanklines ? options.terminator : options.comment;
  auto const comment  = options.comment != '\0' ? options.comment : newline;
  auto const carriage = (options.skipblanklines && options.terminator == '\n') ? '\r' : comment;
  auto new_end        = thrust::remove_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    row_offsets.begin(),
    row_offsets.end(),
    [data = data, d_size, newline, comment, carriage] __device__(uint64_t const pos) {
      return ((pos != d_size) &&
              (data[pos] == newline || data[pos] == comment || data[pos] == carriage));
    });
  return row_offsets.subspan(0, new_end - row_offsets.begin());
}

/**
 * @brief Returns the shared memory size needed to stage every block's rows (one block per
 * `csvparse_block_dim` rows), or 0 if that exceeds `budget` or the data is not 16-byte aligned.
 */
size_t staging_smem_size(device_span<char const> data,
                         device_span<uint64_t const> row_offsets,
                         size_t budget,
                         cuda::stream_ref stream)
{
  auto const num_rows = row_offsets.size() - 1;
  if (num_rows == 0) { return 0; }
  auto const block_size = csvparse_block_dim;
  auto const grid_size  = cudf::util::div_rounding_up_safe<size_t>(num_rows, block_size);
  // Skip the span reduction (and its sync) when an average block span already does not fit
  auto const avg_block_span = data.size() / num_rows * block_size;
  auto const max_block_span = avg_block_span > budget ? budget + 1 : thrust::transform_reduce(
    rmm::exec_policy(stream, cudf::get_current_device_resource_ref()),
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator<size_t>(grid_size),
    cuda::proclaim_return_type<size_t>([row_offsets, num_rows] __device__(size_t block) {
      auto const first = block * csvparse_block_dim;
      auto const last  = cuda::std::min<size_t>(first + csvparse_block_dim, num_rows);
      return static_cast<size_t>(row_offsets[last] - (row_offsets[first] & ~uint64_t{15}));
    }),
    size_t{0},
    cuda::maximum<size_t>{});
  // Room for the final 16-byte load, which may extend past the span
  auto const wanted_smem_size = util::round_up_safe<size_t>(max_block_span + 16, 16);
  // Staging uses 16-byte loads relative to the data start (not aligned e.g. after a skipped BOM in
  // a zero-copy source)
  auto const is_aligned = reinterpret_cast<uintptr_t>(data.data()) % 16 == 0;
  return is_aligned && wanted_smem_size <= budget ? wanted_smem_size : 0;
}

namespace {
/// Shared memory used by the detection histogram (0 when it is kept in global memory)
size_t detection_histogram_smem(size_t num_active_columns)
{
  auto const histogram_bytes = num_active_columns * sizeof(column_type_histogram);
  return histogram_bytes <= 32 * 1024 ? util::round_up_safe<size_t>(histogram_bytes, sizeof(uint4))
                                      : 0;
}
}  // namespace

size_t detection_stage_size(device_span<char const> data,
                            device_span<uint64_t const> row_starts,
                            size_t num_active_columns,
                            cuda::stream_ref stream)
{
  // Stage each block's rows in shared memory when they fit next to the histogram
  constexpr size_t max_smem_size = 48 * 1024;
  return staging_smem_size(
    data, row_starts, max_smem_size - detection_histogram_smem(num_active_columns), stream);
}

cudf::detail::host_vector<column_type_histogram> detect_column_types(
  cudf::io::parse_options_view const& options,
  device_span<char const> const data,
  device_span<column_parse::flags const> const column_flags,
  device_span<uint64_t const> const row_starts,
  size_t const num_active_columns,
  device_span<uint64_t* const> int_values,
  device_span<cudf::bitmask_type* const> int_valids,
  device_span<size_type> int_valid_counts,
  size_t stage_size,
  cuda::stream_ref stream)
{
  // Calculate actual block count to use based on records count
  int const block_size = csvparse_block_dim;
  int const grid_size  = (row_starts.size() + block_size - 1) / block_size;

  auto d_stats = cudf::detail::make_zeroed_device_uvector_async<column_type_histogram>(
    num_active_columns, stream, cudf::get_current_device_resource_ref());

  // Accumulate per-block histograms in shared memory when they fit in the default limit
  auto const histogram_smem       = detection_histogram_smem(num_active_columns);
  bool const use_shared_histogram = histogram_smem != 0;
  auto const smem_bytes           = histogram_smem + stage_size;
  // 32-bit character counters (fewer registers, higher occupancy) are exact unless a single field
  // can exceed INT_MAX characters, which requires a buffer larger than that
  if (data.size() <= static_cast<size_t>(cuda::std::numeric_limits<int>::max())) {
    data_type_detection<int><<<grid_size, block_size, smem_bytes, stream.get()>>>(
      options,
      data,
      column_flags,
      row_starts,
      d_stats,
      use_shared_histogram,
      int_values,
      int_valids,
      int_valid_counts,
      stage_size);
  } else {
    data_type_detection<long><<<grid_size, block_size, smem_bytes, stream.get()>>>(
      options,
      data,
      column_flags,
      row_starts,
      d_stats,
      use_shared_histogram,
      int_values,
      int_valids,
      int_valid_counts,
      stage_size);
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  return cudf::detail::make_host_vector(d_stats, stream);
}

void decode_row_column_data(cudf::io::parse_options_view const& options,
                            device_span<char const> data,
                            char* unescape_buffer,
                            device_span<column_parse::flags const> column_flags,
                            device_span<uint64_t const> row_offsets,
                            device_span<cudf::data_type const> dtypes,
                            device_span<void* const> columns,
                            device_span<cudf::bitmask_type* const> valids,
                            device_span<size_type> valid_counts,
                            cuda::stream_ref stream)
{
  // Calculate actual block count to use based on records count
  auto const block_size = csvparse_block_dim;
  auto const num_rows   = row_offsets.size() - 1;
  auto const grid_size  = cudf::util::div_rounding_up_safe<size_t>(num_rows, block_size);

  // Stage each block's rows in shared memory, sized to fit the longest block span; skip staging
  // when that exceeds the default shared memory limit (blocks then read global memory directly)
  constexpr size_t max_smem_size = 48 * 1024;
  auto const smem_size           = staging_smem_size(data, row_offsets, max_smem_size, stream);
  // Without staging every field is read from global memory, where parsing integers from a
  // register copy of the field saves one load per character
  auto const kernel = smem_size == 0 ? convert_csv_to_cudf<true> : convert_csv_to_cudf<false>;
  kernel<<<grid_size, block_size, smem_size, stream.get()>>>(options,
                                                             data,
                                                             unescape_buffer,
                                                             column_flags,
                                                             row_offsets,
                                                             dtypes,
                                                             columns,
                                                             valids,
                                                             valid_counts,
                                                             smem_size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

rmm::device_uvector<uint64_t> __host__ gather_all_row_offsets(parse_options_view const& options,
                                                             device_span<char const> const data,
                                                             size_t chunk_size,
                                                             size_t parse_pos,
                                                             size_t start_offset,
                                                             size_t data_size,
                                                             size_t byte_range_start,
                                                             cuda::stream_ref stream)
{
  auto const tile_bytes = static_cast<size_t>(rowofs_block_bytes) * slices_per_thread;
  auto const num_tiles  = static_cast<uint32_t>(1 + (chunk_size / tile_bytes));
  auto const mr         = cudf::get_current_device_resource_ref();
  // Kept row counts, with one extra zero so that their exclusive scan ends with the total
  auto tile_kept_rows_buffer = cudf::detail::make_zeroed_device_uvector_async<uint64_t>(
    static_cast<size_t>(num_tiles) + 1, stream, mr);
  auto const tile_kept_rows = tile_kept_rows_buffer.data();
  rmm::device_uvector<tile_transition> tile_transitions(num_tiles, stream);
  rmm::device_uvector<uint8_t> start_states(num_tiles, stream);
  rmm::device_uvector<uint32_t> redo_tiles(num_tiles, stream);
  auto redo_count = cudf::detail::make_zeroed_device_uvector_async<uint32_t>(1, stream, mr);
  rmm::device_uvector<uint32_t> row_bitmaps(
    static_cast<size_t>(num_tiles) * rowofs_block_dim * slices_per_thread, stream);

  auto const launch = [&](uint8_t const* tile_start_states) {
    // The redo launch strides over the (usually empty) list of tiles to redo
    auto const num_blocks = tile_start_states ? std::min<uint32_t>(num_tiles, 256) : num_tiles;
    gather_row_bitmaps_gpu<<<num_blocks, rowofs_block_dim, 0, stream.get()>>>(
      data,
      chunk_size,
      parse_pos,
      start_offset,
      data_size,
      byte_range_start,
      options.terminator,
      options.delimiter,
      (options.quotechar) ? options.quotechar : 0x100,
      (options.comment) ? options.comment : 0x100,
      num_tiles,
      tile_transitions.data(),
      tile_start_states,
      redo_tiles.data(),
      redo_count.data(),
      blank_row_chars::from(options),
      tile_kept_rows,
      row_bitmaps.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  };
  // Assume every tile starts outside of quotes, then redo the tiles where that was wrong
  launch(nullptr);
  tile_start_states_gpu<<<1, start_state_block_dim, 0, stream.get()>>>(
    tile_transitions.data(), num_tiles, start_states.data(), redo_tiles.data(), redo_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  launch(start_states.data());

  // First kept row of each tile; the last entry is the total number of kept rows
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                         tile_kept_rows,
                         tile_kept_rows + num_tiles + 1,
                         tile_kept_rows);
  auto const total_rows =
    cudf::detail::make_host_vector(device_span<uint64_t const>{tile_kept_rows + num_tiles, 1}, stream);
  auto const num_rows = static_cast<size_t>(total_rows[0]);

  rmm::device_uvector<uint64_t> offsets(num_rows, stream);
  auto const parse_off = parse_pos > start_offset ? parse_pos - start_offset : 0;
  row_bitmaps_to_offsets_gpu<<<num_tiles, rowofs_block_dim, 0, stream.get()>>>(
    row_bitmaps.data(), tile_kept_rows, parse_off, offsets);
  CUDF_CUDA_TRY(cudaGetLastError());
  return offsets;
}

uint32_t __host__ gather_row_offsets(parse_options_view const& options,
                                     uint64_t* row_ctx,
                                     device_span<uint64_t> const offsets_out,
                                     device_span<char const> const data,
                                     size_t chunk_size,
                                     size_t parse_pos,
                                     size_t start_offset,
                                     size_t data_size,
                                     size_t byte_range_start,
                                     size_t byte_range_end,
                                     size_t skip_rows,
                                     cuda::stream_ref stream)
{
  uint32_t dim_grid = 1 + (chunk_size / rowofs_block_bytes);

  gather_row_offsets_gpu<<<dim_grid, rowofs_block_dim, 0, stream.get()>>>(
    row_ctx,
    offsets_out,
    data,
    chunk_size,
    parse_pos,
    start_offset,
    data_size,
    byte_range_start,
    byte_range_end,
    skip_rows,
    options.terminator,
    options.delimiter,
    (options.quotechar) ? options.quotechar : 0x100,
    /*(options.escapechar) ? options.escapechar :*/ 0x100,
    (options.comment) ? options.comment : 0x100);
  CUDF_CUDA_TRY(cudaGetLastError());

  return dim_grid;
}

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
