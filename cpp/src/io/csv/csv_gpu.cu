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
#include <cuda/std/algorithm>
#include <cuda/stream>
#include <thrust/count.h>
#include <thrust/detail/copy.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

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

/// Unit in which `seek_field_end_by_words` loads the characters of a row
using char_word = uint64_t;
/// Number of characters in a `char_word`
constexpr int word_chars = sizeof(char_word);

/// Returns a word with every byte set to `c`
__device__ constexpr char_word repeat_char(char c)
{
  return char_word{0x0101'0101'0101'0101} * static_cast<uint8_t>(c);
}

/**
 * @brief Flags the zero bytes of a word.
 *
 * Returns a word with the high bit set in each byte that is zero in `word`, and in no byte below
 * the lowest zero byte. A byte above a zero byte may also be flagged (the subtraction borrows
 * from a zero byte into the next one), so the flags are a superset of the zero bytes whose lowest
 * flag is exact.
 */
__device__ constexpr char_word flag_zero_bytes(char_word word)
{
  constexpr char_word low_bits  = 0x0101'0101'0101'0101;
  constexpr char_word high_bits = 0x8080'8080'8080'8080;
  return (word - low_bits) & ~word & high_bits;
}

/**
 * @brief Equivalent of `cudf::io::gpu::seek_field_end` (without escape characters) that loads the
 * row in aligned 8-byte words rather than one character at a time.
 *
 * The decoding kernels use one thread per row, so the character loads of a warp are scattered over
 * 32 rows and each of them is a separate memory transaction; word loads cut their number by up to
 * 8x. In a field that does not start with a quote, only a delimiter, a terminator or a '\r' can end
 * the field: the positions of these characters in a word are flagged with `flag_zero_bytes` and
 * checked in increasing order with the exact test of `seek_field_end`, so the first position that
 * passes it is the one `seek_field_end` returns (every occurrence is flagged, and false flags fail
 * the test). The characters of a quoted field are checked one by one after the word is loaded.
 *
 * Only words that lie entirely within `[data_begin, end)` are loaded, whatever the alignment of
 * the data, and the characters of the row outside of such words are read one at a time. The lower
 * bound matters when `data_begin` is not the start of an aligned, reader-owned buffer (e.g. when
 * parsing a caller's buffer in place). A word may start before `begin`, in the previous field or
 * in the previous row; these characters are never used, as other threads may be modifying them.
 *
 * @param begin Pointer to the first character of the field
 * @param end Pointer to the end of the row
 * @param data_begin Pointer to the first character of the data that holds the row
 * @param opts A set of parsing options
 *
 * @return Pointer to the character that ends the field, or `end`
 */
__device__ char const* seek_field_end_by_words(char const* begin,
                                               char const* end,
                                               char const* data_begin,
                                               parse_options_view const& opts)
{
  if (opts.multi_delimiter) { return cudf::io::gpu::seek_field_end(begin, end, opts); }

  bool const starts_with_quote = begin < end && *begin == opts.quotechar;
  bool in_quotes               = false;
  // Same test as `seek_field_end` for the character `c` at `pos`
  auto const ends_field = [&](char c, char const* pos) {
    if (starts_with_quote && c == opts.quotechar) {
      in_quotes = !in_quotes;
      return false;
    }
    return !in_quotes && (c == opts.delimiter || c == opts.terminator ||
                          (c == '\r' && pos + 1 < end && pos[1] == '\n'));
  };
  auto const data_address = reinterpret_cast<uintptr_t>(data_begin);
  auto const end_address  = reinterpret_cast<uintptr_t>(end);
  auto current            = begin;
  while (current < end) {
    auto const offset       = static_cast<int>(reinterpret_cast<uintptr_t>(current) % word_chars);
    auto const word_address = reinterpret_cast<uintptr_t>(current) - offset;
    if (word_address < data_address || word_address + word_chars > end_address) {
      if (ends_field(*current, current)) { return current; }
      ++current;
      continue;
    }
    auto const word_begin = current - offset;
    auto const word       = *reinterpret_cast<char_word const*>(word_begin);
    auto const char_at    = [word](int i) { return static_cast<char>(word >> (8 * i)); };
    if (starts_with_quote) {
      for (int i = offset; i < word_chars; ++i) {
        if (ends_field(char_at(i), word_begin + i)) { return word_begin + i; }
      }
    } else {
      auto candidates = (flag_zero_bytes(word ^ repeat_char(opts.delimiter)) |
                         flag_zero_bytes(word ^ repeat_char(opts.terminator)) |
                         flag_zero_bytes(word ^ repeat_char('\r'))) &
                        (~char_word{0} << (8 * offset));
      while (candidates != 0) {
        auto const i = (__ffsll(static_cast<long long>(candidates)) - 1) / 8;
        if (ends_field(char_at(i), word_begin + i)) { return word_begin + i; }
        candidates &= candidates - 1;
      }
    }
    current = word_begin + word_chars;
  }
  return current;
}

/**
 * @brief Marks the field of row `row` in a column as valid.
 *
 * The lanes of a warp decode 32 consecutive rows that start at a multiple of 32 (a one-dimensional
 * grid of `csvparse_block_dim` threads per block, one thread per row), so the validity bits they
 * set in a column's mask are in the same word, at the index of the lane. The lanes that mark
 * fields of the same column together set their bits with a single atomic operation.
 *
 * @param mask Validity mask of the column
 * @param column Index of the column
 * @param row Index of the row
 */
__device__ void set_valid_warp_aggregated(cudf::bitmask_type* mask, int column, size_type row)
{
  static_assert(csvparse_block_dim % cudf::detail::warp_size == 0,
                "The rows of a warp must share the words of the validity masks");
  // The lanes that are active together normally mark the same column, but nothing guarantees it
  // (lanes that diverged in earlier fields may meet here at different columns), so they are grouped
  // by column for correctness
  auto const lanes = __match_any_sync(__activemask(), column);
  // The bit of row `row` in its mask word is `row % warp_size`, which is the index of the lane
  auto const lane = static_cast<int>(threadIdx.x % cudf::detail::warp_size);
  if (lane == __ffs(lanes) - 1) { atomicOr(&mask[cudf::word_index(row)], lanes); }
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
CUDF_KERNEL void __launch_bounds__(csvparse_block_dim)
  data_type_detection(parse_options_view const opts,
                      device_span<char const> csv_text,
                      device_span<column_parse::flags const> const column_flags,
                      device_span<uint64_t const> const row_offsets,
                      device_span<column_type_histogram> d_column_data)
{
  auto const raw_csv = csv_text.data();

  // ThreadIds range per block, so also need the blockId
  // This is entry into the fields; threadId is an element within `num_records`
  auto const rec_id      = grid_1d::global_thread_id();
  auto const rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of the data
  if (rec_id_next >= row_offsets.size()) { return; }

  auto field_start   = raw_csv + row_offsets[rec_id];
  auto const row_end = raw_csv + row_offsets[rec_id_next];

  auto next_field = field_start;
  int col         = 0;
  int actual_col  = 0;

  // Going through all the columns of a given record
  while (col < column_flags.size() && field_start < row_end) {
    // In delim_whitespace mode, collapse leading delimiter runs so leading whitespace does
    // not produce empty fields (matches pandas behavior).
    field_start = cudf::io::gpu::skip_leading_delimiter_run(field_start, row_end, opts);
    if (field_start >= row_end) break;
    auto next_delimiter = seek_field_end_by_words(field_start, row_end, raw_csv, opts);

    // Checking if this is a column that the user wants --- user can filter columns
    if (column_flags[col] & column_parse::inferred) {
      // points to last character in the field
      auto const field_len = static_cast<size_t>(next_delimiter - field_start);
      if (serialized_trie_contains(opts.trie_na, {field_start, field_len})) {
        atomicAdd(&d_column_data[actual_col].null_count, 1);
      } else if (serialized_trie_contains(opts.trie_true, {field_start, field_len}) ||
                 serialized_trie_contains(opts.trie_false, {field_start, field_len})) {
        atomicAdd(&d_column_data[actual_col].bool_count, 1);
      } else if (cudf::io::is_infinity(field_start, next_delimiter)) {
        atomicAdd(&d_column_data[actual_col].float_count, 1);
      } else {
        long count_number    = 0;
        long count_decimal   = 0;
        long count_thousands = 0;
        long count_slash     = 0;
        long count_dash      = 0;
        long count_plus      = 0;
        long count_colon     = 0;
        long count_string    = 0;
        long count_exponent  = 0;

        // Modify field_start & end to ignore whitespace and quotechars
        // This could possibly result in additional empty fields
        auto const trimmed_field_range = trim_whitespaces_quotes(field_start, next_delimiter);
        auto const trimmed_field_len   = trimmed_field_range.second - trimmed_field_range.first;

        if (trimmed_field_len == 0) {
          cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{
            d_column_data[actual_col].string_count};
          ref.fetch_add(1, cuda::memory_order_relaxed);
        } else {
          for (auto cur = trimmed_field_range.first; cur < trimmed_field_range.second; ++cur) {
            if (is_digit(*cur)) {
              count_number++;
              continue;
            }
            if (*cur == opts.decimal) {
              count_decimal++;
              continue;
            }
            if (*cur == opts.thousands) {
              count_thousands++;
              continue;
            }
            // Looking for unique characters that will help identify column types.
            switch (*cur) {
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

          // Integers have to have the length of the string
          // Off by one if they start with a minus sign
          auto const int_req_number_cnt =
            trimmed_field_len - count_thousands -
            ((*trimmed_field_range.first == '-' || *trimmed_field_range.first == '+') &&
             trimmed_field_len > 1);

          if (column_flags[col] & column_parse::as_datetime) {
            // PANDAS uses `object` dtype if the date is unparseable
            if (is_datetime(count_string, count_decimal, count_colon, count_dash, count_slash)) {
              atomicAdd(&d_column_data[actual_col].datetime_count, 1);
            } else {
              atomicAdd(&d_column_data[actual_col].string_count, 1);
            }
          } else if (count_number == int_req_number_cnt) {
            auto const is_negative = (*trimmed_field_range.first == '-');
            auto const data_begin =
              trimmed_field_range.first + (is_negative || (*trimmed_field_range.first == '+'));
            cudf::size_type* ptr = cudf::io::gpu::infer_integral_field_counter(
              data_begin, data_begin + count_number, is_negative, d_column_data[actual_col]);
            atomicAdd(ptr, 1);
          } else if (is_floatingpoint(trimmed_field_len,
                                      count_number,
                                      count_decimal,
                                      count_thousands,
                                      count_dash + count_plus,
                                      count_exponent)) {
            atomicAdd(&d_column_data[actual_col].float_count, 1);
          } else {
            atomicAdd(&d_column_data[actual_col].string_count, 1);
          }
        }
      }
      actual_col++;
    }
    next_field  = next_delimiter + 1;
    field_start = next_field;
    col++;
  }
}

/**
 * @brief Collapses each escaped quote pair (two consecutive `quotechar`) of a quoted field's
 * content into a single `quotechar`, in place.
 *
 * Pairs are matched left to right without overlap, so the result is the same as replacing every
 * occurrence of the two-character string with the one-character string (`""""` becomes `""`).
 * The write position never passes the read position, which makes the in-place update safe.
 * Content without escaped pairs is not written. Bytes past the returned length are left in an
 * unspecified state.
 *
 * @param content First character of the content (after the opening quote)
 * @param length Number of characters in the content (excluding the closing quote)
 * @param quotechar Quote character
 * @return Length of the unescaped content
 */
__device__ __forceinline__ size_t unescape_doublequotes(char* content,
                                                        size_t length,
                                                        char quotechar)
{
  auto const end = content + length;
  auto in        = content;
  while (in + 1 < end && !(in[0] == quotechar && in[1] == quotechar)) {
    ++in;
  }
  if (in + 1 >= end) { return length; }

  auto out = in;
  while (in < end) {
    auto const c = *in;
    *out++       = c;
    in += (c == quotechar && in + 1 < end && in[1] == quotechar) ? 2 : 1;
  }
  return out - content;
}

/**
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed one record at a time. A fixed-width output is written wherever its field is
 * valid, which is where the (zero-initialized) validity bit is set, so fixed-width outputs do not
 * need to be initialized. String outputs do: a (pointer, length) pair is written for valid and NA
 * fields, while fields missing from the end of a short row rely on the zeroed pair reading as null.
 *
 * When `options.doublequote` is set, the escaped quote pairs of quoted string fields are collapsed
 * in place in `data`, and the string pairs point to the unescaped content. This relies on two
 * invariants:
 * - All bytes used for row `i` lie within its byte range `[row_offsets[i], row_offsets[i + 1])`,
 *   and the ranges of different rows do not overlap, so each byte is used by one thread only.
 *   Loads that cover bytes outside the row, such as the word loads of `seek_field_end_by_words`,
 *   must not use them: other threads may be rewriting them.
 * - Within a row, a field is rewritten only after its end (and therefore the start of the next
 *   field) has been found, and the rewrite stays within the field.
 * No other code reads `data` after this kernel, except to copy the strings it describes.
 *
 * @param[in] options A set of parsing options
 * @param[in,out] data The entire CSV data to read; quoted string fields are unescaped in place
 * @param[in] column_flags Per-column parsing behavior flags
 * @param[in] row_offsets The start the CSV data of interest
 * @param[in] dtypes The data type of the column
 * @param[out] columns The output column data
 * @param[out] valids The bitmaps indicating whether the fields of non-string columns are valid
 */
CUDF_KERNEL void __launch_bounds__(csvparse_block_dim)
  convert_csv_to_cudf(cudf::io::parse_options_view options,
                      device_span<char> data,
                      device_span<column_parse::flags const> column_flags,
                      device_span<uint64_t const> row_offsets,
                      device_span<cudf::data_type const> dtypes,
                      device_span<void* const> columns,
                      device_span<cudf::bitmask_type* const> valids)
{
  // Fields are parsed through a read-only view; only string unescaping writes to `data`
  char const* const raw_csv = data.data();
  // thread IDs range per block, so also need the block id.
  // this is entry into the field array - tid is an elements within the num_entries array
  auto const rec_id      = grid_1d::global_thread_id();
  auto const rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of the data
  if (rec_id_next >= row_offsets.size()) return;

  auto field_start   = raw_csv + row_offsets[rec_id];
  auto const row_end = raw_csv + row_offsets[rec_id_next];

  auto next_field = field_start;
  int col         = 0;
  int actual_col  = 0;

  while (col < column_flags.size() && field_start < row_end) {
    // In delim_whitespace mode, collapse leading delimiter runs so leading whitespace does
    // not produce empty fields (matches pandas behavior).
    field_start = cudf::io::gpu::skip_leading_delimiter_run(field_start, row_end, options);
    if (field_start >= row_end) break;
    next_field          = field_start;
    auto next_delimiter = seek_field_end_by_words(field_start, row_end, raw_csv, options);

    if (column_flags[col] & column_parse::enabled) {
      // check if the entire field is a NaN string - consistent with pandas
      auto const is_valid = !serialized_trie_contains(
        options.trie_na, {field_start, static_cast<size_t>(next_delimiter - field_start)});

      // Modify field_start & end to ignore whitespace and quotechars
      auto field_end = next_delimiter;
      if (is_valid && dtypes[actual_col].id() != cudf::type_id::STRING) {
        auto const trimmed_field =
          trim_whitespaces_quotes(field_start, field_end, options.quotechar);
        field_start = trimmed_field.first;
        field_end   = trimmed_field.second;
      }
      if (is_valid) {
        // Type dispatcher does not handle STRING
        if (dtypes[actual_col].id() == cudf::type_id::STRING) {
          auto end        = next_delimiter;
          bool was_quoted = false;
          if (not options.keepquotes) {
            // A quoted string needs an opening and a closing quote; a field consisting of a single
            // quote character (an unterminated quote at the end of the input) is kept as is
            if (not options.detect_whitespace_around_quotes) {
              if (end - field_start >= 2 && (*field_start == options.quotechar) &&
                  (*(end - 1) == options.quotechar)) {
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
          // A quoted field has both of its quotes, so the content length is not negative
          if (was_quoted && options.doublequote) {
            end = field_start + unescape_doublequotes(data.data() + (field_start - raw_csv),
                                                      end - field_start,
                                                      options.quotechar);
          }
          auto str_list = static_cast<std::pair<char const*, size_t>*>(columns[actual_col]);
          str_list[rec_id].first  = field_start;
          str_list[rec_id].second = end - field_start;
        } else {
          if (cudf::type_dispatcher(dtypes[actual_col],
                                    ConvertFunctor{},
                                    field_start,
                                    field_end,
                                    columns[actual_col],
                                    rec_id,
                                    dtypes[actual_col],
                                    options,
                                    column_flags[col] & column_parse::as_hexadecimal)) {
            // set the valid bitmap - all bits were set to 0 to start
            set_valid_warp_aggregated(valids[actual_col], actual_col, rec_id);
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
 * NOTE: This is the most performance-critical piece of the row gathering kernels when a slice is
 * merged character by character. The char_ctx values come from make_char_context with constant
 * arguments (see char_row_context), so that they are compile-time constants selected by branches.
 * Slices that are not merged character by character only merge the characters whose transitions
 * differ from the regular one (see slice_row_contexts).
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
inline __device__ void ctx_merge(packed_rowctx_t* ctxtree, packed_rowctx_t* ctxb, uint32_t t)
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
  uint32_t base, packed_rowctx_t const* ctxtree, uint32_t* ctx, uint32_t* brow4, uint32_t t)
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
static inline __device__ void rowctx_merge_transform(packed_rowctx_t* ctxtree,
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
static inline __device__ rowctx32_t rowctx_inverse_merge_transform(packed_rowctx_t const* ctxtree,
                                                                   uint32_t t)
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

/// Characters per slice: each thread tracks the row contexts of 32-character slices
constexpr uint32_t slice_chars = 32;
static_assert(rowofs_block_bytes == rowofs_block_dim * slice_chars);

/// Value of a disabled quote or comment character, which no character matches
constexpr int no_char = 0x100;

/**
 * @brief Characters that determine where rows start.
 *
 * Characters are compared as `int` values of the `char` data.
 */
struct row_parse_chars {
  int terminator;
  int delimiter;
  int quotechar;    ///< `no_char` if quoting is disabled
  int commentchar;  ///< `no_char` if comments are disabled

  static row_parse_chars from(parse_options_view const& options)
  {
    return {options.terminator,
            options.delimiter,
            options.quotechar ? options.quotechar : no_char,
            options.comment ? options.comment : no_char};
  }
};

/**
 * @brief Returns the row context transitions of the character `c` preceded by `c_prev`, as a
 * per-character context (see make_char_context).
 */
__device__ __forceinline__ uint32_t char_row_context(int c,
                                                     int c_prev,
                                                     row_parse_chars const& chars)
{
  if (c_prev == chars.terminator) {
    if (c == chars.commentchar) {
      // Start of a new comment row
      return make_char_context(ROW_CTX_COMMENT, ROW_CTX_QUOTE, ROW_CTX_COMMENT, 1, 0, 1);
    } else if (c == chars.quotechar) {
      // Quoted string on newrow, or quoted string ending in terminator
      return make_char_context(ROW_CTX_QUOTE, ROW_CTX_NONE, ROW_CTX_QUOTE, 1, 0, 1);
    }
    // Start of a new row unless within a quote
    return make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE, 1, 0, 1);
  } else if (c == chars.quotechar) {
    // Quote handling uses ROW_CTX_COMMENT as a "pending exit" state to correctly handle
    // escaped quotes (""). When in QUOTE state and we see a quote, we can't immediately
    // exit because it might be the first quote of a "" escape sequence. We transition to
    // COMMENT (pending exit) and wait for the next character:
    //   - If next char is quote: it's a "" escape, return to QUOTE
    //   - If next char is anything else: exit confirmed, go to NONE
    // This doesn't conflict with actual comment handling because comments are only
    // detected at row boundaries (after newline), where COMMENT state is set with row
    // counting. Mid-row, COMMENT is purely used for this pending exit mechanism.
    if (c_prev == chars.delimiter) {
      // Quote after delimiter: start field or pending exit
      return make_char_context(ROW_CTX_QUOTE, ROW_CTX_COMMENT);
    } else if (c_prev == chars.quotechar) {
      // Quote after quote: "" escape or stay NONE (Spark compatibility)
      return make_char_context(ROW_CTX_NONE, ROW_CTX_COMMENT, ROW_CTX_QUOTE);
    }
    // Quote after regular char: pending exit or stay NONE
    return make_char_context(ROW_CTX_NONE, ROW_CTX_COMMENT);
  }
  // Non-quote char: stay in current state, or exit from pending
  return make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE);
}

/**
 * @brief Returns a 4-bit mask of the bytes of `word` equal to `c` (bit i for byte i).
 */
__device__ __forceinline__ uint32_t match_bytes(uint32_t word, int c)
{
  if (c == no_char) { return 0; }
  // 0xff in each matching byte
  auto const matches = __vcmpeq4(word, static_cast<uint32_t>(c & 0xff) * 0x0101'0101u);
  // Moves the low bit of byte i (bit 8i) to bit 21 + i: the partial products of the four bits
  // land on distinct bit positions, so the multiplication does not carry into bits 21..24
  return (((matches & 0x0101'0101u) * 0x0020'4081u) >> 21) & 0xf;
}

/**
 * @brief Computes the row contexts of a 32-character slice: for each possible parser state at the
 * start of the slice, the bitmap of the characters that start a row and the state at the end.
 *
 * The result holds the row start bitmaps for the NONE, QUOTE and COMMENT input states in `x`, `y`
 * and `z`, and the output state for each input state in `w` (2 bits per input state, in the order
 * NONE, QUOTE, COMMENT, EOF). Characters at or past `end` do not change the state, except that a
 * row starts at `end` in every state if `end_is_eof` (the row that ends the data).
 *
 * @param slice First character of the slice
 * @param end End of the characters to parse
 * @param end_is_eof Whether `end` is the end of the data
 * @param c_prev Character before the slice, or the terminator at the start of the data
 * @param chars Characters that determine where rows start
 */
__device__ uint4 slice_row_contexts(
  char const* slice, char const* end, bool end_is_eof, int c_prev, row_parse_chars const& chars)
{
  // Initial state is neutral context (no state transitions), zero rows
  uint4 ctx_map = {
    .x = 0,
    .y = 0,
    .z = 0,
    .w = (ROW_CTX_NONE << 0) | (ROW_CTX_QUOTE << 2) | (ROW_CTX_COMMENT << 4) | (ROW_CTX_EOF << 6)};

  if (slice + slice_chars <= end && reinterpret_cast<uintptr_t>(slice) % sizeof(uint4) == 0) {
    // All 32 characters are in range: locate terminators, quotes and comment characters
    auto const vectors      = reinterpret_cast<uint4 const*>(slice);
    uint32_t const words[8] = {vectors[0].x,
                               vectors[0].y,
                               vectors[0].z,
                               vectors[0].w,
                               vectors[1].x,
                               vectors[1].y,
                               vectors[1].z,
                               vectors[1].w};
    uint32_t terminators    = 0;
    uint32_t quotes         = 0;
    uint32_t comments       = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      terminators |= match_bytes(words[i], chars.terminator) << (4 * i);
      quotes |= match_bytes(words[i], chars.quotechar) << (4 * i);
      comments |= match_bytes(words[i], chars.commentchar) << (4 * i);
    }
    // Characters that follow a terminator
    auto const row_starts = (terminators << 1) | (c_prev == chars.terminator ? 1u : 0u);

    if ((quotes | comments) == 0) {
      // Without quote and comment characters, every character after a terminator starts a row
      // in the NONE and COMMENT states and none does in the QUOTE state. The COMMENT state exits
      // to NONE at the first character, so both end in NONE; QUOTE stays QUOTE.
      return {
        row_starts,
        0,
        row_starts,
        (ROW_CTX_NONE << 0) | (ROW_CTX_QUOTE << 2) | (ROW_CTX_NONE << 4) | (ROW_CTX_EOF << 6)};
    }

    // Only the characters that follow a terminator and the quote characters have transitions
    // other than the regular one (NONE->NONE, QUOTE->QUOTE, COMMENT->NONE, no row start). The
    // regular transition is idempotent, so each run of other characters is merged at once.
    auto constexpr regular = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE);
    uint32_t pos           = 0;  // first character not merged yet
    for (auto events = row_starts | quotes; events != 0; events &= events - 1) {
      auto const k = static_cast<uint32_t>(__ffs(events)) - 1;
      if (k > pos) { merge_char_context(ctx_map, regular, pos); }
      int const c_before = k > 0 ? slice[k - 1] : c_prev;
      merge_char_context(ctx_map, char_row_context(slice[k], c_before, chars), k);
      pos = k + 1;
    }
    if (pos < slice_chars) { merge_char_context(ctx_map, regular, pos); }
    return ctx_map;
  }

  // Loop through all 32 bytes and keep a bitmask of row starts for each possible input context
  auto cur = slice;
  int c;
  for (uint32_t pos = 0; pos < slice_chars; pos++, cur++, c_prev = c) {
    uint32_t ctx;
    if (cur < end) {
      c   = cur[0];
      ctx = char_row_context(c, c_prev, chars);
    } else if (end_is_eof && cur == end) {
      // Add a newline at data end (need the extra row offset to infer length of previous row)
      ctx = make_char_context(ROW_CTX_EOF, ROW_CTX_EOF, ROW_CTX_EOF, 1, 1, 1);
    } else {
      // Pass-through context (beyond chunk_size or data_end)
      ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_COMMENT);
    }
    // Merge with current context, keeping track of where new rows occur
    merge_char_context(ctx_map, ctx, pos);
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
 * @param chars Characters that determine where rows start
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
                         row_parse_chars chars)
{
  // Merge tree of the row contexts of the block's 32-character slices
  __shared__ packed_rowctx_t bk_ctxtree[bk_ctxtree_size];

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
  // The end of the data is in this window, and the chunk extends to it
  bool const end_is_eof = data_end_off <= data.size() && end == data_end;
  // Offset of `parse_pos` inside the local `data` window, clamped to avoid underflow
  auto const parse_off = parse_pos > start_offset ? parse_pos - start_offset : 0;
  uint32_t const t     = threadIdx.x;
  size_t block_pos =
    parse_off + blockIdx.x * static_cast<size_t>(rowofs_block_bytes) + t * slice_chars;
  auto const cur   = start + block_pos;
  int const c_prev = (cur > start && cur <= end) ? cur[-1] : chars.terminator;
  auto ctx_map     = slice_row_contexts(cur, end, end_is_eof, c_prev, chars);

  // Eliminate rows that start before byte_range_start
  if (start_offset + block_pos < byte_range_start) {
    uint32_t dist_minus1 =
      cuda::std::min(byte_range_start - (start_offset + block_pos) - 1, UINT64_C(31));
    uint32_t mask = 0xffff'fffe << dist_minus1;
    ctx_map.x &= mask;
    ctx_map.y &= mask;
    ctx_map.z &= mask;
  }

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
 * @brief Predicate that identifies blank and comment rows by the offset of their first character.
 *
 * A row is blank if it starts with the comment character or, when blank lines are skipped, with
 * the terminator or (for a '\n' terminator) a carriage return. Characters that do not apply are
 * replaced with one that does. If none does (no comment character and blank lines are kept), all
 * three are '\0', so rows that start with a NUL character are still removed. The offset of the end
 * of the data, which only marks where the last row ends, is never blank.
 */
struct is_blank_row {
  device_span<char const> data;
  char newline;
  char comment;
  char carriage;

  is_blank_row(parse_options_view const& options, device_span<char const> data)
    : data{data},
      newline{options.skipblanklines ? options.terminator : options.comment},
      comment{options.comment != '\0' ? options.comment : newline},
      carriage{(options.skipblanklines && options.terminator == '\n') ? '\r' : comment}
  {
  }

  __device__ bool operator()(uint64_t pos) const
  {
    return pos != data.size() &&
           (data[pos] == newline || data[pos] == comment || data[pos] == carriage);
  }
};

/**
 * @brief Slices per thread in single-pass row gathering.
 *
 * Determines the tile size (32KB). Larger tiles need less per-tile work (block scans, transitions,
 * row counts) but make the repair of a mis-speculated tile more expensive; with 1, 2 and 4 slices
 * per thread, the total row gathering time of 2 is within 3% of the best on the CSV reader
 * benchmarks with 0% to 100% quoted fields.
 */
constexpr uint32_t tile_slices_per_thread = 2;
/// Slices per tile, the data processed by one thread block in single-pass row gathering
constexpr uint32_t tile_slices = rowofs_block_dim * tile_slices_per_thread;
/// Characters per tile (32KB)
constexpr size_t tile_chars = static_cast<size_t>(tile_slices) * slice_chars;

/**
 * @brief Parser state transition function of a range of characters: the output state for each
 * input state, 2 bits per input state in the order NONE, QUOTE, COMMENT, EOF.
 */
using state_transition = uint8_t;

/// Transition of an empty range
constexpr state_transition identity_transition =
  ROW_CTX_NONE | (ROW_CTX_QUOTE << 2) | (ROW_CTX_COMMENT << 4) | (ROW_CTX_EOF << 6);

/**
 * @brief Composes the transitions of two consecutive ranges (associative, not commutative).
 */
struct compose_transitions {
  __device__ state_transition operator()(state_transition first, state_transition second) const
  {
    state_transition result = 0;
#pragma unroll
    for (uint32_t state = 0; state < 4; ++state) {
      auto const mid = (first >> (2 * state)) & 3;
      result |= ((second >> (2 * mid)) & 3) << (2 * state);
    }
    return result;
  }
};

/**
 * @brief Returns the output state of `transition` for the input state `state`.
 */
__device__ __forceinline__ uint32_t apply_transition(state_transition transition, uint32_t state)
{
  return (transition >> (2 * state)) & 3;
}

/**
 * @brief Computes the row start bitmaps of a tile's slices, given the parser state at the start of
 * the tile.
 *
 * Each thread computes the row contexts of its consecutive slices for every input state. A block
 * scan over the threads' state transitions gives each thread its input state, which selects the
 * row start bitmaps of its slices. Blank and comment rows are then cleared from the bitmaps.
 *
 * @param data Character data
 * @param chars Characters that determine where rows start
 * @param is_blank Identifies the blank and comment rows
 * @param tile Index of the tile
 * @param start_state Parser state at the start of the tile
 * @param[out] row_bitmaps Row start bitmap of every slice
 * @param[out] tile_row_counts Number of rows that start in each tile
 * @return The state transition of the tile, in thread 0
 */
__device__ state_transition gather_tile_row_bitmaps(device_span<char const> data,
                                                    row_parse_chars const& chars,
                                                    is_blank_row const& is_blank,
                                                    size_t tile,
                                                    uint32_t start_state,
                                                    uint32_t* row_bitmaps,
                                                    uint64_t* tile_row_counts)
{
  using block_scan   = cub::BlockScan<state_transition, rowofs_block_dim>;
  using block_reduce = cub::BlockReduce<uint32_t, rowofs_block_dim>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  auto const t           = threadIdx.x;
  auto const first_slice = (tile * rowofs_block_dim + t) * tile_slices_per_thread;
  auto const end         = data.data() + data.size();

  uint4 ctx_maps[tile_slices_per_thread];
  auto thread_transition = identity_transition;
#pragma unroll
  for (uint32_t j = 0; j < tile_slices_per_thread; ++j) {
    auto const slice = data.data() + (first_slice + j) * slice_chars;
    int const c_prev = (slice > data.data() && slice <= end) ? slice[-1] : chars.terminator;
    ctx_maps[j]      = slice_row_contexts(slice, end, true, c_prev, chars);
    thread_transition =
      compose_transitions{}(thread_transition, static_cast<state_transition>(ctx_maps[j].w));
  }

  // Input state of the thread: the tile's start state through the transitions of earlier threads
  state_transition prefix;
  state_transition tile_transition;
  block_scan(scan_storage)
    .ExclusiveScan(
      thread_transition, prefix, identity_transition, compose_transitions{}, tile_transition);
  auto state = apply_transition(prefix, start_state);

  uint32_t num_rows = 0;
#pragma unroll
  for (uint32_t j = 0; j < tile_slices_per_thread; ++j) {
    auto rowmap          = select_rowmap(ctx_maps[j], state);
    auto const slice_pos = (first_slice + j) * slice_chars;
    for (auto rows = rowmap; rows != 0; rows &= rows - 1) {
      auto const bit = __ffs(rows) - 1;
      if (is_blank(slice_pos + bit)) { rowmap &= ~(1u << bit); }
    }
    row_bitmaps[first_slice + j] = rowmap;
    num_rows += __popc(rowmap);
    state = apply_transition(static_cast<state_transition>(ctx_maps[j].w), state);
  }
  num_rows = block_reduce(reduce_storage).Sum(num_rows);
  if (t == 0) { tile_row_counts[tile] = num_rows; }
  return tile_transition;
}

/**
 * @brief Single-pass row gathering, speculative pass: gathers the row start bitmaps of every tile
 * assuming that it starts in the NONE state, and records the state transition of every tile.
 *
 * A tile starts in another state if a quoted field spans its start or ends right before it, so
 * the assumption fails for roughly the proportion of quoted characters in the data.
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  gather_speculative_row_bitmaps_gpu(device_span<char const> data,
                                     row_parse_chars chars,
                                     is_blank_row is_blank,
                                     uint32_t* row_bitmaps,
                                     uint64_t* tile_row_counts,
                                     state_transition* tile_transitions)
{
  auto const tile       = static_cast<size_t>(blockIdx.x);
  auto const transition = gather_tile_row_bitmaps(
    data, chars, is_blank, tile, ROW_CTX_NONE, row_bitmaps, tile_row_counts);
  if (threadIdx.x == 0) { tile_transitions[tile] = transition; }
}

/**
 * @brief Single-pass row gathering, repair pass: gathers the row start bitmaps of a tile again if
 * it does not start in the NONE state assumed by the speculative pass.
 *
 * @param tile_start_transitions Transition from the start of the data to the start of each tile
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  repair_row_bitmaps_gpu(device_span<char const> data,
                         row_parse_chars chars,
                         is_blank_row is_blank,
                         device_span<state_transition const> tile_start_transitions,
                         uint32_t* row_bitmaps,
                         uint64_t* tile_row_counts)
{
  auto const tile = static_cast<size_t>(blockIdx.x);
  // The data starts in the NONE state
  auto const start_state = apply_transition(tile_start_transitions[tile], ROW_CTX_NONE);
  if (start_state == ROW_CTX_NONE) { return; }
  gather_tile_row_bitmaps(data, chars, is_blank, tile, start_state, row_bitmaps, tile_row_counts);
}

/**
 * @brief Single-pass row gathering, output pass: converts the row start bitmaps of each tile into
 * row offsets.
 *
 * @param row_bitmaps Row start bitmap of every slice
 * @param tile_first_rows Index of the first row of each tile
 * @param[out] offsets Offsets of all rows
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  row_bitmaps_to_offsets_gpu(uint32_t const* row_bitmaps,
                             uint64_t const* tile_first_rows,
                             device_span<uint64_t> offsets)
{
  using block_scan = cub::BlockScan<uint32_t, rowofs_block_dim>;
  __shared__ typename block_scan::TempStorage scan_storage;

  auto const tile        = static_cast<size_t>(blockIdx.x);
  auto const first_slice = (tile * rowofs_block_dim + threadIdx.x) * tile_slices_per_thread;
  uint32_t rowmaps[tile_slices_per_thread];
  uint32_t num_rows = 0;
#pragma unroll
  for (uint32_t j = 0; j < tile_slices_per_thread; ++j) {
    rowmaps[j] = row_bitmaps[first_slice + j];
    num_rows += __popc(rowmaps[j]);
  }
  uint32_t first_row_in_tile;
  block_scan(scan_storage).ExclusiveSum(num_rows, first_row_in_tile);

  auto row = tile_first_rows[tile] + first_row_in_tile;
#pragma unroll
  for (uint32_t j = 0; j < tile_slices_per_thread; ++j) {
    auto const slice_pos = (first_slice + j) * slice_chars;
    for (auto rowmap = rowmaps[j]; rowmap != 0; rowmap &= rowmap - 1) {
      offsets[row++] = slice_pos + __ffs(rowmap) - 1;
    }
  }
}

size_t __host__ count_blank_rows(cudf::io::parse_options_view const& opts,
                                 device_span<char const> data,
                                 device_span<uint64_t const> row_offsets,
                                 cuda::stream_ref stream)
{
  return thrust::count_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          row_offsets.begin(),
                          row_offsets.end(),
                          is_blank_row{opts, data});
}

device_span<uint64_t> __host__ remove_blank_rows(cudf::io::parse_options_view const& options,
                                                 device_span<char const> data,
                                                 device_span<uint64_t> row_offsets,
                                                 cuda::stream_ref stream)
{
  auto new_end =
    thrust::remove_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      row_offsets.begin(),
                      row_offsets.end(),
                      is_blank_row{options, data});
  return row_offsets.subspan(0, new_end - row_offsets.begin());
}

cudf::detail::host_vector<column_type_histogram> detect_column_types(
  cudf::io::parse_options_view const& options,
  device_span<char const> const data,
  device_span<column_parse::flags const> const column_flags,
  device_span<uint64_t const> const row_starts,
  size_t const num_active_columns,
  cuda::stream_ref stream)
{
  // Calculate actual block count to use based on records count
  int const block_size = csvparse_block_dim;
  int const grid_size  = (row_starts.size() + block_size - 1) / block_size;

  auto d_stats = cudf::detail::make_zeroed_device_uvector_async<column_type_histogram>(
    num_active_columns, stream, cudf::get_current_device_resource_ref());

  data_type_detection<<<grid_size, block_size, 0, stream.get()>>>(
    options, data, column_flags, row_starts, d_stats);
  CUDF_CUDA_TRY(cudaGetLastError());

  return cudf::detail::make_host_vector(d_stats, stream);
}

void decode_row_column_data(cudf::io::parse_options_view const& options,
                            device_span<char> data,
                            device_span<column_parse::flags const> column_flags,
                            device_span<uint64_t const> row_offsets,
                            device_span<cudf::data_type const> dtypes,
                            device_span<void* const> columns,
                            device_span<cudf::bitmask_type* const> valids,
                            cuda::stream_ref stream)
{
  // Calculate actual block count to use based on records count
  auto const block_size = csvparse_block_dim;
  auto const num_rows   = row_offsets.size() - 1;
  auto const grid_size  = cudf::util::div_rounding_up_safe<size_t>(num_rows, block_size);

  convert_csv_to_cudf<<<grid_size, block_size, 0, stream.get()>>>(
    options, data, column_flags, row_offsets, dtypes, columns, valids);
  CUDF_CUDA_TRY(cudaGetLastError());
}

rmm::device_uvector<uint64_t> gather_all_row_offsets(parse_options_view const& options,
                                                     device_span<char const> data,
                                                     cuda::stream_ref stream)
{
  // The last tile holds the row that ends the data, at data.size()
  auto const num_tiles = data.size() / tile_chars + 1;
  auto const chars     = row_parse_chars::from(options);
  auto const is_blank  = is_blank_row{options, data};

  rmm::device_uvector<uint32_t> row_bitmaps(num_tiles * tile_slices, stream);
  // One more entry than tiles, so that the exclusive scan of the row counts ends with the total
  rmm::device_uvector<uint64_t> tile_rows(num_tiles + 1, stream);
  tile_rows.set_element_to_zero_async(num_tiles, stream);
  rmm::device_uvector<state_transition> tile_transitions(num_tiles, stream);

  gather_speculative_row_bitmaps_gpu<<<num_tiles, rowofs_block_dim, 0, stream.get()>>>(
    data, chars, is_blank, row_bitmaps.data(), tile_rows.data(), tile_transitions.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Transition from the start of the data to the start of each tile
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         tile_transitions.begin(),
                         tile_transitions.end(),
                         tile_transitions.begin(),
                         identity_transition,
                         compose_transitions{});

  // Every tile is processed at most once more, so the total work is at most twice that of a single
  // pass even if the speculation fails for every tile
  repair_row_bitmaps_gpu<<<num_tiles, rowofs_block_dim, 0, stream.get()>>>(
    data, chars, is_blank, tile_transitions, row_bitmaps.data(), tile_rows.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // First row of each tile, followed by the total number of rows
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         tile_rows.begin(),
                         tile_rows.end(),
                         tile_rows.begin());
  rmm::device_uvector<uint64_t> offsets(tile_rows.element(num_tiles, stream), stream);

  row_bitmaps_to_offsets_gpu<<<num_tiles, rowofs_block_dim, 0, stream.get()>>>(
    row_bitmaps.data(), tile_rows.data(), offsets);
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
    row_parse_chars::from(options));
  CUDF_CUDA_TRY(cudaGetLastError());

  return dim_grid;
}

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
