/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "string_columns.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <thrust/scan.h>

#include <algorithm>
#include <limits>

namespace cudf::io::csv::gpu {
namespace {

constexpr int warp_size = cudf::detail::warp_size;

/// Rows of a block of the string column kernels
constexpr size_type block_size = 256;
static_assert(block_size % warp_size == 0, "Blocks of rows must be warp aligned");

/**
 * @brief Decoded size of the string of row `row` of a column: `decoded_strings::null_size` if the
 * row is null or past the last row, and `decoded_strings::oversized_size` if its field is too long
 * for a string.
 */
__device__ inline size_type decoded_size(decoded_strings const& column,
                                         thread_index_type row,
                                         size_type num_rows)
{
  return row < num_rows ? column.size(static_cast<size_type>(row)) : decoded_strings::null_size;
}

/// Number of characters of a string of the given decoded size: 0 for a null row, and for an
/// oversized field, which the reader rejects before any character is copied
__device__ inline size_type string_size(size_type decoded_size)
{
  return cuda::std::max(decoded_size, size_type{0});
}

/**
 * @brief Counts of a block of rows of a string column
 */
struct block_counts {
  int64_t valid;      ///< Valid rows
  int64_t chars;      ///< Characters
  int64_t separate;   ///< Strings copied separately (see `max_warp_copied_size`)
  int64_t oversized;  ///< Fields too long for a string (see `decoded_strings::oversized_size`)

  __device__ block_counts operator+(block_counts const& other) const
  {
    return {valid + other.valid,
            chars + other.chars,
            separate + other.separate,
            oversized + other.oversized};
  }
};

/**
 * @brief Longest string that `write_strings_kernel` copies with the other strings of its warp.
 *
 * A warp copies the strings of its rows at about one window of `warp_size` words per memory
 * latency, so the copy relies on many warps with few characters each: a long string keeps its warp
 * busy long after the others are done, and when most strings are long, there are too few warps to
 * hide the latency. Longer strings are copied separately, by a batched memcpy, which spreads each
 * of them over many threads. With strings of a single size, the warp copy is faster up to about
 * this size, and the batched memcpy from about twice this size.
 */
constexpr size_type max_warp_copied_size = 1024;

/// Whether a string is copied separately (see `max_warp_copied_size`)
__device__ inline bool is_copied_separately(size_type size) { return size > max_warp_copied_size; }

/**
 * @brief Computes the validity masks, valid counts and character counts of string columns.
 *
 * Block `(x, y)` covers the rows `[x * block_size, (x + 1) * block_size)` of column `y`. Blocks of
 * rows are warp aligned, so each warp owns the mask word of its 32 rows and writes it whole. The
 * block adds its counts to the column's counts, and writes its characters to
 * `block_chars[y * gridDim.x + x]`, from which the offsets of the blocks are computed.
 *
 * @param columns The strings of each column
 * @param num_rows Number of rows of each column
 * @param null_masks Validity mask of each column; every word that holds a row is written
 * @param valid_counts Number of valid rows of each column, accumulated into zeroed counts
 * @param chars_counts Number of characters of each column, accumulated into zeroed counts
 * @param separate_count Number of strings of all columns that are copied separately,
 * accumulated into a zeroed count
 * @param oversized_count Number of fields of all columns that are too long for a string,
 * accumulated into a zeroed count
 * @param block_chars Number of characters of each block of rows of each column
 */
CUDF_KERNEL void __launch_bounds__(block_size)
  compute_validity_and_sizes_kernel(decoded_strings const* columns,
                                    size_type num_rows,
                                    bitmask_type* const* null_masks,
                                    int64_t* valid_counts,
                                    int64_t* chars_counts,
                                    int64_t* separate_count,
                                    int64_t* oversized_count,
                                    int64_t* block_chars)
{
  auto const col = blockIdx.y;
  auto const row = static_cast<thread_index_type>(blockIdx.x) * block_size + threadIdx.x;

  auto const decoded      = decoded_size(columns[col], row, num_rows);
  auto const is_valid     = decoded != decoded_strings::null_size;
  auto const is_oversized = decoded == decoded_strings::oversized_size;
  auto const size         = string_size(decoded);

  auto const valid_bits = __ballot_sync(0xffff'ffffu, is_valid);
  if (row < num_rows and threadIdx.x % warp_size == 0) {
    null_masks[col][word_index(static_cast<size_type>(row))] = valid_bits;
  }

  using block_reduce = cub::BlockReduce<block_counts, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;
  auto const counts =
    block_reduce(reduce_storage)
      .Reduce(
        block_counts{
          is_valid ? 1 : 0, size, is_copied_separately(size) ? 1 : 0, is_oversized ? 1 : 0},
        cuda::std::plus<>{});
  if (threadIdx.x == 0) {
    block_chars[static_cast<int64_t>(col) * gridDim.x + blockIdx.x] = counts.chars;
    auto const add = [](int64_t& total, int64_t value) {
      if (value != 0) {
        cuda::atomic_ref<int64_t, cuda::thread_scope_device>{total}.fetch_add(
          value, cuda::memory_order_relaxed);
      }
    };
    add(valid_counts[col], counts.valid);
    add(chars_counts[col], counts.chars);
    add(*separate_count, counts.separate);
    add(*oversized_count, counts.oversized);
  }
}

/**
 * @brief The offsets and characters of an output strings column.
 */
struct strings_column_output {
  cudf::detail::output_offsetalator offsets;
  char* chars;
};

/**
 * @brief The strings copied separately, after `write_strings_kernel`: their sources, output
 * locations and sizes, in slots claimed with `next_slot`. The order of the slots depends on the
 * order in which the warps claim them, which does not matter to the copies.
 */
struct separate_copies {
  char const** sources;
  char** targets;
  size_type* sizes;
  int64_t* next_slot;
};

/**
 * @brief Copies characters of the rows of a warp to their output, in `[begin, end)`.
 *
 * The characters are those of consecutive rows of the warp, whose strings are contiguous in the
 * output. The warp writes them in windows of one 4-byte word per lane, aligned in the output: each
 * lane assembles its word from the bytes of the strings that the word overlaps, and stores it
 * whole, or byte by byte where the word also overlaps characters outside of the range, which
 * other warps (or the separate copies) write. So the stores are coalesced, and neighboring lanes
 * read neighboring source bytes.
 *
 * The byte at output position `p` belongs to the last row whose string starts at or before `p`:
 * the strings are contiguous and in row order, so any later row starts past `p`, and an earlier
 * row that starts at the same position is empty.
 *
 * @param starts Output position of the string of each of the warp's rows, in row order, and the
 * maximum `int64_t` for the rows past the last row
 * @param sources Characters of each of the warp's rows (only read for non-empty strings)
 * @param begin Output position of the first character to copy
 * @param end Output position past the last character to copy
 * @param chars Output characters of the column
 */
__device__ void copy_warp_characters(
  int64_t const* starts, char const* const* sources, int64_t begin, int64_t end, char* chars)
{
  constexpr int word_size = sizeof(uint32_t);
  auto const lane         = static_cast<int>(threadIdx.x % warp_size);
  for (auto window = begin & ~int64_t{word_size - 1}; window < end;
       window += warp_size * word_size) {
    auto const first = window + lane * word_size;
    // Last row that starts at or before `first`, or the first row if none does
    int row = 0;
    for (int delta = warp_size / 2; delta > 0; delta /= 2) {
      if (starts[row + delta] <= first) { row += delta; }
    }
    uint32_t word        = 0;
    uint32_t bytes_owned = 0;
    for (int i = 0; i < word_size; ++i) {
      auto const position = first + i;
      while (row + 1 < warp_size and starts[row + 1] <= position) {
        ++row;
      }
      if (position >= begin and position < end) {
        auto const byte = static_cast<uint8_t>(sources[row][position - starts[row]]);
        word |= static_cast<uint32_t>(byte) << (8 * i);
        bytes_owned |= 1u << i;
      }
    }
    if (bytes_owned == (1u << word_size) - 1) {
      *reinterpret_cast<uint32_t*>(chars + first) = word;
    } else {
      for (int i = 0; i < word_size; ++i) {
        if (bytes_owned & (1u << i)) { chars[first + i] = static_cast<char>(word >> (8 * i)); }
      }
    }
  }
}

/**
 * @brief Writes the offsets and copies the characters of string columns.
 *
 * Block `(x, y)` covers tile `t = y * gridDim.x + x`, if there is such a tile: the rows
 * `[r * block_size, (r + 1) * block_size)` of column `t % num_columns`, where
 * `r = t / num_columns`. The tiles of the same rows of all columns are adjacent, so the strings of
 * these rows, which are close to each other in the source data, are copied together and share
 * cached source reads.
 *
 * The offset of a row is the offset of its block of rows (`block_offsets`) plus the characters
 * of the preceding rows of the block. Each warp copies the strings of its rows with
 * `copy_warp_characters`, except the strings copied separately, which it lists in `separate`
 * instead: they split the warp's rows into runs of rows whose strings are contiguous in the
 * output, which are copied one after the other.
 *
 * The kernel is only launched when no field is oversized, so the sizes it reads are those that
 * `compute_validity_and_sizes_kernel` counted.
 *
 * @param columns The strings of each column
 * @param outputs Output offsets and characters of each column, sized by
 * `compute_validity_and_sizes_kernel`
 * @param block_offsets Offset of the first character of each block of rows of each column, in
 * the layout of the `block_chars` of `compute_validity_and_sizes_kernel`
 * @param num_rows Number of rows of each column
 * @param num_row_blocks Number of blocks of rows of each column
 * @param num_columns Number of columns
 * @param separate The strings copied separately, with room for all of them
 */
CUDF_KERNEL void __launch_bounds__(block_size)
  write_strings_kernel(decoded_strings const* columns,
                       strings_column_output const* outputs,
                       int64_t const* block_offsets,
                       size_type num_rows,
                       size_type num_row_blocks,
                       size_type num_columns,
                       separate_copies separate)
{
  using block_scan = cub::BlockScan<int64_t, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ int64_t starts[block_size];
  __shared__ char const* sources[block_size];

  auto const num_tiles  = static_cast<int64_t>(num_row_blocks) * num_columns;
  auto const lane       = static_cast<int>(threadIdx.x % warp_size);
  auto const warp_first = static_cast<int>(threadIdx.x) - lane;

  // The condition is uniform across the block
  auto const tile = static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  if (tile >= num_tiles) { return; }
  auto const col       = static_cast<size_type>(tile % num_columns);
  auto const row_block = tile / num_columns;
  auto const row       = row_block * block_size + threadIdx.x;
  auto const in_range  = row < num_rows;
  // Null rows and rows past the last row have no characters, and only non-empty strings have
  // pointers to read
  auto const size = string_size(decoded_size(columns[col], row, num_rows));
  auto const source =
    size > 0 ? columns[col].chars(static_cast<size_type>(row)) : static_cast<char const*>(nullptr);

  int64_t offset = 0;
  block_scan(scan_storage).ExclusiveSum(static_cast<int64_t>(size), offset);
  offset += block_offsets[col * static_cast<int64_t>(num_row_blocks) + row_block];
  auto const output = outputs[col];
  if (in_range) {
    output.offsets[static_cast<size_type>(row)] = offset;
    if (row == num_rows - 1) { output.offsets[num_rows] = offset + size; }
  }

  // The warp claims the slots of its strings that are copied separately at once
  auto const copied_separately = is_copied_separately(size);
  auto const separate_rows     = __ballot_sync(0xffff'ffffu, copied_separately);
  if (separate_rows != 0) {
    int64_t first_slot = 0;
    if (lane == 0) {
      first_slot =
        cuda::atomic_ref<int64_t, cuda::thread_scope_device>{*separate.next_slot}.fetch_add(
          __popc(separate_rows), cuda::memory_order_relaxed);
    }
    first_slot = __shfl_sync(0xffff'ffffu, first_slot, 0);
    if (copied_separately) {
      auto const slot        = first_slot + __popc(separate_rows & ((1u << lane) - 1));
      separate.sources[slot] = source;
      separate.targets[slot] = output.chars + offset;
      separate.sizes[slot]   = size;
    }
  }

  // Rows past the last row sort after every row, and have no characters
  starts[threadIdx.x]  = in_range ? offset : cuda::std::numeric_limits<int64_t>::max();
  sources[threadIdx.x] = source;
  // Offset past the characters of the warp's rows: rows past the last row have no characters
  auto const warp_end = __shfl_sync(0xffff'ffffu, offset + size, warp_size - 1);
  __syncwarp();
  // Runs of rows between the rows whose strings are copied separately; rows past the last row
  // have no characters, so the runs can include them
  for (int run_first = 0; run_first < warp_size;) {
    auto const later_separate = separate_rows >> run_first;
    auto const run_end = later_separate != 0 ? run_first + __ffs(later_separate) - 1 : warp_size;
    auto const begin   = starts[warp_first + run_first];
    auto const end     = run_end < warp_size ? starts[warp_first + run_end] : warp_end;
    if (begin < end) {
      copy_warp_characters(starts + warp_first, sources + warp_first, begin, end, output.chars);
    }
    run_first = run_end + 1;
  }
}

}  // namespace

std::vector<std::unique_ptr<column>> make_strings_columns(host_span<decoded_strings const> columns,
                                                          size_type num_rows,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  if (columns.empty()) { return {}; }
  auto const num_columns = columns.size();
  if (num_rows == 0) {
    std::vector<std::unique_ptr<column>> empty_columns;
    std::generate_n(std::back_inserter(empty_columns), num_columns, [] {
      return make_empty_column(type_id::STRING);
    });
    return empty_columns;
  }
  auto const temp_mr        = cudf::get_current_device_resource_ref();
  auto const num_row_blocks = cudf::util::div_rounding_up_safe(num_rows, block_size);

  // Validity masks and counts; the kernel writes every mask word that holds a row
  std::vector<rmm::device_buffer> null_masks;
  auto h_masks = cudf::detail::make_host_vector<bitmask_type*>(num_columns, stream);
  for (size_t col = 0; col < num_columns; ++col) {
    null_masks.emplace_back(bitmask_allocation_size_bytes(num_rows), stream, mr);
    h_masks[col] = static_cast<bitmask_type*>(null_masks.back().data());
  }
  auto const d_columns = cudf::detail::make_device_uvector_async(columns, stream, temp_mr);
  auto const d_masks   = cudf::detail::make_device_uvector_async(h_masks, stream, temp_mr);
  // The valid counts of the columns, their character counts, the number of strings copied
  // separately, and the number of fields too long for a string
  auto counts =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(2 * num_columns + 2, stream, temp_mr);
  auto const valid_counts    = counts.data();
  auto const chars_counts    = counts.data() + num_columns;
  auto const separate_count  = counts.data() + 2 * num_columns;
  auto const oversized_count = counts.data() + 2 * num_columns + 1;
  rmm::device_uvector<int64_t> block_chars(num_columns * num_row_blocks, stream, temp_mr);

  constexpr size_t max_grid_columns = std::numeric_limits<uint16_t>::max();
  for (size_t first = 0; first < num_columns; first += max_grid_columns) {
    dim3 const grid(num_row_blocks, std::min(max_grid_columns, num_columns - first));
    compute_validity_and_sizes_kernel<<<grid, block_size, 0, stream.get()>>>(
      d_columns.data() + first,
      num_rows,
      d_masks.data() + first,
      valid_counts + first,
      chars_counts + first,
      separate_count,
      oversized_count,
      block_chars.data() + first * num_row_blocks);
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  // Offset of each block of rows within its column
  rmm::device_uvector<int64_t> block_offsets(block_chars.size(), stream, temp_mr);
  auto const block_columns = cuda::transform_iterator(
    cuda::counting_iterator<int64_t>{0},
    cuda::proclaim_return_type<int64_t>(
      [num_row_blocks] __device__(int64_t idx) { return idx / num_row_blocks; }));
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                block_columns,
                                block_columns + block_chars.size(),
                                block_chars.begin(),
                                block_offsets.begin());

  auto const h_counts = cudf::detail::make_host_vector(counts, stream);
  CUDF_EXPECTS(h_counts[2 * num_columns + 1] == 0,
               "A string field exceeds the size limit of a string",
               std::overflow_error);

  std::vector<std::unique_ptr<column>> offsets_columns;
  std::vector<rmm::device_uvector<char>> chars;
  auto h_outputs = cudf::detail::make_host_vector<strings_column_output>(num_columns, stream);
  for (size_t col = 0; col < num_columns; ++col) {
    auto const chars_size = h_counts[num_columns + col];
    offsets_columns.push_back(
      cudf::strings::detail::create_offsets_child_column(chars_size, num_rows + 1, stream, mr));
    chars.emplace_back(chars_size, stream, mr);
    h_outputs[col] = {cudf::detail::offsetalator_factory::make_output_iterator(
                        offsets_columns.back()->mutable_view()),
                      chars.back().data()};
  }
  auto const d_outputs = cudf::detail::make_device_uvector_async(h_outputs, stream, temp_mr);

  auto const num_separate = h_counts[2 * num_columns];
  rmm::device_uvector<char const*> separate_sources(num_separate, stream, temp_mr);
  rmm::device_uvector<char*> separate_targets(num_separate, stream, temp_mr);
  rmm::device_uvector<size_type> separate_sizes(num_separate, stream, temp_mr);
  auto next_separate_slot =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(1, stream, temp_mr);
  auto const num_tiles  = static_cast<int64_t>(num_row_blocks * num_columns);
  auto const max_grid_x = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  dim3 const grid(std::min(num_tiles, max_grid_x),
                  cudf::util::div_rounding_up_safe(num_tiles, max_grid_x));
  write_strings_kernel<<<grid, block_size, 0, stream.get()>>>(
    d_columns.data(),
    d_outputs.data(),
    block_offsets.data(),
    num_rows,
    num_row_blocks,
    static_cast<size_type>(num_columns),
    separate_copies{separate_sources.data(),
                    separate_targets.data(),
                    separate_sizes.data(),
                    next_separate_slot.data()});
  CUDF_CUDA_TRY(cudaGetLastError());
  if (num_separate > 0) {
    cudf::detail::batched_memcpy_async(separate_sources.begin(),
                                       separate_targets.begin(),
                                       separate_sizes.begin(),
                                       num_separate,
                                       stream);
  }

  std::vector<std::unique_ptr<column>> strings_columns;
  for (size_t col = 0; col < num_columns; ++col) {
    auto const null_count = num_rows - static_cast<size_type>(h_counts[col]);
    strings_columns.push_back(make_strings_column(
      num_rows,
      std::move(offsets_columns[col]),
      chars[col].release(),
      null_count,
      null_count > 0 ? std::move(null_masks[col]) : rmm::device_buffer{0, stream, mr}));
  }
  return strings_columns;
}

}  // namespace cudf::io::csv::gpu
