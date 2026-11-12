/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "abi/leaf_abi.cuh"
#include "kernel_abi.hpp"

#include <cuda/std/cstdint>

/**
 * @file decode_flat.cuh
 * @brief Tile decode of flat fixed-width Parquet pages.
 *
 * Depends only on the leaf ABI and the kernel ABI, so that the same body can be compiled into
 * libcudf with a leaf known at compile time and into a fragment whose leaf is linked at run time.
 */

namespace cudf::io::parquet::jit {

/// Number of runs a warp keeps in shared memory. A group of 32 values spans at most 32 runs, so
/// any size above 33 lets every refill make progress.
inline constexpr uint32_t run_table_size = 64;

/**
 * @brief Runs of a dictionary index stream that cover the values a warp is decoding.
 */
struct run_table {
  uint32_t start[run_table_size + 1];  ///< First value index of each run; `start[count]` is the
                                       ///< end of the last run
  uint32_t data[run_table_size];       ///< Bit-packed run: offset of its payload from
                                       ///< `page_ref::values`. Repeated run: the index
  uint8_t packed[run_table_size];      ///< Whether each run is bit-packed
  uint32_t count;                      ///< Number of runs in the table
  uint32_t next_offset;                ///< Offset of the first run header not yet parsed
};

/**
 * @brief Loads a little-endian value from an address of any alignment.
 *
 * Reads aligned 32-bit words and only those that hold at least one byte of the value.
 *
 * @tparam in_bytes Size of the value, 4 or 8
 * @param address Address of the first byte of the value
 * @return The value, as `uint32_t` or `uint64_t`
 */
template <int in_bytes>
__device__ inline auto load_value(uint8_t const* address)
{
  auto const bits      = reinterpret_cast<uintptr_t>(address);
  auto const* words    = reinterpret_cast<uint32_t const*>(bits & ~uintptr_t{3});
  uint32_t const shift = static_cast<uint32_t>(bits & 3) * 8;
  if constexpr (in_bytes == 4) {
    uint32_t value = words[0];
    if (shift != 0) { value = __funnelshift_r(value, words[1], shift); }
    return value;
  } else {
    uint32_t low  = words[0];
    uint32_t high = words[1];
    if (shift != 0) {
      uint32_t const next = words[2];
      low                 = __funnelshift_r(low, high, shift);
      high                = __funnelshift_r(high, next, shift);
    }
    return (static_cast<uint64_t>(high) << 32) | low;
  }
}

/**
 * @brief Extracts a bit field from a little-endian bit stream of any alignment.
 *
 * Reads aligned 32-bit words and only those that hold at least one bit of the field.
 *
 * @param base Address of bit 0 of the stream
 * @param bit_offset Offset of the field's lowest bit
 * @param width Width of the field, 1 to 32
 * @return The field, zero extended
 */
__device__ inline uint32_t extract_bits(uint8_t const* base, uint64_t bit_offset, uint32_t width)
{
  uint64_t const bit_address = reinterpret_cast<uintptr_t>(base) * 8 + bit_offset;
  auto const* words          = reinterpret_cast<uint32_t const*>((bit_address >> 5) << 2);
  uint32_t const shift       = static_cast<uint32_t>(bit_address) & 31;
  uint32_t field             = words[0] >> shift;
  if (shift + width > 32) { field |= words[1] << (32 - shift); }
  return width == 32 ? field : field & ((uint32_t{1} << width) - 1);
}

/**
 * @brief Moves a warp's run table forward along the index stream.
 *
 * Drops the runs that end at or before `need`, then parses run headers until the table covers
 * every value below `target` or is full. Values past the end of the stream, or past the part of
 * a run that the stream holds, decode as index 0.
 *
 * @param table The warp's run table
 * @param page The page being decoded
 * @param need Lowest value index still to be decoded
 * @param target Value index up to which coverage is useful
 */
__device__ inline void refill_runs(run_table& table,
                                   page_ref const& page,
                                   uint32_t need,
                                   uint32_t target)
{
  constexpr uint32_t stream_end = 0xffff'ffffu;

  uint32_t end   = table.start[table.count];
  uint32_t kept  = 0;
  uint32_t first = 0;
  while (first < table.count && table.start[first + 1] <= need) {
    ++first;
  }
  for (; first < table.count; ++first, ++kept) {
    table.start[kept]  = table.start[first];
    table.data[kept]   = table.data[first];
    table.packed[kept] = table.packed[first];
  }

  uint32_t offset          = table.next_offset;
  uint32_t const size      = page.values_size;
  uint32_t const bit_width = page.bit_width;
  while (kept < run_table_size && end < target) {
    if (offset >= size) {
      table.start[kept]  = end;
      table.data[kept]   = 0;
      table.packed[kept] = 0;
      ++kept;
      end = stream_end;
      break;
    }
    uint32_t header = 0;
    uint32_t shift  = 0;
    uint32_t byte;
    do {
      byte = page.values[offset++];
      header |= (byte & 0x7f) << shift;
      shift += 7;
    } while ((byte & 0x80) != 0 && offset < size && shift < 35);

    bool const packed = (header & 1) != 0;
    uint64_t num_values;
    uint32_t data;
    if (packed) {
      uint64_t const groups    = header >> 1;
      uint64_t const available = size - offset;
      uint64_t num_bytes       = groups * bit_width;
      num_values               = groups * 8;
      if (num_bytes > available) {
        num_values = (available * 8) / bit_width;
        num_bytes  = available;
      }
      data = offset;
      offset += static_cast<uint32_t>(num_bytes);
    } else {
      uint32_t const num_bytes = (bit_width + 7) >> 3;
      num_values               = header >> 1;
      data                     = 0;
      if (size - offset >= num_bytes) {
        for (uint32_t i = 0; i < num_bytes; ++i) {
          data |= static_cast<uint32_t>(page.values[offset + i]) << (8 * i);
        }
        offset += num_bytes;
      } else {
        offset = size;
      }
    }
    uint64_t const run_end = end + num_values;
    uint32_t const clamped =
      run_end < stream_end - 1 ? static_cast<uint32_t>(run_end) : stream_end - 1;
    if (clamped > need && clamped != end) {
      table.start[kept]  = end;
      table.data[kept]   = data;
      table.packed[kept] = packed;
      ++kept;
    }
    end = clamped;
  }
  table.start[kept] = end;
  table.count       = kept;
  table.next_offset = offset;
}

/**
 * @brief Decodes tiles of flat fixed-width pages, one tile per warp.
 *
 * Must be called by every thread of a block of `decode_flat_block_size` threads. Warp `w` of
 * the grid decodes `tiles[w]`. Rows are handled in groups of 32 aligned to the words of the
 * output validity mask, so that lane `l` of a warp owns output row `32 * j + l`.
 *
 * `leaf` is called once per non-null row of the decode call as
 * `leaf(ctx, output_row, value, dict_idx)` and stores the value.
 *
 * @tparam is_dict Whether the pages are dictionary encoded; PLAIN otherwise
 * @tparam in_bytes Size of a physical value, 4 or 8
 * @tparam nullable Whether the pages' columns have a maximum definition level of 1
 * @tparam Leaf Callable with the signature of the leaf ABI for `in_bytes`
 * @param pages Page descriptors of this decode call
 * @param tiles Tiles to decode, all of pages with this kernel's encoding, width and nullability
 * @param num_tiles Number of entries in `tiles`
 * @param checkpoints Checkpoints of all pages
 * @param contexts Leaf contexts, indexed by `page_ref::ctx_idx`
 * @param null_counts One counter per entry of `pages`; the nulls found are added to it
 * @param leaf The leaf
 */
template <bool is_dict, int in_bytes, bool nullable, typename Leaf>
__device__ inline void decode_flat_tiles(page_ref const* pages,
                                         tile_ref const* tiles,
                                         uint32_t num_tiles,
                                         checkpoint const* checkpoints,
                                         leaf_ctx const* contexts,
                                         int32_t* null_counts,
                                         Leaf leaf)
{
  constexpr uint32_t warp_size = 32;
  constexpr uint32_t stride    = checkpoint_stride(is_dict, nullable);

  uint32_t const lane       = threadIdx.x % warp_size;
  uint32_t const warp       = threadIdx.x / warp_size;
  uint32_t const tile_index = blockIdx.x * (decode_flat_block_size / warp_size) + warp;
  if (tile_index >= num_tiles) { return; }

  tile_ref const tile = tiles[tile_index];
  page_ref const page = pages[tile.page];
  leaf_ctx const ctx  = contexts[page.ctx_idx];

  uint32_t const tile_begin = tile.tile * tile_rows;
  uint32_t const range_end  = page.first_row + page.num_rows;
  uint32_t const row_begin  = tile_begin > page.first_row ? tile_begin : page.first_row;
  uint32_t const row_end = tile_begin + tile_rows < range_end ? tile_begin + tile_rows : range_end;
  if (row_begin >= row_end) { return; }

  int32_t const out_begin = static_cast<int32_t>(row_begin) + page.row_base;
  int32_t const out_end   = static_cast<int32_t>(row_end) + page.row_base;

  // Index of the next value in the page's value stream.
  uint32_t rank                         = row_begin;
  [[maybe_unused]] bool has_nulls       = false;
  [[maybe_unused]] uint32_t num_nulls   = 0;
  [[maybe_unused]] uint32_t level_shift = 0;
  if constexpr (nullable) {
    has_nulls   = (page.flags & page_flag_no_nulls) == 0;
    level_shift = (page.flags & page_flag_wide_levels) != 0 ? 1 : 0;
    if (has_nulls) {
      rank = checkpoints[page.first_checkpoint + tile.tile * stride + (stride - 1)].first_value;
      // The decode call starts inside this tile: count the non-null rows before it.
      for (uint32_t base = tile_begin; base < row_begin; base += warp_size) {
        uint32_t const row = base + lane;
        bool const valid   = row < row_begin && page.levels[size_t{row} << level_shift] != 0;
        rank += __popc(__ballot_sync(0xffff'ffffu, valid));
      }
    }
  }

  [[maybe_unused]] run_table* table     = nullptr;
  [[maybe_unused]] uint32_t covered_end = 0;
  [[maybe_unused]] uint32_t run         = 0;
  if constexpr (is_dict) {
    __shared__ run_table tables[decode_flat_block_size / warp_size];
    table = &tables[warp];
    if (lane == 0) {
      checkpoint const entry =
        checkpoints[page.first_checkpoint + (rank >> tile_rows_log2) * stride];
      table->count       = 0;
      table->start[0]    = entry.first_value;
      table->next_offset = entry.byte_offset;
    }
  }

  for (int32_t out_base = out_begin & ~31; out_base < out_end; out_base += warp_size) {
    int32_t const out_row = out_base + static_cast<int32_t>(lane);
    bool const active     = out_row >= out_begin && out_row < out_end;
    bool valid            = active;
    // Position of this lane's value in the value stream, and the positions the warp needs.
    uint32_t value_index = static_cast<uint32_t>(out_row - page.row_base);
    int32_t const first  = out_base > out_begin ? out_base : out_begin;
    uint32_t need_begin  = static_cast<uint32_t>(first - page.row_base);
    uint32_t need_end =
      static_cast<uint32_t>((out_base + 32 < out_end ? out_base + 32 : out_end) - page.row_base);

    if constexpr (nullable) {
      if (has_nulls) {
        valid                      = active && page.levels[size_t{value_index} << level_shift] != 0;
        uint32_t const active_mask = __ballot_sync(0xffff'ffffu, active);
        uint32_t const valid_mask  = __ballot_sync(0xffff'ffffu, valid);
        uint32_t const null_mask   = active_mask & ~valid_mask;
        value_index                = rank + __popc(valid_mask & ((uint32_t{1} << lane) - 1));
        need_begin                 = rank;
        rank += __popc(valid_mask);
        need_end = rank;
        num_nulls += __popc(null_mask);
        // The mask starts out all valid. A word this warp shares with another tile or page is
        // updated atomically.
        if (lane == 0 && page.valid != nullptr) {
          uint32_t* const word = page.valid + (out_base >> 5);
          if (active_mask == 0xffff'ffffu) {
            *word = valid_mask;
          } else if (null_mask != 0) {
            atomicAnd(word, ~null_mask);
          }
        }
      }
    }

    if constexpr (is_dict) {
      if (need_end > covered_end) {
        if (lane == 0) {
          uint32_t const remaining = static_cast<uint32_t>(out_end - first);
          refill_runs(*table, page, need_begin, need_begin + remaining);
        }
        __syncwarp();
        covered_end = table->start[table->count];
        run         = 0;
      }
    }

    if (valid) {
      uint32_t dict_idx                             = 0;
      decltype(load_value<in_bytes>(nullptr)) value = 0;
      if constexpr (is_dict) {
        while (table->start[run + 1] <= value_index) {
          ++run;
        }
        uint32_t index = table->data[run];
        if (table->packed[run] != 0) {
          index = page.bit_width == 0
                    ? 0
                    : extract_bits(
                        page.values + index,
                        static_cast<uint64_t>(value_index - table->start[run]) * page.bit_width,
                        page.bit_width);
        }
        if (index < page.dict_entries) {
          value    = load_value<in_bytes>(page.dict + size_t{index} * in_bytes);
          dict_idx = index + page.dict_first;
        } else {
          dict_idx = page.dict_entries + page.dict_first;
        }
      } else {
        if (value_index < page.values_size / in_bytes) {
          value = load_value<in_bytes>(page.values + size_t{value_index} * in_bytes);
        }
      }
      leaf(ctx, static_cast<uint32_t>(out_row), value, dict_idx);
    }
  }

  if constexpr (nullable) {
    if (lane == 0 && num_nulls != 0) {
      atomicAdd(null_counts + tile.page, static_cast<int32_t>(num_nulls));
    }
  }
}

}  // namespace cudf::io::parquet::jit
