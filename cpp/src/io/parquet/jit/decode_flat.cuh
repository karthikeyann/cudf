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

/**
 * @brief Runs of a hybrid stream that cover the values a warp is decoding.
 *
 * @tparam size Number of runs the table holds. The values of one loop turn span at most
 * `32 * steps` runs, so any size above `32 * steps + 1` lets every refill make progress.
 */
template <uint32_t size>
struct run_table {
  uint32_t start[size + 1];  ///< First value index of each run; `start[count]` is the end of the
                             ///< last run
  uint32_t data[size];       ///< Bit-packed run: offset of its payload in the stream.
                             ///< Repeated run: the value
  uint8_t packed[size];      ///< Whether each run is bit-packed
  uint32_t count;            ///< Number of runs in the table
  uint32_t next_offset;      ///< Offset of the first run header not yet parsed
};

/**
 * @brief An RLE / bit-packed hybrid stream: dictionary indices or levels.
 */
struct hybrid_stream {
  uint8_t const* data;  ///< First run header
  uint32_t size;        ///< Size in bytes
  uint32_t bit_width;   ///< Width of a value, at most 32
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
 * @brief Moves a warp's run table forward along its stream.
 *
 * Drops the runs that end at or before `need`, then parses run headers until the table covers
 * every value below `target` or is full. Values past the end of the stream, or past the part of
 * a run that the stream holds, decode as 0.
 *
 * @param table The warp's run table
 * @param stream The stream the table describes
 * @param need Lowest value index still to be decoded
 * @param target Value index up to which coverage is useful
 */
template <uint32_t capacity>
__device__ inline void refill_runs(run_table<capacity>& table,
                                   hybrid_stream const& stream,
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
  uint32_t const size      = stream.size;
  uint32_t const bit_width = stream.bit_width;
  while (kept < capacity && end < target) {
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
      byte = stream.data[offset++];
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
          data |= static_cast<uint32_t>(stream.data[offset + i]) << (8 * i);
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
 * @brief Rows of one tile that belong to the decode call, and where they are stored.
 */
struct tile_range {
  uint32_t tile_begin;  ///< First page row of the tile
  uint32_t row_begin;   ///< First page row to decode
  uint32_t row_end;     ///< One past the last page row to decode
  int32_t out_begin;    ///< Output row of `row_begin`
  int32_t out_end;      ///< Output row of `row_end`
};

/**
 * @name Stages of tile decode
 *
 * `decode_tiles` is written once against three stages. A kernel for a column is a composition
 * of one type per stage, chosen at compile time; today's kernels are the compositions listed
 * under `decode_flat_tiles`.
 *
 * **Level source**: turns rows into validity and value stream positions.
 * - `static constexpr uint32_t checkpoints_per_tile`: checkpoint entries it needs per tile.
 * - Constructor `(page, tile, range, entries, stride, slot, lane)`: `entries` are the page's
 *   checkpoint entries, of which tile `k` owns `entries[k * stride + slot]`. Called by all
 *   lanes of the warp; holds the per-tile state.
 * - `uint32_t position() const`: position in the value stream of the next non-null row. The
 *   same in every lane.
 * - `bool step(out_base, lane, out_row, value_index)`: handles the 32 output rows starting at
 *   `out_base`, a multiple of 32, of which lane `l` owns row `out_base + l`. Sets `out_row`,
 *   and returns whether that row is a non-null row of the decode call; if so, `value_index`
 *   is the position of its value. Stores the validity of the rows. Called by all lanes.
 * - `void finish(null_count, lane)`: adds the nulls of the tile to `*null_count`.
 * `required_levels` and `optional_levels` cover flat columns. A source for nested columns
 * would read repetition levels as well, map rows to positions through the offsets it writes,
 * and hand out more than one value per row.
 *
 * **Value source**: returns the value at a position of the value stream. One per encoding.
 * - `using value_type`: `uint32_t` or `uint64_t`, the physical value zero extended.
 * - `static constexpr uint32_t checkpoints_per_tile`.
 * - `static constexpr uint32_t max_turn_values`: most positions one `prepare` may span.
 * - Constructor `(page, entries, stride, slot, position, lane)`: `position` is the first
 *   position the tile needs. Called by all lanes; holds the per-tile state, in registers or
 *   in shared memory it declares itself.
 * - `void prepare(need_begin, need_end, target, lane)`: makes positions
 *   `[need_begin, need_end)` readable; positions below `need_begin` are not read again and
 *   coverage up to `target` will be asked for later. Arguments are the same in every lane.
 * - `value_type get(position, dict_idx) const`: the value at a prepared position. Sets
 *   `dict_idx` on dictionary pages and leaves it alone otherwise. Reads must stay inside the
 *   page; a position the page does not hold yields 0.
 * `plain_values` and `dictionary_values` exist. DELTA_BINARY_PACKED fits as a source whose
 * checkpoint is the value before a mini-block and whose `prepare` unpacks and prefix-sums the
 * mini-blocks of the turn; BYTE_STREAM_SPLIT and BOOLEAN need no checkpoint.
 *
 * **Leaf**: transform, predicate and sink, called as `leaf(ctx, out_row, value, dict_idx)`
 * once per non-null row, with the ABI of `abi/leaf_abi.cuh`. A value wider than 64 bits
 * (DECIMAL128, FIXED_LEN_BYTE_ARRAY, INT96) needs a `value_type` of that width and a third
 * leaf symbol for it. A variable-width sink (strings) is two compositions over the same
 * sources: a leaf that only adds up sizes per row, then, once offsets are known, a leaf that
 * copies bytes; the value source then returns a pointer and a length instead of the bits.
 *
 * Encoding, value width and nullability are compile-time properties of the stage types, so a
 * composition contains only the code of its stages.
 * @{
 */

/**
 * @brief Level source of a REQUIRED column: every row is valid and holds value `row`.
 */
struct required_levels {
  static constexpr uint32_t checkpoints_per_tile = 0;

  __device__ required_levels(page_ref const& page,
                             tile_ref,
                             tile_range const& range,
                             checkpoint const*,
                             uint32_t,
                             uint32_t,
                             uint32_t)
    : _row_base{page.row_base},
      _out_begin{range.out_begin},
      _out_end{range.out_end},
      _position{range.row_begin}
  {
  }

  [[nodiscard]] __device__ uint32_t position() const { return _position; }

  __device__ bool step(int32_t out_base, uint32_t lane, int32_t& out_row, uint32_t& value_index)
  {
    out_row            = out_base + static_cast<int32_t>(lane);
    value_index        = static_cast<uint32_t>(out_row - _row_base);
    int32_t const next = out_base + 32 < _out_end ? out_base + 32 : _out_end;
    if (next > _out_begin) { _position = static_cast<uint32_t>(next - _row_base); }
    return out_row >= _out_begin && out_row < _out_end;
  }

  __device__ void finish(int32_t*, uint32_t) const {}

 private:
  int32_t _row_base;
  int32_t _out_begin;
  int32_t _out_end;
  uint32_t _position;
};

/**
 * @brief Level source of a flat OPTIONAL column, reading the definition level stream.
 *
 * Keeps the runs of the level stream that cover the current step in a shared-memory table
 * owned by the warp. It uses two checkpoint entries per tile: the first locates the run that
 * holds the level of page row `k * tile_rows`, the second holds the number of non-null rows
 * before that row. A page marked `page_flag_no_nulls` behaves like a REQUIRED page.
 */
struct optional_levels {
  static constexpr uint32_t checkpoints_per_tile = 2;

  __device__ optional_levels(page_ref const& page,
                             tile_ref tile,
                             tile_range const& range,
                             checkpoint const* entries,
                             uint32_t stride,
                             uint32_t slot,
                             uint32_t lane)
    : _stream{page.levels, static_cast<uint32_t>(page.values - page.levels), 1},
      _valid{page.valid},
      _table{nullptr},
      _row_base{page.row_base},
      _out_begin{range.out_begin},
      _out_end{range.out_end},
      _position{range.row_begin},
      _num_nulls{0},
      _covered_end{0},
      _run{0},
      _has_nulls{(page.flags & page_flag_no_nulls) == 0}
  {
    if (_has_nulls) {
      __shared__ table_type tables[decode_flat_block_size / 32];
      _table                       = &tables[(threadIdx.x % decode_flat_block_size) / 32];
      checkpoint const* const mine = entries + tile.tile * stride + slot;
      if (lane == 0) {
        _table->count       = 0;
        _table->start[0]    = mine[0].first_value;
        _table->next_offset = mine[0].byte_offset;
      }
      _position = mine[1].first_value;
      // The decode call starts inside this tile: count the non-null rows before it.
      for (uint32_t base = range.tile_begin; base < range.row_begin; base += 32) {
        uint32_t const end = base + 32 < range.row_begin ? base + 32 : range.row_begin;
        cover(base, end, range.row_end, lane);
        uint32_t const row = base + lane;
        bool const valid   = row < end && level_at(row) != 0;
        _position += __popc(__ballot_sync(0xffff'ffffu, valid));
      }
    }
  }

  [[nodiscard]] __device__ uint32_t position() const { return _position; }

  __device__ bool step(int32_t out_base, uint32_t lane, int32_t& out_row, uint32_t& value_index)
  {
    out_row             = out_base + static_cast<int32_t>(lane);
    value_index         = static_cast<uint32_t>(out_row - _row_base);
    bool const active   = out_row >= _out_begin && out_row < _out_end;
    int32_t const first = out_base > _out_begin ? out_base : _out_begin;
    int32_t const last  = out_base + 32 < _out_end ? out_base + 32 : _out_end;
    if (not _has_nulls) {
      if (last > _out_begin) { _position = static_cast<uint32_t>(last - _row_base); }
      return active;
    }
    if (last > first) {
      cover(static_cast<uint32_t>(first - _row_base),
            static_cast<uint32_t>(last - _row_base),
            static_cast<uint32_t>(_out_end - _row_base),
            lane);
    }
    bool const valid           = active && level_at(value_index) != 0;
    uint32_t const active_mask = __ballot_sync(0xffff'ffffu, active);
    uint32_t const valid_mask  = __ballot_sync(0xffff'ffffu, valid);
    uint32_t const null_mask   = active_mask & ~valid_mask;
    value_index                = _position + __popc(valid_mask & ((uint32_t{1} << lane) - 1));
    _position += __popc(valid_mask);
    _num_nulls += __popc(null_mask);
    // The mask starts out all valid. A word this warp shares with another tile or page is
    // updated atomically.
    if (lane == 0 && _valid != nullptr) {
      uint32_t* const word = _valid + (out_base >> 5);
      if (active_mask == 0xffff'ffffu) {
        *word = valid_mask;
      } else if (null_mask != 0) {
        atomicAnd(word, ~null_mask);
      }
    }
    return valid;
  }

  __device__ void finish(int32_t* null_count, uint32_t lane) const
  {
    if (lane == 0 && _num_nulls != 0) { atomicAdd(null_count, static_cast<int32_t>(_num_nulls)); }
  }

 private:
  using table_type = run_table<64>;

  // Makes the levels of page rows `[begin, end)` readable. The same in every lane.
  __device__ void cover(uint32_t begin, uint32_t end, uint32_t target, uint32_t lane)
  {
    if (end > _covered_end) {
      if (lane == 0) { refill_runs(*_table, _stream, begin, target); }
      __syncwarp();
      _covered_end = _table->start[_table->count];
      _run         = 0;
    }
  }

  // Returns the level of a covered page row.
  [[nodiscard]] __device__ uint32_t level_at(uint32_t row)
  {
    while (_table->start[_run + 1] <= row) {
      ++_run;
    }
    uint32_t const data = _table->data[_run];
    if (_table->packed[_run] == 0) { return data; }
    uint32_t const bit = row - _table->start[_run];
    return (_stream.data[data + (bit >> 3)] >> (bit & 7)) & 1;
  }

  hybrid_stream _stream;
  uint32_t* _valid;
  table_type* _table;
  int32_t _row_base;
  int32_t _out_begin;
  int32_t _out_end;
  uint32_t _position;
  uint32_t _num_nulls;
  uint32_t _covered_end;
  uint32_t _run;
  bool _has_nulls;
};

/**
 * @brief Value source of PLAIN pages: value `i` is stored at byte `i * in_bytes`.
 *
 * @tparam in_bytes Size of a physical value, 4 or 8
 */
template <int in_bytes>
struct plain_values {
  using value_type                               = decltype(load_value<in_bytes>(nullptr));
  static constexpr uint32_t checkpoints_per_tile = 0;
  static constexpr uint32_t max_turn_values      = 0xffff'ffffu;

  __device__ plain_values(
    page_ref const& page, checkpoint const*, uint32_t, uint32_t, uint32_t, uint32_t)
    : _values{page.values}, _num_values{page.values_size / in_bytes}
  {
  }

  __device__ void prepare(uint32_t, uint32_t, uint32_t, uint32_t) const {}

  [[nodiscard]] __device__ value_type get(uint32_t position, uint32_t&) const
  {
    return position < _num_values ? load_value<in_bytes>(_values + size_t{position} * in_bytes)
                                  : value_type{0};
  }

 private:
  uint8_t const* _values;
  uint32_t _num_values;
};

/**
 * @brief Value source of RLE_DICTIONARY pages.
 *
 * Keeps the runs of the index stream that cover the current turn in a shared-memory table
 * owned by the warp. Its checkpoint entry locates the run that holds value `k * tile_rows`.
 * An index without a dictionary entry yields 0.
 *
 * @tparam in_bytes Size of a physical value, 4 or 8
 * @tparam turn_values Most positions one `prepare` may span
 */
template <int in_bytes, uint32_t turn_values>
struct dictionary_values {
  using value_type                               = decltype(load_value<in_bytes>(nullptr));
  static constexpr uint32_t checkpoints_per_tile = 1;
  static constexpr uint32_t max_turn_values      = turn_values;

  __device__ dictionary_values(page_ref const& page,
                               checkpoint const* entries,
                               uint32_t stride,
                               uint32_t slot,
                               uint32_t position,
                               uint32_t lane)
    : _page{&page}, _stream{page.values, page.values_size, page.bit_width}, _covered_end{0}, _run{0}
  {
    __shared__ table_type tables[decode_flat_block_size / 32];
    _table = &tables[(threadIdx.x % decode_flat_block_size) / 32];
    if (lane == 0) {
      checkpoint const entry = entries[(position >> tile_rows_log2) * stride + slot];
      _table->count          = 0;
      _table->start[0]       = entry.first_value;
      _table->next_offset    = entry.byte_offset;
    }
  }

  __device__ void prepare(uint32_t need_begin, uint32_t need_end, uint32_t target, uint32_t lane)
  {
    if (need_end > _covered_end) {
      if (lane == 0) { refill_runs(*_table, _stream, need_begin, target); }
      __syncwarp();
      _covered_end = _table->start[_table->count];
      _run         = 0;
    }
  }

  [[nodiscard]] __device__ value_type get(uint32_t position, uint32_t& dict_idx)
  {
    while (_table->start[_run + 1] <= position) {
      ++_run;
    }
    uint32_t index = _table->data[_run];
    if (_table->packed[_run] != 0) {
      index =
        _page->bit_width == 0
          ? 0
          : extract_bits(_page->values + index,
                         static_cast<uint64_t>(position - _table->start[_run]) * _page->bit_width,
                         _page->bit_width);
    }
    if (index < _page->dict_entries) {
      dict_idx = index + _page->dict_first;
      return load_value<in_bytes>(_page->dict + size_t{index} * in_bytes);
    }
    dict_idx = _page->dict_entries + _page->dict_first;
    return value_type{0};
  }

 private:
  using table_type = run_table<turn_values + 32>;
  page_ref const* _page;
  hybrid_stream _stream;
  table_type* _table;
  uint32_t _covered_end;
  uint32_t _run;
};

/** @} */

/**
 * @brief Decodes tiles of pages, one tile per warp.
 *
 * Must be called by every thread of a block of `decode_flat_block_size` threads. Warp `w` of
 * the grid decodes `tiles[w]`. Rows are handled in groups of 32 aligned to the words of the
 * output validity mask, so that lane `l` of a warp owns output row `32 * j + l`.
 *
 * See "Stages of tile decode" for the contracts of the three stage types.
 *
 * @tparam LevelSource Level source of the pages
 * @tparam ValueSource Value source of the pages
 * @tparam steps Number of 32-row groups a warp decodes per loop turn. The loads of one turn are
 * issued before its stores, so this sets how many independent loads each lane has in flight
 * @tparam Leaf Callable with the signature of the leaf ABI for `ValueSource::value_type`
 * @param pages Page descriptors of this decode call
 * @param tiles Tiles to decode, all of pages this composition can decode
 * @param num_tiles Number of entries in `tiles`
 * @param checkpoints Checkpoints of all pages. A page has
 * `ValueSource::checkpoints_per_tile + LevelSource::checkpoints_per_tile` entries per tile,
 * the value source's first
 * @param contexts Leaf contexts, indexed by `page_ref::ctx_idx`
 * @param null_counts One counter per entry of `pages`; the nulls found are added to it
 * @param leaf The leaf
 */
template <typename LevelSource, typename ValueSource, uint32_t steps, typename Leaf>
__device__ inline void decode_tiles(page_ref const* pages,
                                    tile_ref const* tiles,
                                    uint32_t num_tiles,
                                    checkpoint const* checkpoints,
                                    leaf_ctx const* contexts,
                                    int32_t* null_counts,
                                    Leaf leaf)
{
  constexpr uint32_t warp_size = 32;
  constexpr uint32_t stride = ValueSource::checkpoints_per_tile + LevelSource::checkpoints_per_tile;
  static_assert(ValueSource::max_turn_values >= warp_size * steps);

  uint32_t const lane       = threadIdx.x % warp_size;
  uint32_t const warp       = threadIdx.x / warp_size;
  uint32_t const tile_index = blockIdx.x * (decode_flat_block_size / warp_size) + warp;
  if (tile_index >= num_tiles) { return; }

  tile_ref const tile = tiles[tile_index];
  page_ref const page = pages[tile.page];
  leaf_ctx const ctx  = contexts[page.ctx_idx];

  tile_range range;
  range.tile_begin         = tile.tile * tile_rows;
  uint32_t const range_end = page.first_row + page.num_rows;
  range.row_begin          = range.tile_begin > page.first_row ? range.tile_begin : page.first_row;
  range.row_end =
    range.tile_begin + tile_rows < range_end ? range.tile_begin + tile_rows : range_end;
  if (range.row_begin >= range.row_end) { return; }
  range.out_begin = static_cast<int32_t>(range.row_begin) + page.row_base;
  range.out_end   = static_cast<int32_t>(range.row_end) + page.row_base;

  checkpoint const* const entries = checkpoints + page.first_checkpoint;
  LevelSource levels(page, tile, range, entries, stride, ValueSource::checkpoints_per_tile, lane);
  ValueSource values(page, entries, stride, 0, levels.position(), lane);

  constexpr auto turn_rows = static_cast<int32_t>(warp_size * steps);
  for (int32_t turn_base = range.out_begin & ~31; turn_base < range.out_end;
       turn_base += turn_rows) {
    // Output row, validity and value stream position of this lane for each step of the turn.
    int32_t out_row[steps];
    bool valid[steps];
    uint32_t value_index[steps];
    uint32_t const need_begin = levels.position();
    for (uint32_t step = 0; step < steps; ++step) {
      valid[step] = levels.step(
        turn_base + static_cast<int32_t>(step * warp_size), lane, out_row[step], value_index[step]);
    }
    int32_t const first      = turn_base > range.out_begin ? turn_base : range.out_begin;
    uint32_t const remaining = static_cast<uint32_t>(range.out_end - first);
    values.prepare(need_begin, levels.position(), need_begin + remaining, lane);

    // Load every value of the turn before any is stored, so that the loads overlap.
    typename ValueSource::value_type value[steps];
    uint32_t dict_idx[steps];
    for (uint32_t step = 0; step < steps; ++step) {
      value[step]    = 0;
      dict_idx[step] = 0;
      if (valid[step]) { value[step] = values.get(value_index[step], dict_idx[step]); }
    }
    for (uint32_t step = 0; step < steps; ++step) {
      if (valid[step]) {
        leaf(ctx, static_cast<uint32_t>(out_row[step]), value[step], dict_idx[step]);
      }
    }
  }
  levels.finish(null_counts + tile.page, lane);
}

/// Steps per loop turn of the compositions named by `decode_flat_tiles`. More steps keep more
/// loads in flight per lane and cost registers and a larger run table.
inline constexpr uint32_t flat_steps_per_turn = 1;

/**
 * @brief Selects one of two types at compile time.
 */
template <bool condition, typename True, typename False>
struct select_stage {
  using type = True;  ///< The selected type
};

/// @copydoc select_stage
template <typename True, typename False>
struct select_stage<false, True, False> {
  using type = False;  ///< The selected type
};

/**
 * @brief Decodes tiles of flat fixed-width pages: the compositions for PLAIN or dictionary
 * encoding, 4- or 8-byte values, and REQUIRED or OPTIONAL columns.
 *
 * @tparam is_dict Whether the pages are dictionary encoded; PLAIN otherwise
 * @tparam in_bytes Size of a physical value, 4 or 8
 * @tparam nullable Whether the pages' columns have a maximum definition level of 1
 * @tparam Leaf Callable with the signature of the leaf ABI for `in_bytes`
 * @param pages Page descriptors of this decode call
 * @param tiles Tiles to decode, all of pages with this encoding, width and nullability
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
  using level_source = typename select_stage<nullable, optional_levels, required_levels>::type;
  using value_source = typename select_stage<is_dict,
                                             dictionary_values<in_bytes, 32 * flat_steps_per_turn>,
                                             plain_values<in_bytes>>::type;
  decode_tiles<level_source, value_source, flat_steps_per_turn>(
    pages, tiles, num_tiles, checkpoints, contexts, null_counts, leaf);
}

}  // namespace cudf::io::parquet::jit
