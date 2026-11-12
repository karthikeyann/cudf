/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decode_flat.hpp"
#include "io/utilities/hostdevice_vector.hpp"
#include "jit/decode_flat.cuh"
#include "timestamp_utils.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/algorithm>
#include <cuda/std/limits>

#include <algorithm>
#include <array>
#include <bit>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Identifies one page claimed by the tile decode kernels.
 */
struct flat_page_claim {
  uint32_t page_idx;          ///< Index of the page in the subpass page array
  uint32_t first_checkpoint;  ///< Index of the page's first checkpoint entry
  uint32_t shape;             ///< Combination of `flat_shape_*`
};

/// Set in `flat_page_claim::shape` for a dictionary encoded page.
inline constexpr uint32_t flat_shape_dict = 1;
/// Set in `flat_page_claim::shape` for a page of 8-byte values.
inline constexpr uint32_t flat_shape_wide = 2;
/// Set in `flat_page_claim::shape` for a page of a nullable column.
inline constexpr uint32_t flat_shape_nullable = 4;
/// Number of distinct `flat_page_claim::shape` values.
inline constexpr uint32_t num_flat_shapes = 8;

/**
 * @brief State the tile decode kernels keep for the claimed pages of one subpass.
 */
struct flat_decode_plan {
  /**
   * @brief Constructs an empty plan.
   *
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the plan's device memory
   */
  flat_decode_plan(cuda::stream_ref stream, cudf::memory_resources mr)
    : claims(0, stream, mr),
      prepare_order(0, stream, mr),
      tile_offsets(0, stream, mr),
      contexts(0, stream, mr),
      tiles(stream, mr.get_temporary_mr(), 0, cuda::no_init),
      checkpoints(stream, mr.get_temporary_mr(), 0, cuda::no_init),
      page_refs(stream, mr.get_temporary_mr(), 0, cuda::no_init),
      null_counts(stream, mr.get_temporary_mr(), 0, cuda::no_init)
  {
  }

  /// Claimed pages, in subpass page order.
  cudf::detail::hostdevice_vector<flat_page_claim> claims;
  /// Indices of the claims that need checkpoints, in the order the prepare kernels take them.
  cudf::detail::hostdevice_vector<uint32_t> prepare_order;
  /// Position of each claimed page's first tile in `tiles`, rewritten for every decode call.
  cudf::detail::hostdevice_vector<uint32_t> tile_offsets;
  /// One leaf context per input column, rewritten for every decode call.
  cudf::detail::hostdevice_vector<jit::leaf_ctx> contexts;
  /// Tiles of the current decode call, grouped by kernel shape.
  cuda::device_buffer<jit::tile_ref> tiles;
  /// Checkpoints of every claimed page, filled once per subpass.
  cuda::device_buffer<jit::checkpoint> checkpoints;
  /// One descriptor per claimed page, rewritten for every decode call.
  cuda::device_buffer<jit::page_ref> page_refs;
  /// One null counter per claimed page, rewritten for every decode call.
  cuda::device_buffer<int32_t> null_counts;
};

namespace {

constexpr int prepare_block_size = 128;
constexpr int page_block_size    = 256;

/**
 * @brief Location of a page's value section, found by skipping its level sections.
 */
struct value_section {
  uint8_t const* values;  ///< PLAIN values, or the index stream after its bit width byte
  uint32_t size;          ///< Size of `values` in bytes
  uint32_t bit_width;     ///< Dictionary index bit width
  uint8_t const* levels;  ///< Definition level stream of a nullable page
  uint32_t levels_size;   ///< Size of `levels` in bytes
  bool has_nulls;         ///< Whether the definition levels may hold nulls
  uint32_t error;         ///< `decode_error` bits; the page must not be decoded if not 0
};

/**
 * @brief Finds the value section of a claimed page.
 *
 * Mirrors what the generic decoders derive from the page: the byte lengths of the level
 * sections, whether the definition levels may hold nulls, and the dictionary index bit width.
 */
__device__ value_section locate_value_section(PageInfo const& page,
                                              ColumnChunkDesc const& chunk,
                                              bool is_dict)
{
  value_section result{nullptr, 0, 0, nullptr, 0, false, 0};
  uint8_t const* cur       = page.page_data;
  uint8_t const* const end = cur + page.uncompressed_page_size;

  for (auto const lvl : {level_type::REPETITION, level_type::DEFINITION}) {
    uint8_t const* levels_begin = cur;
    int32_t length              = 0;
    if ((page.flags & PAGEINFO_FLAGS_V2) != 0 and page.lvl_bytes[lvl] != 0) {
      length = page.lvl_bytes[lvl];
      cur += length;
    } else if (chunk.level_bits[lvl] != 0) {
      // A V1 page stores the length of the RLE level section in front of it.
      if (cur + 4 < end) {
        length       = cur[0] | (cur[1] << 8) | (cur[2] << 16) | (cur[3] << 24);
        levels_begin = cur + 4;
        cur += length + 4;
      } else {
        result.error |= static_cast<uint32_t>(decode_error::LEVEL_STREAM_OVERRUN);
      }
    }
    if (lvl == level_type::DEFINITION and chunk.max_level[lvl] > 0) {
      // The page has no nulls if one repeated run of the maximum level covers it.
      result.has_nulls                = true;
      uint8_t const* const levels_end = levels_begin + length;
      if (length > 0 and levels_end <= end) {
        result.levels      = levels_begin;
        result.levels_size = static_cast<uint32_t>(length);
        uint32_t run       = 0;
        uint32_t shift     = 0;
        uint32_t byte;
        do {
          byte = *levels_begin++;
          run |= (byte & 0x7f) << shift;
          shift += 7;
        } while ((byte & 0x80) != 0 and levels_begin < levels_end and shift < 35);
        if ((run & 1) == 0 and (run >> 1) == static_cast<uint32_t>(page.num_input_values)) {
          uint32_t const value = levels_begin < levels_end ? *levels_begin : 0;
          result.has_nulls     = value != static_cast<uint32_t>(chunk.max_level[lvl]);
        }
      } else {
        result.error |= static_cast<uint32_t>(decode_error::LEVEL_STREAM_OVERRUN);
      }
    }
  }
  if (cur > end) {
    result.error |= static_cast<uint32_t>(decode_error::DATA_STREAM_OVERRUN);
    return result;
  }
  if (is_dict) {
    result.bit_width = cur < end ? *cur++ : 0;
    if (result.bit_width > 32) {
      result.error |= static_cast<uint32_t>(decode_error::INVALID_DICT_WIDTH);
    }
  }
  result.values = cur;
  result.size   = static_cast<uint32_t>(end - cur);
  return result;
}

/**
 * @brief One run of an RLE / bit-packed hybrid stream.
 */
struct run_extent {
  uint32_t next;     ///< Offset of the next run header, at most the stream size
  uint32_t count;    ///< Number of values in the run
  uint32_t payload;  ///< Offset of the bit-packed values or of the repeated value
  bool packed;       ///< Whether the run is bit-packed
};

/**
 * @brief Parses the run header at `offset`, which must be inside the stream.
 *
 * The header takes at least one byte, so `next` is always past `offset`.
 */
__device__ inline run_extent parse_run(jit::hybrid_stream const& stream, uint32_t offset)
{
  uint32_t header = 0;
  uint32_t shift  = 0;
  uint32_t byte;
  do {
    byte = stream.data[offset++];
    header |= (byte & 0x7f) << shift;
    shift += 7;
  } while ((byte & 0x80) != 0 and offset < stream.size and shift < 35);
  bool const packed    = (header & 1) != 0;
  uint64_t const count = packed ? uint64_t{header >> 1} * 8 : header >> 1;
  uint64_t const bytes =
    packed ? uint64_t{header >> 1} * stream.bit_width : (stream.bit_width + 7) / 8;
  return {static_cast<uint32_t>(cuda::std::min<uint64_t>(offset + bytes, stream.size)),
          static_cast<uint32_t>(
            cuda::std::min<uint64_t>(count, cuda::std::numeric_limits<int32_t>::max())),
          offset,
          packed};
}

/**
 * @brief Counts the non-null rows among page rows `[begin, end)` of a definition level
 * stream with one bit per level, starting from the run at `entry`.
 */
__device__ inline uint32_t count_valid_rows(jit::hybrid_stream const& stream,
                                            jit::checkpoint entry,
                                            uint32_t begin,
                                            uint32_t end)
{
  uint32_t offset = entry.byte_offset;
  uint64_t row    = entry.first_value;
  uint32_t count  = 0;
  while (row < end and offset < stream.size) {
    auto const extent = parse_run(stream, offset);
    uint64_t const lo = cuda::std::max<uint64_t>(row, begin);
    uint64_t const hi = cuda::std::min<uint64_t>(row + extent.count, end);
    if (hi > lo) {
      if (extent.packed) {
        uint64_t const last = hi - row;
        for (uint64_t bit = lo - row; bit < last;) {
          uint64_t const byte_offset = extent.payload + (bit >> 3);
          if (byte_offset >= stream.size) { break; }
          auto const take =
            static_cast<uint32_t>(cuda::std::min<uint64_t>(8 - (bit & 7), last - bit));
          count += __popc((stream.data[byte_offset] >> (bit & 7)) & ((1u << take) - 1));
          bit += take;
        }
      } else if (extent.payload < stream.size and stream.data[extent.payload] != 0) {
        count += static_cast<uint32_t>(hi - lo);
      }
    }
    row += extent.count;
    offset = extent.next;
  }
  return count;
}

/**
 * @brief Checkpoints of a walked stream of one page: entry `k * stride` resumes the stream at
 * value `k * tile_rows`.
 */
struct checkpoint_writer {
  jit::checkpoint* first;  ///< The page's first checkpoint entry
  uint32_t stride;         ///< Entries per tile
  uint32_t num_tiles;      ///< Tiles of the page

  /**
   * @brief Records the run at `offset` that holds values `[value, value + count)` for every
   * tile that starts inside it.
   */
  __device__ void record(uint32_t offset, uint64_t value, uint64_t count) const
  {
    uint64_t const end = value + count;
    for (uint64_t tile = (value + jit::tile_rows - 1) >> jit::tile_rows_log2;
         tile < num_tiles and (tile << jit::tile_rows_log2) < end;
         ++tile) {
      first[tile * stride] = {offset, static_cast<uint32_t>(value)};
    }
  }
};

// Stream bytes each thread owns in one window of the parallel walk.
constexpr uint32_t walk_chunk_bytes = 32;
// Stream bytes covered by one window.
constexpr uint32_t walk_window_bytes = walk_chunk_bytes * prepare_block_size;
// Run headers the walking thread reads on its own before it judges how dense the runs are.
constexpr uint32_t walk_probe_runs = 16;
// The runs are dense if the probe advanced by less than this many bytes per run.
constexpr uint32_t walk_dense_run_bytes = 12;
// A window pays off while the chain of run headers enters at least this many chunks.
constexpr uint32_t walk_min_chunks = 48;

/**
 * @brief Walks the run headers of a hybrid stream and records its checkpoints.
 *
 * The position of a run header depends on every run before it. One thread follows the headers
 * while runs are long. Where runs are short the block works on a window of the stream: every
 * thread computes, for each byte of its chunk taken as a run header, where the chain of runs
 * leaves the chunk and how many values it covers. One thread then follows the chain through
 * the window in one hop per chunk instead of one per run, and every thread walks its chunk
 * again from its now known entry to record the checkpoints.
 *
 * Must be called by every thread of the block.
 *
 * @tparam windowed Whether the block may work on windows; if not, one thread walks the stream
 */
template <bool windowed>
__device__ void walk_hybrid_stream(jit::hybrid_stream const& section,
                                   checkpoint_writer const& writer,
                                   uint32_t num_values)
{
  struct walk_state {
    uint64_t value;   // values before `offset`
    uint32_t offset;  // next run header
    bool window;      // whether the block works on a window next
    bool done;
  };
  // Follows up to `max_runs` run headers one by one.
  auto const walk_serial = [&](walk_state& state, uint32_t max_runs) {
    for (uint32_t run = 0;
         run < max_runs and state.offset < section.size and state.value < num_values;
         ++run) {
      auto const extent = parse_run(section, state.offset);
      writer.record(state.offset, state.value, extent.count);
      state.value += extent.count;
      state.offset = extent.next;
    }
    state.done = not(state.offset < section.size and state.value < num_values);
  };
  // Tiles past the end of the stream start where it ends.
  auto const finish = [&](walk_state const& state) {
    for (uint64_t tile = (state.value + jit::tile_rows - 1) >> jit::tile_rows_log2;
         tile < writer.num_tiles;
         ++tile) {
      writer.first[tile * writer.stride] = {
        section.size,
        static_cast<uint32_t>(
          cuda::std::min<uint64_t>(state.value, cuda::std::numeric_limits<int32_t>::max()))};
    }
  };

  if constexpr (not windowed) {
    if (threadIdx.x == 0) {
      walk_state state{0, 0, false, false};
      walk_serial(state, cuda::std::numeric_limits<uint32_t>::max());
      finish(state);
    }
  } else {
    constexpr uint32_t no_entry = 0xffff'ffffu;
    __shared__ walk_state state;
    // Per window byte, taken as a run header: the first run header at or past the end of its
    // chunk, and the values the chain covers up to there.
    __shared__ uint32_t exit_offset[walk_window_bytes];
    __shared__ uint32_t exit_values[walk_window_bytes];
    // Per chunk: the run header at which the chain enters it, and the values before it.
    __shared__ uint32_t entry_offset[prepare_block_size];
    __shared__ uint32_t entry_value[prepare_block_size];

    uint32_t const t = threadIdx.x;
    if (t == 0) { state = walk_state{0, 0, false, false}; }
    while (true) {
      if (t == 0 and not state.window) {
        uint32_t const begin = state.offset;
        walk_serial(state, walk_probe_runs);
        state.window = state.offset - begin < walk_probe_runs * walk_dense_run_bytes;
      }
      __syncthreads();
      if (state.done) { break; }
      if (state.window) {
        uint32_t const base        = state.offset;
        uint32_t const chunk_begin = base + t * walk_chunk_bytes;
        auto const chunk_end       = static_cast<uint32_t>(
          cuda::std::min<uint64_t>(uint64_t{chunk_begin} + walk_chunk_bytes, section.size));
        if (chunk_begin < section.size) {
          for (uint32_t offset = chunk_end; offset-- > chunk_begin;) {
            auto const extent = parse_run(section, offset);
            if (extent.next >= chunk_end) {
              exit_offset[offset - base] = extent.next;
              exit_values[offset - base] = extent.count;
            } else {
              exit_offset[offset - base] = exit_offset[extent.next - base];
              exit_values[offset - base] = static_cast<uint32_t>(
                cuda::std::min<uint64_t>(uint64_t{extent.count} + exit_values[extent.next - base],
                                         cuda::std::numeric_limits<int32_t>::max()));
            }
          }
        }
        __syncthreads();
        if (t == 0) {
          uint32_t offset  = base;
          uint64_t value   = state.value;
          uint32_t entered = 0;
          for (uint32_t chunk = 0; chunk < prepare_block_size; ++chunk) {
            uint64_t const end = uint64_t{base} + uint64_t{chunk + 1} * walk_chunk_bytes;
            if (offset < end and offset < section.size and value < num_values) {
              entry_offset[chunk] = offset;
              entry_value[chunk]  = static_cast<uint32_t>(value);
              value += exit_values[offset - base];
              offset = exit_offset[offset - base];
              ++entered;
            } else {
              entry_offset[chunk] = no_entry;
            }
          }
          state.offset = offset;
          state.value  = value;
          state.window = entered >= walk_min_chunks;
          state.done   = not(offset < section.size and value < num_values);
        }
        __syncthreads();
        if (entry_offset[t] != no_entry) {
          uint64_t value = entry_value[t];
          for (uint32_t offset = entry_offset[t]; offset < chunk_end;) {
            auto const extent = parse_run(section, offset);
            writer.record(offset, value, extent.count);
            value += extent.count;
            offset = extent.next;
          }
        }
        __syncthreads();
      }
    }
    if (t == 0) { finish(state); }
  }
}

/**
 * @brief Records the checkpoints of claimed pages, one block per page.
 *
 * Walks the definition level stream of a nullable page and the index stream of a dictionary
 * page, and counts the non-null rows before every tile of a nullable page. Launched with one
 * row of blocks, a block does all of it for its page; with two rows, block row 0 handles the
 * levels and block row 1 the index stream.
 *
 * @tparam windowed Whether the walk of a stream may use the whole block
 */
template <bool windowed>
CUDF_KERNEL void __launch_bounds__(prepare_block_size)
  prepare_flat_pages_kernel(PageInfo const* pages,
                            device_span<ColumnChunkDesc const> chunks,
                            flat_page_claim const* claims,
                            uint32_t const* claim_ids,
                            jit::checkpoint* checkpoints)
{
  flat_page_claim const claim  = claims[claim_ids[blockIdx.x]];
  PageInfo const& page         = pages[claim.page_idx];
  ColumnChunkDesc const& chunk = chunks[page.chunk_idx];
  bool const is_dict           = (claim.shape & flat_shape_dict) != 0;
  bool const nullable          = (claim.shape & flat_shape_nullable) != 0;
  uint32_t const stride        = jit::checkpoint_stride(is_dict, nullable);
  auto const num_values        = static_cast<uint32_t>(page.num_input_values);
  uint32_t const num_tiles     = cudf::detail::div_rounding_up_safe(num_values, jit::tile_rows);
  jit::checkpoint* const first = checkpoints + claim.first_checkpoint;
  value_section const section  = locate_value_section(page, chunk, is_dict);
  // With two rows of blocks the level stream and the index stream of a page are walked by
  // different blocks, so that the two walks overlap.
  bool const walks_levels = gridDim.y == 1 or blockIdx.y == 0;
  bool const walks_values = gridDim.y == 1 or blockIdx.y == 1;

  if (walks_levels and nullable and section.has_nulls) {
    // Levels: resume entries from a walk of the level stream, then the non-null rows of
    // every tile, one tile per thread, turned into ranks by one thread.
    jit::hybrid_stream const levels{section.levels, section.levels_size, 1};
    jit::checkpoint* const entries = first + (is_dict ? 1 : 0);
    walk_hybrid_stream<windowed>(levels, checkpoint_writer{entries, stride, num_tiles}, num_values);
    __syncthreads();
    for (uint32_t tile = threadIdx.x; tile < num_tiles; tile += prepare_block_size) {
      uint32_t const tile_begin = tile * jit::tile_rows;
      entries[tile * stride + 1].first_value =
        count_valid_rows(levels,
                         entries[tile * stride],
                         tile_begin,
                         cuda::std::min(tile_begin + jit::tile_rows, num_values));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t rank = 0;
      for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t& entry      = entries[tile * stride + 1].first_value;
        uint32_t const count = entry;
        entry                = rank;
        rank += count;
      }
    }
  }

  if (walks_values and is_dict) {
    __syncthreads();
    walk_hybrid_stream<windowed>(
      jit::hybrid_stream{section.values, section.size, section.bit_width},
      checkpoint_writer{first, stride, num_tiles},
      num_values);
  }
}

/**
 * @brief Writes the descriptor of every claimed page for one decode call, lists its tiles and
 * clears its null counter.
 *
 * `tile_offsets[i]` is the position in `tiles` of the first tile of claimed page `i`.
 */
CUDF_KERNEL void __launch_bounds__(page_block_size)
  setup_flat_pages_kernel(PageInfo const* pages,
                          device_span<ColumnChunkDesc const> chunks,
                          device_span<flat_page_claim const> claims,
                          size_t min_row,
                          size_t num_rows,
                          uint32_t const* tile_offsets,
                          jit::page_ref* page_refs,
                          jit::tile_ref* tiles,
                          int32_t* null_counts,
                          kernel_error::pointer error_code)
{
  size_t const idx = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (idx >= claims.size()) { return; }
  flat_page_claim const claim  = claims[idx];
  PageInfo const& page         = pages[claim.page_idx];
  ColumnChunkDesc const& chunk = chunks[page.chunk_idx];
  bool const is_dict           = (claim.shape & flat_shape_dict) != 0;
  bool const nullable          = (claim.shape & flat_shape_nullable) != 0;
  int const in_bytes           = (claim.shape & flat_shape_wide) != 0 ? 8 : 4;

  jit::page_ref ref{};
  null_counts[idx] = 0;

  // Rows of the page inside [min_row, min_row + num_rows).
  size_t const page_begin = chunk.start_row + page.chunk_row;
  size_t const page_end   = page_begin + page.num_rows;
  size_t const range_end  = min_row + num_rows;
  if (page_begin < range_end and page_end > min_row) {
    size_t const first_row = page_begin >= min_row ? 0 : min_row - page_begin;
    ref.first_row          = static_cast<uint32_t>(first_row);
    ref.num_rows =
      static_cast<uint32_t>(cuda::std::min(page_end, range_end) - page_begin - first_row);
  }
  ref.row_base =
    static_cast<int32_t>(static_cast<int64_t>(page_begin) - static_cast<int64_t>(min_row));

  if (ref.num_rows != 0) {
    // The tiles are listed even if the page turns out to be malformed; they then decode nothing.
    uint32_t const first_tile = ref.first_row / jit::tile_rows;
    uint32_t const last_tile  = (ref.first_row + ref.num_rows - 1) / jit::tile_rows;
    jit::tile_ref* const out  = tiles + tile_offsets[idx];
    for (uint32_t tile = first_tile; tile <= last_tile; ++tile) {
      out[tile - first_tile] = {static_cast<uint32_t>(idx), tile};
    }

    value_section const section = locate_value_section(page, chunk, is_dict);
    if (section.error != 0) {
      cuda::atomic_ref<kernel_error::value_type, cuda::std::thread_scope_device> ref_error{
        *error_code};
      ref_error.fetch_or(section.error, cuda::std::memory_order_relaxed);
      ref.num_rows = 0;
    }
    ref.values      = section.values;
    ref.values_size = section.size;
    ref.bit_width   = static_cast<uint8_t>(section.bit_width);
    if (is_dict) {
      ref.dict         = chunk.dict_page->page_data;
      ref.dict_entries = static_cast<uint32_t>(chunk.dict_page->uncompressed_page_size) / in_bytes;
    }
    if (nullable) {
      ref.levels = section.levels;
      ref.valid  = chunk.valid_map_base[0];
      ref.flags  = section.has_nulls ? 0 : jit::page_flag_no_nulls;
    }
  }
  ref.first_checkpoint = claim.first_checkpoint;
  ref.ctx_idx          = static_cast<uint16_t>(chunk.src_col_index);
  page_refs[idx]       = ref;
}

/**
 * @brief Publishes the null count of every claimed page.
 */
CUDF_KERNEL void __launch_bounds__(page_block_size)
  finish_flat_pages_kernel(PageInfo const* pages,
                           device_span<flat_page_claim const> claims,
                           int32_t const* null_counts)
{
  size_t const idx = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (idx >= claims.size()) { return; }
  pages[claims[idx].page_idx].nesting_decode[0].null_count = null_counts[idx];
}

/**
 * @brief Leaf that stores the physical value unchanged and keeps every row.
 */
struct store_leaf {
  template <typename Value>
  __device__ uint32_t
  operator()(jit::leaf_ctx const& ctx, uint32_t row, Value value, uint32_t) const
  {
    static_cast<Value*>(ctx.out)[row] = value;
    return 1;
  }
};

/**
 * @brief Tile decode kernel with the store leaf compiled in.
 */
template <bool is_dict, int in_bytes, bool nullable, uint32_t steps>
CUDF_KERNEL void __launch_bounds__(jit::decode_flat_block_size)
  decode_flat_tiles_kernel(jit::page_ref const* pages,
                           jit::tile_ref const* tiles,
                           uint32_t num_tiles,
                           jit::checkpoint const* checkpoints,
                           jit::leaf_ctx const* contexts,
                           int32_t* null_counts)
{
  using level_source =
    typename jit::select_stage<nullable, jit::optional_levels, jit::required_levels>::type;
  using value_source = typename jit::select_stage<is_dict,
                                                  jit::dictionary_values<in_bytes, 32 * steps>,
                                                  jit::plain_values<in_bytes>>::type;
  jit::decode_tiles<level_source, value_source, steps>(
    pages, tiles, num_tiles, checkpoints, contexts, null_counts, store_leaf{});
}

void launch_decode_flat_tiles(uint32_t shape,
                              jit::page_ref const* pages,
                              jit::tile_ref const* tiles,
                              uint32_t num_tiles,
                              jit::checkpoint const* checkpoints,
                              jit::leaf_ctx const* contexts,
                              int32_t* null_counts,
                              cuda::stream_ref stream)
{
  constexpr uint32_t tiles_per_block = jit::decode_flat_block_size / cudf::detail::warp_size;
  auto const num_blocks = cudf::detail::div_rounding_up_safe(num_tiles, tiles_per_block);
  auto const launch     = [&]<bool is_dict, int in_bytes, bool nullable>() {
    decode_flat_tiles_kernel<is_dict, in_bytes, nullable, jit::flat_steps_per_turn>
      <<<num_blocks, jit::decode_flat_block_size, 0, stream.get()>>>(
        pages, tiles, num_tiles, checkpoints, contexts, null_counts);
  };
  switch (shape) {
    case 0: launch.template operator()<false, 4, false>(); break;
    case 1: launch.template operator()<true, 4, false>(); break;
    case 2: launch.template operator()<false, 8, false>(); break;
    case 3: launch.template operator()<true, 8, false>(); break;
    case 4: launch.template operator()<false, 4, true>(); break;
    case 5: launch.template operator()<true, 4, true>(); break;
    case 6: launch.template operator()<false, 8, true>(); break;
    case 7: launch.template operator()<true, 8, true>(); break;
    default: CUDF_FAIL("Invalid flat decode shape");
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

uint32_t shape_of(PageInfo const& page, ColumnChunkDesc const& chunk)
{
  auto const wide = chunk.physical_type == Type::INT64 or chunk.physical_type == Type::DOUBLE;
  return (is_dictionary_encoding(page.encoding) ? flat_shape_dict : 0) |
         (wide ? flat_shape_wide : 0) |
         (chunk.max_level[level_type::DEFINITION] > 0 ? flat_shape_nullable : 0);
}

}  // namespace

bool is_flat_decode_eligible(PageInfo const& page,
                             ColumnChunkDesc const& chunk,
                             data_type output_type)
{
  if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0 or page.num_input_values <= 0) { return false; }
  auto const is_dict = page.kernel_mask == decode_kernel_mask::FIXED_WIDTH_DICT;
  if (not is_dict and page.kernel_mask != decode_kernel_mask::FIXED_WIDTH_NO_DICT) { return false; }
  if (is_dict and chunk.dict_page == nullptr) { return false; }

  size_t in_bytes = 0;
  switch (chunk.physical_type) {
    case Type::INT32:
    case Type::FLOAT: in_bytes = 4; break;
    case Type::INT64:
    case Type::DOUBLE: in_bytes = 8; break;
    default: return false;
  }
  // The value must be stored as is: no narrowing, widening or timestamp scaling.
  if (not cudf::is_fixed_width(output_type) or cudf::size_of(output_type) != in_bytes) {
    return false;
  }
  if (chunk.physical_type == Type::INT64 and chunk.ts_clock_rate != 0 and
      calc_timestamp_scale(chunk.logical_type, chunk.ts_clock_rate) != 0) {
    return false;
  }

  if (chunk.max_level[level_type::REPETITION] != 0 or chunk.max_nesting_depth != 1 or
      chunk.src_col_index > std::numeric_limits<uint16_t>::max()) {
    return false;
  }
  auto const max_def_level = chunk.max_level[level_type::DEFINITION];
  if (max_def_level > 1) { return false; }
  if (max_def_level == 1) {
    // The level section must be RLE with a known length.
    auto const is_v2 = (page.flags & PAGEINFO_FLAGS_V2) != 0;
    if (is_v2 ? page.lvl_bytes[level_type::DEFINITION] == 0
              : page.definition_level_encoding != Encoding::RLE) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<flat_decode_plan> prepare_flat_decode(
  cudf::detail::hostdevice_span<PageInfo const> pages,
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  // Dictionary sizes, to tell how wide the indices of a page are.
  std::vector<int32_t> dict_entries(chunks.size(), 0);
  for (size_t idx = 0; idx < pages.size(); ++idx) {
    auto const& page = pages[idx];
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) {
      dict_entries[page.chunk_idx] = page.num_input_values;
    }
  }

  std::vector<flat_page_claim> claims;
  // Claims that need checkpoints, by how their index stream is walked.
  std::vector<uint32_t> serial_ids;
  std::vector<uint32_t> windowed_ids;
  size_t num_checkpoints = 0;
  for (size_t idx = 0; idx < pages.size(); ++idx) {
    auto const& page = pages[idx];
    if (page.kernel_mask != decode_kernel_mask::FLAT_TILES) { continue; }
    auto const& chunk  = chunks[page.chunk_idx];
    auto const shape   = shape_of(page, chunk);
    auto const is_dict = (shape & flat_shape_dict) != 0;
    auto const stride  = jit::checkpoint_stride(is_dict, (shape & flat_shape_nullable) != 0);
    auto const claim   = static_cast<uint32_t>(claims.size());
    claims.push_back({static_cast<uint32_t>(idx), static_cast<uint32_t>(num_checkpoints), shape});
    num_checkpoints +=
      cudf::detail::div_rounding_up_safe<size_t>(page.num_input_values, jit::tile_rows) * stride;
    if (stride == 0) { continue; }
    // Level streams, and index streams much smaller than their bit-packed size, hold mostly
    // short runs. Walking those takes the whole block; long runs are walked by one thread.
    auto const bit_width =
      std::bit_width(static_cast<uint32_t>(std::max(dict_entries[page.chunk_idx], 1) - 1));
    auto const packed_bits = static_cast<uint64_t>(page.num_input_values) * bit_width;
    auto const is_dense =
      (shape & flat_shape_nullable) != 0 or
      static_cast<uint64_t>(page.uncompressed_page_size) * 8 * 20 < packed_bits * 19;
    (is_dense ? windowed_ids : serial_ids).push_back(claim);
  }
  CUDF_EXPECTS(num_checkpoints <= std::numeric_limits<uint32_t>::max(),
               "Too many tiles in a subpass",
               std::overflow_error);

  auto plan    = std::make_shared<flat_decode_plan>(stream, mr);
  plan->claims = cudf::detail::hostdevice_vector<flat_page_claim>(claims.size(), stream, mr);
  std::copy(claims.begin(), claims.end(), plan->claims.begin());
  plan->claims.host_to_device_async(stream);
  plan->tile_offsets = cudf::detail::hostdevice_vector<uint32_t>(claims.size(), stream, mr);
  plan->checkpoints  = cuda::device_buffer<jit::checkpoint>(
    stream, mr.get_temporary_mr(), num_checkpoints, cuda::no_init);
  plan->page_refs =
    cuda::device_buffer<jit::page_ref>(stream, mr.get_temporary_mr(), claims.size(), cuda::no_init);
  plan->null_counts =
    cuda::device_buffer<int32_t>(stream, mr.get_temporary_mr(), claims.size(), cuda::no_init);

  plan->prepare_order =
    cudf::detail::hostdevice_vector<uint32_t>(serial_ids.size() + windowed_ids.size(), stream, mr);
  std::copy(windowed_ids.begin(),
            windowed_ids.end(),
            std::copy(serial_ids.begin(), serial_ids.end(), plan->prepare_order.begin()));
  plan->prepare_order.host_to_device_async(stream);
  if (not serial_ids.empty()) {
    prepare_flat_pages_kernel<false>
      <<<serial_ids.size(), prepare_block_size, 0, stream.get()>>>(pages.device_ptr(),
                                                                   chunks,
                                                                   plan->claims.device_ptr(),
                                                                   plan->prepare_order.device_ptr(),
                                                                   plan->checkpoints.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (not windowed_ids.empty()) {
    prepare_flat_pages_kernel<true>
      <<<dim3(windowed_ids.size(), 2), prepare_block_size, 0, stream.get()>>>(
        pages.device_ptr(),
        chunks,
        plan->claims.device_ptr(),
        plan->prepare_order.device_ptr(serial_ids.size()),
        plan->checkpoints.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return plan;
}

void decode_flat_pages(flat_decode_plan& plan,
                       cudf::detail::hostdevice_span<PageInfo> pages,
                       cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                       cudf::host_span<void* const> column_data,
                       cudf::host_span<flat_column_leaf> column_leaves,
                       size_t min_row,
                       size_t num_rows,
                       kernel_error::pointer error_code,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr)
{
  for (auto& leaf : column_leaves) {
    leaf.num_rows_decoded = 0;
  }
  auto const num_claims = plan.claims.size();
  if (num_claims == 0) { return; }

  // The tiles of one launch: those of one kernel shape decoded with the store leaf, or those of
  // one shape of one column decoded by the column's linked kernel.
  struct launch_group {
    uint32_t shape;                         // Kernel shape of the tiles
    jit::decode_flat_kernel const* kernel;  // The linked kernel, or null for the store leaf
    uint32_t num_tiles = 0;                 // Number of tiles
    uint32_t begin     = 0;                 // Index of the first tile in the tile list
  };
  std::vector<launch_group> groups;
  for (uint32_t shape = 0; shape < num_flat_shapes; ++shape) {
    groups.push_back({shape, nullptr});
  }

  // Tiles that overlap the row range, counted by launch group. The host only counts them; the
  // setup kernel writes the list.
  auto const range_end = min_row + num_rows;
  std::vector<uint32_t> claim_groups(num_claims);
  for (size_t idx = 0; idx < num_claims; ++idx) {
    auto const& claim = plan.claims[idx];
    auto const& page  = pages[claim.page_idx];
    auto const& chunk = chunks[page.chunk_idx];

    auto group = claim.shape;
    if (not column_leaves.empty()) {
      auto const& leaf = column_leaves[chunk.src_col_index];
      auto const* kernel =
        (claim.shape & flat_shape_dict) != 0 ? leaf.dict_kernel : leaf.plain_kernel;
      if (kernel != nullptr) {
        auto const it =
          std::find_if(groups.begin() + num_flat_shapes, groups.end(), [&](auto const& g) {
            return g.kernel == kernel and g.shape == claim.shape;
          });
        group = static_cast<uint32_t>(std::distance(groups.begin(), it));
        if (it == groups.end()) { groups.push_back({claim.shape, kernel}); }
      }
    }
    claim_groups[idx] = group;

    size_t const page_begin = chunk.start_row + page.chunk_row;
    size_t const page_end   = page_begin + page.num_rows;
    if (page_begin >= range_end or page_end <= min_row) {
      plan.tile_offsets[idx] = 0;
      continue;
    }
    auto const first_row = page_begin >= min_row ? 0 : min_row - page_begin;
    auto const last_row  = std::min(page_end, range_end) - page_begin - 1;
    plan.tile_offsets[idx] =
      static_cast<uint32_t>(last_row / jit::tile_rows - first_row / jit::tile_rows + 1);
    groups[group].num_tiles += plan.tile_offsets[idx];
    if (groups[group].kernel != nullptr) {
      column_leaves[chunk.src_col_index].num_rows_decoded += last_row - first_row + 1;
    }
  }
  auto num_tiles = uint32_t{0};
  for (auto& group : groups) {
    group.begin = num_tiles;
    num_tiles += group.num_tiles;
  }
  {
    std::vector<uint32_t> next(groups.size());
    std::transform(
      groups.begin(), groups.end(), next.begin(), [](auto const& group) { return group.begin; });
    for (size_t idx = 0; idx < num_claims; ++idx) {
      auto const count       = plan.tile_offsets[idx];
      plan.tile_offsets[idx] = next[claim_groups[idx]];
      next[claim_groups[idx]] += count;
    }
  }
  plan.tile_offsets.host_to_device_async(stream);
  if (plan.tiles.size() < num_tiles) {
    plan.tiles =
      cuda::device_buffer<jit::tile_ref>(stream, mr.get_temporary_mr(), num_tiles, cuda::no_init);
  }

  // The store leaf uses only the output pointer of its context.
  if (plan.contexts.size() != column_data.size()) {
    plan.contexts = cudf::detail::hostdevice_vector<jit::leaf_ctx>(column_data.size(), stream, mr);
  }
  for (size_t column = 0; column < column_data.size(); ++column) {
    auto ctx              = column_leaves.empty() ? jit::leaf_ctx{} : column_leaves[column].ctx;
    ctx.out               = column_data[column];
    plan.contexts[column] = ctx;
  }
  plan.contexts.host_to_device_async(stream);

  auto const num_page_blocks =
    cudf::detail::div_rounding_up_safe<size_t>(num_claims, page_block_size);
  setup_flat_pages_kernel<<<num_page_blocks, page_block_size, 0, stream.get()>>>(
    pages.device_ptr(),
    chunks,
    plan.claims,
    min_row,
    num_rows,
    plan.tile_offsets.device_ptr(),
    plan.page_refs.data(),
    plan.tiles.data(),
    plan.null_counts.data(),
    error_code);
  CUDF_CUDA_TRY(cudaGetLastError());

  for (auto const& group : groups) {
    if (group.num_tiles == 0) { continue; }
    if (group.kernel == nullptr) {
      launch_decode_flat_tiles(group.shape,
                               plan.page_refs.data(),
                               plan.tiles.data() + group.begin,
                               group.num_tiles,
                               plan.checkpoints.data(),
                               plan.contexts.device_ptr(),
                               plan.null_counts.data(),
                               stream);
    } else {
      constexpr uint32_t tiles_per_block = jit::decode_flat_block_size / cudf::detail::warp_size;
      auto const num_blocks = cudf::detail::div_rounding_up_safe(group.num_tiles, tiles_per_block);
      group.kernel->launch({num_blocks},
                           {jit::decode_flat_block_size},
                           stream,
                           plan.page_refs.data(),
                           plan.tiles.data() + group.begin,
                           group.num_tiles,
                           plan.checkpoints.data(),
                           plan.contexts.device_ptr(),
                           plan.null_counts.data());
    }
  }

  finish_flat_pages_kernel<<<num_page_blocks, page_block_size, 0, stream.get()>>>(
    pages.device_ptr(), plan.claims, plan.null_counts.data());
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
