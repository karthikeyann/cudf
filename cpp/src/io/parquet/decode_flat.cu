/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decode_flat.hpp"
#include "jit/decode_flat.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/limits>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <vector>

namespace cudf::io::parquet::detail {

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
  bool has_nulls;         ///< Whether the definition levels were decoded and may hold nulls
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
  value_section result{nullptr, 0, 0, false, 0};
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
      // The levels are decoded unless one repeated run of the maximum level covers the page.
      result.has_nulls                = true;
      uint8_t const* const levels_end = levels_begin + length;
      if (length > 0 and levels_end <= end) {
        uint32_t run   = 0;
        uint32_t shift = 0;
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
 * @brief Records the checkpoints of the claimed pages, one block per page.
 *
 * For a nullable page the block counts the non-null rows of every tile and turns the counts
 * into ranks. For a dictionary page one thread walks the run headers of the index stream.
 */
CUDF_KERNEL void __launch_bounds__(prepare_block_size)
  prepare_flat_pages_kernel(PageInfo const* pages,
                            device_span<ColumnChunkDesc const> chunks,
                            flat_page_claim const* claims,
                            jit::checkpoint* checkpoints,
                            int level_type_size)
{
  constexpr uint32_t warp_size = cudf::detail::warp_size;
  flat_page_claim const claim  = claims[blockIdx.x];
  PageInfo const& page         = pages[claim.page_idx];
  ColumnChunkDesc const& chunk = chunks[page.chunk_idx];
  bool const is_dict           = (claim.shape & flat_shape_dict) != 0;
  bool const nullable          = (claim.shape & flat_shape_nullable) != 0;
  uint32_t const stride        = jit::checkpoint_stride(is_dict, nullable);
  auto const num_values        = static_cast<uint32_t>(page.num_input_values);
  uint32_t const num_tiles     = cudf::detail::div_rounding_up_safe(num_values, jit::tile_rows);
  jit::checkpoint* const first = checkpoints + claim.first_checkpoint;
  value_section const section  = locate_value_section(page, chunk, is_dict);

  if (nullable and section.has_nulls) {
    uint32_t const lane         = threadIdx.x % warp_size;
    uint32_t const warp         = threadIdx.x / warp_size;
    uint32_t const num_warps    = prepare_block_size / warp_size;
    uint8_t const* const levels = page.lvl_decode_buf[level_type::DEFINITION];
    auto const num_levels       = static_cast<uint32_t>(page.num_decoded_level_values);
    for (uint32_t tile = warp; tile < num_tiles; tile += num_warps) {
      uint32_t const tile_begin = tile * jit::tile_rows;
      uint32_t const tile_end   = cuda::std::min(tile_begin + jit::tile_rows, num_levels);
      uint32_t count            = 0;
      for (uint32_t base = tile_begin; base < tile_end; base += warp_size) {
        uint32_t const row = base + lane;
        bool const valid   = row < tile_end and levels[size_t{row} * level_type_size] != 0;
        count += __popc(__ballot_sync(0xffff'ffffu, valid));
      }
      if (lane == 0) { first[tile * stride + (stride - 1)].first_value = count; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t rank = 0;
      for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t& entry      = first[tile * stride + (stride - 1)].first_value;
        uint32_t const count = entry;
        entry                = rank;
        rank += count;
      }
    }
  }

  if (is_dict and threadIdx.x == 0) {
    // Every header takes at least one byte, so the walk ends with the stream.
    uint32_t offset = 0;
    uint64_t value  = 0;
    uint32_t tile   = 0;
    while (offset < section.size and tile < num_tiles) {
      uint32_t const header_offset = offset;
      uint32_t header              = 0;
      uint32_t shift               = 0;
      uint32_t byte;
      do {
        byte = section.values[offset++];
        header |= (byte & 0x7f) << shift;
        shift += 7;
      } while ((byte & 0x80) != 0 and offset < section.size and shift < 35);
      uint64_t const count = (header & 1) != 0 ? uint64_t{header >> 1} * 8 : header >> 1;
      uint64_t const bytes =
        (header & 1) != 0 ? uint64_t{header >> 1} * section.bit_width : (section.bit_width + 7) / 8;
      offset = static_cast<uint32_t>(cuda::std::min<uint64_t>(offset + bytes, section.size));
      while (tile < num_tiles and uint64_t{tile} * jit::tile_rows < value + count) {
        first[tile * stride] = {header_offset, static_cast<uint32_t>(value)};
        ++tile;
      }
      value = cuda::std::min<uint64_t>(value + count, cuda::std::numeric_limits<int32_t>::max());
    }
    // Tiles past the end of the stream start where it ends.
    for (; tile < num_tiles; ++tile) {
      first[tile * stride] = {section.size, static_cast<uint32_t>(value)};
    }
  }
}

/**
 * @brief Writes the descriptor of every claimed page for one decode call and clears its null
 * counter.
 */
CUDF_KERNEL void __launch_bounds__(page_block_size)
  setup_flat_pages_kernel(PageInfo const* pages,
                          device_span<ColumnChunkDesc const> chunks,
                          device_span<flat_page_claim const> claims,
                          size_t min_row,
                          size_t num_rows,
                          int level_type_size,
                          jit::page_ref* page_refs,
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
      ref.levels = page.lvl_decode_buf[level_type::DEFINITION];
      ref.valid  = chunk.valid_map_base[0];
      ref.flags  = (section.has_nulls ? 0 : jit::page_flag_no_nulls) |
                  (level_type_size == 2 ? jit::page_flag_wide_levels : 0);
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
template <bool is_dict, int in_bytes, bool nullable>
CUDF_KERNEL void __launch_bounds__(jit::decode_flat_block_size)
  decode_flat_tiles_kernel(jit::page_ref const* pages,
                           jit::tile_ref const* tiles,
                           uint32_t num_tiles,
                           jit::checkpoint const* checkpoints,
                           jit::leaf_ctx const* contexts,
                           int32_t* null_counts)
{
  jit::decode_flat_tiles<is_dict, in_bytes, nullable>(
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
    decode_flat_tiles_kernel<is_dict, in_bytes, nullable>
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

flat_decode_plan::flat_decode_plan(cuda::stream_ref stream, cudf::memory_resources mr)
  : claims(0, stream, mr),
    checkpoints(stream, mr.get_temporary_mr(), 0, cuda::no_init),
    page_refs(stream, mr.get_temporary_mr(), 0, cuda::no_init),
    null_counts(stream, mr.get_temporary_mr(), 0, cuda::no_init)
{
}

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
  if (not cudf::is_fixed_width(output_type) or cudf::size_of(output_type) != in_bytes or
      chunk.ts_clock_rate != 0) {
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

std::unique_ptr<flat_decode_plan> prepare_flat_decode(
  cudf::detail::hostdevice_span<PageInfo const> pages,
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  int level_type_size,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  std::vector<flat_page_claim> claims;
  size_t num_checkpoints = 0;
  for (size_t idx = 0; idx < pages.size(); ++idx) {
    auto const& page = pages[idx];
    if (page.kernel_mask != decode_kernel_mask::FLAT_TILES) { continue; }
    auto const& chunk = chunks[page.chunk_idx];
    auto const shape  = shape_of(page, chunk);
    claims.push_back({static_cast<uint32_t>(idx), static_cast<uint32_t>(num_checkpoints), shape});
    num_checkpoints +=
      cudf::detail::div_rounding_up_safe<size_t>(page.num_input_values, jit::tile_rows) *
      jit::checkpoint_stride((shape & flat_shape_dict) != 0, (shape & flat_shape_nullable) != 0);
  }
  CUDF_EXPECTS(num_checkpoints <= std::numeric_limits<uint32_t>::max(),
               "Too many tiles in a subpass",
               std::overflow_error);

  auto plan    = std::make_unique<flat_decode_plan>(stream, mr);
  plan->claims = cudf::detail::hostdevice_vector<flat_page_claim>(claims.size(), stream, mr);
  std::copy(claims.begin(), claims.end(), plan->claims.begin());
  plan->claims.host_to_device_async(stream);
  plan->checkpoints = cuda::device_buffer<jit::checkpoint>(
    stream, mr.get_temporary_mr(), num_checkpoints, cuda::no_init);
  plan->page_refs =
    cuda::device_buffer<jit::page_ref>(stream, mr.get_temporary_mr(), claims.size(), cuda::no_init);
  plan->null_counts =
    cuda::device_buffer<int32_t>(stream, mr.get_temporary_mr(), claims.size(), cuda::no_init);

  if (not claims.empty() and num_checkpoints != 0) {
    prepare_flat_pages_kernel<<<claims.size(), prepare_block_size, 0, stream.get()>>>(
      pages.device_ptr(),
      chunks,
      plan->claims.device_ptr(),
      plan->checkpoints.data(),
      level_type_size);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return plan;
}

void decode_flat_pages(flat_decode_plan& plan,
                       cudf::detail::hostdevice_span<PageInfo> pages,
                       cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                       cudf::host_span<void* const> column_data,
                       size_t min_row,
                       size_t num_rows,
                       int level_type_size,
                       kernel_error::pointer error_code,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr)
{
  auto const num_claims = plan.claims.size();
  if (num_claims == 0) { return; }

  // One tile list per kernel shape, holding the tiles that overlap the row range.
  std::array<std::vector<jit::tile_ref>, num_flat_shapes> shape_tiles;
  auto const range_end = min_row + num_rows;
  for (size_t idx = 0; idx < num_claims; ++idx) {
    auto const& claim       = plan.claims[idx];
    auto const& page        = pages[claim.page_idx];
    auto const& chunk       = chunks[page.chunk_idx];
    size_t const page_begin = chunk.start_row + page.chunk_row;
    size_t const page_end   = page_begin + page.num_rows;
    if (page_begin >= range_end or page_end <= min_row) { continue; }
    auto const first_row = page_begin >= min_row ? 0 : min_row - page_begin;
    auto const last_row  = std::min(page_end, range_end) - page_begin - 1;
    auto& tiles          = shape_tiles[claim.shape];
    for (auto tile = first_row / jit::tile_rows; tile <= last_row / jit::tile_rows; ++tile) {
      tiles.push_back({static_cast<uint32_t>(idx), static_cast<uint32_t>(tile)});
    }
  }
  auto const num_tiles = std::accumulate(
    shape_tiles.begin(), shape_tiles.end(), size_t{0}, [](size_t sum, auto const& v) {
      return sum + v.size();
    });

  cudf::detail::hostdevice_vector<jit::tile_ref> tiles(num_tiles, stream, mr);
  // Every column uses the store leaf, whose context holds only the output pointer.
  cudf::detail::hostdevice_vector<jit::leaf_ctx> d_contexts(column_data.size(), stream, mr);
  std::transform(column_data.begin(), column_data.end(), d_contexts.begin(), [](void* out) {
    jit::leaf_ctx ctx{};
    ctx.out = out;
    return ctx;
  });
  {
    auto out = tiles.begin();
    for (auto const& v : shape_tiles) {
      out = std::copy(v.begin(), v.end(), out);
    }
  }
  tiles.host_to_device_async(stream);
  d_contexts.host_to_device_async(stream);

  auto const num_page_blocks =
    cudf::detail::div_rounding_up_safe<size_t>(num_claims, page_block_size);
  setup_flat_pages_kernel<<<num_page_blocks, page_block_size, 0, stream.get()>>>(
    pages.device_ptr(),
    chunks,
    plan.claims,
    min_row,
    num_rows,
    level_type_size,
    plan.page_refs.data(),
    plan.null_counts.data(),
    error_code);
  CUDF_CUDA_TRY(cudaGetLastError());

  size_t offset = 0;
  for (uint32_t shape = 0; shape < num_flat_shapes; ++shape) {
    auto const count = shape_tiles[shape].size();
    if (count == 0) { continue; }
    launch_decode_flat_tiles(shape,
                             plan.page_refs.data(),
                             tiles.device_ptr(offset),
                             static_cast<uint32_t>(count),
                             plan.checkpoints.data(),
                             d_contexts.device_ptr(),
                             plan.null_counts.data(),
                             stream);
    offset += count;
  }

  finish_flat_pages_kernel<<<num_page_blocks, page_block_size, 0, stream.get()>>>(
    pages.device_ptr(), plan.claims, plan.null_counts.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // The kernels read the tile list and the contexts; keep both alive until they are done.
  stream.sync();
}

}  // namespace cudf::io::parquet::detail
