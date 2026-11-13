/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Datasource over a host buffer that records the device reads it serves
 */
class RecordingSource : public cudf::io::datasource {
 public:
  explicit RecordingSource(std::vector<uint8_t> data) : _data{std::move(data)} {}

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count = std::min(size, _data.size() - offset);
    return buffer::create(
      std::vector<uint8_t>(_data.begin() + offset, _data.begin() + offset + count));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count = std::min(size, _data.size() - offset);
    std::memcpy(dst, _data.data() + offset, count);
    return count;
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cuda::stream_ref stream) override
  {
    auto const count = std::min(size, _data.size() - offset);
    {
      std::lock_guard const lock{_mutex};
      _device_reads.emplace_back(offset, count);
    }
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dst, _data.data() + offset, count, cudaMemcpyDefault, stream.get()));
    stream.sync();
    return std::async(std::launch::deferred, [count] { return count; });
  }

  size_t device_read(size_t offset, size_t size, uint8_t* dst, cuda::stream_ref stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  [[nodiscard]] size_t size() const override { return _data.size(); }

  /// The (offset, size) of every device read served so far, in source order.
  [[nodiscard]] std::vector<std::pair<size_t, size_t>> device_reads() const
  {
    std::lock_guard const lock{_mutex};
    auto reads = _device_reads;
    std::sort(reads.begin(), reads.end());
    return reads;
  }

 private:
  std::vector<uint8_t> _data;
  mutable std::mutex _mutex;
  std::vector<std::pair<size_t, size_t>> _device_reads;
};

/**
 * @brief Datasource that serves host reads only and records them
 */
class HostRecordingSource : public cudf::io::datasource {
 public:
  explicit HostRecordingSource(std::vector<uint8_t> data) : _data{std::move(data)} {}

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count = std::min(size, _data.size() - offset);
    {
      std::lock_guard const lock{_mutex};
      _host_reads.emplace_back(offset, count);
    }
    return buffer::create(
      std::vector<uint8_t>(_data.begin() + offset, _data.begin() + offset + count));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count = std::min(size, _data.size() - offset);
    std::memcpy(dst, _data.data() + offset, count);
    return count;
  }

  [[nodiscard]] size_t size() const override { return _data.size(); }

  /// The (offset, size) of every buffer-returning host read served so far, in source order.
  [[nodiscard]] std::vector<std::pair<size_t, size_t>> host_reads() const
  {
    std::lock_guard const lock{_mutex};
    auto reads = _host_reads;
    std::sort(reads.begin(), reads.end());
    return reads;
  }

 private:
  std::vector<uint8_t> _data;
  mutable std::mutex _mutex;
  std::vector<std::pair<size_t, size_t>> _host_reads;
};

std::vector<uint8_t> make_data(size_t size)
{
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>(i * 31 + (i >> 8));
  }
  return data;
}

}  // namespace

struct ParquetIoUtilsTest : public cudf::test::BaseFixture {
  // Checks that every span holds the bytes of its range.
  static void verify(std::span<cudf::device_span<uint8_t const> const> spans,
                     std::vector<uint8_t> const& data,
                     std::vector<cudf::io::text::byte_range_info> const& byte_ranges)
  {
    auto const stream = cudf::get_default_stream();
    ASSERT_EQ(spans.size(), byte_ranges.size());
    for (size_t i = 0; i < byte_ranges.size(); ++i) {
      auto const offset = static_cast<size_t>(byte_ranges[i].offset());
      auto const size   = static_cast<size_t>(byte_ranges[i].size());
      ASSERT_EQ(spans[i].size(), size) << "range " << i;
      auto const fetched = cudf::detail::make_std_vector(spans[i], stream);
      EXPECT_TRUE(std::equal(fetched.begin(), fetched.end(), data.begin() + offset))
        << "range " << i;
    }
  }

  // Fetches `byte_ranges` and checks that every returned span holds the bytes of its range.
  static void fetch_and_verify(cudf::io::datasource& source,
                               std::vector<uint8_t> const& data,
                               std::vector<cudf::io::text::byte_range_info> const& byte_ranges)
  {
    auto const stream            = cudf::get_default_stream();
    auto const mr                = cudf::get_current_device_resource_ref();
    auto [buffers, spans, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
      source, byte_ranges, cudf::io::parquet::io_submission_policy::SERIALIZE, stream, {mr, mr});
    tasks.get();
    verify(spans, data, byte_ranges);
  }
};

// Column chunks are requested in the order of the selected columns, which need not be their order
// in the file. Ranges that are contiguous in the source are fetched with one read wherever they
// are in the request.
TEST_F(ParquetIoUtilsTest, FetchCoalescesContiguousRangesInAnyRequestOrder)
{
  auto const data = make_data(8'192);
  RecordingSource source{data};

  // In source order: [0, 100) [100, 300) [300, 400) are contiguous, [1000, 1050) is alone,
  // [2000, 2500) [2500, 2600) are contiguous.
  std::vector<cudf::io::text::byte_range_info> const byte_ranges{
    {2'500, 100}, {300, 100}, {1'000, 50}, {0, 100}, {2'000, 500}, {100, 200}};
  fetch_and_verify(source, data, byte_ranges);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source.device_reads(), (reads{{0, 400}, {1'000, 50}, {2'000, 600}}));
}

TEST_F(ParquetIoUtilsTest, FetchKeepsSeparateRangesSeparate)
{
  auto const data = make_data(4'096);
  RecordingSource source{data};

  // A gap of one byte, a repeated range, an overlap and an empty range: none is contiguous with
  // its neighbor in the source.
  std::vector<cudf::io::text::byte_range_info> const byte_ranges{
    {101, 100}, {0, 100}, {1'000, 64}, {1'000, 64}, {2'000, 100}, {2'050, 100}, {3'000, 0}};
  fetch_and_verify(source, data, byte_ranges);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source.device_reads(),
            (reads{{0, 100}, {101, 100}, {1'000, 64}, {1'000, 64}, {2'000, 100}, {2'050, 100}}));
}

TEST_F(ParquetIoUtilsTest, FetchInSourceOrderIsUnchanged)
{
  auto const data = make_data(4'096);
  RecordingSource source{data};

  std::vector<cudf::io::text::byte_range_info> const byte_ranges{
    {0, 256}, {256, 256}, {1'024, 512}, {1'536, 512}, {3'000, 96}};
  fetch_and_verify(source, data, byte_ranges);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source.device_reads(), (reads{{0, 512}, {1'024, 1'024}, {3'000, 96}}));
}

// An empty range is not read and does not split the run it lies in, wherever it is in the request.
TEST_F(ParquetIoUtilsTest, FetchIgnoresEmptyRangesWhenCoalescing)
{
  auto const data = make_data(4'096);
  RecordingSource source{data};

  // An empty range where two ranges meet, one strictly inside a run and one after the last byte.
  std::vector<cudf::io::text::byte_range_info> const byte_ranges{
    {100, 100}, {100, 0}, {0, 100}, {250, 0}, {200, 100}, {300, 0}};
  fetch_and_verify(source, data, byte_ranges);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source.device_reads(), (reads{{0, 300}}));
}

// Column chunks requested row group by row group with the columns out of file order, as the
// readers do: the chunks of both row groups form one contiguous run.
TEST_F(ParquetIoUtilsTest, FetchCoalescesAcrossRowGroups)
{
  auto const data = make_data(4'096);
  RecordingSource source{data};

  std::vector<cudf::io::text::byte_range_info> const byte_ranges{
    {200, 100}, {0, 200}, {500, 100}, {300, 200}};
  fetch_and_verify(source, data, byte_ranges);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source.device_reads(), (reads{{0, 600}}));
}

// A datasource without device reads is read on the host and copied; the coalesced runs and the
// spans are the same.
TEST_F(ParquetIoUtilsTest, FetchCoalescesHostReads)
{
  auto const data = make_data(8'192);
  HostRecordingSource source{data};

  std::vector<cudf::io::text::byte_range_info> const byte_ranges{
    {2'500, 100}, {300, 100}, {1'000, 50}, {0, 100}, {2'000, 500}, {100, 200}};
  fetch_and_verify(source, data, byte_ranges);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source.host_reads(), (reads{{0, 400}, {1'000, 50}, {2'000, 600}}));
}

// Each datasource is coalesced on its own and keeps its own spans.
TEST_F(ParquetIoUtilsTest, FetchCoalescesPerSource)
{
  auto const data_a = make_data(4'096);
  auto data_b       = make_data(4'096);
  std::reverse(data_b.begin(), data_b.end());
  RecordingSource source_a{data_a};
  RecordingSource source_b{data_b};

  std::vector<std::reference_wrapper<cudf::io::datasource>> const sources{source_a, source_b};
  std::vector<std::vector<cudf::io::text::byte_range_info>> const byte_ranges{
    {{100, 100}, {1'000, 24}, {0, 100}}, {{700, 300}, {500, 200}}};

  auto const stream            = cudf::get_default_stream();
  auto const mr                = cudf::get_current_device_resource_ref();
  auto [buffers, spans, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    sources, byte_ranges, cudf::io::parquet::io_submission_policy::SERIALIZE, stream, {mr, mr});
  tasks.get();

  ASSERT_EQ(spans.size(), std::size_t{2});
  verify(spans[0], data_a, byte_ranges[0]);
  verify(spans[1], data_b, byte_ranges[1]);

  using reads = std::vector<std::pair<size_t, size_t>>;
  EXPECT_EQ(source_a.device_reads(), (reads{{0, 200}, {1'000, 24}}));
  EXPECT_EQ(source_b.device_reads(), (reads{{500, 500}}));
}

CUDF_TEST_PROGRAM_MAIN()
