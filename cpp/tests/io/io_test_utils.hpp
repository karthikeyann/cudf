/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <rapids_logger/logger.hpp>

#include <future>
#include <sstream>
#include <string>
#include <vector>

namespace cudf::test {

/**
 * @brief Custom exception for device read async testing
 */
class AsyncException : public std::exception {};

/**
 * @brief Datasource that throws an exception in device_read_async for testing
 */
class ThrowingDeviceReadDatasource : public cudf::io::datasource {
 private:
  std::vector<char> const& data_;

 public:
  using cudf::io::datasource::device_read;

  explicit ThrowingDeviceReadDatasource(std::vector<char> const& data) : data_(data) {}

  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override
  {
    size = std::min(size, data_.size() - offset);
    // Convert char data to bytes for the buffer
    std::vector<std::byte> byte_data(size);
    std::memcpy(byte_data.data(), data_.data() + offset, size);
    return cudf::io::datasource::buffer::create(std::move(byte_data));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, data_.size() - offset);
    std::memcpy(dst, data_.data() + offset, read_size);
    return read_size;
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::unique_ptr<cudf::io::datasource::buffer> device_read(size_t offset,
                                                            size_t size,
                                                            cuda::stream_ref stream) override
  {
    // For testing, just copy the data from the host buffer into a new buffer
    size = std::min(size, data_.size() - offset);
    rmm::device_buffer out_data(size, stream);
    cudaMemcpyAsync(out_data.data(), data_.data() + offset, size, cudaMemcpyDefault, stream.get());
    cudaStreamSynchronize(stream.get());
    return cudf::io::datasource::buffer::create(std::move(out_data));
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cuda::stream_ref stream) override
  {
    // This datasource returns a future that throws a custom exception when accessed for testing
    std::promise<size_t> promise;
    promise.set_exception(std::make_exception_ptr(AsyncException()));
    return promise.get_future();
  }

  [[nodiscard]] size_t size() const override { return data_.size(); }
};

/**
 * @brief Data sink that throws an exception in device_write_async for testing
 */
class ThrowingDeviceWriteDataSink : public cudf::io::data_sink {
 private:
  size_t buffer_size_ = 0;

 public:
  void host_write(void const* data, size_t size) override { buffer_size_ += size; }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, cuda::stream_ref stream) override
  {
    buffer_size_ += size;
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       cuda::stream_ref stream) override
  {
    // This data sink returns a future that throws a custom exception when accessed for testing
    std::promise<void> promise;
    promise.set_exception(std::make_exception_ptr(AsyncException()));
    return promise.get_future();
  }

  void flush() override {}

  size_t bytes_written() override { return buffer_size_; }
};

/**
 * @brief RAII helper that captures log messages and suppresses terminal output.
 *
 * On construction, saves the logger's current sinks, replaces them with an ostream sink that
 * writes to an internal stringstream. On destruction, restores the original sinks.
 */
class log_capture {
 public:
  log_capture()
  {
    auto& logger = cudf::default_logger();
    for (auto& s : logger.sinks()) {
      saved_sinks_.push_back(std::move(s));
    }
    logger.sinks().clear();
    logger.sinks().push_back(std::make_shared<rapids_logger::ostream_sink_mt>(oss_));
  }

  ~log_capture()
  {
    auto& logger = cudf::default_logger();
    logger.sinks().clear();
    for (auto& s : saved_sinks_) {
      logger.sinks().push_back(std::move(s));
    }
  }

  log_capture(log_capture const&)            = delete;
  log_capture& operator=(log_capture const&) = delete;
  log_capture(log_capture&&)                 = delete;
  log_capture& operator=(log_capture&&)      = delete;

  [[nodiscard]] bool has_messages() const { return !oss_.str().empty(); }

 private:
  std::ostringstream oss_;
  std::vector<rapids_logger::sink_ptr> saved_sinks_;
};

/**
 * @brief Returns timestamp strings that test the parsing of the fixed ISO 8601 layout
 * `YYYY-MM-DD[T ]HH:MM:SS[Z|.fraction]`: strings of the layout, and near misses of it.
 *
 * The near misses are generated from strings of the layout: each is truncated to every length,
 * and at every position a character is deleted, or one of a set of characters is substituted or
 * inserted; a few suffixes are also appended. No string contains '/' (see
 * `with_slash_date_separators`).
 */
inline std::vector<std::string> iso_8601_timestamp_test_strings()
{
  std::vector<std::string> strings        = {"2024-02-29T13:45:59.",
                                             "2024-02-29T13:45:59.1",
                                             "2024-02-29T13:45:59.12",
                                             "2024-02-29 13:45:59.1234",
                                             "2024-02-29T13:45:59.12345",
                                             "2024-02-29T13:45:59.123456",
                                             "2024-02-29T13:45:59.1234567",
                                             "2024-02-29T13:45:59.12345678",
                                             "2024-02-29T13:45:59.123456789",
                                             "2024-02-29T13:45:59.123456789012",
                                             "2024-02-29T13:45:59.123Z",
                                             "2024-02-29 13:45:59.5+05:30",
                                             "2024-02-29T13:45:59.25-08:00",
                                             "1969-12-31T23:59:59.999",
                                             "0000-00-00T00:00:00",
                                             "9999-99-99T99:99:99.999",
                                             "2024-02-29T01:45:59.5 PM",
                                             "2024-02-29T01:45:59.5pm",
                                             "2024-02-29 01:45:59.5 AM",
                                             "2024-02-29T01:45:59 PM",
                                             "12:",
                                             "PM"};
  std::string const characters            = "0 -.:MPTZmtxz";
  std::vector<std::string> const suffixes = {"Z5", "Z+05:30", "ZZ", "Zm", "5", "M", " PM", "Z "};
  for (std::string const layout :
       {"2024-02-29T13:45:59", "2024-02-29 13:45:59Z", "2024-02-29T13:45:59.123"}) {
    for (std::size_t position = 0; position <= layout.size(); ++position) {
      auto const head = layout.substr(0, position);
      strings.push_back(head);
      for (char const character : characters) {
        strings.push_back(head + character + layout.substr(position));
      }
      if (position < layout.size()) {
        strings.push_back(head + layout.substr(position + 1));
        for (char const character : characters) {
          strings.push_back(head + character + layout.substr(position + 1));
        }
      }
    }
    for (auto const& suffix : suffixes) {
      strings.push_back(layout + suffix);
    }
  }
  return strings;
}

/**
 * @brief Replaces the first two '-' of a timestamp string with '/'.
 *
 * For a string without '/', timestamp parsing gives the same result for the returned string, which
 * never has the fixed ISO 8601 layout: the parsing treats '-' and '/' alike when it looks for the
 * end of the date, uses the first two separators of the date (all '-' or all '/'), and ignores
 * both in the components and in the time of day.
 *
 * @param timestamp Timestamp string without '/'
 * @return The string with '/' as date separator
 */
inline std::string with_slash_date_separators(std::string timestamp)
{
  for (int i = 0; i < 2; ++i) {
    if (auto const position = timestamp.find('-'); position != std::string::npos) {
      timestamp[position] = '/';
    }
  }
  return timestamp;
}

}  // namespace cudf::test

#define EXPECT_CUDF_LOG_WARN(statement)         \
  do {                                          \
    cudf::test::log_capture _cudf_log_cap_{};   \
    statement;                                  \
    EXPECT_TRUE(_cudf_log_cap_.has_messages()); \
  } while (0)
