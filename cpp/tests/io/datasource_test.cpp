/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstring>
#include <fstream>
#include <memory>
#include <string>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {
/**
 * @brief User source that holds its data in host and device memory. Its device reads that return a
 * buffer return a view of the device memory if `zero_copy` is set (and report zero-copy reads), and
 * an owning copy otherwise.
 */
class user_source : public cudf::io::datasource {
 public:
  user_source(std::string const& data, bool zero_copy)
    : _data{data},
      _zero_copy{zero_copy},
      _device{
        cudf::detail::make_device_uvector(cudf::host_span<char const>{data.data(), data.size()},
                                          cudf::test::get_default_stream(),
                                          cudf::get_current_device_resource_ref())}
  {
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    size = std::min(size, _data.size() - offset);
    return buffer::create(std::vector<char>(_data.begin() + offset, _data.begin() + offset + size));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    size = std::min(size, _data.size() - offset);
    std::memcpy(dst, _data.data() + offset, size);
    return size;
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool supports_zero_copy_device_read() const override { return _zero_copy; }

  std::unique_ptr<buffer> device_read(size_t offset, size_t size, cuda::stream_ref stream) override
  {
    size = std::min(size, _data.size() - offset);
    if (_zero_copy) { return std::make_unique<non_owning_buffer>(device_data() + offset, size); }
    return buffer::create(rmm::device_buffer{device_data() + offset, size, stream});
  }

  size_t device_read(size_t offset, size_t size, uint8_t* dst, cuda::stream_ref stream) override
  {
    size = std::min(size, _data.size() - offset);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dst, device_data() + offset, size, cudaMemcpyDefault, stream.get()));
    return size;
  }

  [[nodiscard]] size_t size() const override { return _data.size(); }

  [[nodiscard]] uint8_t const* device_data() const
  {
    return reinterpret_cast<uint8_t const*>(_device.data());
  }

 private:
  std::string const& _data;
  bool _zero_copy;
  rmm::device_uvector<char> _device;
};
}  // namespace

struct DatasourceTest : public cudf::test::BaseFixture {};

TEST_F(DatasourceTest, ZeroCopyDeviceReads)
{
  // Device buffer sources return views of the buffer; host buffer and file sources do not, and
  // user sources report what they implement
  auto const stream      = cudf::test::get_default_stream();
  std::string const text = "0123456789";
  auto const d_buffer =
    cudf::detail::make_device_uvector(cudf::host_span<char const>{text.data(), text.size()},
                                      stream,
                                      cudf::get_current_device_resource_ref());
  auto const device_source = cudf::io::datasource::create(cudf::device_span<std::byte const>{
    reinterpret_cast<std::byte const*>(d_buffer.data()), d_buffer.size()});
  EXPECT_TRUE(device_source->supports_zero_copy_device_read());
  auto const view = device_source->device_read(2, 5, stream);
  EXPECT_EQ(view->data(), reinterpret_cast<uint8_t const*>(d_buffer.data() + 2));
  EXPECT_EQ(view->size(), 5);

  auto const host_source = cudf::io::datasource::create(
    cudf::host_span<std::byte const>{reinterpret_cast<std::byte const*>(text.data()), text.size()});
  EXPECT_FALSE(host_source->supports_zero_copy_device_read());

  auto const filepath = temp_env->get_temp_filepath("ZeroCopyDeviceReads.txt");
  std::ofstream(filepath) << text;
  EXPECT_FALSE(cudf::io::datasource::create(filepath)->supports_zero_copy_device_read());

  // The wrapper of user sources forwards the query and the device reads
  for (bool const zero_copy : {false, true}) {
    user_source source{text, zero_copy};
    auto const wrapper = cudf::io::datasource::create(&source);
    EXPECT_EQ(wrapper->supports_zero_copy_device_read(), zero_copy);
    auto const buffer = wrapper->device_read(2, 5, stream);
    EXPECT_EQ(buffer->size(), 5);
    EXPECT_EQ(buffer->data() == source.device_data() + 2, zero_copy);
  }
}

CUDF_TEST_PROGRAM_MAIN()
