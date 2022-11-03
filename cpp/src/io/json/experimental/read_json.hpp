/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace cudf::io::detail::json::experimental {

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr);

size_t sources_size(host_span<std::unique_ptr<datasource>> const sources,
                    size_t range_offset,
                    size_t range_size);

size_type find_first_delimiter_in_chunk(host_span<std::unique_ptr<cudf::io::datasource>> sources,
                                        json_reader_options const& reader_opts,
                                        char const delimiter,
                                        rmm::cuda_stream_view stream);

std::vector<table_with_metadata> skeleton_for_parellel_chunk_reader(
  host_span<std::unique_ptr<datasource>> sources,
  json_reader_options const& reader_opts,
  int chunk_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr);
}  // namespace cudf::io::detail::json::experimental
