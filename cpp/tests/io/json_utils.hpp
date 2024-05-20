/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace spark_rapids_jni {

/**
 * path instruction type
 */
enum class diff_path_instruction_type { SUBSCRIPT, WILDCARD, KEY, INDEX, NAMED };


/**
 * Tokenize an input string column and return a column of the form
 * STRUCT<buffer: String, tokens: LIST<STRUCT<tok: INT8, offset: UINT32>>
 * 
 * The buffer holds a potentially modified JSON data. Tokens are the
 * tokens for that buffer. The tok is the ID for the token and will match those in CUDF,
 * except LineEnd will be removed as it is encoded in the buffer itself. 
 * The offset indicates where this token was in buffer relative to the
 * start of the buffer.
 */
std::unique_ptr<cudf::column> tokenize_json(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());


std::unique_ptr<cudf::column> different_get_json_object(
  cudf::column_view const& input,
  std::vector<std::tuple<diff_path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
