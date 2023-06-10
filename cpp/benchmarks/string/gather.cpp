/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_gather(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  if (static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const input_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  data_profile const map_profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, num_rows);
  auto const map_table =
    create_random_table({cudf::type_id::INT32}, row_count{num_rows}, map_profile);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = cudf::strings_column_view(input_table->view().column(0)).chars_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);  // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::gather(
      input_table->view(), map_table->view().column(0), cudf::out_of_bounds_policy::NULLIFY);
  });
}

NVBENCH_BENCH(bench_gather)
  .set_name("gather")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024, 2048, 4096})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216});
