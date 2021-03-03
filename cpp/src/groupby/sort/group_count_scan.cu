/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <groupby/sort/group_scan_util.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> count_scan(rmm::device_vector<size_type> const& group_labels,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  std::unique_ptr<column> result = make_fixed_width_column(
    data_type{type_id::INT32}, group_labels.size(), mask_state::UNALLOCATED, stream, mr);

  if (group_labels.empty()) { return result; }

  auto resultview = result->mutable_view();
  // aggregation::COUNT_ALL
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                group_labels.begin(),
                                group_labels.end(),
                                thrust::make_constant_iterator<size_type>(1),
                                resultview.begin<size_type>());
  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
