/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/sorting.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/compiled_binaryop.hpp>

#include <random>
#include <memory>

class BinOp : public cudf::benchmark {};

// TODO: put it in a struct so `uniform` can be remade with different min, max
template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

void BM_basic_sum1(benchmark::State& state){
  using wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
  //cudf::data_type(cudf::experimental::type_to_id<int64_t>()));
  const cudf::size_type column_size{(cudf::size_type)state.range(0)};
  auto data_it = cudf::test::make_counting_transform_iterator(0,
    [=](cudf::size_type row) { return random_int(0, 100); });
  wrapper keys(data_it, data_it + column_size);
  wrapper vals(data_it, data_it + column_size);
  cudf::column_view lhs = keys;
  cudf::column_view rhs = vals;

  for(auto _ : state) {
    cuda_event_timer timer(state, true);
    auto out1 = cudf::experimental::experimental_binary_operation1(lhs, rhs, lhs.type());
  }
  state.SetComplexityN(state.range(0));
}

void BM_basic_sum2(benchmark::State& state){
  using wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
  //cudf::data_type(cudf::experimental::type_to_id<int64_t>()));
  const cudf::size_type column_size{(cudf::size_type)state.range(0)};
  auto data_it = cudf::test::make_counting_transform_iterator(0,
    [=](cudf::size_type row) { return random_int(0, 100); });
  wrapper keys(data_it, data_it + column_size);
  wrapper vals(data_it, data_it + column_size);
  cudf::column_view lhs = keys;
  cudf::column_view rhs = vals;

  for(auto _ : state) {
    cuda_event_timer timer(state, true);
    auto out1 = cudf::experimental::experimental_binary_operation2(lhs, rhs, lhs.type());
  }
  state.SetComplexityN(state.range(0));
}


void BM_basic_sum3(benchmark::State& state){
  using wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
  //cudf::data_type(cudf::experimental::type_to_id<int64_t>()));
  const cudf::size_type column_size{(cudf::size_type)state.range(0)};
  auto data_it = thrust::make_counting_iterator<int64_t>(0);
  //auto data_it = cudf::test::make_counting_transform_iterator(0,
  //  [=](cudf::size_type row) { return random_int(0, 100); });
  wrapper keys(data_it, data_it + column_size);
  wrapper vals(data_it, data_it + column_size);
  cudf::column_view lhs = keys;
  cudf::column_view rhs = vals;

  for(auto _ : state) {
    cuda_event_timer timer(state, true);
    auto out1 = cudf::experimental::experimental_binary_operation3(lhs, rhs, lhs.type());
  }
  state.SetComplexityN(state.range(0));
}

#define BBM_BENCHMARK_DEFINE(BNAME, name)                                      \
  BENCHMARK_DEFINE_F(BinOp, name)(::benchmark::State & state) {                \
    BNAME(state);                                                              \
  }                                                                            \
  BENCHMARK_REGISTER_F(BinOp, name)                                           \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Arg(1000)       /* 1k  */                                              \
      ->Arg(10000)      /* 10k */                                              \
      ->Arg(100000)     /* 100k*/                                              \
      ->Arg(1000000)    /* 1M  */                                              \
      ->Arg(10000000)   /* 10M */                                              \
      ->Arg(100000000)  /* 100M*/                                              \
      ->Arg(1000000000);/* 1G  */

BBM_BENCHMARK_DEFINE(BM_basic_sum1, exp1);
BBM_BENCHMARK_DEFINE(BM_basic_sum2, exp2);
BBM_BENCHMARK_DEFINE(BM_basic_sum3, exp3);
