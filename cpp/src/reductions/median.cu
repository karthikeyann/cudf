//#include <cudf/cudf.h>

#include <thrust/transform_reduce.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include "reduction_functions.cuh"
#include <bitmask/legacy/legacy_bitmask.hpp>
//#include <rmm/thrust_rmm_allocator.h>

// int any types  -> t=float
// float -> t=float
// double -> t=double
// TODO: works with nan/null? (null ignored)

// TODO need to replace unsigned int with to hold 2^64
typedef thrust::pair<double, unsigned int> Mypair;

template <typename ElementType, typename ResultType>
struct abs_diff : public thrust::unary_function<ElementType, ResultType> {
  const double y;
  ElementType const* elements{};    ///< pointer of cudf data array
  gdf_valid_type const* bitmask{};  ///< pointer of cudf bitmask (null) array
  ResultType const
      identity{};  ///< identity value used when the validity is false

  abs_diff(const gdf_column column, const ResultType _identity, double _y)
      : y(_y),
        elements(static_cast<const ElementType*>(column.data)),
        bitmask(reinterpret_cast<const gdf_valid_type*>(column.valid)),
        identity(_identity) {}

  __host__ __device__ ResultType operator()(gdf_index_type i) const {
    //auto a = static_cast<double>(elements[i]);
    auto a = elements[i];
    return gdf_is_valid(bitmask, i)
               ? ((y >= a) ? Mypair(y-a,1) : Mypair(a-y,0))
               //? Mypair(abs(y - static_cast<double>(elements[i])),
               //         y >= static_cast<double>(elements[i]))
               : identity;
  }
};

template <typename T>
struct plusplus : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(const T& a, const T& b) {
    return T(a.first + b.first, a.second + b.second);
  }
};

template <typename T>
unsigned Objective(const gdf_column col, int n, double t, double* f,
                   double* df) {
  /* calculates the values of the objective and its subgradient */
  // abs_diff<T, Mypair> unary_op(t);
  plusplus<Mypair> binary_op;
  Mypair initpair(0.0, 0);
  auto transformer = abs_diff<T, Mypair>{col, Mypair(0.0, 0), t};
  // auto it = thrust::make_transform_iterator(
  //                               transformer);
  Mypair result = thrust::transform_reduce(
      thrust::counting_iterator<gdf_index_type>(0),
      thrust::counting_iterator<gdf_index_type>(col.size), transformer,
      initpair, binary_op);
  *df = 2.0 * result.second - n;
  *f = result.first;
  return result.second;
}

template <typename T>
struct inside_interval {
  const double L, R;
  T const* elements{};                    ///< pointer of cudf data array
  gdf_valid_type const* bitmask{};        ///< pointer of cudf bitmask (null) array
  inside_interval(const gdf_column column, const double& L, const double& R)
      : L(L),
        R(R),
        elements(static_cast<const T*>(column.data)),
        bitmask(reinterpret_cast<const gdf_valid_type*>(column.valid)) {}
  __host__ __device__ bool operator()(const gdf_index_type& i) {
    return gdf_is_valid(bitmask, i)
               ? (L < elements[i] && elements[i] < R)
               : false;
    // return (L < a && a < R);
  }
};

template <typename T>
double SortZ(const gdf_column col, double L, double R, int index,
             unsigned interval_size)
/* Copies the data satisfying L<data[i]<R into Z and returns the n/2-index
   order statistic of Z after sorting. */
{
  std::cout<<"L="<<L<<",R="<<R<<",index="<<index<<",interval_size="<<interval_size<<std::endl;
  inside_interval<T> pred{col, L, R};
  thrust::device_vector<T> Z(interval_size);

  auto endZ = thrust::copy_if(
      thrust::counting_iterator<gdf_index_type>(0),
      thrust::counting_iterator<gdf_index_type>(col.size), Z.begin(), pred);
  thrust::sort(Z.begin(), endZ);
  std::cout<<"zn="<<interval_size<<"ez="<<thrust::distance(Z.begin(), endZ)<<std::endl;
  auto n = col.size - col.null_count;
  //auto kth = llroundf(n / 2.0) - index;
  auto kth = n / 2 - index;
  std::cout<<"[n/2]-i="<<kth<<",index="<<index<<std::endl;
  if (n%2 == 0)
    return (Z[kth] + Z[kth-1]) / 2.0;
  else
    return Z[kth];
  // TODO return float/double/T ? gdf_scalar? union?
}

namespace detail {
struct median {
  template <typename T,
            typename std::enable_if_t<!std::is_arithmetic<T>::value>* = nullptr>
  gdf_scalar operator()(const gdf_column col) {
    CUDF_FAIL("input data type is not convertible to output data type");
  }
  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  gdf_scalar operator()(const gdf_column col) {
    //TODO: fix this
    //T yL = reinterpret_cast<T>(cudf::reduction::min(col, col.dtype).data);
    //T yR = reinterpret_cast<T>(cudf::reduction::max(col, col.dtype).data);
    double yL = cudf::reduction::min(col, GDF_FLOAT64).data.fp64;
    double yR = cudf::reduction::max(col, GDF_FLOAT64).data.fp64;
    std::cout<<"min: "<<yL<<std::endl;
    std::cout<<"max: "<<yR<<std::endl;
    auto n = col.size - col.null_count;
    decltype(col.size) ltL = 1, ltR = n;
    std::cout<<"ltL="<<ltL
    <<"ltR="<<ltR
    <<std::endl;
    double fL, gL, fR, gR;
    gL = -n + 2;
    gR = n - 2; //wrong: as per paper
    ltL = Objective<T>(col, n, yL, &fL, &gL);
    ltR = Objective<T>(col, n, yR, &fR, &gR);
    std::cout<<"ltL="<<ltL<<",lTR="<<ltR<<std::endl;
    // TODO: need to make sure no overflow
    double sum = cudf::reduction::sum(col, GDF_FLOAT64).data.fp64;
    std::cout<<"sum: "<<sum<<std::endl;
    fL = sum - n * yL;
    fR = n * yR - sum;
    double t = NAN;
    // an approximate solution y, under 30 iterations with n up to 32 million
    // and tolerancef = 10-12.
    for (int i = 0; i < 7; i++) {
      t = (fR - fL + yL * gL - yR * gR) / (gL - gR);
    std::cout<<" fL="<<fL
             <<",fR="<<fR
             <<",gL="<<gL
             <<",gR="<<gR
             <<",yL="<<yL
             <<",yR="<<yR
             <<",t="<<t;
      double ft, gt;
      unsigned ltt = Objective<T>(col, n, t, &ft, &gt);
      if (gt < 0) {
        yL = t;
        fL = ft;
        gL = gt;
        ltL = ltt;
      } else {
        yR = t;
        fR = ft;
        gR = gt;
        ltR = ltt;
      }
    std::cout<<",ft="<<ft
             <<",gt="<<gt
             <<",lt="<<ltt
             <<std::endl;
      //std::cout<<(ltR-ltL)<<",";
      //// stop if interval has <1M elements
      if ((ltR - ltL) < (1LL<<20)) {
        std::cout<<"break-"<<i<<std::endl;
        break;
      }
    }
 
    gdf_scalar result{.data={.fp64=NAN}, .dtype=GDF_FLOAT64, .is_valid=true};
    result.data.fp64 = SortZ<T>(col, yL, yR, ltL, ltR - ltL);
    std::cout<<"\nres="<<result.data.fp64<<std::endl;
    return result;
  }
};
// template double median<double>(const gdf_column col);
}  // namespace detail

gdf_scalar median(const gdf_column col) {
  return cudf::type_dispatcher(col.dtype, detail::median(), col);
}

