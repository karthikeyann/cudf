/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <vector>

/**
 * @file column_view.hpp
 * @brief column view class definitions
 */

namespace cudf {
namespace detail {
/**
 * @brief A non-owning, immutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * A `column_view_base` can be constructed implicitly from a `cudf::column`, or
 *may be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `column_view_base`'s data
 *and bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/memory_layout.html
 *
 * Because `column_view_base` is non-owning, no device memory is allocated nor
 *freed when `column_view_base` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `column_view_base` has an `offset` that
 *indicates the index of the first element in the column relative to the base
 *device memory allocation. By default, `offset()` is zero.
 */
class column_view_base {
 public:
  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   *a column, and instead, accessing the elements should be done via
   *`data<T>()`.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @tparam The type to cast to
   * @return T const* Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  T const* head() const noexcept
  {
    return static_cast<T const*>(_data);
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The type to cast to
   * @return T const* Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T const* data() const noexcept
  {
    return head<T>() + _offset;
  }

  /**
   * @brief Return first element (accounting for offset) after underlying data
   * is casted to the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return T const* Pointer to the first element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T const* begin() const noexcept
  {
    return data<T>();
  }

  /**
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return T const* Pointer to one past the last element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T const* end() const noexcept
  {
    return begin<T>() + size();
  }

  /**
   * @brief Returns the number of elements in the column
   */
  size_type size() const noexcept { return _size; }

  /**
   * @brief Returns true if `size()` returns zero, or false otherwise
   */
  size_type is_empty() const noexcept { return size() == 0; }

  /**
   * @brief Returns the element `data_type`
   */
  data_type type() const noexcept { return _type; }

  /**
   * @brief Indicates if the column can contain null elements, i.e., if it has
   * an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   */
  bool nullable() const noexcept { return nullptr != _null_mask; }

  /**
   * @brief Returns the count of null elements
   *
   * @note If the column was constructed with `UNKNOWN_NULL_COUNT`, or if at any
   * point `set_null_count(UNKNOWN_NULL_COUNT)` was invoked, then the
   * first invocation of `null_count()` will compute and store the count of null
   * elements indicated by the `null_mask` (if it exists).
   */
  size_type null_count() const;

  /**
   * @brief Returns the count of null elements in the range [begin, end)
   *
   * @note If `null_count() != 0`, every invocation of `null_count(begin, end)`
   * will recompute the count of null elements indicated by the `null_mask` in
   * the range [begin, end).
   *
   * @throws cudf::logic_error for invalid range (if `begin < 0`,
   * `begin > end`, `begin >= size()`, or `end > size()`).
   *
   * @param[in] begin The starting index of the range (inclusive).
   * @param[in] end The index of the last element in the range (exclusive).
   */
  size_type null_count(size_type begin, size_type end) const;

  /**
   * @brief Indicates if the column contains null elements,
   * i.e., `null_count() > 0`
   *
   * @return true One or more elements are null
   * @return false All elements are valid
   */
  bool has_nulls() const { return null_count() > 0; }

  /**
   * @brief Indicates if the column contains null elements in the range
   * [begin, end), i.e., `null_count(begin, end) > 0`
   *
   * @throws cudf::logic_error for invalid range (if `begin < 0`,
   * `begin > end`, `begin >= size()`, or `end > size()`).
   *
   * @param begin The starting index of the range (inclusive).
   * @param end The index of the last element in the range (exclusive).
   * @return true One or more elements are null in the range [begin, end)
   * @return false All elements are valid in the range [begin, end)
   */
  bool has_nulls(size_type begin, size_type end) const { return null_count(begin, end) > 0; }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   */
  bitmask_type const* null_mask() const noexcept { return _null_mask; }

  /**
   * @brief Returns the index of the first element relative to the base memory
   * allocation, i.e., what is returned from `head<T>()`.
   */
  size_type offset() const noexcept { return _offset; }

 protected:
  data_type _type{type_id::EMPTY};   ///< Element type
  size_type _size{};                 ///< Number of elements
  void const* _data{};               ///< Pointer to device memory containing elements
  bitmask_type const* _null_mask{};  ///< Pointer to device memory containing
                                     ///< bitmask representing null elements.
                                     ///< Optional if `null_count() == 0`
  mutable size_type _null_count{};   ///< The number of null elements
  size_type _offset{};               ///< Index position of the first element.
                                     ///< Enables zero-copy slicing

  column_view_base()                        = default;
  ~column_view_base()                       = default;
  column_view_base(column_view_base const&) = default;
  column_view_base(column_view_base&&)      = default;
  column_view_base& operator=(column_view_base const&) = default;
  column_view_base& operator=(column_view_base&&) = default;

  /**
   * @brief Construct a `column_view_base` from pointers to device memory for
   *the elements and bitmask of the column.
   *
   * If `null_count()` is zero, `null_mask` is optional.
   *
   * If the null count of the `null_mask` is not specified, it defaults to
   * `UNKNOWN_NULL_COUNT`. The first invocation of `null_count()` will then
   * compute the null count if `null_mask` exists.
   *
   * If `type` is `EMPTY`, the specified `null_count` will be ignored and
   * `null_count()` will always return the same value as `size()`
   *
   * @throws cudf::logic_error if `size < 0`
   * @throws cudf::logic_error if `size > 0` but `data == nullptr`
   * @throws cudf::logic_error if `type.id() == EMPTY` but `data != nullptr`
   *or `null_mask != nullptr`
   * @throws cudf::logic_error if `null_count > 0`, but `null_mask == nullptr`
   * @throws cudf::logic_error if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Optional, pointer to device memory containing the null
   * indicator bitmask
   * @param null_count Optional, the number of null elements.
   * @param offset optional, index of the first element
   * @param children optional, depending on the element type, child columns may
   * contain additional data
   */
  column_view_base(data_type type,
                   size_type size,
                   void const* data,
                   bitmask_type const* null_mask = nullptr,
                   size_type null_count          = UNKNOWN_NULL_COUNT,
                   size_type offset              = 0);
};

class mutable_column_view_base : public column_view_base {
 public:
 protected:
};
}  // namespace detail

/**
 * @brief A non-owning, immutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * @ingroup column_classes
 *
 * A `column_view` can be constructed implicitly from a `cudf::column`, or may
 * be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `column_view`'s data and
 * bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/memory_layout.html
 *
 * Because `column_view` is non-owning, no device memory is allocated nor freed
 * when `column_view` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `column_view` has an `offset` that indicates
 * the index of the first element in the column relative to the base device
 * memory allocation. By default, `offset()` is zero.
 */
class column_view : public detail::column_view_base {
 public:
  column_view() = default;

  // these pragmas work around the nvcc issue where if a column_view is used
  // inside of a __device__ code path, these functions will end up being created
  // as __host__ __device__ because they are explicitly defaulted.  However, if
  // they then end up being called by a simple __host__ function
  // (eg std::vector destructor) you get a compile error because you're trying to
  // call a __host__ __device__ function from a __host__ function.
#pragma nv_exec_check_disable
  ~column_view() = default;
#pragma nv_exec_check_disable
  column_view(column_view const& c) = default;

  column_view(column_view&&) = default;
  column_view& operator=(column_view const&) = default;
  column_view& operator=(column_view&&) = default;

  /**
   * @brief Construct a `column_view` from pointers to device memory for the
   * elements and bitmask of the column.
   *
   * If `null_count()` is zero, `null_mask` is optional.
   *
   * If the null count of the `null_mask` is not specified, it defaults to
   * `UNKNOWN_NULL_COUNT`. The first invocation of `null_count()` will then
   * compute the null count if `null_mask` exists.
   *
   * If `type` is `EMPTY`, the specified `null_count` will be ignored and
   * `null_count()` will always return the same value as `size()`
   *
   * @throws cudf::logic_error if `size < 0`
   * @throws cudf::logic_error if `size > 0` but `data == nullptr`
   * @throws cudf::logic_error if `type.id() == EMPTY` but `data != nullptr`
   *or `null_mask != nullptr`
   * @throws cudf::logic_error if `null_count > 0`, but `null_mask == nullptr`
   * @throws cudf::logic_error if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Optional, pointer to device memory containing the null
   * indicator bitmask
   * @param null_count Optional, the number of null elements.
   * @param offset optional, index of the first element
   * @param children optional, depending on the element type, child columns may
   * contain additional data
   */
  column_view(data_type type,
              size_type size,
              void const* data,
              bitmask_type const* null_mask            = nullptr,
              size_type null_count                     = UNKNOWN_NULL_COUNT,
              size_type offset                         = 0,
              std::vector<column_view> const& children = {});

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   */
  column_view child(size_type child_index) const noexcept { return _children[child_index]; }

  /**
   * @brief Returns the number of child columns.
   */
  size_type num_children() const noexcept { return _children.size(); }

  /**
   * @brief Returns iterator to the beginning of the ordered sequence of child column-views.
   */
  auto child_begin() const noexcept { return _children.cbegin(); }

  /**
   * @brief Returns iterator to the end of the ordered sequence of child column-views.
   */
  auto child_end() const noexcept { return _children.cend(); }

 private:
  friend column_view bit_cast(column_view const& input, data_type type);

  std::vector<column_view> _children{};  ///< Based on element type, children
                                         ///< may contain additional data
};                                       // namespace cudf

/**
 * @brief A non-owning, mutable view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * @ingroup column_classes
 *
 * A `mutable_column_view` can be constructed implicitly from a `cudf::column`,
 * or may be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `mutable_column_view`'s data
 * and bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/memory_layout.html
 *
 * Because `mutable_column_view` is non-owning, no device memory is allocated
 * nor freed when `mutable_column_view` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `mutable_column_view` has an `offset` that
 * indicates the index of the first element in the column relative to the base
 * device memory allocation. By default, `offset()` is zero.
 */
class mutable_column_view : public detail::column_view_base {
 public:
  mutable_column_view() = default;

  ~mutable_column_view() = default;

  mutable_column_view(mutable_column_view const&) = default;

  mutable_column_view(mutable_column_view&&) = default;
  mutable_column_view& operator=(mutable_column_view const&) = default;
  mutable_column_view& operator=(mutable_column_view&&) = default;

  /**
   * @brief Construct a `mutable_column_view` from pointers to device memory for
   *the elements and bitmask of the column.

   * If the null count of the `null_mask` is not specified, it defaults to
   * `UNKNOWN_NULL_COUNT`. The first invocation of `null_count()` will then
   * compute the null count.
   *
   * If `type` is `EMPTY`, the specified `null_count` will be ignored and
   * `null_count()` will always return the same value as `size()`
   *
   * @throws cudf::logic_error if `size < 0`
   * @throws cudf::logic_error if `size > 0` but `data == nullptr`
   * @throws cudf::logic_error if `type.id() == EMPTY` but `data != nullptr`
   *or `null_mask != nullptr`
   * @throws cudf::logic_error if `null_count > 0`, but `null_mask == nullptr`
   * @throws cudf::logic_error if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Optional, pointer to device memory containing the null
   indicator
   * bitmask
   * @param null_count Optional, the number of null elements.
   * @param offset optional, index of the first element
   * @param children optional, depending on the element type, child columns may
   * contain additional data
   */
  mutable_column_view(data_type type,
                      size_type size,
                      void* data,
                      bitmask_type* null_mask                          = nullptr,
                      size_type null_count                             = cudf::UNKNOWN_NULL_COUNT,
                      size_type offset                                 = 0,
                      std::vector<mutable_column_view> const& children = {});

  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of a
   * column, and instead, accessing the elements should be done via `data<T>()`.
   *
   * @tparam The type to cast to
   * @return T* Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  T* head() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::head<T>());
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @tparam T The type to cast to
   * @return T* Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T* data() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::data<T>());
  }

  /**
   * @brief Return first element (accounting for offset) when underlying data is
   * casted to the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return T* Pointer to the first element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T* begin() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::begin<T>());
  }

  /**
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The desired type
   * @return T* Pointer to one past the last element after casting
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  T* end() const noexcept
  {
    return const_cast<T*>(detail::column_view_base::end<T>());
  }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   */
  bitmask_type* null_mask() const noexcept
  {
    return const_cast<bitmask_type*>(detail::column_view_base::null_mask());
  }

  /**
   * @brief Set the null count
   *
   * @throws cudf::logic_error if `new_null_count > 0` and `nullable() == false`
   *
   * @param new_null_count The new null count
   */
  void set_null_count(size_type new_null_count);

  /**
   * @brief Returns a reference to the specified child
   *
   * @param child_index The index of the desired child
   * @return mutable_column_view The requested child `mutable_column_view`
   */
  mutable_column_view child(size_type child_index) const noexcept
  {
    return mutable_children[child_index];
  }

  /**
   * @brief Returns the number of child columns.
   */
  size_type num_children() const noexcept { return mutable_children.size(); }

  /**
   * @brief Returns iterator to the beginning of the ordered sequence of child column-views.
   */
  auto child_begin() const noexcept { return mutable_children.begin(); }

  /**
   * @brief Returns iterator to the end of the ordered sequence of child column-views.
   */
  auto child_end() const noexcept { return mutable_children.end(); }

  /**
   * @brief Converts a mutable view into an immutable view
   *
   * @return column_view An immutable view of the mutable view's elements
   */
  operator column_view() const;

 private:
  friend mutable_column_view bit_cast(mutable_column_view const& input, data_type type);

  std::vector<mutable_column_view> mutable_children;
};

/**
 * @brief Counts the number of descendants of the specified parent.
 *
 * @param parent The parent whose descendants will be counted
 * @return size_type The number of descendants of the parent
 */
size_type count_descendants(column_view parent);

/**
 * @brief Zero-copy cast between types with the same size and compatible underlying representations.
 *
 * This is similar to `reinterpret_cast` or `bit_cast` in that it gives a view of the same raw bits
 * as a different type. Unlike `reinterpret_cast` however, this cast is only allowed on types that
 * have the same width and compatible representations. For example, the way timestamp types are laid
 * out in memory is equivalent to an integer representing a duration since a fixed epoch;
 * bit-casting to the same integer type (INT32 for days, INT64 for others) results in a raw view of
 * the duration count. A FLOAT32 can also be bit-casted into INT32 and treated as an integer value.
 * However, an INT32 column cannot be bit-casted to INT64 as the sizes differ, nor can a string_view
 * column be casted into a numeric type column as their data representations are not compatible.
 *
 * The validity of the conversion can be checked with `cudf::is_bit_castable()`.
 *
 * @throws cudf::logic_error if the specified cast is not possible, i.e.,
 * `is_bit_castable(input.type(), type)` is false.
 *
 * @param input The `column_view` to cast from
 * @param type The `data_type` to cast to
 * @return New `column_view` wrapping the same data as `input` but cast to `type`
 */
column_view bit_cast(column_view const& input, data_type type);

/**
 * @brief Zero-copy cast between types with the same size and compatible underlying representations.
 *
 * This is similar to `reinterpret_cast` or `bit_cast` in that it gives a view of the same raw bits
 * as a different type. Unlike `reinterpret_cast` however, this cast is only allowed on types that
 * have the same width and compatible representations. For example, the way timestamp types are laid
 * out in memory is equivalent to an integer representing a duration since a fixed epoch;
 * bit-casting to the same integer type (INT32 for days, INT64 for others) results in a raw view of
 * the duration count. A FLOAT32 can also be bit-casted into INT32 and treated as an integer value.
 * However, an INT32 column cannot be bit-casted to INT64 as the sizes differ, nor can a string_view
 * column be casted into a numeric type column as their data representations are not compatible.
 *
 * The validity of the conversion can be checked with `cudf::is_bit_castable()`.
 *
 * @throws cudf::logic_error if the specified cast is not possible, i.e.,
 * `is_bit_castable(input.type(), type)` is false.
 *
 * @param input The `mutable_column_view` to cast from
 * @param type The `data_type` to cast to
 * @return New `mutable_column_view` wrapping the same data as `input` but cast to `type`
 */
mutable_column_view bit_cast(mutable_column_view const& input, data_type type);

namespace detail {
/**
 * @brief Computes a hash value on the specified column view based on the shallow state of the
 * column view.
 *
 * Only the shallow states (i.e pointers instead of data pointed by the pointer) of the column view
 * are used in the hash computation. The hash value is computed  recursively on the children of the
 * column view.
 * The states used for the hash computation are: type, size, data pointer, null_mask pointer,
 * offset, and the hash value of the children. Note that `null_count` is not used.
 *
 * Note: This hash function may result in different hash for a copy of the same column with exactly
 * same contents. It is guarenteed to give same hash value for same column_view only, even if the
 * underlying data changes.
 *
 * @param input The `column_view` to compute hash
 * @return The hash value
 */
size_t shallow_hash(column_view const& input);

/**
 * @brief Equality operator for column views based on the shallow state of the column view.
 *
 * Only shallow states used for the hash computation are: type, size, data pointer, null_mask
 * pointer, offset and the column_view of the children recursively. Note that `null_count` is not
 * used.
 *
 * Note: This equality function will consider a column not equal to a copy of the same column with
 * exactly same contents. It is guarenteed to return true for same column_view only, even if the
 * underlying data changes.
 *
 * @param lhs The left `column_view` to compare
 * @param rhs The right `column_view` to compare
 * @return true if the shallow states of the two column views are equal
 */
bool shallow_equal(column_view const& lhs, column_view const& rhs);
}  // namespace detail
}  // namespace cudf
