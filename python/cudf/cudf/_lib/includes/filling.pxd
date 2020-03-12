# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *


cdef extern from "cudf/legacy/filling.hpp" namespace "cudf" nogil:

    cdef void fill(
        gdf_column * column,
        const gdf_scalar & value,
        size_type begin,
        size_type end
    ) except +

    cdef cudf_table repeat(
        const cudf_table & input,
        const gdf_column & count
    ) except +

    cdef cudf_table repeat(
        const cudf_table & input,
        const gdf_scalar & count
    ) except +

    cdef cudf_table tile(
        const cudf_table & input,
        size_type count
    ) except +
