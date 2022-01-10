#ifndef PYTHON_UTILS_H_
#define PYTHON_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cuda.h>

#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"

namespace py = pybind11;

template <typename TType>
inline gpuNUFFT::Array<TType>
readNumpyArray(py::array_t<TType> data)
{
    py::buffer_info myData = data.request();
    TType *t_data = (TType *) myData.ptr;
    gpuNUFFT::Array<TType> dataArray;
    dataArray.data = t_data;
    return dataArray;
}

inline gpuNUFFT::Array<DType2>
readNumpyArray(py::array_t<std::complex<DType>> data)
{
    gpuNUFFT::Array<DType2> dataArray;
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
    dataArray.data = new_data;
    return dataArray;
}

inline void allocate_pinned_memory(gpuNUFFT::Array<DType2> *lin_array, unsigned long int size)
{
  DType2 *new_data;
  cudaMallocHost((void **)&new_data, size);
  lin_array->data = new_data;
}
inline void allocate_pinned_memory(gpuNUFFT::Array<DType> *lin_array, unsigned long int size)
{
  DType *new_data;
  cudaMallocHost((void **)&new_data, size);
  lin_array->data = new_data;
}
template <typename TType>
inline void copyNumpyArray(py::array_t<std::complex<DType>> data, TType *copy_data)
{
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    TType *my_data = reinterpret_cast<TType(&)[0]>(*t_data);
    memcpy(copy_data, my_data, myData.size*sizeof(TType));
}
#endif // PYTHON_UTILS_H_
