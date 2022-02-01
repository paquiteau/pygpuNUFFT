/*
This file holds the python bindings for gpuNUFFT library.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
*/

#include "batch_factory.hpp"
#include "config.hpp"
#include "cuda_runtime.h"
#include "cuda_utils.hpp"
#include "cufft.h"
#include "gpuNUFFT_operator_factory.hpp"
#include "gpuNUFFT_types.hpp"
#include "python_utils.hpp"
#include <algorithm>  // std::sort
#include <cstdint>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <sys/types.h>
#include <vector>  // std::vector

namespace py = pybind11;

BatchGpuNUFFTPythonOperator::BatchGpuNUFFTPythonOperator(
    std::vector<py::array_t<DType>> kspace_loc_list,
    py::array_t<int> image_size, int num_coils,
    py::array_t<std::complex<DType>>sense_maps,
    int kernel_width, int sector_width, int osr, bool balance_workload)
{
  printf("initialize_cuda");
  cudaFree(0);
    // sensitivity maps
    py::buffer_info sense_maps_buffer = sense_maps.request();
    if (sense_maps_buffer.shape.size()==0)
    {
        has_sense_data = false;
        sensArray.data = NULL;
    }
    else
    {
        allocate_pinned_memory(&sensArray, n_coils * imgDims.count() * sizeof(DType2));
        sensArray.dim = imgDims;
        sensArray.dim.channels = n_coils;
        copyNumpyArray(sense_maps, sensArray.data);
        has_sense_data = true;
    }

  n_coils = num_coils;

  const int n_frames = kspace_loc_list.size();


  // initialise every workers
  for(int frame; frame < n_frames; frame++){

  }
}
