/*
This file holds the python bindings for gpuNUFFT library.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
*/

#include "python_factory.hpp"
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
#include <string>
#include <sys/types.h>
#include <vector>  // std::vector

namespace py = pybind11;

GpuNUFFTPythonOperator::GpuNUFFTPythonOperator(
    py::array_t<DType> kspace_loc, py::array_t<int> image_size, int num_coils,
    py::array_t<std::complex<DType>> sense_maps,
    py::array_t<float> density_comp, int kernel_width, int sector_width,
    int osr, bool balance_workload)
{
  // initialize CUDA
  cudaFree(0);
  // k-space coordinates
  py::buffer_info sample_loc = kspace_loc.request();
  trajectory_length = sample_loc.shape[1];
  dimension = sample_loc.shape[0];
  gpuNUFFT::Array<DType> kSpaceTraj = readNumpyArray(kspace_loc);
  kSpaceTraj.dim.length = trajectory_length;

  // density compensation weights
  gpuNUFFT::Array<DType> density_compArray = readNumpyArray(density_comp);
  density_compArray.dim.length = trajectory_length;

  // image size
  py::buffer_info img_dim = image_size.request();
  int *dims = (int *)img_dim.ptr;
  imgDims.width = dims[0];
  imgDims.height = dims[1];
  if (dimension == 3)
    imgDims.depth = dims[2];
  else
    imgDims.depth = 0;

  n_coils = num_coils;

  // sensitivity maps
  py::buffer_info sense_maps_buffer = sense_maps.request();
  if (sense_maps_buffer.shape.size() == 0)
  {
    has_sense_data = false;
    sensArray.data = NULL;
  }
  else
  {
    allocate_pinned_memory(&sensArray,
                           n_coils * imgDims.count() * sizeof(DType2));
    sensArray.dim = imgDims;
    sensArray.dim.channels = n_coils;
    copyNumpyArray(sense_maps, sensArray.data);
    has_sense_data = true;
  }
  factory.setBalanceWorkload(balance_workload);
  gpuNUFFTOp =
      factory.createGpuNUFFTOperator(kSpaceTraj, density_compArray, sensArray,
                                     kernel_width, sector_width, osr, imgDims);
  allocate_pinned_memory(&kspace_data,
                         n_coils * trajectory_length * sizeof(DType2));
  kspace_data.dim.length = trajectory_length;
  kspace_data.dim.channels = n_coils;
  image.dim = imgDims;
  if (has_sense_data == false)
  {
    allocate_pinned_memory(&image, n_coils * imgDims.count() * sizeof(DType2));
    image.dim.channels = n_coils;
  }
  else
  {
    allocate_pinned_memory(&image, imgDims.count() * sizeof(DType2));
    image.dim.channels = 1;
  }
  cudaDeviceSynchronize();
}

py::array_t<std::complex<DType>>
GpuNUFFTPythonOperator::op(py::array_t<std::complex<DType>> input_image,
                           bool interpolate_data)
{
  // Copy array to pinned memory for better memory bandwidths!
  copyNumpyArray(input_image, image.data);
  if (interpolate_data)
    gpuNUFFTOp->performForwardGpuNUFFT(image, kspace_data,
                                       gpuNUFFT::DENSITY_ESTIMATION);
  else
    gpuNUFFTOp->performForwardGpuNUFFT(image, kspace_data);
  cudaDeviceSynchronize();
  std::complex<DType> *ptr =
      reinterpret_cast<std::complex<DType>(&)[0]>(*kspace_data.data);
  auto capsule = py::capsule(ptr, [](void *ptr) { return; });
  return py::array_t<std::complex<DType>>(
      { n_coils, trajectory_length },
      { sizeof(DType2) * trajectory_length, sizeof(DType2) }, ptr, capsule);
}
py::array_t<std::complex<DType>> GpuNUFFTPythonOperator::adj_op(
    py::array_t<std::complex<DType>> input_kspace_data, bool grid_data)
{
  gpuNUFFT::Dimensions myDims = imgDims;
  if (dimension == 2)
    myDims.depth = 1;
  copyNumpyArray(input_kspace_data, kspace_data.data);
  if (grid_data)
    gpuNUFFTOp->performGpuNUFFTAdj(kspace_data, image,
                                   gpuNUFFT::DENSITY_ESTIMATION);
  else
    gpuNUFFTOp->performGpuNUFFTAdj(kspace_data, image);
  cudaDeviceSynchronize();
  std::complex<DType> *ptr =
      reinterpret_cast<std::complex<DType>(&)[0]>(*image.data);
  auto capsule = py::capsule(ptr, [](void *ptr) { return; });
  if (has_sense_data == false)
    return py::array_t<std::complex<DType>>(
        { n_coils, (int)myDims.depth, (int)myDims.height, (int)myDims.width },
        {
            sizeof(DType2) * (int)myDims.depth * (int)myDims.height *
                (int)myDims.width,
            sizeof(DType2) * (int)myDims.height * (int)myDims.width,
            sizeof(DType2) * (int)myDims.width,
            sizeof(DType2),
        },
        ptr, capsule);
  else
    return py::array_t<std::complex<DType>>(
        { (int)myDims.depth, (int)myDims.height, (int)myDims.width },
        {
            sizeof(DType2) * (int)myDims.height * (int)myDims.width,
            sizeof(DType2) * (int)myDims.width,
            sizeof(DType2),
        },
        ptr, capsule);
}

py::array_t<DType>
GpuNUFFTPythonOperator::estimate_density_comp(int num_iter = 10)
{
  IndType n_samples = kspace_data.count();
  gpuNUFFT::Array<CufftType> densArray;
  allocate_pinned_memory(&densArray, n_samples * sizeof(CufftType));
  densArray.dim.length = n_samples;

  // TODO: Allocate directly on device and set with kernel.
  for (int cnt = 0; cnt < n_samples; cnt++)
  {
    densArray.data[cnt].x = 1.0;
    densArray.data[cnt].y = 0.0;
  }

  gpuNUFFT::GpuArray<DType2> densArray_gpu;
  densArray_gpu.dim.length = n_samples;
  allocateDeviceMem(&densArray_gpu.data, n_samples);

  copyToDeviceAsync(densArray.data, densArray_gpu.data, n_samples);

  gpuNUFFT::GpuArray<CufftType> densEstimation_gpu;
  densEstimation_gpu.dim.length = n_samples;
  allocateDeviceMem(&densEstimation_gpu.data, n_samples);

  gpuNUFFT::GpuArray<CufftType> image_gpu;
  image_gpu.dim = imgDims;
  allocateDeviceMem(&image_gpu.data, imgDims.count());

  if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
    printf("error at adj thread synchronization a: %s\n",
           cudaGetErrorString(cudaGetLastError()));
  for (int cnt = 0; cnt < num_iter; cnt++)
  {
    if (DEBUG)
      printf("### update %i\n", cnt);
    gpuNUFFTOp->performGpuNUFFTAdj(densArray_gpu, image_gpu,
                                   gpuNUFFT::DENSITY_ESTIMATION);
    gpuNUFFTOp->performForwardGpuNUFFT(image_gpu, densEstimation_gpu,
                                       gpuNUFFT::DENSITY_ESTIMATION);
    performUpdateDensityComp(densArray_gpu.data, densEstimation_gpu.data,
                             n_samples);
    if (DEBUG && (cudaDeviceSynchronize() != cudaSuccess))
      printf("error at adj thread synchronization d: %s\n",
             cudaGetErrorString(cudaGetLastError()));
  }
  freeDeviceMem(densEstimation_gpu.data);
  freeDeviceMem(image_gpu.data);
  // cudaFreeHost(densArray.data);

  cudaDeviceSynchronize();
  // copy only the real part back to cpu
  DType *tmp_d = (DType *)densArray_gpu.data;

  gpuNUFFT::Array<DType> final_densArray;
  final_densArray.dim.length = n_samples;
  allocate_pinned_memory(&final_densArray, n_samples * sizeof(DType));
  HANDLE_ERROR(cudaMemcpy2DAsync(final_densArray.data, sizeof(DType), tmp_d,
                                 sizeof(DType2), sizeof(DType), n_samples,
                                 cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  freeDeviceMem(densArray_gpu.data);
  DType *ptr = reinterpret_cast<DType(&)[0]>(*final_densArray.data);
  auto capsule = py::capsule(ptr, [](void *ptr) { return; });
  return py::array_t<DType>({ trajectory_length }, { sizeof(DType) }, ptr,
                            capsule);
}

py::array_t<std::complex<DType>> GpuNUFFTPythonOperator::data_consistency(
    py::array_t<std::complex<DType>> input_image,
    py::array_t<std::complex<DType>> obs_data)
{
  gpuNUFFT::Dimensions myDims = imgDims;
  if (dimension == 2)
    myDims.depth = 1;

  copyNumpyArray(input_image, image.data);
  gpuNUFFT::Array<DType2> obsArray = readNumpyArray(obs_data);

  gpuNUFFT::GpuArray<DType2> obsArray_gpu;
  obsArray_gpu.dim = kspace_data.dim;
  gpuNUFFT::GpuArray<DType2> resArray_gpu;
  resArray_gpu.dim = kspace_data.dim;
  gpuNUFFT::GpuArray<DType2> imArray_gpu;
  imArray_gpu.dim = image.dim;
  allocateDeviceMem(&imArray_gpu.data, image.count());
  allocateDeviceMem(&resArray_gpu.data, resArray_gpu.count());
  allocateDeviceMem(&obsArray_gpu.data, obsArray_gpu.count());

  copyToDevice(image.data, imArray_gpu.data, image.count());
  allocateAndCopyToDeviceMem(&obsArray_gpu.data, obsArray.data,
                             obsArray.count());

  HANDLE_ERROR(cudaDeviceSynchronize());
  // F^H(Fx - y) on gpu.
  gpuNUFFTOp->performForwardGpuNUFFT(imArray_gpu, resArray_gpu);
  if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
    printf("### forward done\n");
  diffInPlace(resArray_gpu.data, obsArray_gpu.data, obsArray.count());
  if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
    printf("### diff done\n");
  gpuNUFFTOp->performGpuNUFFTAdj(resArray_gpu, imArray_gpu);
  if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
    printf("### adj done\n");
  copyFromDeviceAsync(imArray_gpu.data, image.data, image.count());
  if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
    printf("### from device done");
  HANDLE_ERROR(cudaDeviceSynchronize());

  freeTotalDeviceMemory(imArray_gpu.data, resArray_gpu.data, obsArray_gpu.data,
                        NULL);
  // return image as numpy array
  std::complex<DType> *ptr =
      reinterpret_cast<std::complex<DType>(&)[0]>(*image.data);
  auto capsule = py::capsule(ptr, [](void *ptr) { return; });
  if (has_sense_data == false)
    return py::array_t<std::complex<DType>>(
        { n_coils, (int)myDims.depth, (int)myDims.height, (int)myDims.width },
        {
            sizeof(DType2) * (int)myDims.depth * (int)myDims.height *
                (int)myDims.width,
            sizeof(DType2) * (int)myDims.height * (int)myDims.width,
            sizeof(DType2) * (int)myDims.width,
            sizeof(DType2),
        },
        ptr, capsule);
  else
    return py::array_t<std::complex<DType>>(
        { (int)myDims.depth, (int)myDims.height, (int)myDims.width },
        {
            sizeof(DType2) * (int)myDims.height * (int)myDims.width,
            sizeof(DType2) * (int)myDims.width,
            sizeof(DType2),
        },
        ptr, capsule);
}

float GpuNUFFTPythonOperator::get_spectral_radius(int max_iter = 20,
                                                  float tolerance = 1e-6)
{
  // TODO: add error checking for curand, cublas and cuda

  int im_size = image.count();

  gpuNUFFT::GpuArray<DType2> x_gpu;
  x_gpu.dim = image.dim;
  allocateDeviceMem(&x_gpu.data, im_size);

  gpuNUFFT::GpuArray<DType2> tmp_kspace_gpu;
  tmp_kspace_gpu.dim = kspace_data.dim;
  allocateDeviceMem(&tmp_kspace_gpu.data, kspace_data.count());

  cudaDeviceSynchronize();
  // __device__ DType *norm_new_d;
  // DType *norm_new_d;
  // cudaMalloc(&norm_new_d, sizeof(DType));
  DType norm_old = 1.0;
  DType norm_new = 1.0;
  DType inv_norm = 1.0;
  // initialisation: create a random complex image.
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(generator, (int)time(NULL));

  // complex value generator by giving twice the size.
  curandGenerateUniform(generator, (DType *)x_gpu.data, 2 * im_size);
  // xold = initialize random x of image size.
  curandDestroyGenerator(generator);
  // Create a handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasScnrm2(handle, im_size, x_gpu.data, 1, &norm_old);
  inv_norm = 1.0 / norm_old;
  cublasCsscal(handle, im_size, &inv_norm, x_gpu.data, 1);

  for (int i = 0; i < max_iter; i++)
  {
    // compute x_new = adj_op(op(x_old))
    gpuNUFFTOp->performForwardGpuNUFFT(x_gpu, tmp_kspace_gpu);
    if (DEBUG && cudaDeviceSynchronize() != cudaSuccess)
      printf("Error at thread sync 1");
    gpuNUFFTOp->performGpuNUFFTAdj(tmp_kspace_gpu, x_gpu);
    if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
      printf("Error at thread sync 2");
    // compute ||x_new||
    cublasScnrm2(handle, im_size, x_gpu.data, 1, &norm_new);
    if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
      printf("Error at thread sync 3");
    // x_new <- x_new/ ||x_new||
    inv_norm = 1.0 / norm_new;
    cublasCsscal(handle, im_size, &inv_norm, x_gpu.data, 1);

    if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
      printf("Error at thread sync 4");
    //   cudaMemcpy(&norm_new, norm_new_d,
    //   sizeof(DType),cudaMemcpyDeviceToHost);

    if (DEBUG && cudaDeviceSynchronize() == cudaSuccess)
      printf("Error at thread sync 5");
    //   std::cout << std::to_string(norm_new - norm_old) << std::endl;
    if (fabs(norm_new - norm_old) < tolerance)
    {
      //   std::cout << "stop after" << i << std::endl;
      break;
    }
    norm_old = norm_new;
  }
  freeTotalDeviceMemory(tmp_kspace_gpu.data, x_gpu.data, NULL);
  cublasDestroy(handle);
  return norm_new;
}
