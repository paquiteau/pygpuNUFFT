#ifndef BATCH_FACTORY_H_
#define BATCH_FACTORY_H_

#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "config.hpp"
#include "gpuNUFFT_operator.hpp"
#include "gpuNUFFT_operator_factory.hpp"

namespace py = pybind11;

/** \brief Python Wrapper for Batch GpuNUFFT Operations
  *
  * This class provides methods for parallele computation of gpuNUFFT operation,
  * for example for fMRI reconstruction.
  */
class BatchGpuNUFFTPythonOperator
{
    gpuNUFFT::GpuNUFFTOperatorFactory factory;
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp;
    int n_frames, n_coils, dimensions;
    bool has_sense_data;
    gpuNUFFT::Dimensions imgDims;
    gpuNUFFT::Array<DType2> sensArray, kspace_data, image_batch;
    std::vector<gpuNUFFT::GpuNUFFTOperator *> operators;
    std::vector<std::thread> workers;

   public:
    /** \brief Constructor
     *
     * @param kspace_locs   The Kspace locations for each frame
     * @param image_size    Size of image space
     * @param num_coils     Number of coils
     * @param sense_maps    Sensitivity Maps
     * @param kernel_width     Width of the interpolation kernel
     * @param sector_width     Width of the sector
     * @param osr              Oversampling ratio of the fourier grid
     * @param balance_workload Flag to indicate load balancing
     */
    BatchGpuNUFFTPythonOperator(
        std::vector<py::array_t<DType>> kspace_loc_list,
        py::array_t<int> image_size, int num_coils,
        py::array_t<std::complex<DType>>sense_maps,
        int kernel_width, int sector_width, int osr, bool balance_workload);

    /** \brief Forward operator (Image to kspace)
     *
     * @param input_image      Input Image
     * @param interpolate_data Flag use for cpu density compensation
     */
    py::array_t<std::complex<DType>>
    op(py::array_t<std::complex<DType>> input_image,
       bool interpolate_data = false);

    /** \brief Adjoint operator (kspace to Image)
     *
     * @param input_kspace_data    Input Kspace data
     * @param grid_data     Flag use for cpu density compensation
     */
    py::array_t<std::complex<DType>>
    adj_op(py::array_t<std::complex<DType>> input_kspace_data,
           bool grid_data = false);


    void clean_memory()
    {
       gpuNUFFTOp->clean_memory();
    }

    ~BatchGpuNUFFTPythonOperator()
    {
        cudaFreeHost(kspace_data.data);
        cudaFreeHost(image.data);
        delete gpuNUFFTOp;
    }

};

#endif // BATCH_FACTORY_H_
