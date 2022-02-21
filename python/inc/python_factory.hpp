#ifndef PYTHON_FACTORY_H_INCLUDED
#define PYTHON_FACTORY_H_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"

namespace py = pybind11;


/** \brief Python Wrapper for gpuNUFFT Operator
  *
  * This wrapper exposes forward and adjoint operation of gpuNUFFT.
  * It also implements standart methods with gpu-only function for better performances.
  */
class GpuNUFFTPythonOperator
{

    gpuNUFFT::GpuNUFFTOperatorFactory factory;
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp;
    int trajectory_length, n_coils, dimension;
    bool has_sense_data;
    gpuNUFFT::Dimensions imgDims;
    // sensitivity maps
    gpuNUFFT::Array<DType2> sensArray, kspace_data, image;

    public:

    /** \brief Constructor
      *
      * @param kspace_loc       The kspace location point
      * @param image_size       Size of image space
      * @param num_coils        Number of coil
      * @param sense_maps       Array of sensitivity maps
      * @param density_comp     Array for density_compensation
      * @param kernel_width     Width of the interpolation kernel
      * @param sector_width     Width of the sector
      * @param osr              Oversampling ratio of the fourier grid
      * @param balance_workload Flag to indicate load balancing
      */
    GpuNUFFTPythonOperator(py::array_t<DType> kspace_loc,
                           py::array_t<int> image_size,
                           int num_coils,
                           py::array_t<std::complex<DType>> sense_maps,
                           py::array_t<float> density_comp,
                           int kernel_width=3,
                           int sector_width=8,
                           int osr=2,
                           bool balance_workload=1);
    /** \brief Forward operator (Image to kspace)
      *
      * @param input_image      Input Image
      * @param interpolate_data Flag use for cpu density compensation
      */
    py::array_t<std::complex<DType>> op(py::array_t<std::complex<DType>> input_image,
                                        bool interpolate_data=false);

    /** \brief Adjoint operator (kspace to Image)
      *
      * @param input_kspace_data    Input Kspace data
      * @param grid_data     Flag use for cpu density compensation
      */
    py::array_t<std::complex<DType>> adj_op(py::array_t<std::complex<DType>> input_kspace_data,
                                            bool grid_data=false);

    /** \brief Estimate density Compensation array using gpu-only functions
      *
      * @param num_iter  number of iterations
      */
    py::array_t<DType> estimate_density_comp(int num_iter);

    /** \brief Estimate the spectral radius using gpu-only functions
      *
      * The spectral radius is estimated using the power method.
      *
      * @param num_iter  number of iterations
      */

    float get_spectral_radius(int max_iter, float tolerance);

    py::array_t<std::complex<DType>> data_consistency(py::array_t<std::complex<DType>> input_image,
                                                     py::array_t<std::complex<DType>> obs_data);
    void clean_memory()
    {
       gpuNUFFTOp->clean_memory();
    }

    void set_smaps(py::array_t<std::complex<DType>> sense_maps)
    {
        py::buffer_info myData = sense_maps.request();
        std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
        DType2 *my_data = reinterpret_cast<DType2(&)[0]>(*t_data);
        memcpy(sensArray.data, my_data, myData.size*sizeof(DType2));
        has_sense_data = true;
        gpuNUFFTOp->setSens(sensArray);
    }

    ~GpuNUFFTPythonOperator()
    {
        cudaFreeHost(kspace_data.data);
        cudaFreeHost(image.data);
        cudaDeviceReset();
        delete gpuNUFFTOp;
    }
};


#endif // PYTHON_FACTORY_H_INCLUDED
