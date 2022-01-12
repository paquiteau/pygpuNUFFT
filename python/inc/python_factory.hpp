#ifndef PYTHON_FACTORY_H_INCLUDED
#define PYTHON_FACTORY_H_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"

namespace py = pybind11;



/** \brief Create the bindings for Python module
 *
 *
 * */

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
    GpuNUFFTPythonOperator(py::array_t<DType> kspace_loc,
                           py::array_t<int> image_size,
                           int num_coils,
                           py::array_t<std::complex<DType>> sense_maps,
                           py::array_t<float> density_comp,
                           int kernel_width=3,
                           int sector_width=8,
                           int osr=2,
                           bool balance_workload=1);

    py::array_t<std::complex<DType>> op(py::array_t<std::complex<DType>> input_image,
                                        bool interpolate_data=false);

    py::array_t<std::complex<DType>> adj_op(py::array_t<std::complex<DType>> input_kspace_data,
                                            bool grid_data=false);

    py::array_t<DType> estimate_density_comp(int num_iter);

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
        delete gpuNUFFTOp;
    }
};


#endif // PYTHON_FACTORY_H_INCLUDED
