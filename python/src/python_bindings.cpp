#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "python_utils.hpp"
#include "python_factory.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gpuNUFFT, m) {
    py::class_<GpuNUFFTPythonOperator>(m, "NUFFTOp")
        .def(py::init<py::array_t<DType>, py::array_t<int>, int, py::array_t<std::complex<DType>>, py::array_t<float>, int, int, int, bool>())
        .def("op", &GpuNUFFTPythonOperator::op)
        .def("adj_op",  &GpuNUFFTPythonOperator::adj_op)
        .def("clean_memory", &GpuNUFFTPythonOperator::clean_memory)
        .def("set_smaps", &GpuNUFFTPythonOperator::set_smaps);
}
