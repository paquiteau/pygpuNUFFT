# gpuNUFFT - GPU Regridding of arbitrary 3-D/2-D MRI data

![CD](https://github.com/paquiteau/pygpuNUFFT/actions/workflows/cd-build.yml/badge.svg)
![CI](https://github.com/paquiteau/pygpuNUFFT/actions/workflows/ci-build.yml/badge.svg)


This is an hard fork of https://github.com/andyschwarzl/gpuNUFFT , which focuses on providing extensive python bindings and more functionality for non cartesian MRI reconstruction.


Original Software by 
- Andreas Schwarzl - andy.schwarzl[at]gmail.com
- Florian Knoll - florian.knoll[at]nyumc.org

Forked and tailored to python by 
- Chaythia GR  [@chaithyagr](https://github.com/chaithyagr/)
- Pierre-Antoine Comby [@paquiteau](https://github.com/paquiteau/)

-------------------------------------------------------------------------------
INFO:
-------------------------------------------------------------------------------
GPU 3D/2D regridding library. 

REQUIREMENTS:
-------------------------------------------------------------------------------

- CUDA capable graphics card and working installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [pybind11](https://github.com/pybind/pybind11) (downloaded by cmake build)
- [CMake 3.15](https://cmake.org/download/) or higher
- [Google test](https://github.com/google/googletest) f(downloaded by cmake build if needed)

CMAKE Options:

- WITH_DEBUG        : DEFAULT OFF, enables Command-Line DEBUG output
- GEN_TESTS         : DEFAULT OFF, generate Unit tests

-------------------------------------------------------------------------------
LINUX, using gcc:
-------------------------------------------------------------------------------

build project via cmake, starting from project root directory:

    > cd CUDA
    > mkdir -p build
    > cd build
    > cmake ..
    > make
	
Note: This version of gpuNUFFT was tested with CUDA 11.0

-------------------------------------------------------------------------------
Doc:
-------------------------------------------------------------------------------
To generate the source code documentation run 

    > make doc

in the build directory. 

Otherwise, you can go to https://paquiteau.github.io/pygpuNUFFT/docs/ to access the up-to-date documentation of master branch.


*Note: Requires doxygen to be installed.*

### Supporting material

The original written documentation and presentations can be found [here](https://www.dropbox.com/sh/gcvcszporj65wnq/AAA3eFsGQnSb7UottCSx0Hiva?dl=0).

### Acknowlegdment

The authors would like to deeply thank the original authors of gpuNUFFT for their hard work and initial development.
