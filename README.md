## gpuNUFFT - GPU Regridding of arbitrary 3-D/2-D MRI data
Original Software by 
- Andreas Schwarzl - andy.schwarzl[at]gmail.com
- Florian Knoll - florian.knoll[at]nyumc.org

Forked and tailored to python by 
- Chaythia GR 
- Carole Lazarus
- Pierre-Antoine Comby

-------------------------------------------------------------------------------
INFO:
-------------------------------------------------------------------------------
GPU 3D/2D regridding library. 

REQUIREMENTS:
-------------------------------------------------------------------------------

- CUDA capable graphics card and working installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CMake 2.8](https://cmake.org/download/) or higher
- [Google test](https://github.com/google/googletest) framework (optional)

CMAKE Options:

- WITH_DEBUG        : DEFAULT OFF, enables Command-Line DEBUG output
- WITH_MATLAB_DEBUG : DEFAULT OFF, enables MATLAB Console DEBUG output
- GEN_TESTS         : DEFAULT OFF, generate Unit tests
- FERMI_GPU         : DEFAULT OFF, set ON to support cards with compute capability 2.0

-------------------------------------------------------------------------------
LINUX, using gcc:
-------------------------------------------------------------------------------

build project via cmake, starting from project root directory:

    > cd CUDA
    > mkdir -p build
    > cd build
    > cmake ..
    > make
	
Note: This version of gpuNUFFT was tested with CUDA 5.0. NVIDIAs nvcc compiler did not support gcc 4.7 or higher at this time. It is therefore suggested to compile gpuNUFFT with gcc 4.6. One comfortable way to toggle between gcc/g++ versions is to use the update-alternatives tool.

*Screencast*: Alternatively follow the installation and usage instructions for Linux:

[![Linux Installation](http://imgur.com/z0bxIIr.png)](https://vimeo.com/161036817 "gpuNUFFT Installation Linux - Click to Watch!")

-------------------------------------------------------------------------------
WINDOWS, with Visual Studio:
-------------------------------------------------------------------------------
Setup a working version of [Visual Studio Community 2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx) or use the Visual Studio professional edition if available. Generate the project solution using the CMake-GUI or from command line by starting from the project root directory:

    > cd CUDA
    > mkdir build 
    > cd build
    > cmake .. -G "Visual Studio 12 2013 Win64"
    > Build created Solution gpuNUFFT.sln using Visual Studio

For the Win64 platform check that all necessary Visual Studio Add-ons are
installed correctly and a Win64 dummy project can be created using VS.

*Screencast*: Alternatively follow the installation and usage instructions for Windows:

[![Windows Installation](http://imgur.com/ZYzx8hL.png)](https://vimeo.com/161037263 "gpuNUFFT Installation Windows - Click to Watch!")

-------------------------------------------------------------------------------
Run:
-------------------------------------------------------------------------------
The compiled CUDA-mexfiles will appear in the bin directory and are also copied 
automatically into the gpuNUFFT/@gpuNUFFT/private directory . Include the gpuNUFFT
directory into the matlab search path, in order to run the provided demo example.


-------------------------------------------------------------------------------
Doc:
-------------------------------------------------------------------------------
To generate the source code documentation run 

    > make doc

in the build directory. 

*Note: Requires doxygen to be installed.*

### Supporting material

Written documentation and presentations can be found [here](https://www.dropbox.com/sh/gcvcszporj65wnq/AAA3eFsGQnSb7UottCSx0Hiva?dl=0).

