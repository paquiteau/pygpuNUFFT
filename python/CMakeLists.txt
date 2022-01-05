

MESSAGE("------setting up Python bindings-------")
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.8.1
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

include_directories(
    inc
    ${GPUNUFFT_INC_DIR}
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    )

message("pybind : ${pybind11_INCLUDE_DIR}")
message("python inc : ${PYTHON_INCLUDE_DIRS}")
message("python lib : ${PYTHON_LIBRARIES}")

set(PYTHON_BINDINGS src/python_factory.cpp
                    src/python_bindings.cpp)
cuda_include_directories(${GPUNUFFT_INC_DIR})
cuda_add_library(gpuNUFFT ${GPU_CU_SOURCES}
                          ${GPU_GPUNUFFT_SOURCES}
                          ${GPUNUFFT_SOURCES}
                          ${GPUNUFFT_INCLUDE}
                          ${PYTHON_BINDINGS}
                          SHARED)

set_target_properties(gpuNUFFT PROPERTIES PREFIX "")
target_link_libraries(gpuNUFFT ${CUDA_LIBRARIES} ${GRID_LIB_ATM_NAME})
CUDA_ADD_CUFFT_TO_TARGET(gpuNUFFT)
CUDA_ADD_CUBLAS_TO_TARGET(gpuNUFFT)
