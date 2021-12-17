#The Name of the project
project(gpuNUFFT)

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")

# find dependencies
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)


#Options
#General DEBUG output
set(DEBUG false)
option(WITH_DEBUG "Enable DEBUG messages" OFF)
if (WITH_DEBUG)
 set (DEBUG true)
endif()
message(STATUS "Setting DEBUG Option to ${DEBUG}")

#Enable Google Tests
set(GEN_TESTS OFF CACHE BOOL "Enable and generate simple GOOGLE test framework unit tests")

#Enable/Disable GPU double precision
set(GPU_DOUBLE_PREC OFF CACHE BOOL "Enable double precision floating point operations on GPU (Compute Capability 1.3 needed)")


if(GPU_DOUBLE_PREC)
  SET(PREC_SUFFIX "_d")
else(GPU_DOUBLE_PREC)
  SET(PREC_SUFFIX "_f")
endif(GPU_DOUBLE_PREC)

set(MY_NCC_FLAGS -gencode=arch=compute_52,code=sm_52)

# if(${CUDA_VERSION_MAJOR} GREATER 7)
#   message("add CUDA 8 flags: Current version: ${CUDA_VERSION}")
#   list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60)
#   list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61)
#   list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_61,code=compute_61)
# endif()

# if(${CUDA_VERSION_MAJOR} GREATER 8)
#   message("add CUDA 9 flags: Current version: ${CUDA_VERSION}")
#   list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_70,code=sm_70)
# endif()

# if(${CUDA_VERSION_MAJOR} GREATER 9)
#   message("add CUDA 10 flags: Current version: ${CUDA_VERSION}")
#   list(APPEND MY_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
# endif()


IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("debug mode")
  list(APPEND CUDA_NVCC_FLAGS ${MY_NVCC_FLAGS} --ptxas-options=-v)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -std=c++11")
ELSE(CMAKE_BUILD_TYPE)
  list(APPEND CUDA_NVCC_FLAGS ${MY_NVCC_FLAGS})
ENDIF()

message(STATUS "setting NVCC FLAGS to: ${CUDA_NVCC_FLAGS}")

#
set(GRID_LIB_ATM_NAME "gpuNUFFT_ATM${PREC_SUFFIX}")

message(STATUS "creating lib with name: ${GRID_LIB_ATM_NAME}")
set(WARNING "/* WARNING: Automatically generated file. Please do not modify this file. */")
configure_file( ${CMAKE_SOURCE_DIR}/CUDA/inc/config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/config.hpp)
configure_file( ${CMAKE_SOURCE_DIR}/CUDA/inc/cufft_config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/cufft_config.hpp)

include_directories(${CMAKE_SOURCE_DIR}/CUDA/inc
                    ${CMAKE_SOURCE_DIR}/python/)

include_directories(${pybind11_INCLUDE_DIRS})


file(GLOB SOURCES_FILES "${CMAKE_SOURCE_DIR}/CUDA/src/*.cpp")
file(GLOB GPU_SOURCES_FILES "${CMAKE_SOURCE_DIR}/CUDA/src/gpu/*.cu")
file(GLOB HEADERS_FILES "${CMAKE_SOURCE_DIR}/CUDA/inc/*.hpp")

cuda_include_directories(${CMAKE_SOURCE_DIR}/CUDA/inc)
cuda_add_library(gpuNUFFT ${GPU_SOURCES_FILES} ${SOURCES_FILES} ${HEADERS_FILES} SHARED)
cuda_add_cufft_to_target(gpuNUFFT)
cuda_add_cublas_to_target(gpuNUFFT)

target_link_libraries(gpuNUFFT ${Python3_LIBRARIES} ${CUDA_LIBRARIES})

set_target_properties(gpuNUFFT PROPERTIES PREFIX "")
