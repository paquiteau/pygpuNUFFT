#The Name of the project
project(gpuNUFFT)

cmake_minimum_required(VERSION 3.15)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

#for older CMAKE versions these vars have to be set
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(ARCHIVE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")

# find dependencies
find_package(CUDA REQUIRED)


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

option(PROFILE "Add profile hook for nvcc" off)
if(PROFILE AND (NOT DEBUG))
  list(APPEND MY_NVCC_FLAGS --generate-line-info)
endif()


cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")


IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("debug mode")
  list(APPEND CUDA_NVCC_FLAGS  ${ARCH_FLAGS} ${MY_NVCC_FLAGS} --ptxas-options=-v -G)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g  -std=c++11")
ELSE(CMAKE_BUILD_TYPE)
  list(APPEND CUDA_NVCC_FLAGS ${MY_NVCC_FLAGS} ${ARCH_FLAGS} --generate-line-info)
ENDIF()

message(STATUS "setting NVCC FLAGS to: ${CUDA_NVCC_FLAGS}")

set(GRID_LIB_ATM_NAME "gpuNUFFT_ATM${PREC_SUFFIX}")

SET(WARNING "/* WARNING: Automatically generated file. Please do not modify this file. */")
configure_file(${CMAKE_SOURCE_DIR}/CUDA/inc/cufft_config.hpp.cmake ${CMAKE_SOURCE_DIR}/CUDA/inc/cufft_config.hpp)
configure_file(${CMAKE_SOURCE_DIR}/CUDA/inc/config.hpp.cmake ${CMAKE_SOURCE_DIR}/CUDA/inc/config.hpp)

message("configure_file called")
#Include dirs
set(GPUNUFFT_INC_DIR ${CMAKE_SOURCE_DIR}/CUDA/inc)
set(GPUNUFFT_INCLUDE ${GPUNUFFT_INC_DIR}/cuda_utils.hpp
                     ${GPUNUFFT_INC_DIR}/cuda_utils.cuh
                     ${GPUNUFFT_INC_DIR}/config.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_utils.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_types.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_kernels.hpp
                     ${GPUNUFFT_INC_DIR}/precomp_kernels.hpp
                     ${GPUNUFFT_INC_DIR}/precomp_utils.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_operator.hpp
                     ${GPUNUFFT_INC_DIR}/balanced_operator.hpp
                     ${GPUNUFFT_INC_DIR}/texture_gpuNUFFT_operator.hpp
                     ${GPUNUFFT_INC_DIR}/balanced_gpuNUFFT_operator.hpp
                     ${GPUNUFFT_INC_DIR}/gpuNUFFT_operator_factory.hpp
                     ${GPUNUFFT_INC_DIR}/balanced_texture_gpuNUFFT_operator.hpp)

SET(CONFIG_INCLUDE ${GPUNUFFT_INC_DIR}/config.hpp
                   ${GPUNUFFT_INC_DIR}/cufft_config.hpp)

include(CUDA/CMakeLists.txt)
add_subdirectory(python)
add_subdirectory(doc)
