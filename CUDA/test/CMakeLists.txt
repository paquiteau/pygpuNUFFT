#setup testframework
MESSAGE("------setting up testframework-------")

include(FetchContent)
FetchContent_Declare(
  gtest
  # Specify the commit you depend on and update it regularly.
  GIT_REPOSITORY https://github.com/google/googletest/
  GIT_TAG release-1.11.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gtest)


# set(GTEST_DIR "${CMAKE_SOURCE_DIR}/../../gtest/googletest")
# MESSAGE("GTEST_DIR " ${GTEST_DIR})

include_directories(${gtest_INCLUDE_DIRS})
MESSAGE("gtest=${gtest_INCLUDE_DIRS}")
# find_library(GTEST_LIB NAMES gtest libgtest HINTS "${GTEST_DIR}/build" "${GTEST_DIR}/build/Debug")
# find_library(GTESTMAIN_LIB NAMES gtest_main libgtest_main HINTS "${GTEST_DIR}/build" "${GTEST_DIR}/build/Debug")

MESSAGE("GTEST_LIBRARIES= ${GTEST_LIBRARIES}")
MESSAGE("GTEST_MAIN_LIBRARIES = ${gtest_MAIN_LIBRARIES}")

#gpu test sources
set(GPU_SOURCES gpu/gpuNUFFT_conv_tests.cpp
				gpu/gpuNUFFT_2D_conv_tests.cpp
				gpu/gpuNUFFT_fft_tests.cpp
				gpu/gpuNUFFT_deapo_tests.cpp
				gpu/forward/gpuNUFFT_forward_tests.cpp)


cuda_add_executable(runGPUATMUnitTests ${GPU_SOURCES}
									   ${GPUNUFFT_SOURCES}
									   ../inc/gpuNUFFT_utils.hpp
									   ../inc/gpuNUFFT_operator_factory.hpp
									   ../inc/gpuNUFFT_operator.hpp
									   ../inc/gpuNUFFT_kernels.hpp)
if(WIN32)
	target_link_libraries(runGPUATMUnitTests ${GRID_LIB_ATM_NAME} gtest_main)
else(WIN32)
	target_link_libraries(runGPUATMUnitTests pthread ${GRID_LIB_ATM_NAME} gtest_main)
endif(WIN32)
