include_directories(inc)

SET(GPUNUFFT_SRC_DIR ${CMAKE_SOURCE_DIR}/CUDA/src)
SET(GPUNUFFT_SOURCES ${GPUNUFFT_SRC_DIR}/gpuNUFFT_utils.cpp
${GPUNUFFT_SRC_DIR}/gpuNUFFT_operator_factory.cpp
${GPUNUFFT_SRC_DIR}/gpuNUFFT_operator.cpp
${GPUNUFFT_SRC_DIR}/texture_gpuNUFFT_operator.cpp
${GPUNUFFT_SRC_DIR}/balanced_gpuNUFFT_operator.cpp
${GPUNUFFT_SRC_DIR}/balanced_texture_gpuNUFFT_operator.cpp)

SET (GPU_GPUNUFFT_SOURCES ${GPUNUFFT_SOURCES}
	${GPUNUFFT_SRC_DIR}/gpu/precomp_kernels.cu)
set(GPU_CU_SOURCES ${GPUNUFFT_SRC_DIR}/gpu/atomic_gpuNUFFT.cu)

if(GEN_TESTS)
	enable_testing()
	add_subdirectory(test)
	add_test(
		NAME runUnitTests
		COMMAND runUnitTests
	)

	add_test(
		NAME runGPUATMUnitTests
		COMMAND runGPUATMUnitTests)
endif(GEN_TESTS)

