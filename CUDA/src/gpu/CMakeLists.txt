SET (GPU_GPUNUFFT_SOURCES ${GPUNUFFT_SOURCES}
		 ${CMAKE_CURRENT_SOURCE_DIR}/precomp_kernels.cu)


include_directories(${GPUNUFFT_INC_DIR})

cuda_include_directories(${GPUNUFFT_INC_DIR})

set(GPU_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/atomic_gpuNUFFT.cu)

cuda_add_library(${GRID_LIB_ATM_NAME}
                 ${GPU_CU_SOURCES}
                 ${GPU_GPUNUFFT_SOURCES}
                 ${GPUNUFFT_SOURCES}
                 ${GPUNUFFT_INCLUDE} SHARED)

cuda_add_cufft_to_target(${GRID_LIB_ATM_NAME})
cuda_add_cublas_to_target(${GRID_LIB_ATM_NAME})
