

configure_file(${CMAKE_SOURCE_DIR}/CUDA/inc/cufft_config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/cufft_config.hpp)
configure_file(${CMAKE_SOURCE_DIR}/CUDA/inc/config.hpp.cmake ${CMAKE_SOURCE_DIR}/inc/config.hpp)

include_directories(inc)

add_subdirectory(src)

if(GEN_TESTS)
	# For make-based builds, defines make target named test.
	# For Visual Studio builds, defines Visual Studio project named RUN_TESTS.
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

