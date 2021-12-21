

include_directories(inc)

add_subdirectory(src)

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

