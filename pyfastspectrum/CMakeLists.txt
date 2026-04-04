cmake_minimum_required(VERSION 3.18)
project(pyfastspectrum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FASTSPECTRUM_ENABLE_CUDA "Build FastSpectrum with CUDA backend support" ON)

find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(ext/pybind11)

set(FASTSPECTRUM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/FastSpectrum/Fast Spectrum")
set(FASTSPECTRUM_LIBIGL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/FastSpectrum/libigl")
set(FASTSPECTRUM_EIGEN_DIR "${FASTSPECTRUM_LIBIGL_DIR}/external/eigen")

add_library(fastspectrum_core STATIC
	"${FASTSPECTRUM_SRC_DIR}/FastSpectrum.cpp"
	"${FASTSPECTRUM_SRC_DIR}/Basis.cpp"
	"${FASTSPECTRUM_SRC_DIR}/EigenSolver.cpp"
	"${FASTSPECTRUM_SRC_DIR}/Sampling.cpp"
	"${FASTSPECTRUM_SRC_DIR}/Utility.cpp")

target_include_directories(fastspectrum_core
	PUBLIC
		"${FASTSPECTRUM_SRC_DIR}"
		"${FASTSPECTRUM_LIBIGL_DIR}/include"
		"${FASTSPECTRUM_EIGEN_DIR}")

target_link_libraries(fastspectrum_core PUBLIC OpenMP::OpenMP_CXX)

if(MSVC)
	target_compile_definitions(fastspectrum_core PUBLIC _USE_MATH_DEFINES)
endif()

if(FASTSPECTRUM_ENABLE_CUDA)
	find_package(CUDAToolkit QUIET)
	if(CUDAToolkit_FOUND)
		target_compile_definitions(fastspectrum_core PUBLIC FASTSPECTRUM_HAS_CUDA)
		target_link_libraries(fastspectrum_core
			PUBLIC
				CUDA::cudart
				CUDA::cublas
				CUDA::cusolver)
	endif()
endif()

pybind11_add_module(pyfastspectrum MODULE src/bindings.cpp)
target_link_libraries(pyfastspectrum PRIVATE fastspectrum_core)

install(TARGETS pyfastspectrum LIBRARY DESTINATION .)
