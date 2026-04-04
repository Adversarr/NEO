cmake_minimum_required(VERSION 3.18)

project(pyspec LANGUAGES CXX)

include(CheckIPOSupported)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PYSPEC_ENABLE_IPO "Enable interprocedural optimization when supported" ON)
option(PYSPEC_ENABLE_NATIVE_ARCH "Enable CPU-specific tuning flags" ON)
option(PYSPEC_ENABLE_FAST_MATH "Enable aggressive floating-point optimizations" OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

add_subdirectory(ext/pybind11)

pybind11_add_module(_pyspec src/pyspec.cpp)
set_target_properties(_pyspec PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyspec")
target_include_directories(_pyspec PRIVATE ext/spectra/include)
target_link_libraries(_pyspec PRIVATE Eigen3::Eigen)
target_compile_definitions(_pyspec PRIVATE EIGEN_NO_DEBUG)

if(PYSPEC_ENABLE_IPO)
    check_ipo_supported(RESULT pyspec_ipo_supported OUTPUT pyspec_ipo_error)
    if(pyspec_ipo_supported)
        set_property(TARGET _pyspec PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO/LTO disabled: ${pyspec_ipo_error}")
    endif()
endif()

target_compile_options(_pyspec PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O3>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(_pyspec PRIVATE
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:RelWithDebInfo>:-funroll-loops>
    )

    if(PYSPEC_ENABLE_NATIVE_ARCH)
        target_compile_options(_pyspec PRIVATE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:RelWithDebInfo>:-march=native>
        )
    endif()

    if(PYSPEC_ENABLE_FAST_MATH)
        target_compile_options(_pyspec PRIVATE
            $<$<CONFIG:Release>:-ffast-math>
            $<$<CONFIG:RelWithDebInfo>:-ffast-math>
        )
    endif()
elseif(MSVC)
    target_compile_options(_pyspec PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2>
    )

    if(PYSPEC_ENABLE_FAST_MATH)
        target_compile_options(_pyspec PRIVATE
            $<$<CONFIG:Release>:/fp:fast>
            $<$<CONFIG:RelWithDebInfo>:/fp:fast>
        )
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(_pyspec PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _pyspec LIBRARY DESTINATION pyspec)
install(FILES pyspec/__init__.py DESTINATION pyspec)
