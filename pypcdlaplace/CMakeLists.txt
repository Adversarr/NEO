cmake_minimum_required(VERSION 3.18)

project(pypcdlaplace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(LAPACK REQUIRED)
find_package(CGAL CONFIG REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/ext/cgal" NO_DEFAULT_PATH)

set(PYBIND11_FINDPYTHON ON CACHE STRING "" FORCE)
add_subdirectory(ext/pybind11 EXCLUDE_FROM_ALL)

add_library(pcdlaplace_core STATIC
  pcdlaplace/src/comp_llpmatrix.cpp
  pcdlaplace/src/lpmatrix.cpp
  pcdlaplace/src/point_cloud.cpp
)
set_target_properties(pcdlaplace_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pcdlaplace_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/pcdlaplace/src"
)

target_link_libraries(pcdlaplace_core PUBLIC CGAL::CGAL)
if(TARGET LAPACK::LAPACK)
  target_link_libraries(pcdlaplace_core PUBLIC LAPACK::LAPACK)
else()
  target_link_libraries(pcdlaplace_core PUBLIC ${LAPACK_LIBRARIES})
endif()

pybind11_add_module(_core MODULE src/pybind_module.cpp)
target_link_libraries(_core PRIVATE pcdlaplace_core)

install(TARGETS _core LIBRARY DESTINATION pypcdlaplace)
install(FILES
  pypcdlaplace/__init__.py
  pypcdlaplace/_api.py
  pypcdlaplace/demo.py
  DESTINATION pypcdlaplace
)
