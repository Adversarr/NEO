# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.


project( pcdlaplace_demo )

cmake_minimum_required(VERSION 2.6.2)
if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER 2.6)
  if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" VERSION_GREATER 2.8.3)
    cmake_policy(VERSION 2.8.4)
  else()
    cmake_policy(VERSION 2.6)
  endif()
endif()

find_package(CGAL QUIET COMPONENTS Core)

if ( CGAL_FOUND )

  include( ${CGAL_USE_FILE} )
  
    include( Qt3Macros-patched )
    qt3_automoc(  comp_llpmatrix.cpp graphlpmatrix.cpp lpmatrix.cpp pcdlp.cpp pcdlpmatrix (copy).C pcdlpmatrix.cpp point_cloud.cpp test_pcd.cpp )

    # Make sure the compiler can find generated .moc files
    include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

    link_directories(/usr/lib)
  
    #include_directories( ${QT3_INCLUDE_DIR} )

    add_executable  (demo test_pcd.cpp point_cloud.cpp lpmatrix.cpp comp_llpmatrix.cpp)
  
    #add_to_cached_list( CGAL_EXECUTABLE_TARGETS demo )
  
  
    # Link the executable to CGAL and third-party libraries
    target_link_libraries(demo ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES} lapack)
  
else()
  
    message(STATUS "NOTICE: This demo requires the CGAL library, and will not be compiled.")
  
endif()

