cmake_minimum_required(VERSION 3.18)
project(gdmgs_mesh_native LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)
find_path(CGAL_INCLUDE_DIR CGAL/Lazy_exact_nt.h
  HINTS "$ENV{GDMGS_CGAL_ROOT}/usr/include" "$ENV{GDMGS_CGAL_ROOT}/include")
find_path(GMP_INCLUDE_DIR gmp.h
  HINTS "$ENV{GDMGS_CGAL_ROOT}/usr/include/x86_64-linux-gnu" "$ENV{GDMGS_CGAL_ROOT}/include")
find_path(MPFR_INCLUDE_DIR mpfr.h
  HINTS "$ENV{GDMGS_CGAL_ROOT}/usr/include" "$ENV{GDMGS_CGAL_ROOT}/include")
find_library(GMP_LIBRARY NAMES gmp libgmp.so.10
  HINTS "$ENV{GDMGS_CGAL_ROOT}/usr/lib/x86_64-linux-gnu")
find_library(MPFR_LIBRARY NAMES mpfr libmpfr.so.6
  HINTS "$ENV{GDMGS_CGAL_ROOT}/usr/lib/x86_64-linux-gnu")
if(NOT CGAL_INCLUDE_DIR OR NOT GMP_INCLUDE_DIR OR NOT MPFR_INCLUDE_DIR OR NOT GMP_LIBRARY OR NOT MPFR_LIBRARY)
  message(FATAL_ERROR "CGAL, GMP and MPFR headers/libraries required; set GDMGS_CGAL_ROOT to a scoped dependency extraction")
endif()
pybind11_add_module(GDMGS_mesh_native mesh_native.cpp)
target_include_directories(GDMGS_mesh_native PRIVATE "${CGAL_INCLUDE_DIR}" "${GMP_INCLUDE_DIR}" "${MPFR_INCLUDE_DIR}")
target_link_libraries(GDMGS_mesh_native PRIVATE "${GMP_LIBRARY}" "${MPFR_LIBRARY}")
target_compile_features(GDMGS_mesh_native PRIVATE cxx_std_17)
target_compile_options(GDMGS_mesh_native PRIVATE -O3 -Wall -Wextra -frounding-math)
