cmake_minimum_required(VERSION 3.18)
project(proxygs_anchor_point_native LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
if(NOT DEFINED PYBIND11_CMAKE_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ProxyGS_anchor_point_native anchor_point_native.cpp)
target_compile_features(ProxyGS_anchor_point_native PRIVATE cxx_std_17)
target_compile_options(
  ProxyGS_anchor_point_native PRIVATE -O3 -Wall -Wextra -ffp-contract=off
)
