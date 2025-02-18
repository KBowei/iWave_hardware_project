cmake_minimum_required(VERSION 3.10)
project(arithmetic_coding)

set(CMAKE_CXX_STANDARD 17)
set(MAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# pybind11
add_subdirectory(extern/pybind11)
include_directories(extern/pybind11/include)

# Source files
set(SOURCES
    arithmetic_coding.cpp
    bind.cpp
)

pybind11_add_module(arithmetic_coding MODULE ${SOURCES})

# Ensure the shared library has a correct name
set_target_properties(arithmetic_coding PROPERTIES PREFIX "" SUFFIX ".so")
