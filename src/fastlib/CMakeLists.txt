cmake_minimum_required(VERSION 3.12)
project(fastlib)

set(CMAKE_CXX_STANDARD 14)
#add_subdirectory(pybind11)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


add_subdirectory(pybind11)
pybind11_add_module(fastlib bind.cpp ZTrie.cpp)

add_executable(main main.cpp)