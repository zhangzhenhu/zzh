cmake_minimum_required(VERSION 3.12)
project(fastlib)

set(CMAKE_CXX_STANDARD 14)

set(PYTHON3_INC_DIR "/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m" CACHE PATH "")
set(PYTHON_EXECUTABLE "/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7")

INCLUDE_DIRECTORIES("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pybind11/include")
INCLUDE_DIRECTORIES("/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m")

add_library(fastlib SHARED lcs.hpp simpleArray.h bind.cpp)