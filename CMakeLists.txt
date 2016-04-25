cmake_minimum_required(VERSION 3.5)
project(phloxar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    PhloxAR/include/array.hpp
    PhloxAR/include/dispatch.hpp
    PhloxAR/include/numpy.hpp
    PhloxAR/include/utils.hpp)

add_executable(phloxar ${SOURCE_FILES})