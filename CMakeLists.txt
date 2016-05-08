cmake_minimum_required(VERSION 3.5)
project(phloxar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        PhloxAR/cxx/include/array.hpp
        PhloxAR/cxx/include/dispatch.hpp
        PhloxAR/cxx/include/numpy.hpp
        PhloxAR/cxx/include/utils.hpp)

add_executable(phloxar ${SOURCE_FILES})