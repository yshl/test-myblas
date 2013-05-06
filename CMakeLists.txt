CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

SET(CMAKE_C_FLAGS_RELEASE "-O2 -march=native -funroll-loops")
SET(CMAKE_C_FLAGS_DEBUG "-g -march=native")
SET(CMAKE_BUILD_TYPE RELEASE)
add_definitions("-Wall")

PROJECT(myblas C)
SET(serial "0.0.1")
SET(soserial "0")

INCLUDE_DIRECTORIES("include")

SUBDIRS(src test time)

include("FindCBLAS.cmake")
enable_testing()
include(CTest)
