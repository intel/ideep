cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(iDeep CXX C)

set(IDEEP_VERSION_MAJOR 0)
set(IDEEP_VERSION_MINOR 0)
set(IDEEP_VERSION_PATCH 1)
set(IDEEP_VERSION
  "${IDEEP_VERSION_MAJOR}.${IDEEP_VERSION_MINOR}.${IDEEP_VERSION_PATCH}")

set(USE_MKLDNN_INTERNAL 1)
include(cmake/profiling.cmake)
include(cmake/mkldnn.cmake)
include(cmake/mkl.cmake)
include(cmake/settings.cmake)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(DIRECTORY include/ DESTINATION include)

add_subdirectory(tests)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -fvisibility=hidden -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-result -Wno-missing-field-initializers")

add_library(ideep SHARED
            total_reduce/compute_request.c
            total_reduce/pal.c
            total_reduce/ring.c
            total_reduce/pending_message.c
            total_reduce/payload.c
            total_reduce/time.c
            total_reduce/TR_interface.c
            total_reduce/total_reduce.c
            )

# total reduce tests
include(cmake/total_reduce.cmake)
