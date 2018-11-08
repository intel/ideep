cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(iDeep CXX C)

set(IDEEP_VERSION_MAJOR 0)
set(IDEEP_VERSION_MINOR 0)
set(IDEEP_VERSION_PATCH 1)
set(IDEEP_VERSION
  "${IDEEP_VERSION_MAJOR}.${IDEEP_VERSION_MINOR}.${IDEEP_VERSION_PATCH}")

option(multinode "Provide non-blocking communciation support for machine learning" OFF)

IF(APPLE)
  SET(CMAKE_INSTALL_NAME_DIR @rpath)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
ENDIF(APPLE)

set(USE_MKLDNN_INTERNAL 1)
include(cmake/profiling.cmake)
include(cmake/mkldnn.cmake)
include(cmake/mkl.cmake)
include(cmake/settings.cmake)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(DIRECTORY include/ DESTINATION include)

add_subdirectory(tests)

if (multinode)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx -Wall -Wextra -Werror")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
endif ()

if (multinode)
    file(GLOB TOTAL_REDUCE_FILES total_reduce/*.c)
else ()
    file(GLOB TOTAL_REDUCE_FILES total_reduce/dummy/TR_dummy_interface.c)
endif ()


add_library(ideep SHARED ${TOTAL_REDUCE_FILES})

if (multinode)

    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(ideep ${MPI_LIBRARIES})

    # total reduce tests
    include(cmake/total_reduce.cmake)

endif ()

install(TARGETS ideep DESTINATION lib${LIB_SUFFIX})
