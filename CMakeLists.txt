cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(iDeep CXX C)

set(IDEEP_VERSION_MAJOR 0)
set(IDEEP_VERSION_MINOR 0)
set(IDEEP_VERSION_PATCH 1)
set(IDEEP_VERSION
  "${IDEEP_VERSION_MAJOR}.${IDEEP_VERSION_MINOR}.${IDEEP_VERSION_PATCH}")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(DIRECTORY include/ DESTINATION include/ideep
  FILES_MATCHING PATTERN *.hpp
  PATTERN ideep.hpp EXCLUDE)
install(FILES include/ideep.hpp DESTINATION include)

add_subdirectory(tests)
