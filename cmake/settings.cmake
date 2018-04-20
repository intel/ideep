if (__settings_included)
  return ()
endif()

set(__settings_included)

add_definitions(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(__cxx_flags "-std=c++11 -fvisibility-inlines-hidden -Wall -Werror -Wno-sign-compare -Wno-unknown-pragmas -fvisibility-inlines-hidden -march=native -mtune=native -pthread -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${__cxx_flags}")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
      # suppress warning on assumptions made regarding overflow (#146)
      set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -Wno-strict-overflow")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHOST")
  # workaround for Intel Compiler 16.0 that produces error caused
  # by pragma omp simd collapse(..)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable:13379")
  endif()
endif()
