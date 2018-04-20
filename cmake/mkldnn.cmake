if (__MKLDNN_INCLUDE)
  return ()
endif()
set(__MKLDNN_INCLUDE)

if (USE_MKLDNN_INTERNAL)
  if (NOT IS_DIRECTORY ${PROJECT_SOURCE_DIR}/mkl-dnn/external)
    execute_process(COMMAND "${PROJECT_SOURCE_DIR}/mkl-dnn/scripts/prepare_mkl.sh"
      RESULT_VARIABLE __result)
  endif()
  add_subdirectory(mkl-dnn)
else()
  include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
  # find mkldnn first
  set(mkldnn_PREFIX ${PROJECT_SOURCE_DIR}/mkl-dnn)

  if (UNIX)
    set(MKLDNN_EXTRA_COMPILER_FLAGS "-fPIC")
  endif()

  set(MKLDNN_CCXX_FLAGS ${CMAKE_CCXX_FLAGS} ${MKLDNN_CCXX_FLAGS})
  set(MKLDNN_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MKLDNN_EXTRA_COMPILER_FLAGS})
  set(MKLDNN_C_FLAGS ${CMAKE_C_FLAGS} ${MKLDNN_EXTRA_COMPILER_FLAGS})

  ExternalProject_Add(mkldnn_exernal
    SOURCE_DIR ${mkldnn_PREFIX}
    BUILD_IN_SOURCE 1
    CMAKE_ARGS
      "-DCMAKE_CCXX_FLAGS=${MKLDNN_CCXX_FLAGS}"
      "-DCMAKE_CXX_FLAGS=${MKLDNN_CXX_FLAGS}"
      "-DCMAKE_C_FLAGS=${MKLDNN_C_FLAGS}"
  )

  ExternalProject_Add_Step(mkldnn_exernal
    prepare_mkl
    DEPENDERS configure
    COMMAND ${mkldnn_PREFIX}/script/prepare_mkl.sh
  )

  set(MKLDNN_FOUND TRUE)
  set(MKLDNN_INTERNAL TRUE)
  set(MKLDNN_INCLUDE_DIR
    ${mkldnn_PREFIX}/include ${mkldnn_PREFIX}/src/cpu 
    ${mkldnn_PREFIX}/src/common)
  set(MKLDNN_LIBRARY_DIR ${mkldnn_PREFIX}/build/src)
  list(APPEND iDeep_EXTERNAL_DEPENDENCIES mkldnn_external)
endif()
