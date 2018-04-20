option(WITH_TEST "include tests" ON)
if (NOT WITH_TEST)
    return()
endif()

add_subdirectory(googletest EXCLUDE_FROM_ALL)
add_subdirectory(rapidcheck EXCLUDE_FROM_ALL)
add_subdirectory(rapidcheck/extras/gtest EXCLUDE_FROM_ALL)
include(../cmake/settings.cmake)
include(../cmake/profiling.cmake)

include_directories(
  ${PROJECT_SOURCE_DIR}/mkl-dnn/include
  ${PROJECT_SOURCE_DIR}/mkl-dnn/tests/gtests
  in
  googletest/googletest/include
  googletest/googlemock/include
  rapidcheck/include
  rapidcheck/extras/gtest/include
  ${vtune_include}
  )

file(GLOB __native_test_src
  test_tensor.cc
  test_lru_cache.cc
  test_ideep_convolution_forward.cc
  test_ideep_convolution_backward_data.cc
  test_ideep_convolution_backward_weights.cc
  test_ideep_lrn_forward.cc
  test_ideep_lrn_backward.cc
  test_ideep_relu.cc
  test_ideep_sum.cc
  test_ideep_batch_normalization.cc
  test_ideep_pooling_forward.cc
  test_ideep_pooling_backward.cc
  test_ideep_inner_product_forward.cc
  test_ideep_inner_product_backward_data.cc
  test_ideep_inner_product_backward_weights.cc
  test_ideep_concat.cc
  test_ideep_reorder.cc
  test_ideep_allocator.cc
  bench_ideep_batch_normalization.cc
  bench_ideep_pooling_forward.cc
  bench_ideep_concat.cc
  )

foreach(__test_file ${__native_test_src})
  get_filename_component(__exe_file_name ${__test_file} NAME_WE)
  add_executable(${__exe_file_name} ${__test_file})
  target_link_libraries(${__exe_file_name}
    mkldnn gtest gtest_main gmock rapidcheck_gtest ${itt_libname} dl)
  add_test(${__exe_file_name} ${__exe_file_name})
endforeach()
