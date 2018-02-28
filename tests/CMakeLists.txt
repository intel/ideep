add_subdirectory(googletest)
add_subdirectory(rapidcheck)
add_subdirectory(rapidcheck/extras/gtest)
include(../cmake/settings.cmake)

include_directories(
  googletest/googletest/include
  googletest/googlemock/include
  rapidcheck/include
  rapidcheck/extras/gtest/include
  ${PROJECT_SOURCE_DIR}/mkl-dnn/include
  ${PROJECT_SOURCE_DIR}/mkl-dnn/tests/gtests
  ${PROJECT_SOURCE_DIR}/mkl-dnn/tests/gtests/in
  )

file(GLOB __native_test_src
  test_tensor.cc
  test_ideep_convolution_forward.cc
  test_ideep_convolution_backward_data.cc
  test_ideep_convolution_backward_weights.cc
  )

foreach(__test_file ${__native_test_src})
  get_filename_component(__exe_file_name ${__test_file} NAME_WE)
  add_executable(${__exe_file_name} ${__test_file})
  target_link_libraries(${__exe_file_name}
    mkldnn gtest gtest_main gmock rapidcheck_gtest iomp5)
  add_test(${__exe_file_name} ${__exe_file_name})
endforeach()
