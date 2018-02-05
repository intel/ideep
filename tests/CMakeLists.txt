add_subdirectory(googletest)
add_subdirectory(rapidcheck)
include(../cmake/settings.cmake)

include_directories(
  googletest/googletest/include
  googletest/googlemock/include
  rapidcheck/include
  ${PROJECT_SOURCE_DIR}/mkl-dnn/tests/gtest
  ${PROJECT_SOURCE_DIR}/mkl-dnn/tests/gtest/in
  )

file(GLOB __native_test_src
  test_lru_cache.cc
  test_tensor.cc
  )

foreach(__test_file ${__native_test_src})
  get_filename_component(__exe_file_name ${__test_file} NAME_WE)
  add_executable(${__exe_file_name} ${__test_file})
  target_link_libraries(${__exe_file_name} mkldnn gtest gmock rapidcheck)
  add_test(${__exe_file_name} ${__exe_file_name})
endforeach()
