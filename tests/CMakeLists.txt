add_subdirectory(googletest)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include
  ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/include
  )

file(GLOB __native_test_src
  test_lru_cache.cc
  test_tensor.cc
  )

foreach(__test_file ${__native_test_src})
  get_filename_component(__exe_file_name ${__test_file} NAME_WE)
  add_executable(${__exe_file_name} ${__test_file})
  target_link_libraries(${__exe_file_name} mkldnn gtest gmock)
  add_test(${__exe_file_name} ${__exe_file_name})
endforeach()
