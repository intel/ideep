include_directories(${PROJECT_SOURCE_DIR}/mkl-dnn/include)


add_executable(tests/test_reduce_avg_float
            total_reduce/test/reduce_avg.cpp
            )
add_executable(tests/test_reduce_avg_int
            total_reduce/test/reduce_int.c
            )

target_link_libraries(tests/test_reduce_avg_float ideep mkldnn)
target_link_libraries(tests/test_reduce_avg_int ideep mkldnn)

if(MPI_COMPILE_FLAGS)
  set_target_properties(tests/test_reduce_avg_float PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(tests/test_reduce_avg_int PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(tests/test_reduce_avg_float PROPERTIES LINK_FLAGS "-lstdc++ ${MPI_LINK_FLAGS}")
  set_target_properties(tests/test_reduce_avg_int PROPERTIES LINK_FLAGS "-lstdc++ ${MPI_LINK_FLAGS}")
endif()

install(FILES ${CMAKE_BINARY_DIR}/libideep.so DESTINATION lib)
