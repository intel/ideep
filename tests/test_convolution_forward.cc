#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include <test_convolution_forward_common.hpp>

using namespace ideep;
class convolution_forward_tests :
  public ::testing::TestWithParam<test_convolution_params_t> {
};
