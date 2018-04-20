#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include <test_convolution_backward_data_common.hpp>

#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t_grady, typename data_t_w,
         typename data_t_acc, typename data_t_gradx>
class convolution_backward_data_test
  : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
  virtual void SetUp () {}
  void TestCommon() {
    auto p = ::testing::TestWithParam<test_convolution_params_t>::GetParam();
    auto cd = p.sizes;

    auto data_type_grady = data_traits<data_t_grady>::data_type;
    auto data_type_w = data_traits<data_t_w>::data_type;

    auto weights_desc = cd.ng > 1 ? tensor::descriptor (
        {cd.ng, cd.oc/cd.ng, cd.ic/cd.ng, cd.kh, cd.kw}, data_type_w,
        static_cast<format>(p.formats.weights_format)) :
      tensor::descriptor (
          {cd.oc, cd.ic, cd.kh, cd.kw }, data_type_w,
          static_cast<format>(p.formats.weights_format));

    tensor::descriptor grady_desc({cd.mb, cd.oc, cd.oh, cd.ow}, data_type_grady,
        static_cast<format>(p.formats.dst_format));

    weights_.init(weights_desc);
    grady_.init(grady_desc);

    fill_data<data_t_w>(
        weights_.get_size() / sizeof(data_t_w),
        reinterpret_cast<data_t_w *>(weights_.get_data_handle()));

    fill_data<data_t_grady>(
        grady_.get_size() / sizeof(data_t_grady),
        reinterpret_cast<data_t_grady *>(grady_.get_data_handle()));

    padR_ = {cd.padh, cd.padw};
    for (int i = 0; i < 2; ++i) {
      if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR_[0])
          / cd.strh + 1 != cd.oh)
          ++padR_[0];
      if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR_[1])
          / cd.strw + 1 != cd.ow)
          ++padR_[1];
    }

    gradx_dims_ = {cd.mb, cd.ic, cd.ih, cd.iw};
  }

  tensor weights_, grady_;
  tensor::dims gradx_dims_;
  tensor::dims padR_;
};

using convolution_test =
  convolution_backward_data_test<float, float, float, float>;

TEST_P(convolution_test, TestCompute) {
  test_convolution_params_t p =
    ::testing::TestWithParam<test_convolution_params_t>::GetParam();
  test_convolution_sizes_t cd = p.sizes;

  tensor gradx;
  auto test = [&] () {
    TestCommon();
    convolution_backward_data::compute(grady_, weights_,
        gradx_dims_, gradx, tensor::dims {cd.strh, cd.strw},
        tensor::dims {cd.dilh, cd.dilw}, tensor::dims {cd.padh, cd.padw},
        padR_);
  };

  if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    return;

  tensor ref_gradx(gradx.get_descriptor());
  compute_ref_conv_bwd_data<float, float, float, float>(cd, ref_gradx,
      weights_, grady_);
  compare_tensor<float>(ref_gradx, gradx);
}

#define FP32
#define DIRECTION_BACKWARD_DATA
#include "convolution_common.h"
// #include "dilated_convolution.h"
