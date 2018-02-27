#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include <test_convolution_backward_data_common.hpp>

#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class convolution_backward_weights_test
  : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
  virtual void SetUp() {
    auto p = ::testing::TestWithParam<test_convolution_params_t>::GetParam();
    auto cd = p.sizes;

    auto data_type = data_traits<data_t>::data_type;

    tensor::descriptor src_desc({cd.mb, cd.ic, cd.ih, cd.iw}, data_type,
        static_cast<format>(p.formats.src_format));

    tensor::descriptor grady_desc({cd.mb, cd.oc, cd.oh, cd.ow}, data_type,
        static_cast<format>(p.formats.dst_format));

    src_.init(src_desc);
    grady_.init(grady_desc);

    fill_data<data_t>(
        src_.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(src_.get_data_handle()));

    fill_data<data_t>(
        grady_.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(grady_.get_data_handle()));

    padR_ = {cd.padh, cd.padw};
    for (int i = 0; i < 2; ++i) {
      if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR_[0])
          / cd.strh + 1 != cd.oh)
          ++padR_[0];
      if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR_[1])
          / cd.strw + 1 != cd.ow)
          ++padR_[1];
    }

    gradw_dims_ = cd.ng > 1 ?
      tensor::dims {cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw } :
      tensor::dims {cd.oc, cd.ic, cd.kh, cd.kw};

    gradb_dims_ = {cd.oc};

    auto gradw_size = std::accumulate(gradw_dims_.begin(), gradw_dims_.end(),
        sizeof(data_t), std::multiplies<int>());
    auto gradb_size = std::accumulate(gradb_dims_.begin(), gradb_dims_.end(),
        sizeof(data_t), std::multiplies<int>());
    raw_gradw_.reset(new char [gradw_size]);
    raw_gradb_.reset(new char [gradb_size]);
  }

  tensor src_, grady_;
  tensor::dims gradw_dims_;
  tensor::dims gradb_dims_;
  tensor::dims padR_;
  std::unique_ptr<char> raw_gradw_;
  std::unique_ptr<char> raw_gradb_;
};

using convolution_test =
  convolution_backward_weights_test<float>;

TEST_P(convolution_test, TestCompute) {
  test_convolution_params_t p =
    ::testing::TestWithParam<test_convolution_params_t>::GetParam();
  test_convolution_sizes_t cd = p.sizes;

  auto gradw_desc = convolution_backward_weights::compute(src_, grady_,
      gradw_dims_, raw_gradw_.get(), raw_gradb_.get(),
      tensor::dims {cd.strh, cd.strw}, tensor::dims {cd.dilh, cd.dilw},
      tensor::dims {cd.padh, cd.padw}, padR_);

  tensor ref_gradw(gradw_desc);
  tensor::descriptor gradb_desc ({grady_.get_dim(1)}, grady_.get_data_type());
  tensor ref_gradb(gradb_desc);
  compute_ref_conv_bwd_weights<float>(cd, src_, grady_, ref_gradw);
  compare_tensor<float>(ref_gradw, tensor {gradw_desc, raw_gradw_.get()});
  compute_ref_conv_bwd_bias<float>(cd, grady_, ref_gradb);
  compare_tensor<float>(ref_gradb, tensor {gradb_desc, raw_gradb_.get()});
}

#define FP32
#define DIRECTION_BACKWARD_DATA
#include "convolution_common.h"
#include "diluted_convolution.h"
