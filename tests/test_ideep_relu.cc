#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class relu_tests:
  public ::testing::TestWithParam<relu_test_params<data_t>> {
private:
  tensor src_;
  tensor grady_;

protected:
  void TestCommon() {
    auto p = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();
    tensor::descriptor src_desc(static_cast<tensor::dims>(p.dims),
        data_traits<data_t>::data_type, static_cast<format>(p.data_format));
    tensor::descriptor grady_desc(static_cast<tensor::dims>(p.dims),
        data_traits<data_t>::data_type, static_cast<format>(p.diff_format));

    src_.init(src_desc);
    grady_.init(grady_desc);

    fill_data<data_t>(
        src_.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(src_.get_data_handle()),
        data_t(0), data_t(1));

    fill_data<data_t>(
        grady_.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(grady_.get_data_handle()),
        data_t(0), data_t(1));
  }

  void Forward() {
    auto p = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();
    tensor dst;
    // auto test = [&]() {
      eltwise_forward::compute(src_, dst,
          algorithm::eltwise_relu, prop_kind::forward, p.negative_slope, 0.0);
    // };

    // if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    //   return;

    check_relu_fwd(p.negative_slope, src_, dst);
  }

  void Backward() {
    auto p = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();
    tensor gradx;
    // auto test = [&]() {
      eltwise_backward::compute(src_, grady_, gradx,
          algorithm::eltwise_relu, p.negative_slope, 0.0);
    // };

    // if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    //   return;

    check_relu_bwd(p.negative_slope, src_, grady_, gradx);
  }
};

using relu_test_float = relu_tests<float>;
using relu_test_params_float = relu_test_params<float>;

TEST_P(relu_test_float, TestsRelu) {
  auto p = ::testing::TestWithParam<relu_test_params<float>>::GetParam();
  tensor gradx;
  auto test = [&]() {
    TestCommon();
    Forward();
    Backward();
  };
  if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    return;
}

#define EXPAND_SIZES(mb, c, h, w) { mb, c, h, w }
#define EXPAND_FORMATS(data) mkldnn::memory::format::data

#define ENGINE mkldnn::engine::kind::cpu

#define PARAMS_EF(data, diff_data, ns, mb, c, h, w, ef, es) \
    relu_test_params_float { ENGINE, \
    EXPAND_FORMATS(data), EXPAND_FORMATS(diff_data), \
    ns, EXPAND_SIZES(mb, c, h, w), ef, es}

#define PARAMS(data, diff_data, ns, mb, c, h, w) \
    PARAMS_EF(data, diff_data, ns, mb, c, h, w, false, mkldnn_success)

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, relu_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(SimpleExpectedFails,
  PARAMS_EF(nchw, nchw, 0.f, 0, 8, 4, 4, true, mkldnn_invalid_arguments),
  PARAMS_EF(nchw, nchw, 0.f, 2, 0, 4, 4, true, mkldnn_invalid_arguments),
  PARAMS_EF(nchw, nchw, 0.f, 2, 8, 0, 4, true, mkldnn_invalid_arguments),
  PARAMS_EF(nchw, nchw, 0.f, 2, 8, 4, 0, true, mkldnn_invalid_arguments)
);

INST_TEST_CASE(SimpleZeroNegativeSlope_NCHW,
  //PARAMS(nchw, nchw, 0.f, 1, 8, 10000, 10000),  // is a tensor of 3 Gb data ok? YES (330 s runtime, slow)
  //PARAMS(nchw, nchw, 0.f, 1, 12, 10000, 10000), // is a tensor of >4 Gb data ok? worked once (release mode)
  PARAMS(nchw, nchw, 0.f, 2, 8, 4, 4),
  PARAMS(nchw, nchw, 0.f, 2, 16, 4, 4),
  PARAMS(nchw, nchw, 0.f, 2, 16, 8, 8),
  PARAMS(nchw, nchw, 0.f, 2, 16, 16, 8),
  PARAMS(nchw, nchw, 0.f, 2, 16, 10, 8),
  PARAMS(nchw, nchw, 0.f, 10, 10, 10, 10),
  PARAMS(nchw, nchw, 0.f, 256, 64, 8, 16),
  PARAMS(nchw, nchw, 0.f, 1, 1, 1, 1),
  PARAMS(nchw, nchw, 0.f, 3, 5, 7, 11)
);

INST_TEST_CASE(Simple_NCHW,
  PARAMS(nchw, nchw, 0.1f, 2, 8, 4, 4),
  PARAMS(nchw, nchw, 0.1f, 2, 16, 4, 4),
  PARAMS(nchw, nchw, 0.1f, 2, 16, 8, 8),
  PARAMS(nchw, nchw, 0.1f, 2, 16, 16, 8),
  PARAMS(nchw, nchw, 0.1f, 2, 16, 10, 8),
  PARAMS(nchw, nchw, 0.1f, 10, 10, 10, 10),
  PARAMS(nchw, nchw, 0.1f, 256, 64, 8, 16),
  PARAMS(nchw, nchw, 0.1f, 1, 1, 1, 1),
  PARAMS(nchw, nchw, 0.1f, 3, 5, 7, 11)
);

INST_TEST_CASE(Simple,
  PARAMS(nchw, nChw8c, 0.1f, 2, 8, 4, 4),
  PARAMS(nChw8c, nchw, 0.1f, 2, 16, 4, 4),
  PARAMS(nchw, nchw, 0.1f, 2, 16, 8, 8),
  PARAMS(nChw8c, nChw8c, 0.1f, 2, 16, 16, 8),
  PARAMS(nhwc, nchw, 0.1f, 2, 16, 10, 8),
  PARAMS(nchw, nhwc, 0.1f, 10, 10, 10, 10)
);

INST_TEST_CASE(AlexNet_NCHW,
  PARAMS(nchw, nchw, 0.f, 2, 96, 55, 55),
  PARAMS(nchw, nchw, 0.f, 2, 256, 27, 27),
  PARAMS(nchw, nchw, 0.f, 2, 384, 13, 13)
);
