#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
struct relu_test_params {
  mkldnn::engine::kind engine_kind;
  mkldnn::memory::format data_format;
  mkldnn::memory::format diff_format;
  data_t negative_slope;
  mkldnn::memory::dims dims;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

template <typename data_t>
void check_relu_fwd(data_t negative_slope, const tensor &src, const tensor &dst)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *dst_data = (data_t *)dst.get_data_handle();

  ASSERT_EQ(src.ndims(), 4);
  ASSERT_EQ(dst.ndims(), 4);
  ASSERT_EQ(src.get_data_type(), mkldnn::memory::data_type::f32);
  ASSERT_EQ(dst.get_data_type(), mkldnn::memory::data_type::f32);

  for (size_t i = 0; i < src.get_size() / sizeof(data_t); ++i) {
    data_t s = src_data[i];
    EXPECT_NEAR(dst_data[i], s > 0 ? s : s * negative_slope, 1.e-7);
  }
}

template <typename data_t>
void check_relu_bwd(data_t negative_slope, const tensor &src, const tensor &grady, const tensor &gradx)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *grady_data = (data_t *)grady.get_data_handle();
  data_t *gradx_data = (data_t *)gradx.get_data_handle();

  const mkldnn::memory::desc src_d =
      mkldnn::memory::desc(*src.get_mkldnn_memory_desc_t());
  const mkldnn::memory::desc gradx_d =
      mkldnn::memory::desc(*gradx.get_mkldnn_memory_desc_t());

  ASSERT_EQ(src.ndims(), 4);
  ASSERT_EQ(grady.ndims(), 4);
  ASSERT_EQ(src.get_data_type(), mkldnn::memory::data_type::f32);
  ASSERT_EQ(grady.get_data_type(), mkldnn::memory::data_type::f32);

  for (size_t i = 0; i < src.get_size() / sizeof(data_t); ++i) {
    data_t ref_x = src_data[map_index(src_d, i)];
    data_t ref_gy = grady_data[map_index(gradx_d, i)];
    data_t ref_gx = ref_gy * ((ref_x > 0) ? data_t{1} : negative_slope);
    EXPECT_NEAR(gradx_data[map_index(gradx_d, i)], ref_gx, 1.e-7);
  }
}

template <typename data_t>
class relu_tests:
  public ::testing::TestWithParam<relu_test_params<data_t>> {
private:
  tensor src_;
  tensor grady_;
  std::unique_ptr<char> raw_dst_;
  std::unique_ptr<char> raw_gradx_;

protected:
  virtual void SetUp() {
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

    raw_dst_.reset(new char [src_.get_size()]);
    raw_gradx_.reset(new char [src_.get_size()]);

    Forward();
    Backward();
  }

  void Forward() {
    auto p = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();
    tensor dst;
    auto test = [&]() {
      dst = eltwise_forward::compute(src_, raw_dst_.get(),
          algorithm::eltwise_relu, prop_kind::forward, p.negative_slope, 0.0);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    check_relu_fwd(p.negative_slope, src_, dst);
  }

  void Backward() {
    auto p = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();
    tensor gradx;
    auto test = [&]() {
      gradx = eltwise_backward::compute(src_, grady_, raw_gradx_.get(),
          algorithm::eltwise_relu, p.negative_slope, 0.0);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    check_relu_bwd(p.negative_slope, src_, grady_, gradx);
  }
};

using relu_test_float = relu_tests<float>;
using relu_test_params_float = relu_test_params<float>;

TEST_P(relu_test_float, TestsRelu) {}

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
