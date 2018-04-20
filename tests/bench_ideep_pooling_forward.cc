#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class pooling_forward_test : public ::testing::TestWithParam<pool_test_params> {
protected:
  virtual void SetUp() {
    pool_test_params p
            = ::testing::TestWithParam<pool_test_params>::GetParam();

    ASSERT_TRUE(p.aprop_kind == mkldnn::prop_kind::forward_training
            || p.aprop_kind == mkldnn::prop_kind::forward_scoring);
    auto data_type = data_traits<data_t>::data_type;
    auto pd = p.test_pd;

    tensor::descriptor src_desc({ pd.mb, pd.c, pd.ih, pd.iw },
        data_type, static_cast<format>(p.src_format));

    src_.init(src_desc);

    fill_data<data_t>(src_.get_size()/ sizeof(data_t),
            (data_t *)src_.get_data_handle());

    std::vector<int> padR = { pd.padt, pd.padl };
    for (int i = 0; i < 2; ++i) {
    if ((pd.ih + pd.padt + padR[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR[0];
    if ((pd.iw + pd.padl + padR[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR[1];
    }
  }

  void test_forward() {
    pool_test_params p
            = ::testing::TestWithParam<pool_test_params>::GetParam();

    ASSERT_TRUE(p.aprop_kind == mkldnn::prop_kind::forward_training
            || p.aprop_kind == mkldnn::prop_kind::forward_scoring);
    auto pd = p.test_pd;

    std::vector<int> padR = { pd.padt, pd.padl };
    for (int i = 0; i < 2; ++i) {
      if ((pd.ih + pd.padt + padR[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR[0];
      if ((pd.iw + pd.padl + padR[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR[1];
    }

    auto dst = make_output();
    auto test = [&]() {
       pooling_forward::compute(src_,
          {pd.mb, pd.c, pd.oh, pd.ow}, dst, {pd.strh, pd.strw},
          {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR, p.aalgorithm,
          p.aprop_kind, padding_kind::zero);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

  }

  tensor src_;
};

using pooling_test_float = pooling_forward_test<float>;
using pool_test_params_float = pool_test_params;

TEST_P(pooling_test_float, TestsPooling) {
  test_forward();
}

namespace mkldnn {

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 128, 512, 14, 14, 14, 14, 3, 3, 1, 1, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 128, 512, 14, 14, 14, 14, 3, 3, 1, 1, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 128, 512, 14, 14, 14, 14, 3, 3, 1, 1, 1, 1 } }
));
}

