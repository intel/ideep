#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class pooling_bwd_test : public ::testing::TestWithParam<pool_bwd_test_params> {
private:
  tensor x_, y_, grady_;
  mkldnn::memory::dims padR_;

protected:
  void TestCommon() {
    auto p = ::testing::TestWithParam<pool_bwd_test_params>::GetParam();
    auto pd = p.test_pd;

    auto data_type = data_traits<data_t>::data_type;
    ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

    tensor::descriptor x_desc({ pd.mb, pd.c, pd.ih, pd.iw },
        data_type, static_cast<format>(p.diff_src_format));
    tensor::descriptor y_desc({ pd.mb, pd.c, pd.oh, pd.ow },
        data_type, static_cast<format>(p.diff_dst_format));

    padR_ = { pd.padt, pd.padl };
    for (int i = 0; i < 2; ++i) {
      if ((pd.ih + pd.padt + padR_[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR_[0];
      if ((pd.iw + pd.padl + padR_[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR_[1];
    }

    x_.init(x_desc);
    fill_data<data_t>(x_.get_nelems(),
        reinterpret_cast<data_t*>(x_.get_data_handle()));

    grady_.init(y_desc);
    fill_data<data_t>(grady_.get_nelems(),
        reinterpret_cast<data_t *>(grady_.get_data_handle()));
  }

  void Forward() {
    auto p = ::testing::TestWithParam<pool_bwd_test_params>::GetParam();
    auto pd = p.test_pd;

 //   auto test = [&]() {
      pooling_forward::compute(x_, tensor::dims {pd.mb, pd.c, pd.oh, pd.ow},
          y_, tensor::dims {pd.strh, pd.strw},
          tensor::dims {pd.kh, pd.kw}, tensor::dims {pd.padt, pd.padl},
          padR_, p.aalgorithm,
          prop_kind::forward_training, padding_kind::zero);
 //   };

 //   if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
 //       return;

    check_pool_fwd<data_t>(p, x_, y_);
  }

  void Backward() {
    auto p = ::testing::TestWithParam<pool_bwd_test_params>::GetParam();
    auto pd = p.test_pd;

    tensor gradx;
//    auto test = [&]() {
      pooling_backward::compute(grady_, y_, x_, gradx,
          {pd.strh, pd.strw}, {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR_,
          p.aalgorithm, padding_kind::zero);
//    };

//    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
//      return;

    check_pool_bwd<data_t>(p, gradx, grady_, y_);
  }
};

using pooling_bwd_test_float = pooling_bwd_test<float>;
using pool_bwd_test_params_float = pool_bwd_test_params;

TEST_P(pooling_bwd_test_float, TestsPoolingBackward) {
    auto p = ::testing::TestWithParam<pool_bwd_test_params>::GetParam();
    auto testcommon = [&] () {
      TestCommon();
      Forward();
      Backward();
    };
    if (catch_expected_failures(testcommon, p.expect_to_fail, p.expected_status))
        return;
}

namespace mkldnn {

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardEF, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 0, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 0, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 0, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 7, 7, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 4, 2, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments}
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMaxAlexNetNCHW, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
    // Reuse cases
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMaxCIFAR10NCHW, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    // Reuse cases
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMax, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1 } },
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1 } },
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMaxBlocked, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 1, 8, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardAvgBlocked, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 8, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 8, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMaxBlocked16, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 1, 16, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardAvgBlocked16, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 5, 5, 2, 2, 3, 3, 0, 0, 2, 2 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 } }
    , pool_bwd_test_params_float{engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 3, 2, 2, 2, 3, 3, 1, 1, 2, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMaxBlockedPerf, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardAvgBlockedPerf, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardMaxBlocked16Perf, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardAvgBlocked16Perf, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_bwd_test_params_float{ engine::kind::cpu,
    pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingBackwardAsymmPadding, pooling_bwd_test_float, ::testing::Values(
    pool_bwd_test_params_float{
    engine::kind::cpu, pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}

    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}

    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}

    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}

    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}

    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
    ,pool_bwd_test_params_float{
    engine::kind::cpu, pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
));

};
