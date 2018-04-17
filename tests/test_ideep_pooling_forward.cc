#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class pooling_forward_test : public ::testing::TestWithParam<pool_test_params> {
protected:
  void TestCommon() {
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
      TestCommon();
      pooling_forward::compute(src_,
          {pd.mb, pd.c, pd.oh, pd.ow}, dst, {pd.strh, pd.strw},
          {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR, p.aalgorithm,
          p.aprop_kind, padding_kind::zero);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    check_pool_fwd<data_t>(p, src_, dst);
  }

  tensor src_;
};

using pooling_test_float = pooling_forward_test<float>;
using pooling_test_s8 = pooling_forward_test<int8_t>;
using pooling_test_u8 = pooling_forward_test<uint8_t>;
using pooling_test_s32 = pooling_forward_test<int32_t>;
using pool_test_params_float = pool_test_params;
TEST_P(pooling_test_s8, TestsPooling) {
  test_forward();
}

TEST_P(pooling_test_u8, TestsPooling){
  test_forward();
}

TEST_P(pooling_test_s32, TestsPooling) {
  test_forward();
}

TEST_P(pooling_test_float, TestsPooling) {
  test_forward();
}

namespace mkldnn {

INSTANTIATE_TEST_CASE_P(
  TestPoolingAlexnetForwardS8, pooling_test_s8, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxS8, pooling_test_s8, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardAvgS8, pooling_test_s8, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxU8, pooling_test_u8, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardAvgU8, pooling_test_u8, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAlexnetForwardS32, pooling_test_s32, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxS32, pooling_test_s32, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardAvgS32, pooling_test_s32, ::testing::Values(
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_avg_include_padding,
    memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_test_params{ prop_kind::forward_inference, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding,
    memory::format::nhwc, memory::format::nhwc,
    {16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardEF, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 0, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 0, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 0, 4, 4, 4, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 7, 7, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments},
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 4, 2, 3, 3, 1, 1, 1, 1 },
    true, mkldnn_invalid_arguments}
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMax, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
));


INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxNHWC, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nhwc,
    memory::format::nhwc, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxBlocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxBlockedPerf, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardAvgBlockedPerf, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 1, 8, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_include_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding, memory::format::nChw8c,
    memory::format::nChw8c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxBlocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardMaxBlocked16Perf, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingForwardAvgBlocked16Perf, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_include_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
    , pool_test_params_float{ prop_kind::forward_training, engine::kind::cpu,
    algorithm::pooling_avg_exclude_padding, memory::format::nChw16c,
    memory::format::nChw16c, { 16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));


INSTANTIATE_TEST_CASE_P(
  TestPoolingAlexnetForwardMaxNCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAlexnetForwardMaxBlocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAlexnetForwardMaxBlocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxBlockedStride1, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxCIFAR10NCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgCIFAR10NCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxCIFAR10Blocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgCIFAR10Blocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxCIFAR10Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgCIFAR10Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 2, 2 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxGoogleNetV1NCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxGoogleNetV1Blocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxResnet50NCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nchw,
    memory::format::nchw, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxResnet50Blocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingMaxResnet50Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw16c,
    memory::format::nChw16c, { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgGoogleNetV1NCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgGoogleNetV1Blocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgGoogleNetV1Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 3, 3 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgResnet50NCHW, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nchw, memory::format::nchw,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgResnet50Blocked, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAvgResnet50Blocked16, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } },
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw16c, memory::format::nChw16c,
    { 2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1 } }
));

INSTANTIATE_TEST_CASE_P(
  TestPoolingAsymmPadding, pooling_test_float, ::testing::Values(
    pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1}}

    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 8, 3, 14, 1, 8, 3, 3, 0, 1, 1, 2}}

    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 3, 100, 1, 51, 3, 3, 0, 1, 1, 2}}

    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 3, 102, 1, 52, 3, 3, 0, 1, 1, 2}}

    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 9, 103, 7, 52, 3, 3, 0, 1, 1, 2}}

    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_max, memory::format::nChw8c,
    memory::format::nChw8c, {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_include_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }
    ,pool_test_params_float{ prop_kind::forward_inference,
    engine::kind::cpu, algorithm::pooling_avg_exclude_padding,
    memory::format::nChw8c, memory::format::nChw8c,
    {1, 96, 300, 500, 151, 251, 3, 3, 1, 1, 2, 2} }

));
};
