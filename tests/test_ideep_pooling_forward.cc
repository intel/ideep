#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

struct test_pool_desc_t {
  int mb, c;
  int ih, iw;
  int oh, ow;
  int kh, kw;
  int padt, padl;
  int strh, strw;
};

struct pool_test_params {
  mkldnn::prop_kind aprop_kind;
  const mkldnn::engine::kind engine_kind;
  mkldnn::algorithm aalgorithm;
  mkldnn::memory::format src_format;
  mkldnn::memory::format dst_format;
  test_pool_desc_t test_pd;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

template <typename data_t>
void check_pool_fwd(const pool_test_params &p, const tensor &src,
        const tensor &dst, const tensor &ws)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *dst_data = (data_t *)dst.get_data_handle();

  auto ws_data = [=](size_t idx) -> int {
    auto w = (unsigned char *)ws.get_data_handle();
    if (w == nullptr) return -1;
    if (ws.get_mkldnn_memory_desc_t()->data_type == mkldnn_u8)
      return (int)w[idx];
    else
      return ((int *)w)[idx];
  };

  const mkldnn::memory::desc src_d =
      mkldnn::memory::desc(*src.get_mkldnn_memory_desc_t());
  const mkldnn::memory::desc dst_d =
      mkldnn::memory::desc(*dst.get_mkldnn_memory_desc_t());
  const mkldnn::memory::desc ws_d  =
      mkldnn::memory::desc(*ws.get_mkldnn_memory_desc_t());

  auto pd = p.test_pd;

#pragma omp parallel for collapse(4) schedule(static)
  for (int n = 0; n < pd.mb; n++) {
    for (int c = 0; c < pd.c; c++) {
      for (int oh = 0; oh < pd.oh; oh++) {
        for (int ow = 0; ow < pd.ow; ow++) {
          int oidx = n * pd.c * pd.oh * pd.ow + c * pd.oh * pd.ow
                  + oh * pd.ow + ow;
          data_t out = dst_data[map_index(dst_d, oidx)];
          int out_index = -1;
          if(p.aalgorithm == mkldnn::pooling_max
              && p.aprop_kind == mkldnn::prop_kind::forward_training) {
            out_index = ws_data(map_index(ws_d, oidx));
          }
          data_t out_ref = data_t(0);
          int out_ref_index = 0;
          bool is_initialized = false;
          int num_summands = 0;

          for (int kh = 0; kh < pd.kh; ++kh) {
            for (int kw = 0; kw < pd.kw; ++kw) {
              const int ih = oh * pd.strh - pd.padt + kh;
              const int iw = ow * pd.strw - pd.padl + kw;

              if (ih < 0 || ih >= pd.ih) continue;
              if (iw < 0 || iw >= pd.iw) continue;

              int iidx = n * pd.c * pd.ih * pd.iw
                      + c * pd.ih * pd.iw + ih * pd.iw + iw;

              data_t d = src_data[map_index(src_d, iidx)];
              if (p.aalgorithm == mkldnn::pooling_max) {
                if (!is_initialized) {
                  out_ref = d;
                  out_ref_index = kh* pd.kh + kw;
                  is_initialized = true;
                } else {
                  if (out_ref < d) {
                    out_ref = d;
                    out_ref_index = kh* pd.kh + kw;
                  }
                }
              } else if (p.aalgorithm == mkldnn::pooling_avg_include_padding ||
                       p.aalgorithm == mkldnn::pooling_avg_exclude_padding) {
                out_ref += d;
                num_summands++;
              }
            }
          }

          if (p.aalgorithm == mkldnn::pooling_avg_include_padding) {
            num_summands = pd.kw * pd.kh;
          }

          if (p.aalgorithm == mkldnn::pooling_avg_include_padding ||
            p.aalgorithm == mkldnn::pooling_avg_exclude_padding) {
            out_ref = out_round<data_t>(
                    (float)out_ref / num_summands);
          }
          EXPECT_NEAR(out, out_ref, 1e-6);
          if(p.aalgorithm == mkldnn::pooling_max
            && p.aprop_kind == mkldnn::forward_training) {
            EXPECT_EQ(out_index, out_ref_index) << " n = " << n
                 << " c = " << c << " oh = " << oh << " ow = " << ow;
          }
        }
      }
    }
  }
}

template <typename data_t>
class pooling_test : public ::testing::TestWithParam<pool_test_params> {
protected:
  virtual void SetUp()
  {
    pool_test_params p
            = ::testing::TestWithParam<pool_test_params>::GetParam();

    ASSERT_TRUE(p.engine_kind == mkldnn::engine::kind::cpu);
    ASSERT_TRUE(p.aprop_kind == mkldnn::prop_kind::forward_training
            || p.aprop_kind == mkldnn::prop_kind::forward_scoring);
    mkldnn::memory::data_type data_type = data_traits<data_t>::data_type;

    test_pool_desc_t pd = p.test_pd;

    tensor::descriptor src_desc({ pd.mb, pd.c, pd.ih, pd.iw },
        data_type, static_cast<format>(p.src_format));

    tensor src;
    src.init(src_desc);

    fill_data<data_t>(src.get_size()/ sizeof(data_t),
            (data_t *)src.get_data_handle());

    std::vector<int> padR = { pd.padt, pd.padl };
    for (int i = 0; i < 2; ++i) {
    if ((pd.ih + pd.padt + padR[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR[0];
    if ((pd.iw + pd.padl + padR[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR[1];
    }

    tensor dst, ws;
    auto test = [&]() {
      size_t dst_sz = pd.mb * pd.c * pd.oh * pd.ow;
      auto dst_r = new char [dst_sz * sizeof(data_t)];
      dst = pooling_forward::compute(src,
          {pd.mb, pd.c, pd.oh, pd.ow}, dst_r, {pd.strh, pd.strw},
          {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR, p.aalgorithm,
          p.aprop_kind, padding_kind::zero);

      bool with_workspace = true
          && p.aprop_kind == mkldnn::prop_kind::forward_training
          && p.aalgorithm == mkldnn::pooling_max;

      if (with_workspace)
        ws = *dst.get_extra();
      else
        ws.init(tensor::descriptor({}, data_type,
            static_cast<format>(p.dst_format)));
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    check_pool_fwd<data_t>(p, src, dst, ws);
  }
};

using pooling_test_float = pooling_test<float>;
using pooling_test_s8 = pooling_test<int8_t>;
using pooling_test_u8 = pooling_test<uint8_t>;
using pooling_test_s32 = pooling_test<int32_t>;
using pool_test_params_float = pool_test_params;
TEST_P(pooling_test_s8, TestsPooling)
{
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

TEST_P(pooling_test_u8, TestsPooling)
{
}

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

TEST_P(pooling_test_s32, TestsPooling)
{
}

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

TEST_P(pooling_test_float, TestsPooling)
{
}

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
