#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

struct test_pool_bwd_desc_t {
  int mb, c;
  int ih, iw;
  int oh, ow;
  int kh, kw;
  int padt, padl;
  int strh, strw;
};

struct pool_bwd_test_params {
  mkldnn::engine::kind engine_kind;
  mkldnn::algorithm aalgorithm;
  mkldnn::memory::format diff_src_format;
  mkldnn::memory::format diff_dst_format;
  test_pool_bwd_desc_t test_pd;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

template <typename data_t>
void check_pool_fwd(const pool_bwd_test_params &p, const tensor &src,
        const tensor &dst)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *dst_data = (data_t *)dst.get_data_handle();

  const mkldnn::memory::desc src_d =
      mkldnn::memory::desc(*src.get_mkldnn_memory_desc_t());
  const mkldnn::memory::desc dst_d =
      mkldnn::memory::desc(*dst.get_mkldnn_memory_desc_t());

  auto pd = p.test_pd;

  auto apply_offset = [=](int index, int offset) {
    return (index > offset) ? index - offset : 0;
  };

#pragma omp parallel for collapse(4) schedule(static)
  for (int n = 0; n < pd.mb; n++) {
    for (int c = 0; c < pd.c; c++) {
      for (int oh = 0; oh < pd.oh; oh++) {
        for (int ow = 0; ow < pd.ow; ow++) {
          int oidx = n * pd.c * pd.oh * pd.ow + c * pd.oh * pd.ow
                  + oh * pd.ow + ow;
          data_t out = dst_data[map_index(dst_d, oidx)];
          data_t out_ref = data_t(0);
          bool is_initialized = false;

          auto ih_start = apply_offset(oh*pd.strh, pd.padt);
          auto iw_start = apply_offset(ow*pd.strw, pd.padl);
          auto ih_end =
              std::min(oh*pd.strh - pd.padt + pd.kh, pd.ih);
          auto iw_end =
              std::min(ow*pd.strw - pd.padl + pd.kw, pd.iw);

          auto num_summands = (p.aalgorithm != mkldnn::pooling_avg_exclude_padding)
              ? pd.kw*pd.kh : (ih_end - ih_start)*(iw_end - iw_start);

          for (int ih = ih_start; ih < ih_end; ++ih) {
            for (int iw = iw_start; iw < iw_end; ++iw) {
              int iidx = n * pd.c * pd.ih * pd.iw
                      + c * pd.ih * pd.iw + ih * pd.iw + iw;

              data_t d = src_data[map_index(src_d, iidx)];
              if (p.aalgorithm == mkldnn::pooling_max) {
                if (!is_initialized) {
                  out_ref = d;
                  is_initialized = true;
                } else {
                  if (out_ref < d)
                    out_ref = d;
                }
              } else if (p.aalgorithm == mkldnn::pooling_avg_include_padding
                      || p.aalgorithm == mkldnn::pooling_avg_exclude_padding) {
                out_ref += d;
              }
            }
          }

          if (p.aalgorithm == mkldnn::pooling_avg_include_padding ||
            p.aalgorithm == mkldnn::pooling_avg_exclude_padding) {
            out_ref /= num_summands;
          }
          EXPECT_NEAR(out, out_ref, 1e-6f);
        }
      }
    }
  }
}

template <typename data_t>
void check_pool_bwd(const pool_bwd_test_params &p, const tensor &diff_src,
        const tensor &diff_dst, const tensor &ws)
{
  data_t *diff_src_data = (data_t *)diff_src.get_data_handle();
  data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

  auto ws_data = [=](size_t idx) -> int {
    auto w = (unsigned char *)ws.get_data_handle();
    if (w == nullptr) return -1;
    if (ws.get_mkldnn_memory_desc_t()->data_type == mkldnn_u8)
      return (int)w[idx];
    else
      return ((int *)w)[idx];
  };

  const mkldnn::memory::desc diff_src_d =
      mkldnn::memory::desc(*diff_src.get_mkldnn_memory_desc_t());
  const mkldnn::memory::desc diff_dst_d =
      mkldnn::memory::desc(*diff_dst.get_mkldnn_memory_desc_t());
  const mkldnn::memory::desc ws_d =
      mkldnn::memory::desc(*ws.get_mkldnn_memory_desc_t());

  auto pd = p.test_pd;
  data_t *ref_diff_src = new data_t[pd.mb*pd.c*pd.ih*pd.iw];

  auto apply_offset = [=](int index, int offset) {
    return (index > offset) ? index - offset : 0;
  };

#pragma omp parallel for collapse(4) schedule(static)
  for (int n = 0; n < pd.mb; n++) {
    for (int c = 0; c < pd.c; c++) {
      for (int ih = 0; ih < pd.ih; ih++) {
        for (int iw = 0; iw < pd.iw; iw++) {
          int iidx = n * pd.c * pd.ih * pd.iw
              + c * pd.ih * pd.iw + ih * pd.iw + iw;
          ref_diff_src[iidx] = 0.;
        }
      }
    }
  }

#pragma omp parallel for collapse(2) schedule(static)
  for (int n = 0; n < pd.mb; n++) {
    for (int c = 0; c < pd.c; c++) {
      for (int oh = 0; oh < pd.oh; oh++) {
        for (int ow = 0; ow < pd.ow; ow++) {
          int oidx = n * pd.c * pd.oh * pd.ow + c * pd.oh * pd.ow
                  + oh * pd.ow + ow;
          data_t diff_dst = diff_dst_data[map_index(diff_dst_d, oidx)];
          if (p.aalgorithm == mkldnn::pooling_max) {
            int kh_max = ws_data(map_index(ws_d, oidx)) / pd.kw;
            int kw_max = ws_data(map_index(ws_d, oidx)) % pd.kw;
            for (int kh = 0; kh < pd.kh; kh++) {
              for (int kw = 0; kw < pd.kw; kw++) {
                int iw = ow * pd.strw - pd.padl + kw;
                int ih = oh * pd.strh - pd.padt + kh;
                if (iw < 0 || iw >= pd.iw) continue;
                if (ih < 0 || ih >= pd.ih) continue;
                int iidx = n * pd.c * pd.ih * pd.iw
                        + c * pd.ih * pd.iw + ih * pd.iw + iw;

                if (kh == kh_max && kw == kw_max)
                  ref_diff_src[iidx] += diff_dst;
              }
            }
          } else if (p.aalgorithm == mkldnn::pooling_avg_include_padding
                || p.aalgorithm == mkldnn::pooling_avg_exclude_padding) {
            auto ih_start = apply_offset(oh*pd.strh, pd.padt);
            auto iw_start = apply_offset(ow*pd.strw, pd.padl);
            auto ih_end =
                std::min(oh*pd.strh - pd.padt + pd.kh, pd.ih);
            auto iw_end =
                std::min(ow*pd.strw - pd.padl + pd.kw, pd.iw);

            auto num_summands = (p.aalgorithm != mkldnn::pooling_avg_exclude_padding)
                ? pd.kw*pd.kh : (ih_end - ih_start)*(iw_end - iw_start);

            for (int ih = ih_start; ih < ih_end; ih++) {
              for (int iw = iw_start; iw < iw_end; iw++) {
                int iidx = n * pd.c * pd.ih * pd.iw
                        + c * pd.ih * pd.iw + ih * pd.iw + iw;
                ref_diff_src[iidx] += diff_dst / num_summands;
              }
            }
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(4) schedule(static)
  for (auto n = 0; n < pd.mb; n++)
    for (auto c = 0; c < pd.c; c++)
      for (auto ih = 0; ih < pd.ih; ih++)
        for (auto iw = 0; iw < pd.iw; iw++) {
          int iidx = n * pd.c * pd.ih * pd.iw
              + c * pd.ih * pd.iw + ih * pd.iw + iw;
          EXPECT_NEAR(ref_diff_src[iidx],
                      diff_src_data[map_index(diff_src_d, iidx)],
                      1e-5f);
        }
}

template <typename data_t>
class pooling_bwd_test : public ::testing::TestWithParam<pool_bwd_test_params> {
private:
  std::shared_ptr<tensor::descriptor> src_desc_;
  std::shared_ptr<tensor::descriptor> dst_desc_;
  std::shared_ptr<tensor> dst_;
  std::shared_ptr<tensor> dst2_;
  pool_bwd_test_params p;
  mkldnn::memory::dims padR;
  mkldnn::memory::data_type data_type;

protected:
  virtual void SetUp()
  {
    p = ::testing::TestWithParam<pool_bwd_test_params>::GetParam();
    test_pool_bwd_desc_t pd = p.test_pd;

    ASSERT_TRUE(p.engine_kind == mkldnn::engine::kind::cpu);
    data_type = data_traits<data_t>::data_type;
    ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

    src_desc_.reset(new tensor::descriptor({ pd.mb, pd.c, pd.ih, pd.iw },
        data_type, static_cast<format>(p.diff_src_format)));
    dst_desc_.reset(new tensor::descriptor({ pd.mb, pd.c, pd.oh, pd.ow },
        data_type, static_cast<format>(p.diff_dst_format)));

    padR = { pd.padt, pd.padl };
    for (int i = 0; i < 2; ++i) {
    if ((pd.ih + pd.padt + padR[0] - pd.kh)/pd.strh + 1 < pd.oh) ++padR[0];
    if ((pd.iw + pd.padl + padR[1] - pd.kw)/pd.strw + 1 < pd.ow) ++padR[1];
    }
  }

  void Forward()
  {
    tensor src;
    std::shared_ptr<char> dst_r;
    test_pool_bwd_desc_t pd = p.test_pd;

    auto test = [&]() {
      src.init(*src_desc_);
      fill_data<data_t>(src.get_size()/ sizeof(data_t),
              (data_t *)src.get_data_handle());

      size_t dst_sz = pd.mb * pd.c * pd.oh * pd.ow;
      dst_r.reset(new char [dst_sz * sizeof(data_t)]);

      auto dst_tmp = pooling_forward::compute(src,
          {pd.mb, pd.c, pd.oh, pd.ow}, dst_r.get(), {pd.strh, pd.strw},
          {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR, p.aalgorithm,
          prop_kind::forward_training, padding_kind::zero);
      dst_.reset(new tensor(dst_tmp));

    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
        return;

    check_pool_fwd<data_t>(p, src, *(dst_.get()));
  }

  void Backward()
  {
    tensor diff_src;
    tensor diff_dst;
    std::shared_ptr<char> diff_src_r;
    test_pool_bwd_desc_t pd = p.test_pd;

    auto test = [&]() {
      diff_dst.init(*dst_desc_);
      fill_data<data_t>(diff_dst.get_size() / sizeof(data_t),
              (data_t *)diff_dst.get_data_handle());

      size_t diff_src_sz = pd.mb * pd.c * pd.ih * pd.iw;
      diff_src_r.reset(new char [diff_src_sz * sizeof(data_t)]);

      diff_src = pooling_backward::compute(diff_dst, *(dst_.get()),
          {pd.mb, pd.c, pd.ih, pd.iw}, diff_src_r.get(), src_desc_.get(),
          {pd.strh, pd.strw}, {pd.kh, pd.kw},
          {pd.padt, pd.padl}, padR, p.aalgorithm,
          padding_kind::zero);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    tensor ws;
    bool with_workspace = p.aalgorithm == mkldnn::pooling_max;
    if (!with_workspace) {
      auto ws_desc = tensor::descriptor({}, data_type,
          static_cast<format>(p.diff_dst_format));
      ws.init(ws_desc);
    } else {
      ws = *dst_->get_extra();
    }
    check_pool_bwd<data_t>(p, diff_src, diff_dst, ws);
  }

  void Forward2()
  {
    tensor src;
    test_pool_bwd_desc_t pd = p.test_pd;

    auto test = [&]() {
      src.init(*src_desc_);
      fill_data<data_t>(src.get_size()/ sizeof(data_t),
              (data_t *)src.get_data_handle());

      auto dst_tmp = pooling_forward::compute(src,
          {pd.mb, pd.c, pd.oh, pd.ow}, {pd.strh, pd.strw},
          {pd.kh, pd.kw}, {pd.padt, pd.padl}, padR, p.aalgorithm,
          prop_kind::forward_training, padding_kind::zero);
      dst2_.reset(new tensor(dst_tmp));

    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
        return;

    check_pool_fwd<data_t>(p, src, *(dst2_.get()));
  }

  void Backward2()
  {
    tensor diff_src;
    tensor diff_dst;
    test_pool_bwd_desc_t pd = p.test_pd;

    // dst2_ == NULL ??
    bool with_workspace = (p.aalgorithm == mkldnn::pooling_max)
                          && dst2_;
    auto test = [&]() {
      diff_dst.init(*dst_desc_);
      fill_data<data_t>(diff_dst.get_size() / sizeof(data_t),
              (data_t *)diff_dst.get_data_handle());

      diff_src = pooling_backward::compute(diff_dst,
          with_workspace ? dst2_->get_extra() : NULL,
          {pd.mb, pd.c, pd.ih, pd.iw}, src_desc_.get(),
          {pd.strh, pd.strw}, {pd.kh, pd.kw},
          {pd.padt, pd.padl}, padR, p.aalgorithm,
          padding_kind::zero);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    tensor ws;
    if (!with_workspace) {
      auto ws_desc = tensor::descriptor({}, data_type,
          static_cast<format>(p.diff_dst_format));
      ws.init(ws_desc);
    } else {
      ws = *dst2_->get_extra();
    }

    check_pool_bwd<data_t>(p, diff_src, diff_dst, ws);
  }
};

using pooling_bwd_test_float = pooling_bwd_test<float>;
using pool_bwd_test_params_float = pool_bwd_test_params;

TEST_P(pooling_bwd_test_float, TestsPoolingBackward)
{
    Forward();
    Backward();

}

TEST_P(pooling_bwd_test_float, TestsPoolingBackward2)
{
    Forward2();
    Backward2();

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
