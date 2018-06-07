#pragma once

#include "mkldnn_test_common.hpp"
#include <gtest/gtest.h>
#include <cmath>

#include <ideep.hpp>
#include <ideep_pin_singletons.hpp>

namespace ideep {

// Helpers for migrating MKL-DNN test
inline size_t map_index(const mkldnn_memory_desc_t *md, size_t index) {
  using fmt = mkldnn::memory::format;

  const fmt fwd_weights_g_qvnni = fmt::gOIhw8i16o2i;
  const fmt fwd_weights_qvnni = fmt::OIhw8i16o2i;
  const fmt bwd_weights_g_qvnni = fmt::gOIhw8o16i2o;
  const fmt bwd_weights_qvnni = fmt::OIhw8o16i2o;

  const fmt fwd_weights_g_vnni = fmt::gOIhw4i16o4i;
  const fmt fwd_weights_vnni = fmt::OIhw4i16o4i;

  const bool with_groups = (md->format == fwd_weights_g_qvnni)
    || (md->format == bwd_weights_g_qvnni)
    || (md->format == fwd_weights_g_vnni);

  const bool qvnni = (md->format == fwd_weights_g_qvnni)
    || (md->format == bwd_weights_g_qvnni)
    || (md->format == fwd_weights_qvnni)
    || (md->format == bwd_weights_qvnni);

  const bool vnni = (md->format == fwd_weights_g_vnni)
    || (md->format == fwd_weights_vnni);

  const bool fwd_wei = (md->format == fwd_weights_g_qvnni)
    || (md->format == fwd_weights_qvnni)
    || (md->format == fwd_weights_g_vnni)
    || (md->format == fwd_weights_vnni);

  const bool bwd_wei = (md->format == bwd_weights_g_qvnni)
    || (md->format == bwd_weights_qvnni);

  const int ndims = md->ndims;
  const int *dims = md->dims;
  const int *pdims = md->layout_desc.blocking.padding_dims;
  const int *optd = md->layout_desc.blocking.offset_padding_to_data;

  auto *strides_block = md->layout_desc.blocking.strides[0];
  auto *strides_within_block = md->layout_desc.blocking.strides[1];

  size_t ph_index = 0;
  size_t oc_lb = 0, ic_sb = 0,
      oc_sb = 0, ic_lb = 0;

  for (int rd = 0; rd < ndims; ++rd) {
      int d = ndims - rd - 1;

      EXPECT_LE(dims[d], pdims[d]);

      int cur_dim = dims[d];
      EXPECT_GT(cur_dim, 0);
      int cur_block = md->layout_desc.blocking.block_dims[d];

      size_t pos_d = /*static_cast<ssize_t>*/(index % cur_dim);
      EXPECT_GE(optd[d], 0);
      size_t cur_pos = optd[d] + pos_d;

      size_t cur_pos_block = cur_pos / cur_block;
      size_t cur_pos_within_block = cur_pos % cur_block;

      if (d == (with_groups + 0)) {
        if (qvnni) { oc_lb = pos_d % 16; oc_sb = pos_d % 2; }
        else if (vnni) {oc_lb = pos_d % 16;}
      }
      if (d == (with_groups + 1)) {
        if (qvnni) { ic_sb = pos_d %2; ic_lb = pos_d % 16; }
        else if (vnni) { ic_sb = pos_d % 4; }
      }

      ph_index += cur_pos_block*strides_block[d];
      ph_index += cur_pos_within_block*strides_within_block[d];

      index /= cur_dim;
  }
  int scale = (vnni) ? 3:1;
  if (fwd_wei) {
    //ph_index += -16 * ic_2 + oc_16 + ic_2;
    ph_index += scale * oc_lb + ic_sb;
    EXPECT_GE(ph_index, 16*ic_sb);
    ph_index -= 16*ic_sb;
  } else if (bwd_wei) {
    //ph_index += -16 * oc_2 + ic_16 + oc_2;
    ph_index += ic_lb + oc_sb;
    EXPECT_GE(ph_index, 16 * oc_sb);
    ph_index -= 16 * oc_sb;
  }
  ph_index += md->layout_desc.blocking.offset_padding;

  return ph_index;
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_fwd(const test_convolution_sizes_t &c,
    const test_convolution_attr_t &attr, const tensor &src,
    const tensor &weights, const tensor &bias, const tensor &dst)
{
  const bool w_bias = dst.get_internal_format() != format::format_undef;
  data_t_src *src_data = (data_t_src *)src.get_data_handle();
  data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();

  data_t_dst *bias_data = w_bias ? (data_t_dst *)bias.get_data_handle() : nullptr;
  data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();
  const auto *src_d = src.get_mkldnn_memory_desc_t();
  const auto *weights_d = weights.get_mkldnn_memory_desc_t();
  const auto *bias_d = bias.get_mkldnn_memory_desc_t();
  const auto *dst_d = dst.get_mkldnn_memory_desc_t();

#pragma omp parallel for collapse(5) schedule(static)
  for (int n = 0; n < c.mb; n++) {
    for (int g = 0; g < c.ng; g++) {
      for (int oc = 0; oc < c.oc / c.ng; oc++) {
        for (int oh = 0; oh < c.oh; oh++) {
          for (int ow = 0; ow < c.ow; ow++) {
            data_t_acc a = 0;
            for (int ic = 0; ic < c.ic / c.ng; ic++) {
              for (int kh = 0; kh < c.kh; kh++) {
                for (int kw = 0; kw < c.kw; kw++) {
                  int iw = ow * c.strw
                        - c.padw + kw * (1 + c.dilw);
                  int ih = oh * c.strh
                        - c.padh + kh * (1 + c.dilh);
                  if (iw < 0 || iw >= c.iw) continue;
                  if (ih < 0 || ih >= c.ih) continue;
                  int iidx = n * c.ic * c.ih * c.iw
                          + g * c.ic / c.ng * c.ih * c.iw
                          + ic * c.ih * c.iw + ih * c.iw + iw;
                  int widx = g * c.oc / c.ng * c.ic
                                  / c.ng * c.kh * c.kw
                          + oc * c.ic / c.ng * c.kh * c.kw
                          + ic * c.kh * c.kw + kh * c.kw + kw;
                  a += ((data_t_acc) src_data[map_index(src_d, iidx)])
                    * weights_data[map_index(weights_d, widx)];
                }
              }
            }

            float a_fp = (float)a;

            a_fp += (float)(bias_data ?
                bias_data[map_index(bias_d, g * c.oc / c.ng + oc)] : 0);

            if (attr.oscale.is_def()) {
              const auto &s = attr.oscale;
              using P = test_convolution_attr_t::scale_t;
              if (s.policy == P::policy_t::COMMON) {
                  a_fp *= s.scale;
              }
            }

            if (data_traits<data_t_dst>::data_type != tensor::data_type::f32) {
              using R = mkldnn::round_mode;
              switch (attr.rmode) {
                case R::round_down: a_fp = floorf(a_fp); break;
                case R::round_nearest: a_fp = nearbyintf(a_fp); break;
              }
            }

            int oidx = n * c.oc * c.oh * c.ow
                   + g * c.oc / c.ng * c.oh * c.ow
                   + oc * c.oh * c.ow + oh * c.ow + ow;
            dst_data[map_index(dst_d, oidx)] = (data_t_dst)a_fp;
          }
        }
      }
    }
  }
}

template <typename data_t_diff_dst, typename data_t_wei,
          typename data_t_acc, typename data_t_diff_src>
void compute_ref_conv_bwd_data(const test_convolution_sizes_t &c,
        const tensor& diff_src, const tensor& weights, const tensor& diff_dst)
{
    data_t_diff_dst *diff_dst_data = (data_t_diff_dst *)diff_dst.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
    data_t_diff_src *diff_src_data = (data_t_diff_src *)diff_src.get_data_handle();

    const auto *diff_src_d = diff_src.get_mkldnn_memory_desc_t();
    const auto *weights_d = weights.get_mkldnn_memory_desc_t();
    const auto *diff_dst_d = diff_dst.get_mkldnn_memory_desc_t();

# pragma omp parallel for collapse(5) schedule(static)
  for (int mb = 0; mb < c.mb; ++mb) {
    for (int g = 0; g < c.ng; ++g) {
      for (int ic = 0; ic < c.ic / c.ng; ++ic) {
        for (int ih = 0; ih < c.ih; ++ih) {
          for (int iw = 0; iw < c.iw; ++iw) {
            int sidx = mb * c.ic * c.ih * c.iw
                    + g * c.ic / c.ng * c.ih * c.iw
                    + ic * c.ih * c.iw + ih * c.iw + iw;
            data_t_acc a = data_t_acc(0);
            for (int oc = 0; oc < c.oc / c.ng; oc++) {
              for (int kh = 0; kh < c.kh; kh++) {
                for (int kw = 0; kw < c.kw; kw++) {
                  if (iw + c.padw < kw * (1 + c.dilw) ||
                      ih + c.padh < kh * (1 + c.dilh))
                    continue;
                  int ow = iw - kw * (1 + c.dilw) + c.padw;
                  int oh = ih - kh * (1 + c.dilh) + c.padh;
                  if (ow % c.strw != 0 || oh % c.strh != 0)
                    continue;
                  ow /= c.strw;
                  oh /= c.strh;
                  if (oh < c.oh && ow < c.ow) {
                    int didx = mb * c.oc * c.oh * c.ow
                      + g * c.oc / c.ng * c.oh * c.ow
                      + oc * c.oh * c.ow + oh * c.ow
                      + ow;
                    int widx = g * c.oc / c.ng * c.ic
                      / c.ng * c.kh * c.kw
                      + oc * c.ic / c.ng * c.kh * c.kw
                      + ic * c.kh * c.kw + kh * c.kw
                      + kw;

                    a += (data_t_acc)(
                      diff_dst_data[map_index(diff_dst_d, didx)]
                      * weights_data[map_index(weights_d, widx)]);
                  }
                }
              }
            }
            diff_src_data[map_index(diff_src_d, sidx)] = (data_t_diff_src)a;
          }
        }
      }
    }
  }
}

template <typename data_t>
void compute_ref_conv_bwd_bias(const test_convolution_sizes_t &c,
        const tensor& diff_dst, const tensor& diff_bias)
{
  data_t *diff_bias_data = (data_t *)diff_bias.get_data_handle();
  data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

  const auto *bias_d = diff_bias.get_mkldnn_memory_desc_t();
  const auto *dst_d = diff_dst.get_mkldnn_memory_desc_t();

# pragma omp parallel for collapse(2) schedule(static)
  for (int g = 0; g < c.ng; ++g) {
    for (int oc = 0; oc < c.oc / c.ng; ++oc) {
      int bidx = g * c.oc / c.ng + oc;
      diff_bias_data[map_index(bias_d, bidx)] = 0.0;
      for (int mb = 0; mb < c.mb; ++mb) {
        for (int oh = 0; oh < c.oh; ++oh) {
          for (int ow = 0; ow < c.ow; ++ow) {
            int oidx = mb * c.oc * c.oh * c.ow
                    + g * c.oc / c.ng * c.oh * c.ow
                    + oc * c.oh * c.ow + oh * c.ow + ow;
            diff_bias_data[map_index(bias_d, bidx)]
                += diff_dst_data[map_index(dst_d, oidx)];
          }
        }
      }
    }
  }
}

template <typename data_t>
void compute_ref_conv_bwd_weights(const test_convolution_sizes_t &c,
        const tensor& src, const tensor& diff_dst, const tensor& diff_weights)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *diff_weights_data = (data_t *)diff_weights.get_data_handle();
  data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

  const auto *src_d = src.get_mkldnn_memory_desc_t();
  const auto *weights_d = diff_weights.get_mkldnn_memory_desc_t();
  const auto *dst_d = diff_dst.get_mkldnn_memory_desc_t();

# pragma omp parallel for collapse(5) schedule(static)
  for (int g = 0; g < c.ng; ++g) {
    for (int oc = 0; oc < c.oc / c.ng; oc++) {
      for (int ic = 0; ic < c.ic / c.ng; ++ic) {
        for (int kh = 0; kh < c.kh; kh++) {
          for (int kw = 0; kw < c.kw; kw++) {
            int widx = g * c.oc / c.ng * c.ic / c.ng * c.kh * c.kw
              + oc * c.ic / c.ng * c.kh * c.kw
              + ic * c.kh * c.kw + kh * c.kw + kw;
            diff_weights_data[map_index(weights_d, widx)] = 0.0;
            for (int mb = 0; mb < c.mb; ++mb) {
              for (int oh = 0; oh < c.oh; ++oh) {
                for (int ow = 0; ow < c.ow; ++ow) {
                  if (ow*c.strw + kw * (1 + c.dilw) < c.padw ||
                      oh*c.strh + kh * (1 + c.dilh) < c.padh ||
                      ow*c.strw + kw * (1 + c.dilw) >= c.iw + c.padw ||
                      oh*c.strh + kh * (1 + c.dilh)>= c.ih + c.padh)
                      continue;

                  int ih = oh * c.strh - c.padh + kh * (1 + c.dilh);
                  int iw = ow * c.strw - c.padw + kw * (1 + c.dilw);
                  int sidx = mb * c.ic * c.ih * c.iw
                    + g * c.ic / c.ng * c.ih * c.iw
                    + ic * c.ih * c.iw + ih * c.iw + iw;
                  int didx = mb * c.oc * c.oh * c.ow
                    + g * c.oc / c.ng * c.oh * c.ow
                    + oc * c.oh * c.ow + oh * c.ow + ow;

                  diff_weights_data[map_index(weights_d, widx)]
                    += src_data[map_index(src_d, sidx)]
                    * diff_dst_data[map_index(dst_d, didx)];
                }
              }
            }
          }
        }
      }
    }
  }
}

enum {ACROSS = 0, WITHIN = 1};

struct test_lrn_desc_t {
  int mb, c, h, w;
  float alpha, beta, k;
  int local_size, kind;
};

struct lrn_test_params {
  prop_kind aprop_kind;
  const engine::kind engine_kind;
  algorithm aalgorithm;
  mkldnn::memory::format src_format;
  mkldnn::memory::format dst_format;
  test_lrn_desc_t test_ld;
};

template <typename data_t>
void check_lrn_fwd(const test_lrn_desc_t &ld, const tensor& src,
    const tensor& dst) {
  data_t *src_ptr = (data_t *)src.get_data_handle();
  data_t *dst_ptr = (data_t *)dst.get_data_handle();

  auto *src_d = src.get_mkldnn_memory_desc_t();
  auto *dst_d = dst.get_mkldnn_memory_desc_t();

  const int C = ld.c;
  const int H = ld.h;
  const int W = ld.w;
  const int size = ld.local_size;
  const int CSIZE = ld.kind == ACROSS ? size : 1;
  const int HWSIZE = size + 1 - CSIZE;
  const int summands = ld.kind == ACROSS ? size : size*size;

  auto off = [=](int n, int c, int h, int w) {
    return ((n * ld.c + c) * ld.h + h) * ld.w + w;
  };

  auto ker = [=](data_t *d, int n, int oc, int oh, int ow) {
    data_t sum = 0.0;
    for (int c = oc; c < oc + CSIZE; ++c) {
      if (c < (CSIZE - 1) / 2)
        continue;
      if (c >= C + (CSIZE - 1) / 2)
        continue;
      for (int h = oh; h < oh + HWSIZE; ++h) {
        if (h < (HWSIZE - 1) / 2)
          continue;
        if (h >= H + (HWSIZE - 1) / 2)
          continue;
        for (int w = ow; w < ow + HWSIZE; ++w) {
          if (w < (HWSIZE - 1) / 2)
            continue;
          if (w >= W + (HWSIZE - 1) / 2)
            continue;
          data_t s = src_ptr[map_index(src_d,off(n, c - (CSIZE - 1) / 2,
                h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2))];
          sum += s * s;
        }
      }
    }
    data_t norm_coef = powf(static_cast<float>(ld.k + ld.alpha * sum / summands),
        static_cast<float>(ld.beta));
    data_t ref_out = src_ptr[map_index(src_d, off(n, oc, oh, ow))]/norm_coef;
    data_t eps = static_cast<data_t>(1.e-7f*(2*summands+5));
    data_t out = d[0];
    data_t norm_max = std::max(fabs(out), fabs(ref_out));
    if (norm_max < eps) norm_max = 1.;
    EXPECT_NEAR(out, ref_out, eps*norm_max);
  };

  const int N = ld.mb;
# pragma omp parallel for collapse(4) schedule(static)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          ker(&dst_ptr[map_index(dst_d,off(n, c, h, w))], n, c, h, w);
        }
      }
    }
  }
}

template <typename data_t>
void check_lrn_bwd(const lrn_test_params &p, const tensor& src,
    const tensor& diff_dst, const tensor& diff_src) {
  data_t *src_ptr = (data_t *)src.get_data_handle();
  data_t *diff_dst_ptr = (data_t *)diff_dst.get_data_handle();
  data_t *diff_src_ptr = (data_t *)diff_src.get_data_handle();

  const int MB = p.test_ld.mb;
  const int C = p.test_ld.c;
  const int H = p.test_ld.h;
  const int W = p.test_ld.w;
  const int local_size = p.test_ld.local_size;

  data_t *ref_diff_src_ptr = new data_t[MB*C*H*W];

  const auto *src_d = src.get_mkldnn_memory_desc_t();
  const auto *diff_dst_d = diff_dst.get_mkldnn_memory_desc_t();
  const auto *diff_src_d = diff_src.get_mkldnn_memory_desc_t();

  auto off = [=](int n, int c, int h, int w) {
    return ((n * C + c) * H + h) * W + w;
  };

  auto get_omega = [=](data_t c_k, int kernel_size, float alpha, int C,
      const data_t *src, int n, int c, int h, int w) {
    data_t sum = 0.0;

    int half_kernel_size = (kernel_size - 1) / 2;
    int c_start = (c < half_kernel_size) ? 0 : c - half_kernel_size;
    int c_end = c + kernel_size - half_kernel_size;
    c_end = c_end < C ? c_end : C;
    for (int i = c_start; i < c_end; ++i) {
      data_t value = src[map_index(src_d, off(n, i, h, w))];
      sum += value * value;
    }
    sum *= alpha / kernel_size;
    return c_k + sum;
  };

  auto ker = [=](data_t *d, int mb, int oc, int oh, int ow) {
    const float alpha = p.test_ld.alpha;
    const float beta = p.test_ld.beta;
    const float k = p.test_ld.k;
    const int kernel_size = p.test_ld.local_size;
    int ks_start = kernel_size/2 > oc ? kernel_size/2 - oc : 0;
    int ks_stop = C - oc <= kernel_size/2 ?
      C - oc + kernel_size/2 : kernel_size;

    data_t A = 0, B = 0, omega_mid = 0;

    for (int ks = ks_start; ks < ks_stop; ks++) {
      int _t = oc + ks - (kernel_size/2);
      data_t omega = get_omega(static_cast<data_t>(k), kernel_size, alpha, C,
              src_ptr, mb, _t, oh, ow);

      if (ks == kernel_size/2) omega_mid = omega;

      data_t t = src_ptr[map_index(src_d, off(mb, _t, oh, ow))]
        / powf((float)omega, (float)beta);
      B +=  (1.0f / omega) * t * diff_dst_ptr[
        map_index(diff_dst_d, off(mb, _t, oh, ow))];
    }

    A = (1.0f / powf((float)omega_mid, (float)beta))
        * diff_dst_ptr[map_index(diff_dst_d, off(mb, oc, oh, ow))];
    B *= src_ptr[map_index(src_d, off(mb, oc, oh, ow))];
    B *= (2.0f * alpha * beta) / kernel_size;
    *d = A - B;
  };

# pragma omp parallel for collapse(4) schedule(static)
  for (int mb = 0; mb < MB; ++mb) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          ker(&ref_diff_src_ptr[map_index(diff_src_d, off(mb, c, h, w))],
                  mb, c, h, w);
          auto A = ref_diff_src_ptr[map_index(diff_src_d, off(mb, c, h, w))];
          auto B = diff_src_ptr[map_index(diff_src_d, off(mb, c, h, w))];
          data_t eps = static_cast<data_t>(1.e-6*((2*(2*local_size + 3) + 6)
                *local_size + (2*local_size + 3) + 9) );
          data_t norm_max = std::max(fabs(A), fabs(B));
          if (norm_max < eps) norm_max = 1.;
          EXPECT_NEAR(A, B, eps*norm_max);
        }
      }
    }
  }
}

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
void check_pool_fwd(const pool_test_params &p, const tensor &src,
        const tensor &dst) {
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *dst_data = (data_t *)dst.get_data_handle();
  auto *ws = dst.get_extra();

  auto ws_data = [=](size_t idx) -> int {
    auto w = (unsigned char *)ws->get_data_handle();
    if (w == nullptr) return -1;
    if (ws->get_mkldnn_memory_desc_t()->data_type == mkldnn_u8)
      return (int)w[idx];
    else
      return ((int *)w)[idx];
  };

  const auto *src_d = src.get_mkldnn_memory_desc_t();
  const auto *dst_d = dst.get_mkldnn_memory_desc_t();
  const auto *ws_d = ws == nullptr? nullptr : ws->get_mkldnn_memory_desc_t();

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
void check_pool_fwd(const pool_bwd_test_params &p, const tensor& src,
        const tensor& dst)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *dst_data = (data_t *)dst.get_data_handle();

  const auto *src_d = src.get_mkldnn_memory_desc_t();
  const auto *dst_d = dst.get_mkldnn_memory_desc_t();

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

          auto num_summands =
            (p.aalgorithm != mkldnn::pooling_avg_exclude_padding)
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
              } else if (p.aalgorithm ==
                mkldnn::pooling_avg_include_padding ||
                p.aalgorithm == mkldnn::pooling_avg_exclude_padding) {
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
void check_pool_bwd(const pool_bwd_test_params &p, const tensor &gradx,
        const tensor &grady, const tensor &y)
{
  data_t *diff_src_data = (data_t *)gradx.get_data_handle();
  data_t *diff_dst_data = (data_t *)grady.get_data_handle();
  auto *ws = y.get_extra();

  auto ws_data = [=](size_t idx) -> int {
    auto w = (unsigned char *)ws->get_data_handle();
    if (w == nullptr) return -1;
    if (ws->get_data_type() == mkldnn_u8)
      return (int)w[idx];
    else
      return ((int *)w)[idx];
  };

  const auto *diff_src_d = gradx.get_mkldnn_memory_desc_t();
  const auto *diff_dst_d = grady.get_mkldnn_memory_desc_t();
  const auto *ws_d = ws == nullptr ? nullptr : ws->get_mkldnn_memory_desc_t();

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

struct test_bnrm_sizes_t {
  int mb, c, d, h, w;
};

struct test_bnrm_formats_t {
  format data_format;
  format diff_format;
};

struct test_bnrm_params_t {
  mkldnn::engine::kind engine_kind;
  test_bnrm_formats_t formats;
  test_bnrm_sizes_t sizes;
  float eps;
  int ndims;
};

template <typename data_t>
void check_bnrm_fwd(const test_bnrm_params_t &p,
        const tensor& src, const tensor& mean, const tensor& variance,
        const tensor& scale, const tensor& shift, const tensor& dst,
        unsigned flags, prop_kind pk) {
  const bool use_weights = flags & mkldnn::use_scale_shift;
  const bool calculate_stats = !(flags & mkldnn::use_global_stats);
  const bool is_training = (pk == prop_kind::forward_training);

  const data_t *src_data = (const data_t *)src.get_data_handle();

  const auto *scale_data = use_weights ?
    reinterpret_cast<const data_t *>(scale.get_data_handle()) : nullptr;
  const auto *shift_data = use_weights ?
    reinterpret_cast<const data_t *>(shift.get_data_handle()) : nullptr;

  const data_t *mean_data = (!calculate_stats || is_training) ?
         (const data_t *)mean.get_data_handle() : nullptr;
  const data_t *variance_data = (!calculate_stats || is_training) ?
         (const data_t *)variance.get_data_handle() : nullptr;
  const data_t *dst_data = (data_t *)dst.get_data_handle();

  const auto *src_d = src.get_mkldnn_memory_desc_t();
  const auto* dst_d = dst.get_mkldnn_memory_desc_t();

  test_bnrm_sizes_t bp = p.sizes;
  data_t eps = static_cast<data_t>(1.e-4 * bp.mb * bp.d * bp.h * bp.w);

#pragma omp parallel for
  for (int c = 0; c < bp.c; c++) {
    data_t ref_mean = calculate_stats ? data_t(0) : mean_data[c];
    data_t ref_variance = calculate_stats ? data_t(0) : variance_data[c];
    if (calculate_stats) {
      for (int n = 0; n < bp.mb; n++)
      for (int d = 0; d < bp.d; d++)
      for (int h = 0; h < bp.h; h++)
      for (int w = 0; w < bp.w; w++) {
        int sidx = n * bp.c * bp.d * bp.h * bp.w
          + c * bp.d * bp.h * bp.w
          + d * bp.h * bp.w + h * bp.w + w;
        ref_mean += src_data[map_index(src_d, sidx)];
      }
      ref_mean /= bp.mb * bp.d * bp.h * bp.w;
      if (is_training) {
        data_t mean_norm_max = std::max(fabs(mean_data[c]), fabs(ref_mean));
        if (mean_norm_max < eps) mean_norm_max = data_t(1);
        EXPECT_NEAR((mean_data[c] - ref_mean) / mean_norm_max, 0., eps);
      }

      for (int n = 0; n < bp.mb; n++)
      for (int d = 0; d < bp.d; d++)
      for (int h = 0; h < bp.h; h++)
      for (int w = 0; w < bp.w; w++) {
        int sidx = n * bp.c * bp.d * bp.h * bp.w
          + c * bp.d * bp.h * bp.w + d * bp.h * bp. w + h * bp.w + w;
        data_t tmp = src_data[map_index(src_d, sidx)] - ref_mean;
        ref_variance += tmp * tmp;
      }
      ref_variance /= bp.mb * bp.d * bp.h * bp.w;
      if (is_training) {
          data_t variance_norm_max = std::max(fabs(variance_data[c]),
              fabs(ref_variance));
          if (variance_norm_max < eps) variance_norm_max = data_t(1);
          EXPECT_NEAR((variance_data[c] - ref_variance) /
              variance_norm_max, 0., eps);
      }
    }
    data_t ref_sqrt_variance = static_cast<data_t>(sqrt(ref_variance + p.eps));
    data_t ref_rsqrt_variance = data_t(1) / (ref_sqrt_variance);

    if (use_weights) {
      auto *scale_d = scale.get_mkldnn_memory_desc_t();
      auto *shift_d = shift.get_mkldnn_memory_desc_t();
      for (int n = 0; n < bp.mb; n++)
      for (int d = 0; d < bp.d; d++)
      for (int h = 0; h < bp.h; h++)
      for (int w = 0; w < bp.w; w++) {
        int sdidx = n * bp.c * bp.d * bp.h * bp.w
          + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
        data_t ref_dst = scale_data[map_index(scale_d, c)]
                * (src_data[map_index(src_d, sdidx)]
                - ref_mean) * ref_rsqrt_variance
                + shift_data[map_index(shift_d, c)];
        data_t out = dst_data[map_index(dst_d, sdidx)];
        data_t norm_max = std::max(fabs(out), fabs(ref_dst));
        if (norm_max < 10e-3) norm_max = data_t(1);
        EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
      }
    } else {
      for (int n = 0; n < bp.mb; n++)
      for (int d = 0; d < bp.d; d++)
      for (int h = 0; h < bp.h; h++)
      for (int w = 0; w < bp.w; w++) {
        int sdidx = n * bp.c * bp.d * bp.h * bp.w
          + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
        data_t ref_dst = (src_data[map_index(src_d, sdidx)]
                - ref_mean) * ref_rsqrt_variance;
        data_t out = dst_data[map_index(dst_d, sdidx)];
        data_t norm_max = std::max(fabs(out), fabs(ref_dst));
        if (norm_max < 10e-3) norm_max = data_t(1);
        EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
      }
    }
  }
}

template <typename data_t>
void check_bnrm_bwd(const test_bnrm_params_t &p,
        const tensor& src, const tensor& diff_dst, const tensor& mean,
        const tensor& variance, const tensor& scale, const tensor& diff_src,
        const tensor& diff_scale, const tensor& diff_shift,
        unsigned flags, prop_kind pk)
{
  const bool use_weights = flags & mkldnn::use_scale_shift;
  const bool calculate_diff_stats = !(flags & mkldnn::omit_stats);

  const data_t *src_data = (const data_t *)src.get_data_handle();
  const data_t *scale_data = use_weights ?
    (const data_t *)scale.get_data_handle() : nullptr;
  const data_t *diff_dst_data = (const data_t *)diff_dst.get_data_handle();
  const data_t *mean_data = (const data_t *)mean.get_data_handle();
  const data_t *variance_data = (const data_t *)variance.get_data_handle();
  const data_t *diff_src_data = (data_t *)diff_src.get_data_handle();

  const data_t *diff_scale_data = (pk == prop_kind::backward) ?
    reinterpret_cast<data_t *>(diff_scale.get_data_handle()) : nullptr;
  const auto *diff_shift_data = (pk == prop_kind::backward) ?
    reinterpret_cast<data_t *>(diff_shift.get_data_handle()) : nullptr;

  const auto* src_d = src.get_mkldnn_memory_desc_t();
  const auto* diff_dst_d = diff_dst.get_mkldnn_memory_desc_t();
  const auto* scale_d = scale.get_mkldnn_memory_desc_t();
  const auto* diff_src_d = diff_src.get_mkldnn_memory_desc_t();
  const auto* diff_scale_d = diff_scale.get_mkldnn_memory_desc_t();
  const auto* diff_shift_d = diff_shift.get_mkldnn_memory_desc_t();

  test_bnrm_sizes_t bp = p.sizes;

  const data_t eps = static_cast<data_t>(1.e-4 * bp.mb * bp.d * bp.h * bp.w);

#pragma omp parallel for
  for (int c = 0; c < bp.c; c++) {
    data_t ref_diff_gamma = data_t(0);
    data_t ref_diff_beta = data_t(0);

    auto v_mean = mean_data[c];
    auto v_variance = variance_data[c];
    const data_t sqrt_variance = data_t(1.0 / sqrt(v_variance + p.eps));

    auto gamma = use_weights ? scale_data[map_index(scale_d, c)] : 1;

    for (int n = 0; n < bp.mb; n++)
    for (int d = 0; d < bp.d; d++)
    for (int h = 0; h < bp.h; h++)
    for (int w = 0; w < bp.w; w++) {
      int sidx = n * bp.c * bp.d * bp.h * bp.w + c * bp.d * bp.h * bp.w
              + d * bp.h * bp.w + h * bp.w + w;
      ref_diff_gamma += (src_data[map_index(src_d, sidx)] - v_mean)
          * diff_dst_data[map_index(diff_dst_d, sidx)];
      ref_diff_beta += diff_dst_data[map_index(diff_dst_d, sidx)];
    }
    ref_diff_gamma *= sqrt_variance;

    if (pk == prop_kind::backward) {
      auto diff_gamma = diff_scale_data[map_index(diff_scale_d, c)];
      data_t norm_max = std::max(fabs(diff_gamma), fabs(ref_diff_gamma));
      if (norm_max < 10e-3) norm_max = data_t(1);
      EXPECT_NEAR((diff_gamma - ref_diff_gamma) / norm_max, 0., eps);

      auto diff_beta = diff_shift_data[map_index(diff_shift_d, c)];
      norm_max = std::max(fabs(diff_beta), fabs(ref_diff_beta));
      if (norm_max < 10e-3) norm_max = data_t(1);
      EXPECT_NEAR((diff_beta - ref_diff_beta) / norm_max, 0., eps);
    }

    for (int n = 0; n < bp.mb; n++)
    for (int d = 0; d < bp.d; d++)
    for (int h = 0; h < bp.h; h++)
    for (int w = 0; w < bp.w; w++) {
      int sidx = n * bp.c * bp.d * bp.h * bp.w
        + c * bp.d * bp.h * bp.w + d * bp.h * bp.w + h * bp.w + w;
      data_t ref_diff_src = diff_dst_data[map_index(diff_dst_d, sidx)];
      if (calculate_diff_stats) {
        ref_diff_src -= ref_diff_beta/(bp.mb*bp.d*bp.h*bp.w)
        + (src_data[map_index(src_d, sidx)] - v_mean)
        *ref_diff_gamma*sqrt_variance/(bp.mb*bp.d*bp.h*bp.w);
      }
      ref_diff_src *= gamma*sqrt_variance;
      data_t out_diff_src = diff_src_data[map_index(diff_src_d, sidx)];
      data_t norm_max = std::max(fabs(out_diff_src), fabs(ref_diff_src));
      if (norm_max < eps) norm_max = data_t(1);
      EXPECT_NEAR((out_diff_src - ref_diff_src) / norm_max, 0., eps);
    }
  }
}

struct test_inner_product_descr_t {
  int mb, ic, oc, kh, kw;
};

struct inprod_test_forward_params {
  prop_kind aprop_kind;
  const engine::kind engine_kind;
  mkldnn::memory::format src_format;
  mkldnn::memory::format weights_format;
  mkldnn::memory::format bias_format;
  mkldnn::memory::format dst_format;
  test_inner_product_descr_t test_ipd;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

struct inprod_test_bwd_data_params {
  const engine::kind engine_kind;
  mkldnn::memory::format diff_src_format;
  mkldnn::memory::format weights_format;
  mkldnn::memory::format diff_dst_format;
  test_inner_product_descr_t test_ipd;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

struct inprod_test_bwd_weights_params {
  const engine::kind engine_kind;
  mkldnn::memory::format src_format;
  mkldnn::memory::format diff_weights_format;
  mkldnn::memory::format diff_bias_format;
  mkldnn::memory::format diff_dst_format;
  test_inner_product_descr_t test_ipd;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

template <typename data_t>
void compute_ref_inner_product_fwd(test_inner_product_descr_t ipd,
    const tensor& src, const tensor& weights,
    const tensor& bias, const tensor& dst)
{
  const bool w_bias = (bias.get_internal_format() != format::format_undef);
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *weights_data = (data_t *)weights.get_data_handle();
  data_t *bias_data = w_bias ? (data_t *)bias.get_data_handle() : nullptr;
  data_t *dst_data = (data_t *)dst.get_data_handle();

  const auto* src_d = src.get_mkldnn_memory_desc_t();
  const auto* weights_d = weights.get_mkldnn_memory_desc_t();
  const auto* bias_d = bias.get_mkldnn_memory_desc_t();
  const auto* dst_d = dst.get_mkldnn_memory_desc_t();

#pragma omp parallel for collapse(2) schedule(static)
  for (int n = 0; n < ipd.mb; n++) {
    for (int oc = 0; oc < ipd.oc; oc++) {
      int oidx = n * ipd.oc + oc;
      dst_data[map_index(dst_d, oidx)] = bias_data ?
        bias_data[map_index(bias_d, oc)] : data_t{0};
      for (int ic = 0; ic < ipd.ic; ic++) {
        for (int kh = 0; kh < ipd.kh; kh++) {
          for (int kw = 0; kw < ipd.kw; kw++) {
            int iidx = n * ipd.ic * ipd.kh * ipd.kw
                    + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
            int widx = oc * ipd.ic * ipd.kh * ipd.kw
                    + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
            dst_data[map_index(dst_d, oidx)]
                    += src_data[map_index(src_d, iidx)]
                    * weights_data[map_index(weights_d, widx)];
          }
        }
      }
    }
  }
}

template <typename data_t>
void compute_ref_inner_product_bwd_data(const test_inner_product_descr_t &ipd,
        const tensor& diff_dst, const tensor& weights, const tensor& diff_src)
{
  data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();
  data_t *weights_data = (data_t *)weights.get_data_handle();
  data_t *diff_src_data = (data_t *)diff_src.get_data_handle();

  const auto* diff_dst_d = diff_dst.get_mkldnn_memory_desc_t();
  const auto* weights_d = weights.get_mkldnn_memory_desc_t();
  const auto* diff_src_d = diff_src.get_mkldnn_memory_desc_t();

  bool has_spatial = ipd.kh > 1 && ipd.kw > 1;

#pragma omp parallel for collapse(2) schedule(static)
  for (int n = 0; n < ipd.mb; n++) {
    for (int ic = 0; ic < ipd.ic; ic++) {
      if (has_spatial) {
        for (int kh = 0; kh < ipd.kh; ++kh) {
          for (int kw = 0; kw < ipd.kw; ++kw) {
            int dsidx = n * ipd.ic * ipd.kh * ipd.kw
                    + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
            data_t *ds = &diff_src_data[map_index(diff_src_d, dsidx)];
            *ds = data_t(0);
            for (int oc = 0; oc < ipd.oc; ++oc) {
              int ddidx = n * ipd.oc + oc;
              int widx = oc * ipd.ic * ipd.kh * ipd.kw +
                ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
              *ds += diff_dst_data[map_index(diff_dst_d, ddidx)]
                * weights_data[map_index(weights_d, widx)];
            }
          }
        }
      } else {
        int dsidx = n * ipd.ic + ic;
        data_t *ds = &diff_src_data[map_index(diff_src_d, dsidx)];
        *ds = data_t(0);
        for (int oc = 0; oc < ipd.oc; ++oc) {
          int ddidx = n * ipd.oc + oc;
          int widx = oc * ipd.ic + ic;
          *ds += diff_dst_data[map_index(diff_dst_d, ddidx)]
            * weights_data[map_index(weights_d, widx)];
        }
      }
    }
  }
}

template <typename data_t>
void compute_ref_inner_product_bwd_bias(const test_inner_product_descr_t &ipd,
        const tensor& diff_dst, const tensor& diff_bias) {
  data_t *diff_bias_data = (data_t *)diff_bias.get_data_handle();
  data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

  const auto* diff_bias_d = diff_bias.get_mkldnn_memory_desc_t();
  const auto* diff_dst_d = diff_dst.get_mkldnn_memory_desc_t();

# pragma omp parallel for schedule(static)
  for (int oc = 0; oc < ipd.oc; ++oc) {
    data_t *db = &diff_bias_data[map_index(diff_bias_d, oc)];
    *db = data_t(0);
    for (int n = 0; n < ipd.mb; ++n) {
      *db += diff_dst_data[map_index(diff_dst_d, n*ipd.oc + oc)];
    }
  }
}

template <typename data_t>
void compute_ref_inner_product_bwd_weights(const test_inner_product_descr_t &ipd,
        const tensor& src, const tensor& diff_dst, const tensor& diff_weights)
{
  data_t *src_data = (data_t *)src.get_data_handle();
  data_t *diff_weights_data = (data_t *)diff_weights.get_data_handle();
  data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

  const auto* src_d = src.get_mkldnn_memory_desc_t();
  const auto* diff_weights_d = diff_weights.get_mkldnn_memory_desc_t();
  const auto* diff_dst_d = diff_dst.get_mkldnn_memory_desc_t();

  bool has_spatial = ipd.kh > 1 && ipd.kw > 1;

# pragma omp parallel for collapse(2) schedule(static)
  for (int oc = 0; oc < ipd.oc; ++oc) {
    for (int ic = 0; ic < ipd.ic; ++ic) {
      if (has_spatial) {
        for (int kh = 0; kh < ipd.kh; ++kh) {
          for (int kw = 0; kw < ipd.kw; ++kw) {
            int dwidx = oc * ipd.ic * ipd.kh * ipd.kw
                    + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
            data_t *dw = &diff_weights_data[map_index(diff_weights_d, dwidx)];
            *dw = data_t(0);
            for (int n = 0; n < ipd.mb; ++n) {
              int ddidx = n * ipd.oc + oc;
              int sidx = n * ipd.ic * ipd.kh * ipd.kw
                      + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
              *dw += diff_dst_data[map_index(diff_dst_d, ddidx)] *
                  src_data[map_index(src_d, sidx)];
            }
          }
        }
      } else {
        int dwidx = oc * ipd.ic + ic;
        data_t *dw = &diff_weights_data[map_index(diff_weights_d, dwidx)];
        *dw = data_t(0);
        for (int n = 0; n < ipd.mb; ++n) {
          int ddidx = n * ipd.oc + oc;
          int sidx = n * ipd.ic + ic;
          *dw += diff_dst_data[map_index(diff_dst_d, ddidx)] *
              src_data[map_index(src_d, sidx)];
        }
      }
    }
  }
}

template <typename data_i_t, typename data_o_t>
inline void check_reorder(const tensor::descriptor &md_i,
    const tensor::descriptor &md_o, const data_i_t *src, const data_o_t *dst) {
  const auto dims = md_i.get_dims();
  const size_t nelems = std::accumulate(
          dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  auto mkldnn_md_i = md_i.get_mkldnn_memory_desc_t();
  auto mkldnn_md_o = md_o.get_mkldnn_memory_desc_t();

  for (size_t i = 0; i < nelems; ++i) {
    data_i_t s_raw = src[map_index(mkldnn_md_i, i)];
    data_o_t s = static_cast<data_o_t>(s_raw);
    data_o_t d = dst[map_index(mkldnn_md_o, i)];
    ASSERT_EQ(s, d) << "mismatch at position " << i;
  }
}

template <typename reorder_types>
struct test_simple_params {
    engine::kind engine_kind;
    mkldnn::memory::format fmt_i;
    mkldnn::memory::format fmt_o;
    mkldnn::memory::dims dims;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

void fill_tensor(tensor& t) {
  switch (t.get_data_type()) {
    case tensor::data_type::f32:
      fill_data<float>(t.get_size() / sizeof(float),
          reinterpret_cast<float *>(t.get_data_handle()));
      break;
    case tensor::data_type::s32:
      fill_data<int>(t.get_size() / sizeof(float),
          reinterpret_cast<int *>(t.get_data_handle()));
      break;
    default:
      break;
  }
}

template <typename data_t>
static void compare_tensor(const tensor& ref, const tensor& dst) {
  ASSERT_TRUE(data_traits<data_t>::data_type == mkldnn::memory::data_type::f32 ||
      data_traits<data_t>::data_type == mkldnn::memory::data_type::s32);

  auto *ref_desc = ref.get_mkldnn_memory_desc_t();
  auto *dst_desc = dst.get_mkldnn_memory_desc_t();

  ASSERT_TRUE(ref_desc->ndims == dst_desc->ndims);
  auto ndims = ref_desc->ndims;
  for (auto d = 0; d < ndims; ++d) {
    ASSERT_TRUE(ref_desc->dims[d] == dst_desc->dims[d]);
  }

  auto num = std::accumulate(ref_desc->dims, &ref_desc->dims[ref_desc->ndims],
      1, std::multiplies<int>());

  data_t *ref_data = (data_t *)ref.get_data_handle();
  data_t *dst_data = (data_t *)dst.get_data_handle();

# pragma omp parallel for schedule(static)
  for (ptrdiff_t i = 0; i < num; i ++) {
    data_t ref = ref_data[map_index(ref_desc, i)];
    data_t got = dst_data[map_index(dst_desc, i)];

    if (data_traits<data_t>::data_type == mkldnn::memory::data_type::f32) {
      data_t diff = got - ref;
      data_t e = (std::abs(ref) > (data_t)1e-4) ? diff / ref : diff;
      EXPECT_NEAR(e, (data_t)0.0, (data_t)1e-4)
        << "Ref: " << ref << " GOT: " << got
        << " Index: " << i << " Total: " << num;
    } else {
      EXPECT_EQ(ref, got) << "Index: " << i << " Total: " << num;
    }
  }
}

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

  auto *src_d = src.get_mkldnn_memory_desc_t();
  auto *gradx_d = gradx.get_mkldnn_memory_desc_t();
  auto *grady_d = grady.get_mkldnn_memory_desc_t();

  ASSERT_EQ(src.ndims(), 4);
  ASSERT_EQ(grady.ndims(), 4);
  ASSERT_EQ(src.get_data_type(), mkldnn::memory::data_type::f32);
  ASSERT_EQ(grady.get_data_type(), mkldnn::memory::data_type::f32);

  for (size_t i = 0; i < src.get_size() / sizeof(data_t); ++i) {
    data_t ref_x = src_data[map_index(src_d, i)];
    data_t ref_gy = grady_data[map_index(grady_d, i)];
    data_t ref_gx = ref_gy * ((ref_x > 0) ? data_t{1} : negative_slope);
    EXPECT_NEAR(gradx_data[map_index(gradx_d, i)], ref_gx, 1.e-7);
  }
}

struct concat_test_params {
  const engine::kind engine_kind;
  size_t concat_dimension;
  std::vector<mkldnn::memory::format> srcs_format;
  mkldnn::memory::format dst_format;
  std::vector<mkldnn::memory::dims> srcs_cds;
  mkldnn::memory::dims dst_cds;
};

template <typename data_t>
void check_data(const std::vector<tensor> &srcs, const tensor &dst,
      int concat_dim) {
  const data_t *dst_data = (const data_t *)dst.get_data_handle();
  const auto &dst_d = dst.get_mkldnn_memory_desc_t();
  const auto dst_dims = dst.get_dims();

  int acc_concat_dim = 0;

  for (size_t num = 0; num < srcs.size(); num++) {
    const data_t *src_data = (const data_t *)srcs[num].get_data_handle();
    const auto &src_d = srcs[num].get_mkldnn_memory_desc_t();
    const std::vector<int> src_dims = srcs[num].get_dims();
    for (auto n = 0; n < src_dims[0]; n++)
    for (auto c = 0; c < src_dims[1]; c++)
    for (auto h = 0; h < src_dims[2]; h++)
    for (auto w = 0; w < src_dims[3]; w++) {
      auto src_idx = w
        + src_dims[3]*h
        + src_dims[2]*src_dims[3]*c
        + src_dims[1]*src_dims[2]*src_dims[3]*n;

      auto adj_dst_dim = [&](int dim, int dim_sz) {
        if (concat_dim == dim) return dim_sz + acc_concat_dim;
        return dim_sz;
      };
      auto dst_idx = adj_dst_dim(3, w)
        + dst_dims[3]*adj_dst_dim(2, h)
        + dst_dims[2]*dst_dims[3]*adj_dst_dim(1, c)
        + dst_dims[1]*dst_dims[2]*dst_dims[3]*adj_dst_dim(0, n);

      EXPECT_NEAR(src_data[map_index(src_d, src_idx)],
                dst_data[map_index(dst_d, dst_idx)],
                1e-7);
    }

    acc_concat_dim += src_dims[concat_dim];
  }
}

}
