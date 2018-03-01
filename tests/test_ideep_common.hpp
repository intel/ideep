#pragma once

#include "mkldnn_test_common.hpp"
#include <gtest/gtest.h>
#include <cmath>

#include <ideep.hpp>

namespace ideep {

INIT_GLOBAL_ENGINE

// Helpers for migrating MKL-DNN test
inline size_t map_index(const mkldnn_memory_desc_t *md, size_t index) {
  using fmt = mkldnn::memory::format;
  const fmt fwd_weights_g = fmt::gOIhw8i16o2i;
  const fmt fwd_weights = fmt::OIhw8i16o2i;
  const fmt bwd_weights_g = fmt::gOIhw8o16i2o;
  const fmt bwd_weights = fmt::OIhw8o16i2o;

  const bool with_groups = (md->format == fwd_weights_g)
                        || (md->format == bwd_weights_g);

  const int ndims = md->ndims;
  const int *dims = md->dims;
  const int *pdims = md->layout_desc.blocking.padding_dims;
  const int *optd = md->layout_desc.blocking.offset_padding_to_data;

  auto *strides_block = md->layout_desc.blocking.strides[0];
  auto *strides_within_block = md->layout_desc.blocking.strides[1];

  size_t ph_index = 0;
  size_t oc_16 = 0, ic_2 = 0,
      oc_2 = 0, ic_16 = 0;

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

      if (d == (with_groups + 0)) { oc_16 = pos_d % 16; oc_2 = pos_d % 2; }
      if (d == (with_groups + 1)) { ic_2 = pos_d % 2; ic_16 = pos_d % 16; }

      ph_index += cur_pos_block*strides_block[d];
      ph_index += cur_pos_within_block*strides_within_block[d];

      index /= cur_dim;
  }
  if (md->format == fwd_weights_g || md->format == fwd_weights) {
      //ph_index += -16 * ic_2 + oc_16 + ic_2;
      ph_index += oc_16 + ic_2;
      EXPECT_GE(ph_index, 16*ic_2);
      ph_index -= 16*ic_2;
  } else
      if (md->format == bwd_weights_g || md->format == bwd_weights) {
          //ph_index += -16 * oc_2 + ic_16 + oc_2;
          ph_index += ic_16 + oc_2;
          EXPECT_GE(ph_index, 16 * oc_2);
          ph_index -= 16 * oc_2;
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
}
