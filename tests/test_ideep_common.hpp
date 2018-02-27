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
        << "Index: " << i << " Total: " << num;
    } else {
      EXPECT_EQ(ref, got) << "Index: " << i << " Total: " << num;
    }
  }
}
}
