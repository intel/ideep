#pragma once

#include "mkldnn_test_common.hpp"
#include <gtest/gtest.h>
#include <cmath>

#include <ideep.hpp>

namespace ideep {

inline size_t map_index(const mkldnn_memory_desc_t *md, size_t index) {
  mkldnn::memory::desc dup_md(*md);
  return ::map_index(dup_md, index);
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
}
