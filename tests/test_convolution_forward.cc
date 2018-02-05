#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include <test_convolution_forward_common.hpp>

#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t_src, typename data_t_wei,
         typename data_t_acc, typename data_t_dst>
class convolution_forward_tests :
  public ::testing::TestWithParam<test_convolution_params_t> {
protected:
  virtual void SetUp() {
    test_convolution_params_t p = 
      ::testing::TestWithParam<test_convolution_params_t>::GetParam();
    test_convolution_sizes_t cd = p.sizes;

    tensor::descriptor src_desc ({cd.mb, cd.ic, cd.ih, cd.iw},
        data_traits<data_t_src>::data_type, p.formats.src_format);

    auto weights_desc = cd.ng > 1 ? 
      tensor::descriptor(
          {cd.ng, cd.oc/cd.ng, cd.ic/cd.ng, cd.kh, cd.kw},
          data_traits<data_t_wei>::data_type, p.formats.weights_format) :
      tensor::descriptor(
          {cd.oc, cd.ic, cd.kh, cd.kw},
          data_traits<data_t_wei>::data_type, p.formats.weights_format);

    bool with_bias = p.formats.bias_format != format::format_undef;
    auto bias_desc = with_bias ?
          tensor::descriptor({cd.oc}, data_traits<data_t_dst>::data_type,
              p.formats.dst_format) :
            tensor::descriptor({}, data_traits<data_t_dst>::data_type,
                p.formats.dst_format);

    tensor src(src_desc);
    tensor weights(weights_desc);
    tensor bias(bias_desc);

    fill_data<data_t_src>(
        src.get_size() / sizeof(data_t_src),
        reinterpret_cast<data_t_src *>(src.get_data_handle()));
    fill_data<data_t_wei>(
        weights.get_size() / sizeof(data_t_src),
        reinterpret_cast<data_t_src *>(weights.get_data_handle()));

    if (with_bias) {
      fill_data<data_t_dst>(
          bias.get_size() / sizeof(data_t_dst),
          reinterpret_cast<data_t_src *>(bias.get_data_handle()));
    }

    tensor::dims padR {cd.padh, cd.padw};
    for (int i = 0; i < 2; ++ i) {
      if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
        / cd.strh + 1 != cd.oh)
        ++padR[0];
      if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
        / cd.strw + 1 != cd.ow)
        ++padR[1];
    }

    tensor::dims dst_dims {cd.mb, cd.oc, cd.oh, cd.ow};
    auto dst_size = std::accumulate(dst_dims.begin(), dst_dims.end(),
        1, std::multiplies<int>());
    auto raw_dst = std::unique_ptr<char>(new char [dst_size]);

    auto dst_desc = with_bias ?
      convolution_forward::compute(src, weights, bias, dst_dims,
          raw_dst.get(), tensor::dims {cd.strh, cd.strw },
          tensor::dims {cd.dilh, cd.dilw}, tensor::dims {cd.padh, cd.padw },
          padR) :
      convolution_forward::compute(src, weights, dst_dims,
          raw_dst.get(), tensor::dims {cd.strh, cd.strw },
          tensor::dims {cd.dilh, cd.dilw}, tensor::dims {cd.padh, cd.padw },
          padR);

    tensor ref_dst(dst_desc);
    test_convolution_attr_t attr = p.attr;
    attr.mkldnn_attr_recreate();
    compute_ref_conv_fwd<data_t_src, data_t_wei, data_t_acc, data_t_dst>(
        cd, attr, src, weights, bias, ref_dst);
  }
};
