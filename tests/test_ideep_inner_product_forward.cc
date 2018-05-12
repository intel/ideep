#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class inner_product_test :
  public ::testing::TestWithParam<inprod_test_forward_params> {
protected:
  virtual void SetUp() {
    auto p = ::testing::TestWithParam<inprod_test_forward_params>::GetParam();
    auto ipd = p.test_ipd;
    bool has_spatial = ipd.kh > 1 && ipd.kw > 1;
    bool with_bias = p.bias_format != mkldnn::memory::format::format_undef;

    auto data_type = data_traits<data_t>::data_type;

    auto src_desc = has_spatial ?
      tensor::descriptor({ipd.mb, ipd.ic, ipd.kh, ipd.kw}, data_type,
          static_cast<format>(p.src_format)) :
        tensor::descriptor ({ipd.mb, ipd.ic}, data_type,
          static_cast<format>(p.src_format));
    auto weights_desc = has_spatial ?
      tensor::descriptor({ipd.oc, ipd.ic, ipd.kh, ipd.kw}, data_type,
          static_cast<format>(p.weights_format)) :
        tensor::descriptor ({ipd.oc, ipd.ic}, data_type,
          static_cast<format>(p.weights_format));

    auto bias_desc = with_bias ?
      tensor::descriptor({ipd.oc}, data_type,
          static_cast<format>(p.bias_format)) :
      tensor::descriptor({}, data_type,
          static_cast<format>(p.bias_format));

    src_.init(src_desc);
    weights_.init(weights_desc);
    bias_.init(bias_desc);

    tensor::descriptor dst_desc({ipd.mb, ipd.oc}, data_type,
        static_cast<format>(p.dst_format));
    dst_ref_.init(dst_desc);
  }

  tensor src_, weights_, bias_, dst_ref_;
};

using inner_product_test_float = inner_product_test<float>;

TEST_P(inner_product_test_float, TestsForward) {
  auto p = ::testing::TestWithParam<inprod_test_forward_params>::GetParam();
  bool with_bias = p.bias_format != mkldnn::memory::format::format_undef;

  fill_tensor(src_);
  fill_tensor(weights_);
  if (with_bias)
    fill_tensor(bias_);

  tensor dst;
  auto test = [&] () {
    if (with_bias)
      inner_product_forward::compute(src_, weights_, bias_, dst);
    else
      inner_product_forward::compute(src_, weights_, dst);
  };

  if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    return;

  compute_ref_inner_product_fwd<float>(
      p.test_ipd, src_, weights_, bias_, dst_ref_);
  compare_tensor<float>(dst_ref_, dst);
}

using inprod_test_params_float = inprod_test_forward_params;

INSTANTIATE_TEST_CASE_P(TestInnerProductForwardNoBias, inner_product_test_float,
  ::testing::Values(
    // inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    // mkldnn::memory::format::any, mkldnn::memory::format::any,
    // mkldnn::memory::format::format_undef, mkldnn::memory::format::any,
    // { 2, 32, 48, 6, 6 } },
    // inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    // mkldnn::memory::format::any, mkldnn::memory::format::any,
    // mkldnn::memory::format::format_undef, mkldnn::memory::format::any,
    // { 2, 512, 48, 2, 2 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nchw, mkldnn::memory::format::oihw,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nChw8c, mkldnn::memory::format::oIhw8i,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nChw16c, mkldnn::memory::format::oIhw16i,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 2, 4, 1, 1 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::nc,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::nc,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 2, 4, 1, 1 } }
));

// INSTANTIATE_TEST_CASE_P(TestInnerProductForwardEF, inner_product_test_float,
//   ::testing::Values(
//     inprod_test_params_float { prop_kind::forward, engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     { 0, 32, 48, 6, 6 }, true, mkldnn_invalid_arguments},
//     inprod_test_params_float { prop_kind::forward, engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     { 2, 0, 48, 6, 6 }, true, mkldnn_invalid_arguments})
// );

INSTANTIATE_TEST_CASE_P(TestInnerProductForward, inner_product_test_float,
  ::testing::Values(
    // inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    // mkldnn::memory::format::any, mkldnn::memory::format::any,
    // mkldnn::memory::format::any, mkldnn::memory::format::any,
    // { 2, 32, 48, 6, 6 } },
    // inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    // mkldnn::memory::format::any, mkldnn::memory::format::any,
    // mkldnn::memory::format::any, mkldnn::memory::format::any,
    // { 2, 512, 48, 2, 2 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nchw, mkldnn::memory::format::oihw,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nChw8c, mkldnn::memory::format::oIhw8i,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nChw16c, mkldnn::memory::format::oIhw16i,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 2, 4, 1, 1 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::nc,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::nc,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 2, 4, 1, 1 } }
));
