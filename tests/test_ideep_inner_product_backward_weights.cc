#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include <vector>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class inner_product_test_bwd_weights :
  public ::testing::TestWithParam<inprod_test_bwd_weights_params> {
protected:
  virtual void SetUp() {
    auto p =
      ::testing::TestWithParam<inprod_test_bwd_weights_params>::GetParam();
    auto ipd = p.test_ipd;
    bool has_spatial = ipd.kh > 1 && ipd.kw > 1;
    bool with_bias =
      p.diff_bias_format != mkldnn::memory::format::format_undef;
    auto data_type = data_traits<data_t>::data_type;

    auto src_desc = has_spatial ?
      tensor::descriptor({ipd.mb, ipd.ic, ipd.kh, ipd.kw}, data_type,
          static_cast<format>(p.src_format)) :
        tensor::descriptor({ipd.mb, ipd.ic}, data_type,
            static_cast<format>(p.src_format));
    src_.init(src_desc);

    tensor::descriptor grady_desc ({ipd.mb, ipd.oc}, data_type,
          static_cast<format>(p.diff_dst_format));
    grady_.init(grady_desc);
    
    auto gradw_desc = has_spatial ?
      tensor::descriptor({ipd.oc, ipd.ic, ipd.kh, ipd.kw}, data_type,
          static_cast<format>(p.diff_weights_format)) :
        tensor::descriptor({ipd.oc, ipd.ic}, data_type,
            static_cast<format>(p.diff_weights_format));
    gradw_ref_.init(gradw_desc);

    auto gradb_desc = with_bias ?
      tensor::descriptor({ipd.oc}, data_type,
          static_cast<format>(p.diff_bias_format)) :
        tensor::descriptor(tensor::dims{}, data_type,
            static_cast<format>(p.diff_bias_format));
    gradb_ref_.init(gradb_desc);
  }

  tensor src_, grady_, gradw_ref_, gradb_ref_;
};

using inner_product_test_float = inner_product_test_bwd_weights<float>;
using inprod_test_params_float = inprod_test_bwd_weights_params;

TEST_P(inner_product_test_float, TestBackwardWeights) {
  auto p =
    ::testing::TestWithParam<inprod_test_bwd_weights_params>::GetParam();
  auto ipd = p.test_ipd;
  fill_tensor(src_);
  fill_tensor(grady_);

  bool with_bias =
    p.diff_bias_format != mkldnn::memory::format::format_undef;

  tensor gradw;
  tensor gradb;
  auto test = [&] () {
    if (with_bias) {
      inner_product_backward_weights::compute(src_, grady_, gradw, gradb);
    } else
      inner_product_backward_weights::compute(src_, grady_, gradw);
  };

  if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    return;

  compute_ref_inner_product_bwd_weights<float>(ipd, src_, grady_, gradw_ref_);
  compare_tensor<float>(gradw_ref_, gradw);

  if (with_bias) {
    compute_ref_inner_product_bwd_bias<float>(ipd, grady_, gradb_ref_);
    compare_tensor<float>(gradb_ref_, gradb);
  }
}

INSTANTIATE_TEST_CASE_P(TestInnerProductBackwardWeightsNoBias,
    inner_product_test_float,
  ::testing::Values(
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::format_undef, mkldnn::memory::format::any,
//     { 2, 32, 48, 6, 6 } },
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::format_undef, mkldnn::memory::format::any,
//     { 2, 1024, 48, 2, 2 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nchw, mkldnn::memory::format::oihw,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nChw8c, mkldnn::memory::format::oIhw8i,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nChw16c, mkldnn::memory::format::oIhw16i,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 1000, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::format_undef, mkldnn::memory::format::nc,
    { 2, 2, 4, 1, 1 } })
);

// INSTANTIATE_TEST_CASE_P(TestInnerProductBackwardWeightsEF,
//     inner_product_test_float,
//   ::testing::Values(
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     { 0, 32, 48, 6, 6 }, true, mkldnn_invalid_arguments},
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     { 2, 0, 48, 6, 6 }, true, mkldnn_invalid_arguments})
// );

INSTANTIATE_TEST_CASE_P(TestInnerProductBackwardWeights,
  inner_product_test_float,
  ::testing::Values(
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     { 2, 32, 48, 6, 6 } },
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     { 2, 32, 1024, 2, 2 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nchw, mkldnn::memory::format::oihw,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nChw8c, mkldnn::memory::format::oIhw8i,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nChw16c, mkldnn::memory::format::oIhw16i,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 1000, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::x, mkldnn::memory::format::nc,
    { 2, 2, 4, 1, 1 } })
);
