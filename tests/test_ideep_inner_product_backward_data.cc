#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class inner_product_test_bwd_data :
  public ::testing::TestWithParam<inprod_test_bwd_data_params> {
protected:
  virtual void SetUp() {
    auto p = ::testing::TestWithParam<inprod_test_bwd_data_params>::GetParam();
    auto ipd = p.test_ipd;
    bool has_spatial = ipd.kh > 1 && ipd.kw > 1;

    auto data_type = data_traits<data_t>::data_type;

    auto grady_desc = tensor::descriptor ({ipd.mb, ipd.oc}, data_type,
          static_cast<format>(p.diff_dst_format));
    auto weights_desc = has_spatial ?
      tensor::descriptor({ipd.oc, ipd.ic, ipd.kh, ipd.kw}, data_type,
          static_cast<format>(p.weights_format)) :
        tensor::descriptor ({ipd.oc, ipd.ic}, data_type,
          static_cast<format>(p.weights_format));

    grady_.init(grady_desc);
    weights_.init(weights_desc);

    auto gradx_desc = has_spatial ?
      tensor::descriptor ({ipd.mb, ipd.ic, ipd.kh, ipd.kw}, data_type,
        static_cast<format>(p.diff_src_format)) :
        tensor::descriptor ({ipd.mb, ipd.ic}, data_type,
            static_cast<format>(p.diff_src_format));

    gradx_ref_.init(gradx_desc);
  }

  tensor grady_, weights_, gradx_ref_;
};

using inner_product_test_float = inner_product_test_bwd_data<float>;

TEST_P(inner_product_test_float, TestsBackwardData) {
  auto p = ::testing::TestWithParam<inprod_test_bwd_data_params>::GetParam();
  auto ipd = p.test_ipd;
  fill_tensor(grady_);
  fill_tensor(weights_);

  tensor gradx;
  auto gradx_dims = ipd.kh > 1 && ipd.kw > 1 ?
    tensor::dims {ipd.mb, ipd.ic, ipd.kh, ipd.kw} :
    tensor::dims {ipd.mb, ipd.ic};

  auto test = [&] () {
    inner_product_backward_data::compute(grady_, weights_, gradx_dims, gradx);
  };

  if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
    return;

  compute_ref_inner_product_bwd_data<float>(
      p.test_ipd, grady_, weights_, gradx_ref_);
  compare_tensor<float>(gradx_ref_, gradx);
}

using inprod_test_params_float = inprod_test_bwd_data_params;

// INSTANTIATE_TEST_CASE_P(TestInnerProductBackwardDataEF, inner_product_test_float,
//   ::testing::Values(
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any,
//     { 0, 32, 48, 6, 6 }, true, mkldnn_invalid_arguments},
//     inprod_test_params_float{ engine::kind::cpu,
//     mkldnn::memory::format::any, mkldnn::memory::format::any,
//     mkldnn::memory::format::any,
//     { 2, 0, 48, 6, 6 }, true, mkldnn_invalid_arguments})
// );

INSTANTIATE_TEST_CASE_P(TestInnerProductBackwardData, inner_product_test_float,
  ::testing::Values(
//    inprod_test_params_float{ engine::kind::cpu,
//    mkldnn::memory::format::any, mkldnn::memory::format::any,
//    mkldnn::memory::format::any, { 2, 32, 48, 6, 6 } },
//    inprod_test_params_float{ engine::kind::cpu,
//    mkldnn::memory::format::any, mkldnn::memory::format::any,
//    mkldnn::memory::format::any, { 2, 1024, 48, 2, 2 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nchw, mkldnn::memory::format::oihw,
    mkldnn::memory::format::nc, { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nChw8c, mkldnn::memory::format::oIhw8i,
    mkldnn::memory::format::nc, { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nChw16c, mkldnn::memory::format::oIhw16i,
    mkldnn::memory::format::nc, { 2, 32, 48, 6, 6 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::nc, { 2, 32, 1152, 1, 1 } },
    inprod_test_params_float{ engine::kind::cpu,
    mkldnn::memory::format::nc, mkldnn::memory::format::oi,
    mkldnn::memory::format::nc, { 2, 2, 4, 1, 1 } }));
