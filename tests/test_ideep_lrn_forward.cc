#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class lrn_forward_test :
  public ::testing::TestWithParam<lrn_test_params> {
protected:
  virtual void SetUp() {
    lrn_test_params p
      = ::testing::TestWithParam<lrn_test_params>::GetParam();

    auto data_type = data_traits<data_t>::data_type;
    ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

    auto ld = p.test_ld;

    auto src_desc = tensor::descriptor({ ld.mb, ld.c, ld.h, ld.w },
        data_type, static_cast<format>(p.src_format));

    src_.init(src_desc);

    // Only true for dense format
    fill_data<data_t>(src_.get_size() / sizeof(data_t),
            (data_t *)src_.get_data_handle());

    tensor::dims dst_dims = {ld.mb, ld.c, ld.h, ld.w};
  }

  tensor src_;
};

using lrn_forward_test_float = lrn_forward_test<float>;
using lrn_fwd_test_params_float = lrn_test_params;

TEST_P (lrn_forward_test_float, TestsLRN) {
  auto p = ::testing::TestWithParam<lrn_test_params>::GetParam();
  auto ld = p.test_ld;
  auto dst = make_output();
  lrn_forward::compute(src_, dst, ld.local_size, ld.alpha,
      ld.beta, ld.k, p.aalgorithm, p.aprop_kind);

  check_lrn_fwd<float>(ld, src_, dst);
}

INSTANTIATE_TEST_CASE_P(TestLRNForward, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNForwardNHWC, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.85f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.85f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNForward_nChw8c, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNForward_nChw16c, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.7f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNAlexnetForwardNCHW, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels, mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNAlexnetForwardNHWC, lrn_forward_test_float,
    ::testing::Values(
      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nhwc,
      mkldnn::memory::format::nhwc,
      { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nhwc,
      mkldnn::memory::format::nhwc,
      { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nhwc,
      mkldnn::memory::format::nhwc,
      { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nhwc,
      mkldnn::memory::format::nhwc,
      { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNAlexnetForward_nChw8c, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNAlexnetForward_nChw16c, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNGoogleNetV1ForwardNCHW, lrn_forward_test_float,
    ::testing::Values(
    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_fwd_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNGoogleNetV1Forward_nChw8c, lrn_forward_test_float,
    ::testing::Values(
      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNGoogleNetV1Forward_nChw16c, lrn_forward_test_float,
    ::testing::Values(
      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw16c,
      mkldnn::memory::format::nChw16c,
      { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw16c,
      mkldnn::memory::format::nChw16c,
      { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw16c,
      mkldnn::memory::format::nChw16c,
      { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

      lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_across_channels,
      mkldnn::memory::format::nChw16c,
      mkldnn::memory::format::nChw16c,
      { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(
    TestLRNRCNNForwardBlocked, lrn_forward_test_float,
    ::testing::Values(
      lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 3, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_training,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } }

      , lrn_fwd_test_params_float{ prop_kind::forward_scoring,
      engine::kind::cpu, algorithm::lrn_within_channel,
      mkldnn::memory::format::nChw8c,
      mkldnn::memory::format::nChw8c,
      { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, WITHIN } })
);
