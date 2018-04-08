#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class lrn_test : public ::testing::TestWithParam<lrn_test_params> {
protected:
  virtual void SetUp() {
    auto p = ::testing::TestWithParam<lrn_test_params>::GetParam();
    auto data_type = data_traits<data_t>::data_type;
    auto ld = p.test_ld;

    tensor::descriptor src_desc({ld.mb, ld.c, ld.h, ld.w}, data_type,
        static_cast<format>(p.src_format));
    tensor::descriptor grady_desc({ld.mb, ld.c, ld.h, ld.w}, data_type,
        static_cast<format>(p.dst_format));

    src_.init(src_desc);
    fill_data<data_t>(src_.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(src_.get_data_handle()));

    grady_.init(grady_desc);
    fill_data<data_t>(grady_.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(grady_.get_data_handle()));
  }

  tensor src_, grady_;
};

using lrn_test_float = lrn_test<float>;
using lrn_test_params_float = lrn_test_params;

TEST_P(lrn_test_float, TestsLRN) {
  auto p = ::testing::TestWithParam<lrn_test_params>::GetParam();
  auto ld = p.test_ld;

  auto dst = make_output();
  lrn_forward::compute(src_, dst, ld.local_size, ld.alpha,
      ld.beta, ld.k, p.aalgorithm, p.aprop_kind);
  check_lrn_fwd<float>(ld, src_, dst);

  if (p.aprop_kind == prop_kind::forward_training) {
    auto gradx = make_output();
    lrn_backward::compute(src_, grady_, dst, gradx,
        ld.local_size, ld.alpha, ld.beta, ld.k, p.aalgorithm);
    check_lrn_bwd<float>(p, src_, grady_, gradx);
  }
}

INSTANTIATE_TEST_CASE_P(TestLRN, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 4.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 1.0f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 1.0f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 6.5f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 20, 12, 7, 7, 1.0e-2f, 0.5f, 6.5f, 3, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNNHWC, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 10, 4, 4, 1.0e-4f, 0.75f, 3.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRN_nChw8c, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 8, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRN_nChw16c, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 16, 4, 4, 1.0e-4f, 0.75f, 5.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 16, 1, 1, 1.0e-4f, 0.75f, 2.2f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 1.0f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 1, 32, 5, 5, 1.0e-2f, 0.7f, 0.1f, 3, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNAlexnetNCHW, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNAlexnetNHWC, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nhwc,
    mkldnn::memory::format::nhwc,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNAlexnet_nChw8c, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNAlexnet_nChw16c, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNGoogleNetV1NCHW, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nchw,
    mkldnn::memory::format::nchw,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNGoogleNetV1_nChw8c, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

INSTANTIATE_TEST_CASE_P(TestLRNGoogleNetV1_nChw16c, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 64, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } },

    lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_across_channels,
    mkldnn::memory::format::nChw16c,
    mkldnn::memory::format::nChw16c,
    { 2, 192, 56, 56, 1.0e-4f, 0.75f, 1.0f, 5, ACROSS } })
);

// Backward does not support WITHIN yet.
/*
INSTANTIATE_TEST_CASE_P(TestLRNRCNNBlocked, lrn_test_float,
  ::testing::Values(
    lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 3, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 3, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 3, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 3, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 5, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 96, 55, 55, 1.0e-4f, 0.75f, 5, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_training,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 5, WITHIN } }

    , lrn_test_params_float{ prop_kind::forward_scoring,
    engine::kind::cpu, algorithm::lrn_within_channel,
    mkldnn::memory::format::nChw8c,
    mkldnn::memory::format::nChw8c,
    { 2, 256, 27, 27, 1.0e-4f, 0.75f, 5, WITHIN } })
);
*/

