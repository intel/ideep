#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class batch_normalization_test :
  public ::testing::TestWithParam<test_bnrm_params_t> {
protected:
  virtual void SetUp() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    auto data_type = data_traits<data_t>::data_type;
    auto bs = p.sizes;

    tensor::descriptor src_desc({bs.mb, bs.c, bs.h, bs.w}, data_type,
        static_cast<format>(p.formats.data_format));
    tensor::descriptor grady_desc({bs.mb, bs.c, bs.h, bs.w}, data_type,
        static_cast<format>(p.formats.diff_format));
    statistic_desc_ = tensor::descriptor ({bs.c}, data_type);

    src_.init(src_desc);
    grady_.init(grady_desc);
    mean_.init(statistic_desc_);
    variance_.init(statistic_desc_);
    scale_.init(statistic_desc_);
    shift_.init(statistic_desc_);
  }

  void test_forward_inference() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    fill_tensor(src_);
    fill_tensor(mean_);
    fill_tensor(variance_);
    fill_tensor(scale_);
    fill_tensor(shift_);
    raw_dst_.reset(new char [src_.get_size()]);

    auto dst = batch_normalization_forward_inference::compute(
        src_, mean_, variance_, scale_, shift_, raw_dst_.get(), p.eps);

    check_bnrm_fwd<data_t>(p, src_, mean_, variance_, scale_, shift_, dst,
        batch_normalization_flag::use_scale_shift |
        batch_normalization_flag::use_global_stats,
        prop_kind::forward_inference);
  }

  void test_forward_training() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    fill_tensor(src_);
    fill_tensor(scale_);
    fill_tensor(shift_);
    raw_dst_.reset(new char [src_.get_size()]);
    raw_mean_.reset(new char [statistic_desc_.get_size()]);
    raw_var_.reset(new char [statistic_desc_.get_size()]);
    running_mean_.reset(new char [statistic_desc_.get_size()]);
    running_var_.reset(new char [statistic_desc_.get_size()]);

    auto dst = batch_normalization_forward_training::compute(
        src_, scale_, shift_, raw_dst_.get(), raw_mean_.get(), raw_var_.get(),
        running_mean_.get(), running_var_.get(), 0.9f, p.eps);

    mean_.init(statistic_desc_, raw_mean_.get());
    variance_.init(statistic_desc_, raw_var_.get());

    check_bnrm_fwd<data_t>(p, src_, mean_, variance_, scale_, shift_, dst,
        batch_normalization_flag::use_scale_shift,
        prop_kind::forward_training);
  }

  void test_backward() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    fill_tensor(src_);
    fill_tensor(mean_);
    fill_tensor(variance_);
    fill_tensor(grady_);
    fill_tensor(scale_);
    raw_gradx_.reset(new char [src_.get_size()]);
    raw_gscale_.reset(new char [statistic_desc_.get_size()]);
    raw_gshift_.reset(new char [statistic_desc_.get_size()]);

    auto gradx = batch_normalization_backward::compute(src_, mean_,
        variance_, grady_, scale_, raw_gradx_.get(), raw_gscale_.get(),
        raw_gshift_.get(), p.eps);

    tensor gscale(statistic_desc_, raw_gscale_.get());
    tensor gshift(statistic_desc_, raw_gshift_.get());

    check_bnrm_bwd<data_t>(p, src_, grady_, mean_, variance_, scale_,
        gradx, gscale, gshift, batch_normalization_flag::use_scale_shift,
        prop_kind::backward);
  }

  void test_forward_inference2() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    fill_tensor(src_);
    fill_tensor(mean_);
    fill_tensor(variance_);
    fill_tensor(scale_);
    fill_tensor(shift_);

    auto dst = batch_normalization_forward_inference::compute(
        src_, mean_, variance_, scale_, shift_, p.eps);

    check_bnrm_fwd<data_t>(p, src_, mean_, variance_, scale_, shift_, dst,
        batch_normalization_flag::use_scale_shift |
        batch_normalization_flag::use_global_stats,
        prop_kind::forward_inference);
  }

  void test_forward_training2() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    fill_tensor(src_);
    fill_tensor(scale_);
    fill_tensor(shift_);

    auto dst = batch_normalization_forward_training::compute(
        src_, scale_, shift_, 0.9f, p.eps);

    check_bnrm_fwd<data_t>(p, src_, std::get<1>(dst), std::get<2>(dst), scale_, shift_, std::get<0>(dst),
        batch_normalization_flag::use_scale_shift,
        prop_kind::forward_training);
  }

  void test_backward2() {
    auto p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();
    fill_tensor(src_);
    fill_tensor(mean_);
    fill_tensor(variance_);
    fill_tensor(grady_);
    fill_tensor(scale_);

    auto gs = batch_normalization_backward::compute(src_, mean_,
        variance_, grady_, scale_, p.eps);

    tensor gscale;
    tensor gshift;
    tensor gw = std::get<1>(gs);

    gscale.init({{gw.get_nelems() / 2}, gw.get_data_type(), format::x},
                gw.get_data_handle());
    gshift.init({{gw.get_nelems() / 2}, gw.get_data_type(), format::x},
                (void *)((unsigned long long)gw.get_data_handle() + gw.get_size() / 2));

    check_bnrm_bwd<data_t>(p, src_, grady_, mean_, variance_, scale_,
        std::get<0>(gs), gscale, gshift,
        batch_normalization_flag::use_scale_shift, prop_kind::backward);
  }

  tensor::descriptor statistic_desc_;
  tensor src_, grady_, mean_, variance_, scale_, shift_;
  std::unique_ptr<char []> raw_dst_, raw_gradx_, raw_mean_;
  std::unique_ptr<char []> raw_var_, raw_gscale_, raw_gshift_;
  std::unique_ptr<char []> running_mean_, running_var_;
};

using bnrm_test_float = batch_normalization_test<float>;

TEST_P(bnrm_test_float, TestsInference) {
  test_forward_inference();
  test_forward_inference2();
}

TEST_P(bnrm_test_float, TestsForward) {
  test_forward_training();
  test_forward_training2();
}

TEST_P(bnrm_test_float, TestsBackward) {
  test_backward();
  test_backward2();
}

#define EXPAND_ARGS(args) args
#define EXPAND_SIZES(mb, c, h, w) { mb, c, h, w }
#define EXPAND_FORMATS(data, diff) \
    { static_cast<format>(mkldnn::memory::format::data), \
      static_cast<format>(mkldnn::memory::format::diff) }

#define ENGINE engine::kind::cpu
#define EPS 1e-5f

#define PARAMS(data, diff, mb, c, h, w, eps) \
    test_bnrm_params_t { ENGINE, \
    EXPAND_FORMATS(data, diff), EXPAND_SIZES(mb, c, h, w), eps }

#define PARAMS_N(...) EXPAND_ARGS(PARAMS(nchw, nchw, __VA_ARGS__))
#define PARAMS_B8(...) EXPAND_ARGS(PARAMS(nChw8c, nChw8c, __VA_ARGS__))
#define PARAMS_B16(...) EXPAND_ARGS(PARAMS(nChw16c, nChw16c, __VA_ARGS__))

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, bnrm_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(Simple_NCHW,
    PARAMS_N(2, 8, 1, 1, EPS),
    PARAMS_N(2, 10, 1, 1, EPS),
    PARAMS_N(2, 8, 4, 4, EPS),
    PARAMS_N(2, 10, 4, 4, EPS)
);

INST_TEST_CASE(Simple_Blocked,
    PARAMS_B8(2, 8, 1, 1, EPS),
    PARAMS_B8(2, 8, 4, 4, EPS),
    PARAMS_B8(2, 8, 6, 6, EPS),
    PARAMS_B8(2, 16, 4, 4, EPS),
    PARAMS_B8(2, 16, 4, 4, EPS),
    PARAMS_B8(2, 16, 8, 8, EPS),
    PARAMS_B8(2, 16, 8, 8, EPS),
    PARAMS_B8(2, 16, 16, 8, EPS),
    PARAMS_B8(2, 16, 16, 8, EPS),
    PARAMS_B8(2, 16, 10, 8, EPS),
    PARAMS_B8(2, 16, 10, 8, EPS),
    PARAMS_B16(2, 16, 4, 4, EPS),
    PARAMS_B16(2, 16, 4, 4, EPS),
    PARAMS_B16(2, 16, 8, 8, EPS),
    PARAMS_B16(2, 16, 8, 8, EPS),
    PARAMS_B16(2, 16, 16, 8, EPS),
    PARAMS_B16(2, 16, 16, 8, EPS),
    PARAMS_B16(2, 16, 10, 8, EPS),
    PARAMS_B16(2, 16, 10, 8, EPS)
);

INST_TEST_CASE(GoogleNet_NCHW,
    PARAMS_N(2, 64, 112, 112, EPS),
    PARAMS_N(2, 64, 56, 56, EPS),
    PARAMS_N(2, 192, 56, 56, EPS),
    PARAMS_N(2, 96, 28, 28, EPS),
    PARAMS_N(2, 16, 28, 28, EPS),
    PARAMS_N(2, 64, 28, 28, EPS),
    PARAMS_N(2, 128, 28, 28, EPS),
    PARAMS_N(2, 32, 28, 28, EPS),
    PARAMS_N(2, 96, 28, 28, EPS),
    PARAMS_N(2, 96, 14, 14, EPS),
    PARAMS_N(2, 16, 14, 14, EPS),
    PARAMS_N(2, 192, 14, 14, EPS),
    PARAMS_N(2, 208, 14, 14, EPS),
    PARAMS_N(2, 48, 14, 14, EPS),
    PARAMS_N(2, 64, 14, 14, EPS),
    PARAMS_N(2, 112, 14, 14, EPS),
    PARAMS_N(2, 24, 14, 14, EPS),
    PARAMS_N(2, 160, 14, 14, EPS),
    PARAMS_N(2, 224, 14, 14, EPS),
    PARAMS_N(2, 128, 4, 4, EPS),
    PARAMS_N(2, 128, 14, 14, EPS),
    PARAMS_N(2, 512, 14, 14, EPS),
    PARAMS_N(2, 256, 14, 14, EPS),
    PARAMS_N(2, 144, 14, 14, EPS),
    PARAMS_N(2, 32, 14, 14, EPS),
    PARAMS_N(2, 228, 14, 14, EPS),
    PARAMS_N(2, 528, 14, 14, EPS),
    PARAMS_N(2, 320, 14, 14, EPS),
    PARAMS_N(2, 160, 7, 7, EPS),
    PARAMS_N(2, 32, 7, 7, EPS),
    PARAMS_N(2, 256, 7, 7, EPS),
    PARAMS_N(2, 320, 7, 7, EPS),
    PARAMS_N(2, 128, 7, 7, EPS),
    PARAMS_N(2, 192, 7, 7, EPS),
    PARAMS_N(2, 48, 7, 7, EPS),
    PARAMS_N(2, 384, 7, 7, EPS)
);

INST_TEST_CASE(GoogleNet_Blocked_8,
    PARAMS_B8(2, 64, 112, 112, EPS),
    PARAMS_B8(2, 64, 56, 56, EPS),
    PARAMS_B8(2, 192, 56, 56, EPS),
    PARAMS_B8(2, 96, 28, 28, EPS),
    PARAMS_B8(2, 16, 28, 28, EPS),
    PARAMS_B8(2, 64, 28, 28, EPS),
    PARAMS_B8(2, 128, 28, 28, EPS),
    PARAMS_B8(2, 32, 28, 28, EPS),
    PARAMS_B8(2, 96, 28, 28, EPS),
    PARAMS_B8(2, 96, 14, 14, EPS),
    PARAMS_B8(2, 16, 14, 14, EPS),
    PARAMS_B8(2, 192, 14, 14, EPS),
    PARAMS_B8(2, 208, 14, 14, EPS),
    PARAMS_B8(2, 48, 14, 14, EPS),
    PARAMS_B8(2, 64, 14, 14, EPS),
    PARAMS_B8(2, 112, 14, 14, EPS),
    PARAMS_B8(2, 24, 14, 14, EPS),
    PARAMS_B8(2, 160, 14, 14, EPS),
    PARAMS_B8(2, 224, 14, 14, EPS),
    PARAMS_B8(2, 128, 4, 4, EPS),
    PARAMS_B8(2, 128, 14, 14, EPS),
    PARAMS_B8(2, 512, 14, 14, EPS),
    PARAMS_B8(2, 256, 14, 14, EPS),
    PARAMS_B8(2, 144, 14, 14, EPS),
    PARAMS_B8(2, 32, 14, 14, EPS),
    PARAMS_B8(2, 528, 14, 14, EPS),
    PARAMS_B8(2, 320, 14, 14, EPS),
    PARAMS_B8(2, 160, 7, 7, EPS),
    PARAMS_B8(2, 32, 7, 7, EPS),
    PARAMS_B8(2, 256, 7, 7, EPS),
    PARAMS_B8(2, 320, 7, 7, EPS),
    PARAMS_B8(2, 128, 7, 7, EPS),
    PARAMS_B8(2, 192, 7, 7, EPS),
    PARAMS_B8(2, 48, 7, 7, EPS),
    PARAMS_B8(2, 384, 7, 7, EPS)
);

INST_TEST_CASE(GoogleNet_Blocked_16,
    PARAMS_B16(2, 64, 112, 112, EPS),
    PARAMS_B16(2, 64, 56, 56, EPS),
    PARAMS_B16(2, 192, 56, 56, EPS),
    PARAMS_B16(2, 96, 28, 28, EPS),
    PARAMS_B16(2, 16, 28, 28, EPS),
    PARAMS_B16(2, 64, 28, 28, EPS),
    PARAMS_B16(2, 128, 28, 28, EPS),
    PARAMS_B16(2, 32, 28, 28, EPS),
    PARAMS_B16(2, 96, 28, 28, EPS),
    PARAMS_B16(2, 96, 14, 14, EPS),
    PARAMS_B16(2, 16, 14, 14, EPS),
    PARAMS_B16(2, 192, 14, 14, EPS),
    PARAMS_B16(2, 208, 14, 14, EPS),
    PARAMS_B16(2, 48, 14, 14, EPS),
    PARAMS_B16(2, 64, 14, 14, EPS),
    PARAMS_B16(2, 112, 14, 14, EPS),
    //PARAMS_B16(2, 24, 14, 14, EPS),
    PARAMS_B16(2, 160, 14, 14, EPS),
    PARAMS_B16(2, 224, 14, 14, EPS),
    PARAMS_B16(2, 128, 4, 4, EPS),
    PARAMS_B16(2, 128, 14, 14, EPS),
    PARAMS_B16(2, 512, 14, 14, EPS),
    PARAMS_B16(2, 256, 14, 14, EPS),
    PARAMS_B16(2, 144, 14, 14, EPS),
    PARAMS_B16(2, 32, 14, 14, EPS),
    PARAMS_B16(2, 528, 14, 14, EPS),
    PARAMS_B16(2, 320, 14, 14, EPS),
    PARAMS_B16(2, 160, 7, 7, EPS),
    PARAMS_B16(2, 32, 7, 7, EPS),
    PARAMS_B16(2, 256, 7, 7, EPS),
    PARAMS_B16(2, 320, 7, 7, EPS),
    PARAMS_B16(2, 128, 7, 7, EPS),
    PARAMS_B16(2, 192, 7, 7, EPS),
    PARAMS_B16(2, 48, 7, 7, EPS),
    PARAMS_B16(2, 384, 7, 7, EPS)
);
