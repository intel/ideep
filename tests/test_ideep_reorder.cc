#include <numeric>
#include <immintrin.h>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename reorder_types>
class reorder_simple_test:
  public ::testing::TestWithParam<test_simple_params<reorder_types>> {
  using data_i_t = typename reorder_types::first_type;
  using data_o_t = typename reorder_types::second_type;
protected:
  virtual void SetUp() {
    test_simple_params<reorder_types> p
        = ::testing::TestWithParam<decltype(p)>::GetParam();

    const size_t nelems_i = std::accumulate(p.dims.begin(), p.dims.end(),
            size_t(1), std::multiplies<size_t>());
    const size_t nelems_o = std::accumulate(p.dims.begin(), p.dims.end(),
            size_t(1), std::multiplies<size_t>());
    ASSERT_EQ(nelems_i, nelems_o);

    void *src_mem, *dst_mem;

    ::posix_memalign(&src_mem, 64, nelems_i * sizeof(data_i_t));
    ::posix_memalign(&dst_mem, 64, nelems_o * sizeof(data_o_t));

    src_data_.reset(new (src_mem) data_i_t[nelems_i]);
    dst_data_.reset(new (dst_mem) data_o_t[nelems_o]);

    auto prec_i = data_traits<data_i_t>::data_type;
    auto prec_o = data_traits<data_o_t>::data_type;

    mpd_i_ = tensor::descriptor(p.dims, prec_i,
        static_cast<format>(p.fmt_i));
    mpd_o_ = tensor::descriptor(p.dims, prec_o,
        static_cast<format>(p.fmt_o));

    src_.init(mpd_i_, src_data_.get());
    dst_.init(mpd_o_, dst_data_.get());

    // fill_tensor(src_);

    /* initialize input data, TODO: expand fill_tensor for it */
    auto mkldnn_mpd_i = mpd_i_.get_mkldnn_memory_desc_t();
    auto *src_data = src_data_.get();
    for (size_t i = 0; i < nelems_i; ++i) {
      src_data[map_index(mkldnn_mpd_i, i)] = data_i_t(i);
    }

    auto *dst_data = dst_data_.get();
    // Invalidate all cache line for testing
    for (size_t i = 0; i < nelems_i; ++i) {
      _mm_clflush(&src_data[i]);
      _mm_clflush(&dst_data[i]);
    }
  }

  void test_reorder() {
    auto test = [&]() {
      reorder::compute(src_, dst_);
    };

    test_simple_params<reorder_types> p
        = ::testing::TestWithParam<decltype(p)>::GetParam();
    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    check_reorder(mpd_i_, mpd_o_, src_data_.get(), dst_data_.get());
  }

  tensor src_, dst_;
  tensor::descriptor mpd_i_, mpd_o_;
  std::unique_ptr<data_i_t> src_data_;
  std::unique_ptr<data_o_t> dst_data_;
};

using f32_f32 = std::pair<float, float>;
using s32_s32 = std::pair<int32_t, int32_t>;
using s16_s16 = std::pair<int16_t, int16_t>;
using s8_s8 = std::pair<int8_t, int8_t>;

using reorder_simple_expected_fail_f32_f32 = reorder_simple_test<f32_f32>;
using reorder_simple_test_data_f32_f32 = reorder_simple_test<f32_f32>;
using reorder_simple_test_weights_f32_f32 = reorder_simple_test<f32_f32>;
using reorder_simple_test_weights_f32_f32_IOhw16o16i = reorder_simple_test<f32_f32>;
using reorder_simple_test_weights_f32_f32_IOhw16i16o = reorder_simple_test<f32_f32>;
using reorder_simple_test_weights_f32_f32_IOhw16i16o_1 = reorder_simple_test<f32_f32>;
using reorder_simple_test_s32_s32 = reorder_simple_test<s32_s32>;
using reorder_simple_test_s16_s16 = reorder_simple_test<s16_s16>;
using reorder_simple_test_s8_s8 = reorder_simple_test<s8_s8>;

using eng = engine::kind;
using fmt = mkldnn::memory::format;

using test_simple_params_s32_s32 = test_simple_params<s32_s32>;
using test_simple_params_f32_f32 = test_simple_params<f32_f32>;
using test_simple_params_s16_s16 = test_simple_params<s16_s16>;
using test_simple_params_s8_s8 = test_simple_params<s8_s8>;

using cfg_f32= test_simple_params_f32_f32;
using cfg_s32= test_simple_params_s32_s32;
using cfg_s16= test_simple_params_s16_s16;
using cfg_s8= test_simple_params_s8_s8;

TEST_P(reorder_simple_expected_fail_f32_f32, TestsReorder) {
  test_reorder();
}
// INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_expected_fail_f32_f32,
//   ::testing::Values(
//     cfg_f32{eng::cpu, fmt::nchw, fmt::nchw, {0, 16, 8, 8},
//         true, mkldnn_invalid_arguments},
//     cfg_f32{eng::cpu, fmt::nchw, fmt::nChw8c, {0, 16, 8, 8},
//         true, mkldnn_invalid_arguments},
//     cfg_f32{eng::cpu, fmt::nchw, fmt::nChw16c, {0, 16, 8, 8},
//         true, mkldnn_invalid_arguments},
//     cfg_f32{eng::cpu, fmt::OIhw8o8i, fmt::oihw, {32, 0, 3, 3},
//         true, mkldnn_invalid_arguments},
//     cfg_f32{eng::cpu, fmt::OIhw8i8o, fmt::OIhw8o8i, {0, 32, 3, 3},
//         true, mkldnn_invalid_arguments},
//     cfg_f32{eng::cpu, fmt::OIhw16o16i, fmt::oihw, {32, 32, 0, 3},
//         true, mkldnn_invalid_arguments},
//     cfg_f32{eng::cpu, fmt::OIhw16i16o, fmt::OIhw16o16i, {32, 32, 3, 0},
//         true, mkldnn_invalid_arguments})
// );

TEST_P(reorder_simple_test_data_f32_f32, TestsReorder) {
  test_reorder();
}
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_data_f32_f32,
  ::testing::Values(
    cfg_f32{eng::cpu, fmt::nchw, fmt::nchw, {10, 10, 13, 13}},
    cfg_f32{eng::cpu, fmt::nchw, fmt::nhwc, {10, 10, 10, 10}},
    cfg_f32{eng::cpu, fmt::nhwc, fmt::nchw, {10, 10, 10, 10}},
    cfg_f32{eng::cpu, fmt::nchw, fmt::chwn, {28, 3, 10, 10}},
    cfg_f32{eng::cpu, fmt::chwn, fmt::nchw, {28, 3, 10, 10}},
    cfg_f32{eng::cpu, fmt::nhwc, fmt::nhwc, {10, 10, 13, 13}},
    cfg_f32{eng::cpu, fmt::nchw, fmt::nChw8c, {2, 32, 4, 4}},
    cfg_f32{eng::cpu, fmt::nChw8c, fmt::nchw, {2, 32, 4, 4}},
    cfg_f32{eng::cpu, fmt::chwn, fmt::nChw8c, {28, 96, 10, 10}},
    cfg_f32{eng::cpu, fmt::nChw8c, fmt::chwn, {28, 96, 10, 10}},
    cfg_f32{eng::cpu, fmt::nhwc, fmt::nChw8c, {3, 64, 16, 16}},
    cfg_f32{eng::cpu, fmt::nChw8c, fmt::nhwc, {3, 64, 16, 16}},
    cfg_f32{eng::cpu, fmt::nChw8c, fmt::nChw16c, {10, 96, 27, 27}},
    cfg_f32{eng::cpu, fmt::nChw16c, fmt::nChw8c, {10, 96, 27, 27}},
    cfg_f32{eng::cpu, fmt::nchw, fmt::nChw16c, {2, 64, 4, 4}},
    cfg_f32{eng::cpu, fmt::nChw16c, fmt::nchw, {2, 64, 4, 4}},
    cfg_f32{eng::cpu, fmt::chwn, fmt::nChw16c, {28, 96, 10, 10}},
    cfg_f32{eng::cpu, fmt::nChw16c, fmt::chwn, {28, 96, 10, 10}},
    cfg_f32{eng::cpu, fmt::nhwc, fmt::nChw16c, {2, 64, 4, 4}},
    cfg_f32{eng::cpu, fmt::nChw16c, fmt::nhwc, {2, 64, 4, 4}})
);

TEST_P(reorder_simple_test_weights_f32_f32, TestsReorder) {
  test_reorder();
}
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_weights_f32_f32,
  ::testing::Values(
    cfg_f32{eng::cpu, fmt::hwio, fmt::oihw, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::hwio, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::hwio, fmt::Ohwi8o, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::Ohwi8o, fmt::hwio, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::hwio, fmt::Ohwi16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::Ohwi16o, fmt::hwio, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw8i8o, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw8i8o, fmt::oihw, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::ihwo, fmt::OIhw8i8o, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw8i8o, fmt::ihwo, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw8o8i, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw8o8i, fmt::oihw, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw8i8o, fmt::OIhw8o8i, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw8o8i, fmt::OIhw8i8o, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::hwio, fmt::OIhw8i8o, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw8i8o, fmt::hwio, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::hwigo, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::hwigo, fmt::goihw, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::gOIhw8i8o, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw8i8o, fmt::goihw, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::gOIhw8o8i, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw8o8i, fmt::goihw, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw8i8o, fmt::gOIhw8o8i, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw8o8i, fmt::gOIhw8i8o, {2, 32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16i16o, fmt::oihw, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::ihwo, fmt::OIhw16i16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16i16o, fmt::ihwo, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16o16i, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16o16i, fmt::oihw, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::hwio, fmt::OIhw16i16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16i16o, fmt::hwio, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::gOIhw16i16o, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw16i16o, fmt::goihw, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::gOIhw16o16i, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw16o16i, fmt::goihw, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16i16o, fmt::OIhw16o16i, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16o16i, fmt::OIhw16i16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw16i16o, fmt::gOIhw16o16i, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw16o16i, fmt::gOIhw16i16o, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::Oihw16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::Oihw16o, fmt::oihw, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::gOihw16o, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOihw16o, fmt::goihw, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::Ohwi16o, fmt::Oihw16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::Oihw16o, fmt::Ohwi16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOhwi16o, fmt::gOihw16o, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOihw16o, fmt::gOhwi16o, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::Goihw8g, {16, 16, 16, 3, 3}},
    cfg_f32{eng::cpu, fmt::Goihw8g, fmt::goihw, {16, 16, 16, 3, 3}})
);

TEST_P(reorder_simple_test_weights_f32_f32_IOhw16i16o, TestsGoogleNetReorder) {
  test_reorder();
}

TEST_P(reorder_simple_test_weights_f32_f32_IOhw16i16o_1, TestsGoogleNetReorder) {
  test_reorder();
}

INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_weights_f32_f32_IOhw16i16o,
  ::testing::Values(
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {16, 192, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {16, 480, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 16, 5, 5}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 192, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 256, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 512, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 528, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {32, 832, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {48, 16, 5, 5}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {48, 32, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {48, 832, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 32, 5, 5}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 64, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 192, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 256, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 480, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 512, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {64, 576, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {96, 32, 5, 5}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {96, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {96, 96, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {96, 192, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {96, 480, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {96, 576, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {112, 512, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 32, 5, 5}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 48, 5, 5}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 96, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 128, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 256, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 512, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 528, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 576, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {128, 832, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {192, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {192, 128, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {192, 480, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {192, 832, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {384, 192, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {384, 256, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {384, 832, 3, 3}})
);

INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_weights_f32_f32_IOhw16i16o_1,
  ::testing::Values(
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {160, 128, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {160, 512, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {160, 528, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {160, 832, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {208, 96, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {224, 112, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {224, 576, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {256, 128, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {256, 528, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {256, 832, 1, 1}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {288, 144, 3, 3}},
    cfg_f32{eng::cpu, fmt::oihw, fmt::OIhw16i16o, {320, 160, 3, 3}})
);

TEST_P(reorder_simple_test_weights_f32_f32_IOhw16o16i, TestsReorder) {
  test_reorder();
}
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_weights_f32_f32_IOhw16o16i,
  ::testing::Values(
    cfg_f32{eng::cpu, fmt::oihw, fmt::IOhw16o16i, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::IOhw16o16i, fmt::oihw, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::OIhw16i16o, fmt::IOhw16o16i, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::IOhw16o16i, fmt::OIhw16i16o, {64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::goihw, fmt::gOIhw16o16i, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gIOhw16o16i, fmt::goihw, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gOIhw16i16o, fmt::gIOhw16o16i, {2, 64, 64, 3, 3}},
    cfg_f32{eng::cpu, fmt::gIOhw16o16i, fmt::gOIhw16i16o, {2, 64, 64, 3, 3}} )
);

TEST_P(reorder_simple_test_s32_s32, TestsReorder) {
  test_reorder();
}
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_s32_s32,
  ::testing::Values(
    cfg_s32{eng::cpu, fmt::nchw, fmt::nChw16c, {2, 64, 4, 4}},
    cfg_s32{eng::cpu, fmt::nChw16c, fmt::nchw, {2, 64, 4, 4}})
);

TEST_P(reorder_simple_test_s16_s16, TestsReorder) {
  test_reorder();
}
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_s16_s16,
  ::testing::Values(
    cfg_s16{eng::cpu, fmt::oihw, fmt::OIhw8i16o2i, {64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::OIhw8i16o2i, fmt::oihw, {64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::goihw, fmt::gOIhw8i16o2i, {2, 64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::gOIhw8i16o2i, fmt::goihw, {2, 64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::OIhw8i16o2i, fmt::OIhw8o16i2o, {64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::OIhw8o16i2o, fmt::OIhw8i16o2i, {64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::gOIhw8i16o2i, fmt::gOIhw8o16i2o, {2, 64, 64, 3, 3}},
    cfg_s16{eng::cpu, fmt::gOIhw8o16i2o, fmt::gOIhw8i16o2i, {2, 64, 64, 3, 3}} )
);

TEST_P(reorder_simple_test_s8_s8, TestsReorder) {
  test_reorder();
}
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_s8_s8,
  ::testing::Values(
    cfg_s8{eng::cpu, fmt::oihw, fmt::OIhw4i16o4i, {64, 64, 3, 3}},
    cfg_s8{eng::cpu, fmt::OIhw4i16o4i, fmt::oihw, {64, 64, 3, 3}},
    cfg_s8{eng::cpu, fmt::goihw, fmt::gOIhw4i16o4i, {2, 64, 64, 3, 3}},
    cfg_s8{eng::cpu, fmt::gOIhw4i16o4i, fmt::goihw, {2, 64, 64, 3, 3}})
);
