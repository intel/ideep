#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class async_tests:
  public ::testing::TestWithParam<async_test_params<data_t>> {

  using scratch_allocator = ideep::utils::scratch_allocator;
protected:
  void TestMain() {
    // auto p = ::testing::TestWithParam<async_test_params<data_t>>::GetParam();
    tensor::descriptor src_desc({2, 256, 13, 13}, mkldnn::memory::data_type::f32, ideep::nchw);
    tensor::descriptor wgt1_desc({384, 256, 3, 3}, mkldnn::memory::data_type::f32, ideep::oihw);
    tensor::descriptor wgt2_desc({384, 384, 3, 3}, mkldnn::memory::data_type::f32, ideep::oihw);
    tensor::descriptor wgt3_desc({256, 384, 3, 3}, mkldnn::memory::data_type::f32, ideep::oihw);

    tensor src, wgt1, wgt2, wgt3;

    src.init(src_desc);
    wgt1.init(wgt1_desc);
    wgt2.init(wgt2_desc);
    wgt3.init(wgt3_desc);

    auto out_dims1 = {2, 384, 13, 13};
    auto out_dims2 = {2, 384, 13, 13};
    auto out_dims3 = {2, 256, 13, 13};

    test_convolution_sizes_t cs1(2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1);
    test_convolution_sizes_t cs2(2, 1, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1);
    test_convolution_sizes_t cs3(2, 1, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1);

    fill_data<data_t>(
        src.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(src.get_data_handle()),
        data_t(0), data_t(1));

    fill_data<data_t>(
        wgt1.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(wgt1.get_data_handle()),
        data_t(0), data_t(1));

    fill_data<data_t>(
        wgt2.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(wgt2.get_data_handle()),
        data_t(0), data_t(1));

    fill_data<data_t>(
        wgt3.get_size() / sizeof(data_t),
        reinterpret_cast<data_t *>(wgt3.get_data_handle()),
        data_t(0), data_t(1));

    tensor dst1;
    convolution_forward::compute<scratch_allocator, true>(
        src, wgt1, out_dims1, dst1,
        tensor::dims{1, 1}, tensor::dims{0, 0},
        tensor::dims{1, 1}, tensor::dims{1, 1});
    tensor dst2;
    convolution_forward::compute<scratch_allocator, true>(
        dst1, wgt2, out_dims2, dst2,
        tensor::dims{1, 1}, tensor::dims{0, 0},
        tensor::dims{1, 1}, tensor::dims{1, 1});
    tensor dst3;
    convolution_forward::compute<scratch_allocator, true>(
        dst2, wgt3, out_dims3, dst3,
        tensor::dims{1, 1}, tensor::dims{0, 0},
        tensor::dims{1, 1}, tensor::dims{1, 1});

    test_convolution_attr_t ca;
    auto bias_desc = tensor::descriptor({}, mkldnn::memory::data_type::f32);
    tensor bias(bias_desc), ref_dst(dst3.get_descriptor());
    compute_ref_conv_fwd<float, float, float, float>(cs3, ca, dst2, wgt3, bias, ref_dst);
  }
};

using async_test_float = async_tests<float>;
using async_test_params_float = async_test_params<float>;

TEST_P(async_test_float, TestsAsync) {
  // auto p = ::testing::TestWithParam<async_test_params<float>>::GetParam();
  TestMain();
}

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, async_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(Simple_ALEXNET,
  async_test_params_float{"Simple_ALEXNET"}
);
