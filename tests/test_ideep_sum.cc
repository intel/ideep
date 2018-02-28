#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

struct sum_test_params {
  const mkldnn::engine::kind engine_kind;
  std::vector<mkldnn::memory::format> srcs_format;
  mkldnn::memory::format dst_format;
  mkldnn::memory::dims dims;
  std::vector<float> scale;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

template <typename data_t, typename acc_t>
class sum_test: public ::testing::TestWithParam<sum_test_params> {
  void check_data(const std::vector<tensor> &srcs,
                  const std::vector<float> scale,
                  const tensor &dst)
  {
    const data_t *dst_data = (const data_t *)dst.get_data_handle();
    const auto &dst_d =
        mkldnn::memory::desc(*dst.get_mkldnn_memory_desc_t());
    const auto dst_dims = dst.get_dims();

#   pragma omp parallel for collapse(4) schedule(static)
    for (auto n = 0; n < dst_dims[0]; n++)
    for (auto c = 0; c < dst_dims[1]; c++)
    for (auto h = 0; h < dst_dims[2]; h++)
    for (auto w = 0; w < dst_dims[3]; w++) {
      acc_t src_sum = 0.0;
      for (size_t num = 0; num < srcs.size(); num++) {
        const data_t *src_data =
            (const data_t *)srcs[num].get_data_handle();
        const auto &src_d =
            mkldnn::memory::desc(*srcs[num].get_mkldnn_memory_desc_t());
        const auto src_dims = srcs[num].get_dims();

        auto src_idx = w
            + src_dims[3]*h
            + src_dims[2]*src_dims[3]*c
            + src_dims[1]*src_dims[2]*src_dims[3]*n;
        if (num == 0) {
          src_sum = data_t(scale[num]) * src_data[map_index(src_d, src_idx)];
        } else {
          src_sum += data_t(scale[num])* src_data[map_index(src_d, src_idx)];
        }

        src_sum = std::max(std::min(src_sum,
            std::numeric_limits<acc_t>::max()),
            std::numeric_limits<acc_t>::lowest());

      }

      auto dst_idx = w
          + dst_dims[3]*h
          + dst_dims[2]*dst_dims[3]*c
          + dst_dims[1]*dst_dims[2]*dst_dims[3]*n;
      auto diff = src_sum - dst_data[map_index(dst_d, dst_idx)];
      auto e = (std::abs(src_sum) > 1e-4) ? diff / src_sum : diff;
      EXPECT_NEAR(e, 0.0, 1.2e-7);
    }
  }

protected:
  virtual void SetUp() {
    sum_test_params p =
        ::testing::TestWithParam<sum_test_params>::GetParam();

    ASSERT_EQ(p.srcs_format.size(), p.scale.size());
    const auto num_srcs = p.srcs_format.size();

    ASSERT_TRUE(p.engine_kind == mkldnn::engine::kind::cpu);

    std::vector<tensor> srcs;
    for (size_t i = 0; i < num_srcs; i++) {
      tensor src;
      tensor::descriptor src_desc(static_cast<tensor::dims>(p.dims),
          data_traits<data_t>::data_type, static_cast<format>(p.srcs_format[i]));
      src.init(src_desc);

      fill_data<data_t>(src.get_size() / sizeof(data_t),
          reinterpret_cast<data_t *>(src.get_data_handle()));
      srcs.push_back(src);
    }

    tensor dst;
    tensor::descriptor dst_desc(static_cast<tensor::dims>(p.dims),
        data_traits<data_t>::data_type, static_cast<format>(p.dst_format));
    dst.init(dst_desc);
    auto test = [&](){
      // ASSERT_EQ(sum_pd.dst_primitive_desc().desc().data.format,
      //         dst_desc.data.format);
      // ASSERT_EQ(sum_pd.dst_primitive_desc().desc().data.ndims,
      //         dst_desc.data.ndims);

      data_t *dst_data = (data_t *)dst.get_data_handle();
      const size_t sz = dst.get_size() / sizeof(data_t);
      // overwriting dst to prevent false positives for test cases.
# pragma parallel for
      for (size_t i = 0; i < sz; i++) {
        dst_data[i] = -32;
      }

      sum::compute(p.scale, srcs, dst);
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    check_data(srcs, p.scale, dst);
  }
};

namespace mkldnn {
#define INST_TEST_CASE(test) \
TEST_P(test, TestsSum) {} \
INSTANTIATE_TEST_CASE_P(TestSum, test, ::testing::Values( \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw, \
    {0, 8, 2, 2}, {1.0f, 1.0f}, true, mkldnn_invalid_arguments}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw, \
    {1, 0, 2, 2}, {1.0f, 1.0f}, true, mkldnn_invalid_arguments}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw, \
    {2, 8, 2, 2}, {1.0f, 1.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nChw8c, \
    {2, 16, 3, 4}, {1.0f, 1.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nChw8c, \
    {2, 16, 2, 2}, {1.0f, 1.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nchw, \
    {2, 16, 3, 4}, {1.0f, 1.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nchw, \
    {2, 8, 2, 2}, {2.0f, 3.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nChw8c,\
    {2, 16, 3, 4}, {2.0f, 3.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nchw}, memory::format::nChw8c, \
    {2, 16, 2, 2}, {2.0f, 3.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw8c, memory::format::nChw8c}, memory::format::nchw, \
    {2, 16, 3, 4}, {2.0f, 3.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {5, 8, 3, 3}, {2.0f, 3.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nchw, memory::format::nChw8c}, memory::format::nchw, \
    {32, 32, 13, 14}, {2.0f, 3.0f}}, \
    sum_test_params{engine::kind::cpu, \
    {memory::format::nChw16c, memory::format::nChw8c}, \
    memory::format::nChw16c, \
    {2, 16, 3, 3}, {2.0f, 3.0f}} \
));

using sum_test_float = sum_test<float,float>;
using sum_test_u8 = sum_test<uint8_t,float>;
using sum_test_s32 = sum_test<int32_t,float>;

INST_TEST_CASE(sum_test_float)
INST_TEST_CASE(sum_test_u8)
INST_TEST_CASE(sum_test_s32)

#undef INST_TEST_CASE
}
