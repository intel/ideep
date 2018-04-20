#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include <ideep.hpp>
#include "test_ideep_common.hpp"

using namespace ideep;

template <typename data_t>
class concat_test: public ::testing::TestWithParam<concat_test_params> {
using scratch_allocator = ideep::utils::scratch_allocator;

protected:
  virtual void SetUp() {
    concat_test_params p
        = ::testing::TestWithParam<concat_test_params>::GetParam();

    int src_dim_sum = 0;
    for (size_t i = 0; i < p.srcs_cds.size(); i++) {
      for (size_t dim = 0; dim < p.dst_cds.size(); dim++) {
        if (dim == p.concat_dimension)
          src_dim_sum += p.srcs_cds[i][dim];
        else
          ASSERT_TRUE(p.srcs_cds[i][dim] == p.dst_cds[dim]);
      }
    }
    ASSERT_TRUE(src_dim_sum == p.dst_cds[p.concat_dimension]);
    ASSERT_TRUE(p.engine_kind == engine::kind::cpu);

    std::vector<tensor> inputs;
    mkldnn::memory::data_type data_type = data_traits<data_t>::data_type;
    for (size_t i = 0; i < p.srcs_cds.size(); i++) {
      auto input_desc = tensor::descriptor(p.srcs_cds[i],
           data_type, static_cast<format>(p.srcs_format[i]));
      tensor input;
      input.init(input_desc);

      fill_data<data_t>(input.get_size() / sizeof(data_t),
          reinterpret_cast<data_t *>(input.get_data_handle()));

      inputs.push_back(input);
    }

    // auto dst_desc = tensor::descriptor(p.dst_cds, data_type, p.dst_format);
    auto dst = make_output();
    concat::compute(inputs, p.concat_dimension, dst);

    check_data<data_t>(inputs, dst, static_cast<int>(p.concat_dimension));

    // test concat backward
    std::vector<tensor> gxs;
    std::vector<int> axis_len;
    for (size_t i = 0; i < p.srcs_cds.size(); i++) {
      axis_len.push_back(p.srcs_cds[i][p.concat_dimension]);
    }

    tensor::dims offset_dims(p.dst_cds.size(), 0);
    tensor::dims gx_dims(dst.get_dims());

    for (int i = 0; i < p.srcs_cds.size(); i++) {
      gx_dims[p.concat_dimension] = axis_len[i];
      auto gx = ideep::reorder::compute<scratch_allocator>(dst, gx_dims, offset_dims);
      gxs.push_back(gx);
      offset_dims[p.concat_dimension] += axis_len[i];
    }

    check_data<data_t>(gxs, dst, static_cast<int>(p.concat_dimension));
  }

};

using concat_test_float = concat_test<float>;
using concat_test_s8 = concat_test<int8_t>;

TEST_P(concat_test_float, TestsConcat) {}
namespace mkldnn {
INSTANTIATE_TEST_CASE_P(TestConcat, concat_test_float, ::testing::Values(
  concat_test_params{engine::kind::cpu, 1,
  {memory::format::nchw, memory::format::nchw}, memory::format::nchw,
  {{2, 8, 0, 1}, {2, 16, 0, 1}}, {2, 24, 0, 1}},
  concat_test_params{engine::kind::cpu, 1,
  {memory::format::nchw, memory::format::nchw}, memory::format::nchw,
  {{2, 0, 1, 1}, {2, 0, 1, 1}}, {2, 0, 1, 1}},
  concat_test_params{engine::kind::cpu, 1,
  {memory::format::nchw, memory::format::nchw}, memory::format::nchw,
  {{0, 8, 1, 1}, {0, 16, 1, 1}}, {0, 24, 1, 1}}
));
}
