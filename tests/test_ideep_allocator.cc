/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


#include <numeric>
#include <mkldnn_test_common.hpp>
#include <gtest/gtest.h>
#include <ideep.hpp>

#include "test_ideep_common.hpp"

using namespace ideep;

struct allocator_params {
  enum {
    default_alloc = 0,
    scratch_alloc,
  };
  int alloc;
  test_inner_product_descr_t test_ipd;
  bool expect_to_fail;
  mkldnn_status_t expected_status;
};

template <typename data_t>
class allocator_test : public ::testing::TestWithParam<allocator_params> {
protected:
  virtual void SetUp() {
    auto p = ::testing::TestWithParam<allocator_params>::GetParam();

    auto data_type = data_traits<data_t>::data_type;

    // Test by inner product forward.
    auto ipd = p.test_ipd;
    // Expected format is nchw.
    // To bring reorder in computation, src is initialized by nChw16c.
    auto src_desc =
        tensor::descriptor({ipd.mb, ipd.ic, ipd.kh, ipd.kw}, data_type,
        static_cast<format>(mkldnn::memory::format::nChw16c));
    auto wgt_desc =
        tensor::descriptor({ipd.oc, ipd.ic, ipd.kh, ipd.kw}, data_type,
        static_cast<format>(mkldnn::memory::format::OIhw16o16i));
    auto dst_desc =
        tensor::descriptor({ipd.mb, ipd.oc}, data_type);
    src_.init(src_desc);
    wgt_.init(wgt_desc);
    dst_ref_.init(dst_desc);

    // tensor cases
    std::shared_ptr<tensor> src1(new tensor()), src2(new tensor()),
                            src3(new tensor()), src4(new tensor()),
                            wgt1(new tensor()), wgt2(new tensor());
    src1->init<SCRATCH_ALLOCATOR(convolution_forward)>(src_desc);
    auto raw_src1 = (unsigned long long)src1->get_data_handle();
    src1.reset();
    src2->init<SCRATCH_ALLOCATOR(convolution_forward)>(src_desc);
    auto raw_src2 = (unsigned long long)src2->get_data_handle();
    ASSERT_EQ(raw_src1, raw_src2);

    src3->init<SCRATCH_ALLOCATOR(convolution_backward_data)>(src_desc);
    auto raw_src3 = (unsigned long long)src3->get_data_handle();
    src3.reset();
    src4->init<SCRATCH_ALLOCATOR(convolution_backward_data)>(src_desc);
    auto raw_src4 = (unsigned long long)src4->get_data_handle();
    ASSERT_EQ(raw_src3, raw_src4);

    ASSERT_TRUE(raw_src2 != raw_src4);

    wgt1->init<SCRATCH_ALLOCATOR(convolution_forward)>(wgt_desc);
    auto raw_wgt1 = (unsigned long long)wgt1->get_data_handle();
    wgt1.reset();
    wgt2->init<SCRATCH_ALLOCATOR(convolution_forward)>(wgt_desc);
    auto raw_wgt2 = (unsigned long long)wgt2->get_data_handle();
    ASSERT_EQ(raw_wgt1, raw_wgt2);
    ASSERT_TRUE(raw_wgt2 != raw_src2);

    src2.reset();
    src4.reset();
    wgt2.reset();

    forward();
  }

  void forward() {
    auto p = ::testing::TestWithParam<allocator_params>::GetParam();

    fill_tensor(src_);
    fill_tensor(wgt_);

    tensor dst;
    auto test = [&] () {
      if (p.alloc == allocator_params::default_alloc) {
        inner_product_forward::compute(src_, wgt_, dst);
      } else if (p.alloc == allocator_params::scratch_alloc) {
        inner_product_forward::compute<ideep::utils::scratch_allocator>(
            src_, wgt_, dst);
      } else {
        throw std::invalid_argument("bad arg");
      }
    };

    if (catch_expected_failures(test, p.expect_to_fail, p.expected_status))
      return;

    compute_ref_inner_product_fwd<data_t>(
        p.test_ipd, src_, wgt_, tensor(), dst_ref_);

    compare_tensor<float>(dst_ref_, dst);
  }

  tensor src_, wgt_, dst_ref_;
};

using allocator_test_float = allocator_test<float>;
using allocator_test_params_float = allocator_params;

TEST_P(allocator_test_float, TestsAllocator) {}

INSTANTIATE_TEST_CASE_P(
    TestAllocators, allocator_test_float, ::testing::Values(
  // default alloc
  allocator_test_params_float{ allocator_test_params_float::default_alloc,
  { 256, 256, 96, 3, 3 } },
  // push mem
  allocator_test_params_float{ allocator_test_params_float::scratch_alloc,
  { 256, 256, 96, 3, 3 } },
  // pop mem
  allocator_test_params_float{ allocator_test_params_float::scratch_alloc,
  { 256, 256, 96, 3, 3 } },
  // pop mem in same size
  allocator_test_params_float{ allocator_test_params_float::scratch_alloc,
  { 256, 256, 96, 9, 1 } }
));
