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


#ifndef _BN_PY_H_
#define _BN_PY_H_

#include <vector>
#include <memory>
#include <omp.h>
#include "mdarray.h"
#include "mkl_vml_functions.h"
#include "ideep.hpp"

class batch_normalization {
public:
  using tensor = ideep::tensor;
  using alloc = ideep::utils::allocator;
  using descriptor = ideep::param::descriptor;
  using batch_normalization_forward_training = ideep::batch_normalization_forward_training;
  using batch_normalization_forward_inference = ideep::batch_normalization_forward_inference;
  using batch_normalization_backward = ideep::batch_normalization_backward;

  static std::vector<mdarray> Forward(mdarray *src,
                                      mdarray *weights,
                                      mdarray *mean,
                                      mdarray *variance,
                                      float eps) {
    std::vector<mdarray> outs;

    if (mean) {
      auto dst = batch_normalization_forward_inference::compute(*src->get(),
                     *mean->get(), *variance->get(), *weights->get(), eps);

      outs.push_back(mdarray(dst));
    } else {
      auto tensors = batch_normalization_forward_training::compute(*src->get(),
                     *weights->get(), eps);

      auto dst = std::get<0>(tensors);
      auto mean = std::get<1>(tensors);
      auto variance = std::get<2>(tensors);

      tensor inv;
      inv.init({variance.get_dims(), src->get()->get_data_type(),
                descriptor::public_compatible_format(variance.get_descriptor())});

      batch_normalization_inv((float *)variance.get_data_handle(), eps, variance.get_size(),
                              (float *)inv.get_data_handle());

      outs.push_back(mdarray(dst));
      outs.push_back(mdarray(mean));
      outs.push_back(mdarray(variance));
      outs.push_back(mdarray(inv));
    }

    return outs;
  }

  static std::vector<mdarray> Backward(mdarray *src, mdarray *grady,
                                       mdarray *mean, mdarray *variance,
                                       mdarray *weights, float eps) {
    std::vector<mdarray> outs;
    auto tensors = batch_normalization_backward::compute(*src->get(),
                       *mean->get(), *variance->get(), *grady->get(),
                       *weights->get(), eps);

    outs.push_back(mdarray(tensors.first));
    outs.push_back(mdarray(tensors.second));

    return outs;
  }

private:
  static void batch_normalization_inv(float *var, float eps, int size, float *inv) {
    int blk_nthr = omp_get_max_threads(),
      blk_num = blk_nthr,
      blk_len = size / blk_num,
      blk_len_ex = size % blk_num;

    if (!blk_len)
      blk_nthr = size;

    float *var_eps = reinterpret_cast<float *>(alloc::malloc(size * sizeof(float)));

    # pragma omp parallel num_threads(blk_nthr)
    {
      int ithr = omp_get_thread_num();
      int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
      int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
          blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
      int bend = bstart + blen;

      for (int b = bstart; b < bend; b++)
        var_eps[b] = var[b] + eps;
    }

    vsPowx(size, var_eps, -0.5, inv);

    alloc::free(reinterpret_cast<char *>(var_eps));
  }
};

#endif // _BN_PY_H_
