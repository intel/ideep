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


#ifndef _LINEAR_PY_H_
#define _LINEAR_PY_H_

#include <vector>
#include <memory>
#include "param.h"
#include "mdarray.h"
#include "ideep.hpp"

class linear
{
public:
  using scratch_allocator = ideep::utils::scratch_allocator;
  using tensor = ideep::tensor;
  using dims_t = mkldnn::memory::dims;
  using inner_product_forward = ideep::inner_product_forward;
  using inner_product_backward_data = ideep::inner_product_backward_data;
  using inner_product_backward_weights = ideep::inner_product_backward_weights;

  static mdarray Forward(mdarray *src,
                         mdarray *weights,
                         mdarray *bias) {
    tensor dst;
    if (bias)
      inner_product_forward::compute<scratch_allocator>(
          *src->get(), *weights->get(), *bias->get(), dst);
    else
      inner_product_forward::compute<scratch_allocator>(
          *src->get(), *weights->get(), dst);

    auto out = mdarray(dst);
    return out;
  }

  static mdarray BackwardWeights(mdarray *src,
                                 mdarray *grady) {
    tensor gW;
    inner_product_backward_weights::compute<scratch_allocator>(
        *src->get(), *grady->get(), gW);

    auto out = mdarray(gW);
    return out;
  }

  static std::vector<mdarray> BackwardWeightsBias(mdarray *src,
                                                  mdarray *grady) {
    tensor gW, gb;
    inner_product_backward_weights::compute<scratch_allocator>(
        *src->get(), *grady->get(), gW, gb);

    std::vector<mdarray> outs;
    outs.push_back(mdarray(gW));
    outs.push_back(mdarray(gb));
    return outs;
  }

  static mdarray BackwardData(mdarray *weights,
                              mdarray *grady) {
    // TODO: only 2-D supported
    dims_t gradx_dims = {grady->get()->get_dims()[0],
                         weights->get()->get_dims()[1]};

    tensor gradx_tensor;
    inner_product_backward_data::compute<scratch_allocator>(
        *grady->get(), *weights->get(), gradx_dims, gradx_tensor);

    auto out = mdarray(gradx_tensor);
    return out;
  }
};

#endif //_LINEAR_PY_H
