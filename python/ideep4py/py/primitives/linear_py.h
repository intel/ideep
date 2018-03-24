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
#include "op_param.h"
#include "mdarray.h"
#include "ideep.hpp"

class linear
{
public:
  using scratch_allocator = ideep::utils::scratch_allocator;
  using tensor = ideep::tensor;
  using inner_product_forward = ideep::inner_product_forward;
  using inner_product_backward_data = ideep::inner_product_backward_data;
  using inner_product_backward_weights = ideep::inner_product_backward_weights;

  static mdarray Forward(mdarray *src,
                         mdarray *weights,
                         mdarray *bias) {
    auto dst = bias ?
               inner_product_forward::compute<scratch_allocator>(
                   *src->get(), *weights->get(), *bias->get()) :
               inner_product_forward::compute<scratch_allocator>(
                   *src->get(), *weights->get());

    auto out = mdarray(dst);
    return out;
  }

  static mdarray BackwardWeights(mdarray *src,
                                 mdarray *grady) {
    auto gW = inner_product_backward_weights::compute<scratch_allocator>(
                  *(src->get()), *(grady->get()));

    auto out = mdarray(gW);
    return out;
  }

  static std::vector<mdarray> BackwardWeightsBias(mdarray *src,
                                                  mdarray *grady) {
    auto gWb = inner_product_backward_weights::compute<scratch_allocator>(
                  *(src->get()), *(grady->get()), true);

    std::vector<mdarray> outs;
    outs.push_back(mdarray(gWb.first));
    outs.push_back(mdarray(gWb.second));
    return outs;
  }

  static mdarray BackwardData(mdarray *weights,
                              mdarray *grady) {
    // auto gx = inner_product_backward_data::compute<scratch_allocator>(
    //               *(weights->get()), *(grady->get()),
    //               tensor::dims {grady->get_dims()[0], weights->get_dims()[1]});

    // auto out = mdarray(gx);
    // return out;
  }
};

#endif //_LINEAR_PY_H
