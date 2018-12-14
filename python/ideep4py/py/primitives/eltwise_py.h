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


#ifndef _ELTWISE_PY_H_
#define _ELTWISE_PY_H_

#include <vector>
#include <memory>
#include "mdarray.h"
#include "ideep.hpp"

class Relu
{
public:
  using scratch_allocator = ideep::utils::scratch_allocator;
  using tensor = ideep::tensor;
  using algorithm = ideep::algorithm;
  using prop_kind = ideep::prop_kind;
  using eltwise_forward = ideep::eltwise_forward;
  using eltwise_backward = ideep::eltwise_backward;

  static mdarray Forward(mdarray &src, float slope = 0.0) {
    tensor dst;
    eltwise_forward::compute<scratch_allocator, _IDEEP4PY_WEB_OPT_>(
                  *src.get(), dst, algorithm::eltwise_relu,
                  prop_kind::forward, slope);

    auto out = mdarray(dst);
    return out;
  }

  static mdarray Backward(mdarray &src, mdarray &grady, float slope = 0.0) {
    tensor gradx;
    eltwise_backward::compute<scratch_allocator, _IDEEP4PY_WEB_OPT_>(
        *src.get(), *grady.get(), gradx, algorithm::eltwise_relu, slope);

    auto out = mdarray(gradx);
    return out;
  }
};

class Tanh
{
public:
  using scratch_allocator = ideep::utils::scratch_allocator;
  using tensor = ideep::tensor;
  using eltwise_forward = ideep::eltwise_forward;
  using eltwise_backward = ideep::eltwise_backward;
  using algorithm = ideep::algorithm;

  static mdarray Forward(mdarray &src) {
    tensor dst;
    eltwise_forward::compute<scratch_allocator, _IDEEP4PY_WEB_OPT_>(
        *(src.get()), dst, algorithm::eltwise_tanh);

    auto out = mdarray(dst);
    return out;
  }

  static mdarray Backward(mdarray &src, mdarray &grady) {
    tensor gradx;
    eltwise_backward::compute<scratch_allocator, _IDEEP4PY_WEB_OPT_>(
        *(src.get()), *(grady.get()), gradx, algorithm::eltwise_tanh);

    auto out = mdarray(gradx);
    return out;
  }
};

#endif
