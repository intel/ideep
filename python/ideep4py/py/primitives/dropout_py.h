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


#ifndef _DROPOUT_PY_H_
#define _DROPOUT_PY_H_

#include <vector>
#include <memory>
#include "param.h"
#include "mdarray.h"
#include "ideep.hpp"

class dropout {
public:
  using dropout_forward = ideep::dropout_forward;
  using dropout_backward = ideep::dropout_backward;

  static std::vector<mdarray> Forward(mdarray *src, float ratio) {
    std::vector<mdarray> outs;
    ideep::tensor dst, mask;
    dropout_forward::compute(*src->get(), ratio, dst, mask);

    outs.push_back(mdarray(mask));
    outs.push_back(mdarray(dst));

    return outs;
  }

  static mdarray Backward(mdarray *mask, mdarray *grady) {
    ideep::tensor gradx;
    dropout_backward::compute(*mask->get(), *grady->get(), gradx);

    auto out = mdarray(gradx);
    return out;
  }
};

#endif // _DROPOUT_PY_H_
