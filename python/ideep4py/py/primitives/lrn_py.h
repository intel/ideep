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


#ifndef _LRN_PY_H_
#define _LRN_PY_H_

#include <vector>
#include <memory>
#include "utils.h"
#include "mdarray.h"
#include "ideep.hpp"

class localResponseNormalization
{
public:
  using tensor = ideep::tensor;
  using scratch_allocator = ideep::utils::scratch_allocator;
  using lrn_forward = ideep::lrn_forward;
  using lrn_backward = ideep::lrn_backward;
  using algorithm = ideep::algorithm;

  static std::vector<mdarray> Forward(mdarray *src, lrn_param_t *pp) {
    std::vector<mdarray> outs;

    tensor dst;
    lrn_forward::compute<scratch_allocator>(*src->get(), dst, pp->n,
        pp->alpha, pp->beta, pp->k, lrn_algo_convert(pp->algo_kind));

    outs.push_back(mdarray(dst));
    outs.push_back(mdarray(*dst.get_extra()));

    return outs;
  }

  static mdarray Backward(mdarray *src, mdarray *grady, mdarray *ws, lrn_param_t *pp) {
    tensor dst;
    if (ws)
      dst.init_extra(ws->get()->get_descriptor(), ws->get()->get_data_handle());

    tensor gradx;
    lrn_backward::compute<scratch_allocator>(*src->get(), *grady->get(),
        dst, gradx, pp->n, pp->alpha, pp->beta, pp->k,
        lrn_algo_convert(pp->algo_kind));

    auto out = mdarray(gradx);
    return out;
  }

};

#endif // _LRN_PY_H_
