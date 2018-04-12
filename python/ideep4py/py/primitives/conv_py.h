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


#ifndef _CONV_PY_H_
#define _CONV_PY_H_

#include <vector>
#include <memory>
#include "utils.h"
#include "mdarray.h"
#include "ideep.hpp"

class convolution2D
{
public:
  using tensor = ideep::tensor;
  using scratch_allocator = ideep::utils::scratch_allocator;
  using convolution_forward = ideep::convolution_forward;
  using convolution_backward_data = ideep::convolution_backward_data;
  using convolution_backward_weights = ideep::convolution_backward_weights;


  static mdarray Forward(mdarray *src,
                         mdarray *weights,
                         mdarray *bias,
                         conv_param_t *cp) {
    tensor dst;
    if (bias)
      convolution_forward::compute<scratch_allocator>(
          *(src->get()), *(weights->get()),
          *(bias->get()), cp->out_dims, dst,
          tensor::dims {cp->sy, cp->sx},
          tensor::dims {cp->dilate_y, cp->dilate_x},
          tensor::dims {cp->pad_lh, cp->pad_lw},
          tensor::dims {cp->pad_rh, cp->pad_rw});
    else
      convolution_forward::compute<scratch_allocator>(
          *(src->get()), *(weights->get()), cp->out_dims, dst,
          tensor::dims {cp->sy, cp->sx},
          tensor::dims {cp->dilate_y, cp->dilate_x},
          tensor::dims {cp->pad_lh, cp->pad_lw},
          tensor::dims {cp->pad_rh, cp->pad_rw});

    auto out = mdarray(dst);
    return out;
  }


  static mdarray BackwardWeights(mdarray *src,
                                 mdarray *grady,
                                 conv_param_t *cp) {
    tensor gW;
    convolution_backward_weights::compute<scratch_allocator>(
        *(src->get()), *(grady->get()), cp->out_dims, gW,
        tensor::dims {cp->sy, cp->sx},
        tensor::dims {cp->dilate_y, cp->dilate_x},
        tensor::dims {cp->pad_lh, cp->pad_lw},
        tensor::dims {cp->pad_rh, cp->pad_rw});

    auto out = mdarray(gW);
    return out;
  }


  static std::vector<mdarray> BackwardWeightsBias(mdarray *src,
                                                  mdarray *grady,
                                                  conv_param_t *cp) {
    tensor gW, gb;
    convolution_backward_weights::compute<scratch_allocator>(
        *(src->get()), *(grady->get()), cp->out_dims, gW, gb,
        tensor::dims {cp->sy, cp->sx},
        tensor::dims {cp->dilate_y, cp->dilate_x},
        tensor::dims {cp->pad_lh, cp->pad_lw},
        tensor::dims {cp->pad_rh, cp->pad_rw});

    std::vector<mdarray> outs;
    outs.push_back(mdarray(gW));
    outs.push_back(mdarray(gb));

    return outs;
  }


  static mdarray BackwardData(mdarray *weights,
                              mdarray *diff_dst,
                              conv_param_t *cp) {
    tensor gx;
    convolution_backward_data::compute<scratch_allocator>(
        *(diff_dst->get()), *(weights->get()), cp->out_dims, gx,
        tensor::dims {cp->sy, cp->sx},
        tensor::dims {cp->dilate_y, cp->dilate_x},
        tensor::dims {cp->pad_lh, cp->pad_lw},
        tensor::dims {cp->pad_rh, cp->pad_rw});

    auto out = mdarray(gx);
    return out;
  }
};

#endif // _CONV_PY_H_
