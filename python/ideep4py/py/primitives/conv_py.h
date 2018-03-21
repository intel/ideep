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
#include "op_param.h"
#include "mdarray.h"
#include "conv.h"
#include "ideep.hpp"

using tensor = ideep::tensor;

template <typename T>
class Convolution2D_Py
{
public:
    /*
     * Python Convolution Forward
     * Y = W*X + b
     * params:
     * src: input, x
     * weight: weights, w
     * dst: output, y
     * bias: bias, b
     * cp: convolution parameters
     */
    static mdarray Forward(mdarray *src,
                           mdarray *weights,
                           mdarray *bias,
                           conv_param_t *cp) {
        tensor *src_ = reinterpret_cast<tensor *>(src->get()->tensor());
        tensor *weights_ = reinterpret_cast<tensor *>(weights->get()->tensor());

        // TODO
        // allocate buffer by user
        tensor dst({cp->out_dims, src_->get_data_type()});

        if (bias) {
            tensor *bias_ = reinterpret_cast<tensor *>(bias->get()->tensor());

            dst = ideep::convolution_forward::compute(
                    *src_, *weights_, *bias_,
                    cp->out_dims, dst.get_data_handle(),
                    tensor::dims {cp->sy, cp->sx},
                    tensor::dims {cp->dilate_y, cp->dilate_x},
                    tensor::dims {cp->pad_lh, cp->pad_lw},
                    tensor::dims {cp->pad_rh, cp->pad_rw});
        } else {
            dst = ideep::convolution_forward::compute(
                    *src_, *weights_,
                    cp->out_dims, dst.get_data_handle(),
                    tensor::dims {cp->sy, cp->sx},
                    tensor::dims {cp->dilate_y, cp->dilate_x},
                    tensor::dims {cp->pad_lh, cp->pad_lw},
                    tensor::dims {cp->pad_rh, cp->pad_rw});
        }

        return mdarray(dst);
    }

    /*
     * Python Convolution backward weights
     * gW = gy*x
     * params:
     * src: input, x
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static mdarray BackwardWeights(mdarray *src,
                                   mdarray *diff_dst,
                                   conv_param_t *cp) {
        auto tensor = Convolution2D<T>::BackwardWeights(
                          (src->get()->tensor()),
                          (diff_dst->get()->tensor()), cp);

        auto out = mdarray(tensor);
        return out;
    }

    /*
     * Python Convolution backward weights & bias
     * gW = gy*x
     * params:
     * src: input, x
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static std::vector<mdarray> BackwardWeightsBias(mdarray *src,
                                                    mdarray *diff_dst,
                                                    conv_param_t *cp) {
        std::vector<mdarray> outs;
        auto tensors = Convolution2D<T>::BackwardWeightsBias(
                           (src->get()->tensor()),
                           (diff_dst->get()->tensor()), cp);

        for (int i = 0; i < tensors.size(); i++)
            outs.push_back(mdarray(tensors[i]));

        return outs;
    }

    /*
     * Python Convolution backward data
     * gx = gy*w
     * param:
     * weights: weights, w
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static mdarray BackwardData(mdarray *weights,
                                mdarray *diff_dst,
                                conv_param_t *cp) {
        auto tensor = Convolution2D<T>::BackwardData(
                          (weights->get()->tensor()),
                          (diff_dst->get()->tensor()), cp);

        auto out = mdarray(tensor);
        return out;
    }

};

#endif // _CONV_PY_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
