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


#ifndef _CONV_H_
#define _CONV_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "op_param.h"
#include "tensor.h"

template <typename T>
class Convolution2D : public Layer<T>
{
public:
    Convolution2D();
    ~Convolution2D();

    /*
     * Convolution Forward
     * Y = W*X + b
     * params:
     * src: input, x
     * weight: weights, w
     * dst: output, y
     * bias: bias, b
     * cp: convolution parameters
     */
    static Tensor *Forward(Tensor *src, 
                          Tensor *weights, 
                          Tensor *bias,
                          conv_param_t *cp);

    /*
     * Convolution backward weights
     * gW = gy*x
     * params:
     * src: input, x
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static Tensor *BackwardWeights(Tensor *src,
                                   Tensor *diff_dst,
                                   conv_param_t *cp);

    /*
     * Convolution backward weights & bias
     * gW = gy*x
     * params:
     * src: input, x
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static std::vector<Tensor *> BackwardWeightsBias(Tensor *src,
                                                     Tensor *diff_dst,
                                                     conv_param_t *cp);

    /*
     * Convolution backward data
     * gx = gy*w
     * param:
     * weights: weights, w
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static Tensor *BackwardData(Tensor *weights, 
                               Tensor *diff_dst,
                               conv_param_t *cp);

};

#endif // _CONV_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
