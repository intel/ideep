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


#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "op_param.h"
#include "tensor.h"

template <typename T>
class Linear : public Layer<T>
{
public:
    Linear();
    ~Linear();
    /*
     *Linear forward
     * Y = W*X + b
     * params:
     * src: input, x
     * weights: weights, w
     * dst: output, y
     * bias: bias, b
     */
    static Tensor *Forward( Tensor* src,
                            Tensor* weights,
                            Tensor* bias);
    /*
     * Linear backward weights
     * gW = gy*x
     * params:
     * src: input, x
     * diff_dst: diff dst, gy
     */
    static std::vector<Tensor*> BackwardWeights(Tensor* src,
                                                Tensor* diff_dst,
                                                bool need_bias);
    /*
     * Linear backward data
     * gx = gy*w
     * param:
     * weights: weights, w
     * diff_dst: diff dst, gy
     */
    static Tensor *BackwardData(Tensor* weights,
                                Tensor* diff_dst);
};
#endif //_LINEAR_H_

