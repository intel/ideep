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


#pragma once

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "tensor.h"

typedef enum _eltwise_algorithm {
    ELTWISE_RELU = mkldnn::eltwise_relu,
    ELTWISE_TANH = mkldnn::eltwise_tanh,
    ELTWISE_ELU = mkldnn::eltwise_elu,
    ELTWISE_SQUARE = mkldnn::eltwise_square,
    ELTWISE_ABS = mkldnn::eltwise_abs,
    ELTWISE_SQRT = mkldnn::eltwise_sqrt,
    ELTWISE_LINEAR = mkldnn::eltwise_linear,
    ELTWISE_BOUNDED_RELU = mkldnn::eltwise_bounded_relu,
    ELTWISE_SOFT_RELU = mkldnn::eltwise_soft_relu,
    ELTWISE_LOGISTIC = mkldnn::eltwise_logistic,
} eltwise_algorithm_t;


static inline mkldnn::algorithm ideepy2mkldnn_eltwise_algorithm(eltwise_algorithm_t alg_kind) {
    return (mkldnn::algorithm)alg_kind;
}

template <typename...> class Eltwise;
template <typename T1, typename T2>
class Eltwise<T1, T2> : public Layer<T1>
{
public:
    Eltwise();
    ~Eltwise();
    
    /*
     * Eltwise Forward
     * params:
     * src: input, x
     * dst: output, y
     * y = max(x, 0)
     */
    static Tensor *Forward(Tensor *src, eltwise_algorithm_t alg_kind, T2 alpha, T2 beta); 

    /*
     * Eltwise backward data
     * params:
     * src: input, x
     * diff_dst: input, gy
     * dst: output, gx
     * gx = gy*y
     */
    static Tensor *Backward(Tensor *src, Tensor *diff_dst, eltwise_algorithm_t alg_kind, T2 alpha, T2 beta);
};


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
