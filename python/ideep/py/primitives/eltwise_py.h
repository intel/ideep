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

#include <vector>
#include <memory>
#include "mdarray.h"
#include "eltwise.h"

template <typename T>
class Relu_Py
{
public:
    static mdarray Forward(mdarray &src) {
        // Shoule be removed in future????
        implementation::mdarray *src_internal = src.get();
        Tensor *dst_tensor = Eltwise<T, float>::Forward(
                src_internal->tensor(), ELTWISE_RELU, 0.0 , 0.0);

        mdarray dst_mdarray = mdarray(dst_tensor);
        return dst_mdarray;
    }

    static mdarray Backward(mdarray& src, mdarray& diff_dst) {
        //FIXME
        //Should be removed in future
        Tensor *src_tensor = src.get()->tensor();
        Tensor *diff_dst_tensor = diff_dst.get()->tensor();

        Tensor *diff_src_tensor = Eltwise<T, float>::Backward(src_tensor, diff_dst_tensor, ELTWISE_RELU, 0.0, 0.0);

        // FIXME
        // In future, mdarray will have a Tensor member, no need to create a new one
        mdarray diff_src_mdarray = mdarray(diff_src_tensor);
        return diff_src_mdarray;
    }

};

template <typename T>
class Tanh_Py
{
public:
    static mdarray Forward(mdarray &src) {
        // Shoule be removed in future????
        implementation::mdarray *src_internal = src.get();
        Tensor *dst_tensor = Eltwise<T, float>::Forward(
                src_internal->tensor(), ELTWISE_TANH, 0.0 , 0.0); 
        
        mdarray dst_mdarray = mdarray(dst_tensor);
        return dst_mdarray;
    }

    static mdarray Backward(mdarray& src, mdarray& diff_dst) {
        //FIXME
        //Should be removed in future
        Tensor *src_tensor = src.get()->tensor();
        Tensor *diff_dst_tensor = diff_dst.get()->tensor();

        Tensor *diff_src_tensor = Eltwise<T, float>::Backward(src_tensor, diff_dst_tensor, ELTWISE_TANH, 0.0, 0.0);

        // FIXME
        // In future, mdarray will have a Tensor member, no need to create a new one
        mdarray diff_src_mdarray = mdarray(diff_src_tensor);
        return diff_src_mdarray;
    }

};

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
