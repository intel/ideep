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


#ifndef _POOLING_PY_H_
#define _POOLING_PY_H_

#include <vector>
#include <memory>
#include "op_param.h"
#include "mdarray.h"
#include "pooling.h"

template <typename T>
class Pooling2D_Py
{
public:
    /*
     * Python Pooling Forward
     * params:
     * src: input, x
     * pp: pooling parameters
     */
    static std::vector<mdarray> Forward(mdarray *src, 
                                        pooling_param_t *pp) {
        std::vector<mdarray> outputs;

        // Shoule be removed in future????
        implementation::mdarray *src_internal = src->get();
        
        std::vector<Tensor *> outputs_tensor = Pooling2D<T>::Forward(
                                                    (src_internal->tensor()),
                                                    pp);
        // FIXME
        //FIXME
        for (int i = 0; i < outputs_tensor.size(); i++) {
            outputs.push_back( mdarray(outputs_tensor[i]) );
        }

        return outputs;
    }

    /*
     * Python Pooling backward
     * param:
     * diff_dst: diff dst, gy
     * ws: workspace
     * pp: pooling parameters
     */
    static mdarray Backward(mdarray *diff_dst,
                            mdarray *ws,
                            pooling_param_t *pp) {
        //FIXME
        //Should be removed in future
        implementation::mdarray *diff_dst_internal = diff_dst->get();
        implementation::mdarray *ws_internal;
        if ( pp->algo_kind == pooling_param_t::algorithm::pooling_max)
            ws_internal = ws->get();
        
        Tensor *diff_src_tensor;
        if ( pp->algo_kind == pooling_param_t::algorithm::pooling_max) {
            diff_src_tensor = Pooling2D<T>::Backward(
                                    (diff_dst_internal->tensor()),
                                    (ws_internal->tensor()),
                                    pp);
        } else {
            diff_src_tensor = Pooling2D<T>::Backward(
                                    (diff_dst_internal->tensor()),
                                    NULL,
                                    pp);
        }

        // FIXME
        // In future, mdarray will have a Tensor member, no need to create a new one
        mdarray diff_src_mdarray = mdarray(diff_src_tensor);
        return diff_src_mdarray;
    }

};

#endif // _POOLING_PY_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
