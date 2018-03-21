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


#ifndef _CONCAT_PY_H_
#define _CONCAT_PY_H_

#include <vector>
#include <memory>
#include "mdarray.h"
#include "concat.h"

template <typename T>
class Concat_Py
{
public:
    /*
     * Python Concat Forward
     * params:
     * src: input, xs
     * axis
     */
    static mdarray Forward(std::vector<mdarray> src, int axis) {
        std::vector<Tensor*> src_tensor;

        for (int i = 0; i < src.size(); i++) {
            src_tensor.push_back(src[i].get()->tensor());
        }

        Tensor *dst_tensor = Concat<T>::Forward(src_tensor, axis);

        mdarray dst_mdarray = mdarray(dst_tensor);
        return dst_mdarray;
    }

    /*
     * Python Concat Backward
     */
    static std::vector<mdarray> Backward(mdarray *diff_dst,
                                         std::vector<int> offsets,
                                         int axis) {
        std::vector<mdarray> gxs;

        std::vector<Tensor *> gxs_tensor = Concat<T>::Backward(
                                            (diff_dst->get()->tensor()),
                                            offsets,
                                            axis);

        //
        for (int i = 0; i < gxs_tensor.size(); i++){
            gxs.push_back(mdarray(gxs_tensor[i]));
        }

        return gxs;
    }

};

#endif // _CONCAT_PY_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
