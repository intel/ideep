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


#ifndef _BN_PY_H_
#define _BN_PY_H_

#include <vector>
#include <memory>
#include "op_param.h"
#include "mdarray.h"
#include "bn.h"

template <typename T>
class batch_normalization_py {
public:
    static std::vector<mdarray> Forward(mdarray *src,
        mdarray *w, mdarray *mean, mdarray *var, float eps) {

        std::vector<mdarray> outs;
        auto tensors = batch_normalization<T>::Forward(
                           (src->get()->tensor()),
                           (w ? w->get()->tensor() : nullptr),
                           (mean ? mean->get()->tensor() : nullptr),
                           (var ? var->get()->tensor() : nullptr), eps);

        for (int i = 0; i < tensors.size(); i++)
            outs.push_back(mdarray(tensors[i]));

        return outs;
    }

    static std::vector<mdarray> Backward(mdarray *src, mdarray *diff_dst,
        mdarray *mean, mdarray *var, mdarray *w, float eps) {

        std::vector<mdarray> outs;
        auto tensors = batch_normalization<T>::Backward(
                           (src->get()->tensor()),
                           (diff_dst->get()->tensor()),
                           (mean->get()->tensor()),
                           (var->get()->tensor()),
                           (w ? w->get()->tensor() : nullptr),
                           eps);

        for (int i = 0; i < tensors.size(); i++)
            outs.push_back(mdarray(tensors[i]));

        return outs;
    }
};

#endif // _BN_PY_H_
