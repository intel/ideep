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
#include "ideep.hpp"

using tensor = ideep::tensor;

#if 0
inline static mkldnn::memory reorder_if_must(mkldnn::memory user
        , mkldnn::memory::primitive_desc expect
        , std::unique_ptr<mkldnn::memory> &mreorder
        , std::vector<mkldnn::primitive> *dag) {

    if (user.get_primitive_desc() != expect) {
        mkldnn::memory interm(expect);
        dag->push_back(mkldnn::reorder(user, interm));
        return interm;
    }

    return user;
}
#endif

template<typename T>
inline static void axpby(tensor *dst, T a, tensor *x, T b, tensor *y) {
// TODO: computation sum
#if 0
    std::vector<mkldnn::primitive> prims;
    std::unique_ptr<mkldnn::memory> mreorder;

    /// Reorder to x's format
    auto mid = reorder_if_must(y->mkldnn_memory(), x->mkldnn_memory().get_primitive_desc()
            , mreorder, &prims);

    mkldnn::sum::primitive_desc sum_pd(std::vector<float>({(float)a, (float)b})
            , {x->mkldnn_memory().get_primitive_desc(), mid.get_primitive_desc()});

    std::vector<mkldnn::memory::primitive::at> inputs_at {x->mkldnn_memory(), mid};

    mkldnn::sum sum_prim(sum_pd, inputs_at, dst->mkldnn_memory());
    prims.push_back(sum_prim);

    mkldnn::stream s(mkldnn::stream::kind::eager);
    s.submit(prims).wait();
#endif
  return;
}

