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


#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "tensor.h"
#include "eltwise.h"
#include "eltwise_fwd.h"
#include "eltwise_bwd.h"
#include "prim_factory.h"
#include "reorder_op.h"

using namespace mkldnn;

const mkldnn::memory::dims NONE_DIMS = {};
extern engine cpu_engine;

template<typename T1, typename T2>
Eltwise<T1, T2>::Eltwise()
{
}

template<typename T1, typename T2>
Eltwise<T1, T2>::~Eltwise()
{
}

template<typename T1, typename T2>
Tensor *Eltwise<T1, T2>::Forward(Tensor *src, eltwise_algorithm_t alg_kind, T2 alpha, T2 beta)
{
    //sanity check for data type
    assert(memory_data_type<T1>() == src.cxx_data_type());

    // get a eltwise fwd from primitive pool
    EltwiseFwd<T1, T2> *eltwise_fwd = nullptr;
    // FIXME: in this model, every call to eltwise_fwd will create a new tensor, when to free???
    mkldnn::memory::format src_fmt = src->cxx_format(); // src fmt in tensor
    mkldnn::algorithm malg_kind = ideepy2mkldnn_eltwise_algorithm(alg_kind);
    eltwise_fwd = EltwiseFwdFactory<T1, T2>::get(src->dims(), malg_kind, src_fmt, alpha, beta);

    // create tensor based on primitive's dst 
    // assume dst and src have same data type
    auto data = Allocator::malloc(src->dims(), type2size(src->type()), MPOOL_ELTWISE_FWD);
    Tensor *dst_tensor = new Tensor(src->ndims(), src->dims(), data,
            (mkldnn_memory_format_t)eltwise_fwd->dst_fmt_,
            src->type());

    // do forward
    eltwise_fwd->execute(src->data(), dst_tensor->data());

    return dst_tensor;
}

template<typename T1, typename T2>
Tensor *Eltwise<T1, T2>::Backward(Tensor *src, Tensor *diff_dst, eltwise_algorithm_t alg_kind, T2 alpha, T2 beta)
{
    // sanity check for data type
    assert(memory_data_type<T1>() == diff_dst->cxx_data_type());
    assert(src->ndims() == diff_dst->ndims());
    assert(src->size() == diff_dst->size());

    // get a eltwise bwd data from primitive pool
    EltwiseBwd<T1, T2> *eltwise_bwd = nullptr;
    mkldnn::algorithm malg_kind = ideepy2mkldnn_eltwise_algorithm(alg_kind);
    eltwise_bwd = EltwiseBwdFactory<T1, T2>::get(diff_dst->dims(), malg_kind, diff_dst->cxx_format(), alpha, beta);

    void *src_buf = src->data();

    if (src->cxx_format() != diff_dst->cxx_format()) {
        //LOG(INFO) << "eltwise bwd data fmt not match, need to reorder";
        //LOG(INFO) << "diff_dst_fmt=" << diff_dst->cxx_format() <<", src format=" << src->cxx_format();
        ReorderOp<T1>* reorder_src_op = ReorderFactory<T1>::get(src->dims(), src->cxx_format(), diff_dst->cxx_format());
        //src_reorder = new avx::byte[diff_dst->len()];
        auto src_reorder = Allocator::malloc(diff_dst->len(), MPOOL_REORDER);
        reorder_src_op->execute(src_buf, src_reorder.get());
        src_buf = static_cast<void *>(src_reorder.get());
    }

    // create tensor based on selected primitive
    // assume dst and src have same data type
    auto data = Allocator::malloc(src->dims(), type2size(src->type()), MPOOL_ELTWISE_BWD);
    Tensor *diff_src = new Tensor(src->ndims(), src->dims(), data,
                                    (mkldnn_memory_format_t)eltwise_bwd->src_diff_fmt_,
                                    src->type());
    
    eltwise_bwd->execute(src_buf, diff_dst->data(), diff_src->data());

    return diff_src;
}

template class Eltwise<float, float>;

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
