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


#include <glog/logging.h>
#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include "tensor.h"
#include "mem.h"
#include "lrn.h"
#include "utils.h"
#include "lrn_fwd.h"
#include "lrn_bwd.h"
#include "prim_factory.h"
#include "reorder_op.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
LocalResponseNormalization<T>::LocalResponseNormalization()
{
}

template<typename T>
LocalResponseNormalization<T>::~LocalResponseNormalization()
{
}

template<typename T>
std::vector<Tensor *> LocalResponseNormalization<T>::Forward(
    Tensor *src, lrn_param_t* pp)
{
    //sanity check for data type
    assert(memory_data_type<T>() == src.cxx_data_type());

    // get a conv2d fwd from primitive pool
    mkldnn::memory::format src_fmt = src->cxx_format(); // src fmt in tensor
    LocalResponseNormalizationFwd<T> *lrn_forward = NULL;
    lrn_forward = LocalResponseNormalizationFwdFactory<T>::get(
        src->dims(), src_fmt,
        pp->n, pp->k,
        pp->alpha, pp->beta, 
        lrn_algo_convert(pp->algo_kind));
    
    // mkldnn::memory::format src_fmt = src->cxx_format(); // src fmt in tensor

    void *src_tmp = src->data();
    shared_ptr<avx::byte> src_reorder;
    
    // check wehther fmt is same
    if (src_fmt == lrn_forward->src_fmt_) {
        //LOG(INFO) << "lrn forward fmt matched";
    } else {
        //LOG(INFO) << "lrn fwd fmt not match, need to reorder";
       // LOG(INFO) << "src_fmt=" << src_fmt <<", lrn_forward->src_fmt_=" << lrn_forward->src_fmt_;
        // FIXME: when to free the reordered memory
        ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src->dims(), src_fmt, lrn_forward->src_fmt_);
        src_reorder = Allocator::malloc(src->len(), MPOOL_REORDER);
        //src_reorder = new avx::byte[src->len()];
        reorder_src_op->execute(src_tmp, src_reorder.get());
        src_tmp = src_reorder.get();
    }

    // create tensor based on primitive's dst 
    // assume dst and src have same data type
    //Tensor *dst_tensor = new Tensor(src->dims(), src->cxx_data_type(), lrn_forward->dst_fmt_, cpu_engine);
    auto data = Allocator::malloc(src->dims(), type2size(src->type()), MPOOL_LRN_FWD);
    Tensor *dst_tensor = new Tensor(src->ndims(), src->dims(), data,
            (mkldnn_memory_format_t)lrn_forward->dst_fmt_,
            src->type());
    
    // do forward
    // to return workspace
    // LOG(INFO) << "ws_dt_=" << lrn_forward->ws_dt_;
    // workspace must be int tensor
    //Tensor *ws_tensor = new Tensor((lrn_forward->ws_dims_), lrn_forward->ws_dt_, lrn_forward->ws_fmt_, cpu_engine);
    auto ws_data = Allocator::malloc(lrn_forward->ws_size_, MPOOL_LRN_FWD);
    Tensor *ws_tensor = new Tensor(lrn_forward->ws_dims_,
            static_cast<mkldnn_data_type_t>(lrn_forward->ws_dt_),
            lrn_forward->ws_fmt_, ws_data);

    lrn_forward->execute(src_tmp, dst_tensor->data(), ws_tensor->data());
    std::vector<Tensor *> outputs;
    outputs.push_back(dst_tensor);
    outputs.push_back(ws_tensor);

    //LOG(INFO) << "Succ exec lrn forward";
    return outputs;
}

template<typename T>
Tensor *LocalResponseNormalization<T>::Backward(
                Tensor *src, Tensor *diff_dst, Tensor *ws, lrn_param_t* pp)
{
    //sanity check
    assert(src->ndims() == diff_dst->ndims());
    assert(src->size() == diff_dst->size());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());

    mkldnn::memory::dims ws_dims;
    mkldnn::memory::data_type ws_dt;
    ws_dims = ws->cxx_dims();
    ws_dt = ws->cxx_data_type();

    // get a conv2d bwd data from primitive pool
    LocalResponseNormalizationBwd<T> *lrn_bwd = NULL;
    lrn_bwd = LocalResponseNormalizationBwdFactory<T>::get(src->dims(), diff_dst->dims(), ws_dims, ws_dt,
            pp->n, pp->k, pp->alpha, pp->beta, lrn_algo_convert(pp->algo_kind));

    // FIXME: in this model, every call to conv_forward will create a new tensor, when to free???
    shared_ptr<avx::byte> ws_reorder;
    mkldnn::memory::format ws_fmt = ws->cxx_format();
    void* ws_tmp = ws->data();
    assert(ws_tmp == NULL);

    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();
    void* diff_dst_tmp = diff_dst->data();
    shared_ptr<avx::byte> diff_dst_reorder;

    if (ws_fmt != lrn_bwd->ws_fmt_) {
        //LOG(INFO) << "lrn bwd data ws fmt not match, need to reorder";
        //LOG(INFO) << "ws_fmt=" << ws_fmt << ", lrn_bwd->ws_fmt_="<< lrn_bwd->ws_fmt_;
        ReorderOp<T>* reorder_ws_op = ReorderFactory<T>::get(ws_dims, ws_fmt, lrn_bwd->ws_fmt_);
        ws_reorder = Allocator::malloc(ws->len(), MPOOL_REORDER);
        //ws_reorder = new avx::byte[ws->len()];
        reorder_ws_op->execute(ws_tmp, ws_reorder.get());
        ws_tmp = ws_reorder.get();
    } 
    if (diff_dst_fmt != lrn_bwd->diff_dst_fmt_) {
        //LOG(INFO) << "lrn bwd data diff dst fmt not match, need to reorder";
        //LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", lrn_bwd->diff_dst_fmt_=" << lrn_bwd->diff_dst_fmt_;
        ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst->dims(), diff_dst_fmt, lrn_bwd->diff_dst_fmt_);
        diff_dst_reorder = Allocator::malloc(diff_dst->len(), MPOOL_REORDER);
        //diff_dst_reorder = new avx::byte[diff_dst->len()];
        reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder.get());
        diff_dst_tmp = diff_dst_reorder.get();
    }
    void *src_buf = src->data();
    shared_ptr<avx::byte> src_reorder;
    if (src->cxx_format() != diff_dst->cxx_format()) {
        //LOG(INFO) << "lrn bwd data src fmt not match, need to reorder";
       // LOG(INFO) << "diff_dst_fmt=" << diff_dst->cxx_format() <<", src format=" << src->cxx_format();
        ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src->dims(), src->cxx_format(), diff_dst->cxx_format());
        //src_reorder = new avx::byte[diff_dst->len()];
        src_reorder = Allocator::malloc(src->len(), MPOOL_REORDER);
        reorder_src_op->execute(src_buf, src_reorder.get());
        src_buf = src_reorder.get();
    }

    // create tensor based on selected primitive
    // assume dst and src have same data type
    //Tensor *diff_src_tensor = new Tensor(src->dims(), diff_dst->cxx_data_type(), lrn_bwd->diff_src_fmt_, cpu_engine);
    auto data = Allocator::malloc(src->dims(), type2size(src->type()), MPOOL_LRN_BWD);
    Tensor *diff_src_tensor = new Tensor(src->ndims(), src->dims(), data,
            (mkldnn_memory_format_t)lrn_bwd->diff_src_fmt_,
            src->type());
    
    lrn_bwd->execute(src_buf, diff_src_tensor->data(), diff_dst_tmp, ws_tmp);

    return diff_src_tensor;
}


template class LocalResponseNormalization<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
