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
#include "conv.h"
#include "utils.h"
#include "conv_fwd.h"
#include "conv_bwd_data.h"
#include "conv_bwd_weights.h"
#include "prim_factory.h"
#include "reorder_op.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Convolution2D<T>::Convolution2D()
{
}

template<typename T>
Convolution2D<T>::~Convolution2D()
{
}

template<typename T>
Tensor *Convolution2D<T>::Forward(
                Tensor *src, Tensor *weights,
                Tensor *bias,
                conv_param_t *cp)
{
    // sanity check
    mkldnn::memory::dims src_dims = (mkldnn::memory::dims)src->dims();
    mkldnn::memory::dims w_dims = (mkldnn::memory::dims)weights->dims();
    mkldnn::memory::dims dst_dims = (mkldnn::memory::dims)cp->out_dims;
    mkldnn::memory::dims b_dims;
    if (bias)
        b_dims = (mkldnn::memory::dims)bias->dims();

    //sanity check for data type
    //assuem all x/w/b should have same data type as T
    //FIXME
    //yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == weights->cxx_data_type());
    if (bias)
        assert(memory_data_type<T>() == bias->cxx_data_type());
    
    // get a conv2d fwd from primitive pool
    Convolution2DFwd<T> *conv2d_forward = NULL;
    if (bias)
        conv2d_forward = Convolution2DFwdFactory<T>::get(src_dims, w_dims, b_dims, dst_dims,
                cp->dilate_y, cp->dilate_x, cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);
    else
        conv2d_forward = Convolution2DFwdFactory<T>::get(src_dims, w_dims, NONE_DIMS, dst_dims,
                cp->dilate_y, cp->dilate_x, cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);
    
    // FIXME: in this model, every call to conv_forward will create a new tensor, when to free???
    mkldnn::memory::format src_fmt = src->cxx_format(); // src fmt in tensor
    mkldnn::memory::format w_fmt = weights->cxx_format(); // weight fmt in tensor

    void *src_tmp = src->data();
    void *w_tmp = weights->data();
    shared_ptr<avx::byte> src_reorder;
    shared_ptr<avx::byte> w_reorder;
    
    // check wehther fmt is same
    if (src_fmt == conv2d_forward->src_fmt_ && w_fmt == conv2d_forward->weights_fmt_) {
        //LOG(INFO) << "primitive fmt matched";
    } else {
        //LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_forward->src_fmt_) {
            //LOG(INFO) << "src_fmt=" << src_fmt <<", conv2d_forward->src_fmt_=" << conv2d_forward->src_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, conv2d_forward->src_fmt_);
            src_reorder = Allocator::malloc(src->len(), MPOOL_REORDER);
            //src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder.get());
            src_tmp = src_reorder.get();
        }

        if (w_fmt != conv2d_forward->weights_fmt_) {
            //LOG(INFO) << "weight_fmt=" << w_fmt <<", conv2d_forward->weight_fmt_=" << conv2d_forward->weights_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_w_op = ReorderFactory<T>::get(w_dims, w_fmt, conv2d_forward->weights_fmt_);
            w_reorder = Allocator::malloc(weights->len(), MPOOL_REORDER);
            //w_reorder = new avx::byte[weights->len()];
            reorder_w_op->execute(w_tmp, w_reorder.get());
            w_tmp = w_reorder.get();
            
            
            // set internal fmt back to weight tensor
            weights->reset_memory(
                    static_cast<mkldnn_memory_format_t>(conv2d_forward->weights_fmt_),
                    w_reorder);
        }
    }

    // create tensor based on primitive's dst 
    // assume dst and src have same data type
    //Tensor *dst_tensor = new Tensor(dst_dims, src->cxx_data_type(), conv2d_forward->dst_fmt_, cpu_engine);
    auto data = Allocator::malloc(dst_dims, type2size(src->type()), MPOOL_CONV_FWD);
    Tensor *dst_tensor = new Tensor(dst_dims.size(), dst_dims, data,
            (mkldnn_memory_format_t)conv2d_forward->dst_fmt_,
            src->type());
    
    // do forward
    if (bias) {
        conv2d_forward->execute(src_tmp, w_tmp, bias->data(), dst_tensor->data());
    } else {
        conv2d_forward->execute(src_tmp, w_tmp, dst_tensor->data());
    }

    return dst_tensor;
}

/*
 * gW = gy *x
 */
template<typename T>
Tensor *Convolution2D<T>::BackwardWeights(
                Tensor *src, Tensor *diff_dst,
                conv_param_t *cp)
{
    std::vector<Tensor *> bwd_weight_vec;

    // sanity check
    mkldnn::memory::dims src_dims = (mkldnn::memory::dims)src->dims();
    mkldnn::memory::dims diff_dst_dims = (mkldnn::memory::dims)diff_dst->dims();
    mkldnn::memory::dims diff_w_dims = (mkldnn::memory::dims)cp->out_dims;

    assert(src_dims == src->cxx_dims() && diff_dst_dims = diff_dst->cxx_dims());

    // sanity check for data type
    // FIXME
    // is it possible y and w have different data type??
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());

    // get a conv2d bwd weights from primitive pool
    Convolution2DBwdWeights<T> *conv2d_bwd_weights = NULL;
    conv2d_bwd_weights = Convolution2DBwdWeightsFactory<T>::get(src_dims, diff_w_dims, NONE_DIMS, diff_dst_dims,
        cp->dilate_y, cp->dilate_x, cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);

    // create tensor based on selected primitive
    mkldnn::memory::format src_fmt = src->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();

    //assum dst and src have same data type
    void* src_tmp = src->data();
    void* diff_dst_tmp = diff_dst->data();
    shared_ptr<avx::byte> src_reorder;
    shared_ptr<avx::byte> diff_dst_reorder;

    //check whether fmt is same
    if (src_fmt == conv2d_bwd_weights->src_fmt_ && diff_dst_fmt == conv2d_bwd_weights->diff_dst_fmt_) {
       // LOG(INFO) << "primitive fmt matched";
    } else {
       // LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_bwd_weights->src_fmt_) {
            //LOG(INFO) << "src_fmt=" << src_fmt << ", conv2d_bwd_weights->src_fmt_=" << conv2d_bwd_weights->src_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, conv2d_bwd_weights->src_fmt_);
            src_reorder = Allocator::malloc(src->len(), MPOOL_REORDER);
            //src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder.get());
            src_tmp = src_reorder.get();
        }
        if (diff_dst_fmt != conv2d_bwd_weights->diff_dst_fmt_) {
           // LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_weights->diff_dst_fmt_=" << conv2d_bwd_weights->diff_dst_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, conv2d_bwd_weights->diff_dst_fmt_);
            diff_dst_reorder = Allocator::malloc(diff_dst->len(), MPOOL_REORDER);
            //diff_dst_reorder = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder.get());
            diff_dst_tmp = diff_dst_reorder.get();
        }
    }

    //assum dst and src have same data type
    //Tensor *diff_w_tensor = new Tensor(diff_w_dims, src->cxx_data_type(), conv2d_bwd_weights->diff_weights_fmt_, cpu_engine);
    auto w_data = Allocator::malloc(diff_w_dims, type2size(src->type()), MPOOL_CONV_BWD);
    Tensor *diff_w_tensor = new Tensor(diff_w_dims.size(), diff_w_dims, w_data,
            (mkldnn_memory_format_t)conv2d_bwd_weights->diff_weights_fmt_,
            src->type());

    // do execute
    conv2d_bwd_weights->execute(src_tmp, diff_w_tensor->data(), diff_dst_tmp);
    return diff_w_tensor;
}

template<typename T>
std::vector<Tensor *> Convolution2D<T>::BackwardWeightsBias(
                Tensor *src, Tensor *diff_dst,
                conv_param_t *cp)
{
    std::vector<Tensor *> bwd_weight_vec;

    // sanity check
    mkldnn::memory::dims src_dims = (mkldnn::memory::dims)src->dims();
    mkldnn::memory::dims diff_dst_dims = (mkldnn::memory::dims)diff_dst->dims();
    mkldnn::memory::dims diff_w_dims = (mkldnn::memory::dims)cp->out_dims;
    mkldnn::memory::dims diff_b_dims = {diff_w_dims[0]};

    assert(src_dims == src->cxx_dims() && diff_dst_dims = diff_dst->cxx_dims());

    // sanity check for data type
    // FIXME
    // is it possible y and w have different data type??
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());

    // get a conv2d bwd weights from primitive pool
    Convolution2DBwdWeights<T> *conv2d_bwd_weights = NULL;
    conv2d_bwd_weights = Convolution2DBwdWeightsFactory<T>::get(src_dims, diff_w_dims, diff_b_dims, diff_dst_dims,
        cp->dilate_y, cp->dilate_x, cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);

    // create tensor based on selected primitive
    mkldnn::memory::format src_fmt = src->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();

    //assum dst and src have same data type
    void* src_tmp = src->data();
    void* diff_dst_tmp = diff_dst->data();
    shared_ptr<avx::byte> src_reorder;
    shared_ptr<avx::byte> diff_dst_reorder;

    //check whether fmt is same
    if (src_fmt == conv2d_bwd_weights->src_fmt_ && diff_dst_fmt == conv2d_bwd_weights->diff_dst_fmt_) {
       // LOG(INFO) << "primitive fmt matched";
    } else {
       // LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_bwd_weights->src_fmt_) {
            //LOG(INFO) << "src_fmt=" << src_fmt << ", conv2d_bwd_weights->src_fmt_=" << conv2d_bwd_weights->src_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, conv2d_bwd_weights->src_fmt_);
            src_reorder = Allocator::malloc(src->len(), MPOOL_REORDER);
            //src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder.get());
            src_tmp = src_reorder.get();
        }
        if (diff_dst_fmt != conv2d_bwd_weights->diff_dst_fmt_) {
           // LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_weights->diff_dst_fmt_=" << conv2d_bwd_weights->diff_dst_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, conv2d_bwd_weights->diff_dst_fmt_);
            diff_dst_reorder = Allocator::malloc(diff_dst->len(), MPOOL_REORDER);
            //diff_dst_reorder = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder.get());
            diff_dst_tmp = diff_dst_reorder.get();
        }
    }

    //assum dst and src have same data type
    //Tensor *diff_w_tensor = new Tensor(diff_w_dims, src->cxx_data_type(), conv2d_bwd_weights->diff_weights_fmt_, cpu_engine);
    auto w_data = Allocator::malloc(diff_w_dims, type2size(src->type()), MPOOL_CONV_BWD);
    Tensor *diff_w_tensor = new Tensor(diff_w_dims.size(), diff_w_dims, w_data,
            (mkldnn_memory_format_t)conv2d_bwd_weights->diff_weights_fmt_,
            src->type());

    auto b_data = Allocator::malloc(diff_b_dims, type2size(src->type()), MPOOL_CONV_BWD);
    Tensor *diff_b_tensor = new Tensor(diff_b_dims.size(), diff_b_dims, b_data,
            (mkldnn_memory_format_t)mkldnn::memory::format::x, src->type());

    conv2d_bwd_weights->execute(src_tmp, diff_w_tensor->data(), diff_b_tensor->data(), diff_dst_tmp);
    bwd_weight_vec.push_back(diff_w_tensor);
    bwd_weight_vec.push_back(diff_b_tensor);

    return bwd_weight_vec;
}

template<typename T>
Tensor *Convolution2D<T>::BackwardData(
                Tensor *weights, Tensor *diff_dst,
                conv_param_t *cp)
{
    //sanity check
    mkldnn::memory::dims diff_src_dims = (mkldnn::memory::dims)cp->out_dims;
    mkldnn::memory::dims w_dims = (mkldnn::memory::dims)weights->dims();
    mkldnn::memory::dims diff_dst_dims = (mkldnn::memory::dims)diff_dst->dims();
    assert(w_dims == weights->cxx_dims() && diff_dst_dims == diff_dst->cxx_dims());

    // sanity check for data type
    // assuem all x/w/b should have same data type as T
    // FIXME
    // yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == weights->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());

    // get a conv2d bwd data from primitive pool
    Convolution2DBwdData<T> *conv2d_bwd_data = NULL;
    conv2d_bwd_data = Convolution2DBwdDataFactory<T>::get( diff_src_dims, w_dims, diff_dst_dims,
            cp->dilate_y, cp->dilate_x, cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);

    // FIXME: in this model, every call to conv_forward will create a new tensor, when to free???
    mkldnn::memory::format w_fmt = weights->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();
    
    void* w_tmp = weights->data();
    void* diff_dst_tmp = diff_dst->data();
    shared_ptr<avx::byte> w_reorder;
    shared_ptr<avx::byte> diff_dst_reorder;

    if (w_fmt == conv2d_bwd_data->weights_fmt_ && diff_dst_fmt == conv2d_bwd_data->diff_dst_fmt_) {
        //LOG(INFO) << "conv2d bwd data primitive fmt matched";
    } else {
        //LOG(INFO) << "conv2d bwd data fmt not match, need to reorder";

        if (w_fmt != conv2d_bwd_data->weights_fmt_) {
            //LOG(INFO) << "weight_fmt=" << w_fmt << ", conv2d_bwd_data->weights_fmt_="<< conv2d_bwd_data->weights_fmt_;
            ReorderOp<T>* reorder_w_op = ReorderFactory<T>::get(w_dims, w_fmt, conv2d_bwd_data->weights_fmt_);
            w_reorder = Allocator::malloc(weights->len(), MPOOL_REORDER);
            //w_reorder = new avx::byte[weights->len()];
            reorder_w_op->execute(w_tmp, w_reorder.get());
            w_tmp = w_reorder.get();
        } 
        if (diff_dst_fmt != conv2d_bwd_data->diff_dst_fmt_) {
            //LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_data->diff_dst_fmt_=" << conv2d_bwd_data->diff_dst_fmt_;
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, conv2d_bwd_data->diff_dst_fmt_);
            diff_dst_reorder = Allocator::malloc(diff_dst->len(), MPOOL_REORDER);
            //diff_dst_reorder = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder.get());
            diff_dst_tmp = diff_dst_reorder.get();
        }
    }

    // create tensor based on selected primitive
    // assume dst and src have same data type
    //Tensor *diff_src_tensor = new Tensor(diff_src_dims, diff_dst->cxx_data_type(), conv2d_bwd_data->diff_src_fmt_, cpu_engine);
    auto data = Allocator::malloc(diff_src_dims, type2size(diff_dst->type()), MPOOL_CONV_BWD);
    Tensor *diff_src_tensor = new Tensor(diff_src_dims.size(), diff_src_dims, data,
            (mkldnn_memory_format_t)conv2d_bwd_data->diff_src_fmt_,
            diff_dst->type());
    
    conv2d_bwd_data->execute(diff_src_tensor->data(), w_tmp, diff_dst_tmp);

    return diff_src_tensor;
}


template class Convolution2D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
