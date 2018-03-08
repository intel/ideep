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


#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include "pooling_bwd.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Pooling2DBwd<T>::Pooling2DBwd(mkldnn::memory::dims diff_src_d,
                              mkldnn::memory::dims diff_dst_d,
                              mkldnn::memory::dims ws_d,
                              mkldnn::memory::data_type ws_dt,
                              int ker_h, int ker_w,
                              int sy, int sx,
                              int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                              mkldnn::algorithm alg_kind )
{
    bwd_stream_.reset(new stream(stream::kind::eager));
    // setup
    if ( bwd_ == NULL) 
        setup(diff_src_d, diff_dst_d, ws_d, ws_dt, ker_h, ker_w, sy, sx,
                pad_lh, pad_lw, pad_rh, pad_rw, alg_kind);
}

template<typename T>
Pooling2DBwd<T>::~Pooling2DBwd()
{
}

template<typename T>
void Pooling2DBwd<T>::setup(mkldnn::memory::dims diff_src_d,
                           mkldnn::memory::dims diff_dst_d,
                           mkldnn::memory::dims ws_d,
                           mkldnn::memory::data_type ws_dt,
                           int ker_h, int ker_w,
                           int sy, int sx,
                           int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                           mkldnn::algorithm alg_kind )
{
    //LOG(INFO) << "Pooling backward_setup";

    if (alg_kind != pooling_max && alg_kind != pooling_avg
            && alg_kind != pooling_avg_include_padding && alg_kind != pooling_avg_exclude_padding) {
        //LOG(ERROR) << "alg_kind must be either pooling_max or "
         //          << "pooling_avg";
    }
    
    alg_kind_ = alg_kind;
    memory::dims strides   = {sy, sx};
    memory::dims padding_l = {pad_lh, pad_lw};
    memory::dims padding_r = {pad_rh, pad_rw};
    memory::dims kernel    = {ker_h, ker_w};

    // create memory desc
    diff_src_md_.reset(new memory::desc({diff_src_d}, memory_data_type<T>(),
                            memory::format::any)); //
    // FIXME
    // Pooling doesn't expose to get the diff_dst_primitive_desc, so we need to hard set the fmt for diff dst
    // a util function is used to do this, may be broken the condition in future
    diff_dst_md_.reset(new memory::desc({diff_dst_d}, memory_data_type<T>(),
                            get_desired_format(diff_dst_d[1]))); // use diff dst chanel to decide fmt

    // create a pooling descriptor
    bwd_desc_.reset(new pooling_backward::desc(
                                         alg_kind,
                                         *diff_src_md_, *diff_dst_md_,
                                         strides, kernel, padding_l, padding_r,
                                         padding_kind::zero));
    
    //FIXME
    //Need a forward hint to create backward, will be removed in future
    // create a pooling descriptor
    fwd_desc_.reset(new pooling_forward::desc(prop_kind::forward_training, 
                alg_kind,
                *diff_src_md_, *diff_dst_md_,
                strides, kernel, padding_l, padding_r,
                padding_kind::zero));
    fwd_pd_.reset(new pooling_forward::primitive_desc( *fwd_desc_, cpu_engine));

    bwd_pd_.reset(new pooling_backward::primitive_desc(
                                *bwd_desc_, cpu_engine, *fwd_pd_));

    // store expected primitive format
    diff_src_fmt_ = static_cast<mkldnn::memory::format>(bwd_pd_.get()->diff_src_primitive_desc().desc().data.format);
    diff_dst_fmt_ = get_desired_format(diff_dst_d[1]);

    // create MKL-DNN internal memory object with dummy data
    diff_src_mem_.reset(new memory(bwd_pd_.get()->diff_src_primitive_desc(), dummy));
    diff_dst_mem_.reset(new memory({{{diff_dst_d}, memory_data_type<T>(), diff_dst_fmt_}, cpu_engine}, dummy));
    
    // for max pooling, need to return workspace for backward
    if (alg_kind == pooling_max) {
        //FIXME
        //Pooling backward doesn't expose to get the workspace_primitive_desc, we need to hard set here
        // store workspace's dims and fmt to create ws tensor
        ws_fmt_ = get_desired_format(ws_d[1]);
        ws_mem_.reset(new memory({{{ws_d}, ws_dt, ws_fmt_}, cpu_engine}, dummy)); // use ws dims's channel to decide format
        
        bwd_.reset(new pooling_backward(
                *bwd_pd_, *diff_dst_mem_, *ws_mem_, *diff_src_mem_));
    } else {
        bwd_.reset(new pooling_backward(
                *bwd_pd_, *diff_dst_mem_, *diff_src_mem_));
    }

    bwd_primitives_.push_back(*bwd_);
    return;
}

template<typename T>
void Pooling2DBwd<T>::execute(void *diff_src, void *diff_dst, void *ws)
{
    //LOG(INFO) << "Pooling backward";

    diff_src_mem_->set_data_handle(diff_src); // input
    diff_dst_mem_->set_data_handle(diff_dst); // output dst
    if ( alg_kind_ == pooling_max ) { // max pooling must have ws
        assert(ws!=NULL);
        ws_mem_->set_data_handle(ws); // output workspace
    }
       
    bwd_stream_->submit(bwd_primitives_);

    // set back data handle
    diff_src_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    if ( alg_kind_ == pooling_max ) { // max pooling must have ws
        assert(ws!=NULL);
        ws_mem_->set_data_handle(dummy);
    }
    
    //LOG(INFO) << "Pooling backward finish";
    return;
}

template class Pooling2DBwd<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
