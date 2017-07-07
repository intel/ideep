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
#include "pooling_fwd.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Pooling2DFwd<T>::Pooling2DFwd(mkldnn::memory::dims src_d,
                              mkldnn::memory::dims dst_d,
                              int ker_h, int ker_w,
                              int sy, int sx,
                              int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                              mkldnn::algorithm alg_kind )
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    // setup
    if ( fwd_ == NULL) 
        setup(src_d, dst_d, ker_h, ker_w, sy, sx,
                pad_lh, pad_lw, pad_rh, pad_rw, alg_kind);
}

template<typename T>
Pooling2DFwd<T>::~Pooling2DFwd()
{
}

template<typename T>
void Pooling2DFwd<T>::setup(mkldnn::memory::dims src_d,
                           mkldnn::memory::dims dst_d,
                           int ker_h, int ker_w,
                           int sy, int sx,
                           int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                           mkldnn::algorithm alg_kind )
{
    //LOG(INFO) << "Pooling forward_setup";

    if (alg_kind != pooling_max && alg_kind != pooling_avg
            && alg_kind != pooling_avg_include_padding && alg_kind != pooling_avg_exclude_padding) {
        //LOG(ERROR) << "alg_kind must be either pooling_max or "
                   //<< "pooling_avg";
    }
    
    alg_kind_ = alg_kind;
    memory::dims strides   = {sy, sx};
    memory::dims padding_l = {pad_lh, pad_lw};
    memory::dims padding_r = {pad_rh, pad_rw};
    memory::dims kernel    = {ker_h, ker_w};

    // create memory desc
    // FIXME
    // Pooling doesn't expose to get the src_primitive_desc, so we need to hard set the fmt for src
    // a util function is used to do this, may be broken the condition in future
    src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(),
                            get_desired_format(src_d[1]))); // use src's input channel to decide expected fmt
    dst_md_.reset(new memory::desc({dst_d}, memory_data_type<T>(),
                            memory::format::any));

    // create a pooling descriptor
    fwd_desc_.reset(new pooling_forward::desc(prop_kind::forward_training,
                                         alg_kind,
                                         *src_md_, *dst_md_,
                                         strides, kernel, padding_l, padding_r,
                                         padding_kind::zero));

    fwd_pd_.reset(new pooling_forward::primitive_desc(
                                *fwd_desc_, cpu_engine));

    // store expected primitive format
    src_fmt_ = get_desired_format(src_d[1]);
    dst_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->dst_primitive_desc().desc().data.format);

    // create MKL-DNN internal memory object with dummy data
    src_mem_.reset(new memory({{{src_d}, memory_data_type<T>(), src_fmt_}, cpu_engine}, dummy));
    dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc(), dummy));
    
    // for max pooling, need to return workspace for backward
    if (alg_kind == pooling_max) {
        auto ws_pd = fwd_pd_.get()->workspace_primitive_desc().desc().data;

        // store workspace's dims and fmt to create ws tensor
        ws_fmt_ = static_cast<mkldnn::memory::format>(ws_pd.format);
        ws_dims_.assign(ws_pd.dims, ws_pd.dims+ws_pd.ndims);
        ws_dt_ = static_cast<mkldnn::memory::data_type>(ws_pd.data_type);
        ws_size_ = fwd_pd_.get()->workspace_primitive_desc().get_size();

        ws_mem_.reset(new memory(fwd_pd_.get()->workspace_primitive_desc(), dummy));
        fwd_.reset(new pooling_forward(
                *fwd_pd_, *src_mem_, *dst_mem_, *ws_mem_));
    } else {
        fwd_.reset(new pooling_forward(
                *fwd_pd_, *src_mem_, *dst_mem_));
    }

    fwd_primitives_.push_back(*fwd_);
    return;
}

template<typename T>
void Pooling2DFwd<T>::execute(void *src, void *dst, void *ws)
{
    //LOG(INFO) << "Pooling forward";

    src_mem_->set_data_handle(src); // input
    dst_mem_->set_data_handle(dst); // output dst
    if ( alg_kind_ == pooling_max ) { // max pooling must have ws
        assert(ws!=NULL);
        ws_mem_->set_data_handle(ws); // output workspace
    }
       
    fwd_stream_->submit(fwd_primitives_);

    // set back data handle
    src_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);
    if ( alg_kind_ == pooling_max ) { // max pooling must have ws
        assert(ws!=NULL);
        ws_mem_->set_data_handle(dummy);
    }
    
    //LOG(INFO) << "Pooling forward finish";
    return;
}

template class Pooling2DFwd<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
