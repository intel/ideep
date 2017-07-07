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
#include "mkldnn.hpp"
#include "conv_fwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Convolution2DFwd<T>::Convolution2DFwd( mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
                                       mkldnn::memory::dims b_d, mkldnn::memory::dims dst_d,
                                       int dilate_y, int dilate_x,
                                       int sy, int sx,
                                       int pad_lh, int pad_lw, int pad_rh, int pad_rw)
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    // create conv primitive
    if (conv_fwd_ == NULL) {
        setup(src_d, w_d, b_d, dst_d,
                dilate_y, dilate_x,
                sy, sx,
                pad_lh, pad_lw,
                pad_rh, pad_rw);
    }
}

template<typename T>
Convolution2DFwd<T>::~Convolution2DFwd()
{
}

template<typename T>
void Convolution2DFwd<T>::setup(mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
        mkldnn::memory::dims b_d, mkldnn::memory::dims dst_d,
        int dilate_y, int dilate_x,
        int s1, int s2,
        int pl1, int pl2,
        int pr1, int pr2)
{
    //LOG(INFO) << "Convolution forward_setup";
    assert(src_d != NULL);
    assert(w_d != NULL);
    assert(bias_d != NULL); // no bias case, expect as NONE_DIMS, not NULL
    assert(dst_d != NULL);

    dilates_ = {dilate_y, dilate_x};
    strides_ = {s1, s2};
    padding_l_ = {pl1, pl2};
    padding_r_ = {pr1, pr2};

    //LOG(INFO) << "src_d1=" << src_d[0] << ", src_d2=" << src_d[1] << "; src_d3=" << src_d[2] << ", src_d4=" << src_d[3];
    //LOG(INFO) << "w_d1=" << w_d[0] << ", w_d2=" << w_d[1] << "; w_d3=" << w_d[2] << ", w_d4=" << w_d[3];
    //LOG(INFO) << "dst_d1=" << dst_d[0] << ", dst_d2=" << dst_d[1] << "; dst_d3=" << dst_d[2] << ", dst_d4=" << dst_d[3];
    //LOG(INFO) << "dialte_y=" << dilate_y << ", dilate_x=" << dilate_x;
    //LOG(INFO) << "sy=" << s1 << ", sx=" << s2;
    //LOG(INFO) << "pl1=" << pl1 << ", pl2=" << pl2 << ", pr1=" << pr1 << ", pr2=" << pr2;

    /* create memory descriptors for convolution data w/ no specified format */
    src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(),
                                   memory::format::any));
    weights_md_.reset(new memory::desc({w_d},
                                       memory_data_type<T>(), memory::format::any));
    dst_md_.reset(new memory::desc({dst_d}, memory_data_type<T>(),
                                   memory::format::any));
    if (!b_d.empty())
        bias_md_.reset(new memory::desc({b_d}, memory_data_type<T>(),
                                   memory::format::any));
    /* create a convolution */
    if (!b_d.empty()) {
        fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                                                 convolution_direct, *src_md_, *weights_md_, *bias_md_,
                                                 *dst_md_, strides_, dilates_, padding_l_, padding_r_,
                                                 padding_kind::zero));
    } else {
        fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                                                 convolution_direct, *src_md_, *weights_md_,
                                                 *dst_md_, strides_, dilates_, padding_l_, padding_r_,
                                                 padding_kind::zero));
    }

    fwd_pd_.reset(new convolution_forward::primitive_desc(*fwd_desc_, cpu_engine));

    //store the expected memory format
    src_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->src_primitive_desc().desc().data.format);
    weights_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->weights_primitive_desc().desc().data.format);
    dst_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->dst_primitive_desc().desc().data.format);
    
    // create memory primitive based on dummy data
    src_mem_.reset(new memory(fwd_pd_.get()->src_primitive_desc(), dummy));
    weights_mem_.reset(new memory(fwd_pd_.get()->weights_primitive_desc(), dummy));
    dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc(), dummy));

    /* create convolution primitive and add it to net */
    if (!b_d.empty()) {
        bias_mem_.reset(new memory({{{b_d}, memory_data_type<T>(), memory::format::x}, cpu_engine}, dummy));
        conv_fwd_.reset(new convolution_forward(*fwd_pd_, *src_mem_,
                                      *weights_mem_, *bias_mem_, *dst_mem_));
    } else {
        conv_fwd_.reset(new convolution_forward(*fwd_pd_, *src_mem_,
                                      *weights_mem_, *dst_mem_));
    }

    fwd_primitives_.push_back(*conv_fwd_);
    return;
}

template<typename T>
void Convolution2DFwd<T>::execute(void* src, void* w, void* b, void* dst)
{
    //LOG(INFO) << "Convolution forward";
    //LOG(INFO) << "conv_fwd_:" << conv_fwd_;
    //LOG(INFO) << "x=" << x << "; x_size=" << x_d1*x_d2*x_d3*x_d4*4;
    src_mem_->set_data_handle(src);
    weights_mem_->set_data_handle(w);
    bias_mem_->set_data_handle(b);
    dst_mem_->set_data_handle(dst);
    //conv_fwd_->execute();
    fwd_stream_->submit(fwd_primitives_);

    //after exec, set data handle back
    src_mem_->set_data_handle(dummy);
    weights_mem_->set_data_handle(dummy);
    bias_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);

    return;
}

template<typename T>
void Convolution2DFwd<T>::execute(void* src, void* w, void* dst)
{
    //LOG(INFO) << "Convolution forward without bias";
//    LOG(INFO) << conv_fwd_;

    src_mem_->set_data_handle(src);
    weights_mem_->set_data_handle(w);
    dst_mem_->set_data_handle(dst);
    //conv_fwd_->execute();
    fwd_stream_->submit(fwd_primitives_);
    
    //after exec, set data handle back
    src_mem_->set_data_handle(dummy);
    weights_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);
    
    return;
}

template class Convolution2DFwd<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
