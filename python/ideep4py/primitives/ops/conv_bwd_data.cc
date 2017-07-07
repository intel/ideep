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
#include "conv_bwd_data.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Convolution2DBwdData<T>::Convolution2DBwdData(
        mkldnn::memory::dims diff_src_d,
        mkldnn::memory::dims w_d,
        mkldnn::memory::dims diff_dst_d,
        int dilate_y, int dilate_x,
        int sy, int sx,
        int pad_lh, int pad_lw, int pad_rh, int pad_rw)
{
    bwd_data_stream_.reset(new stream(stream::kind::eager));
    // create conv primitive
    if (conv_bwd_data_ == NULL) {
        setup(diff_src_d, w_d, diff_dst_d,
                dilate_y, dilate_x,
                sy, sx,
                pad_lh, pad_lw,
                pad_rh, pad_rw);
    }
}

template<typename T>
Convolution2DBwdData<T>::~Convolution2DBwdData()
{
}

template<typename T>
void Convolution2DBwdData<T>::setup(
        mkldnn::memory::dims diff_src_d, 
        mkldnn::memory::dims w_d,
        mkldnn::memory::dims diff_dst_d,
        int dilate_y, int dilate_x,
        int sy, int sx,
        int pad_lh, int pad_lw,
        int pad_rh, int pad_rw)
{
    //LOG(INFO) << "Convolution backward data setup";
    assert(diff_src_d != NULL);
    assert(w_d != NULL);
    assert(diff_dst_d != NULL);

    dilates_ = {dilate_y, dilate_x};
    strides_ = {sy, sx};
    padding_l_ = {pad_lh, pad_lw};
    padding_r_ = {pad_rh, pad_rw};

    //LOG(INFO) << "diff_src[0]=" << diff_src_d[0] << ", diff_src[1]=" << diff_src_d[1] << ", diff_src[2]=" << diff_src_d[2] << ", diff_src[3]=" << diff_src_d[3];
    //LOG(INFO) << "w[0]=" << w_d[0] << ", w[1]=" << w_d[1] << ", w=" << w_d[2] << ", w[3]=" << w_d[3];
    //LOG(INFO) << "diff_dst[0]=" << diff_dst_d[0] << ", diff_dst[1]=" << diff_dst_d[1] << ", diff_dst[2]=" << diff_dst_d[2] << ", diff_dst[3]=" << diff_dst_d[3];

    //LOG(INFO) << "sy=" << sy << ", sx=" << sx;
   // LOG(INFO) << "pl1=" << pad_lh << ", pl2=" << pad_lw << ", pr1=" << pad_rh << ", pr2=" << pad_rw;

    /* create memory descriptors for convolution data w/ no specified format */
    diff_src_md_.reset(new memory::desc({diff_src_d}, memory_data_type<T>(),
                                   memory::format::any));
    weights_md_.reset(new memory::desc({w_d},
                                       memory_data_type<T>(), memory::format::any));
    diff_dst_md_.reset(new memory::desc({diff_dst_d}, memory_data_type<T>(),
                                   memory::format::any));
    /* create a convolution */
    bwd_data_desc_.reset(new convolution_backward_data::desc(
                    convolution_direct, *diff_src_md_, *weights_md_,
                    *diff_dst_md_, strides_, dilates_, padding_l_, padding_r_, padding_kind::zero));

    // FIXME
    // yli135: Current conv bwd need a fwd pd as hint, will remove in future
    fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                convolution_direct, *diff_src_md_, *weights_md_,
                *diff_dst_md_, strides_, dilates_, padding_l_, padding_r_, padding_kind::zero));
    fwd_pd_.reset(new convolution_forward::primitive_desc(*fwd_desc_, cpu_engine));

    /* create backward conv prim desc*/
    bwd_data_pd_.reset(new convolution_backward_data::primitive_desc(
                *bwd_data_desc_, cpu_engine, *fwd_pd_));


    //store the expected memory format
    diff_src_fmt_ = static_cast<mkldnn::memory::format>(bwd_data_pd_.get()->diff_src_primitive_desc().desc().data.format);
    weights_fmt_ = static_cast<mkldnn::memory::format>(bwd_data_pd_.get()->weights_primitive_desc().desc().data.format);
    diff_dst_fmt_ = static_cast<mkldnn::memory::format>(bwd_data_pd_.get()->diff_dst_primitive_desc().desc().data.format);
    
    // create memory primitive based on dummy data
    diff_src_mem_.reset(new memory(bwd_data_pd_.get()->diff_src_primitive_desc(), dummy));
    weights_mem_.reset(new memory(bwd_data_pd_.get()->weights_primitive_desc(), dummy));
    diff_dst_mem_.reset(new memory(bwd_data_pd_.get()->diff_dst_primitive_desc(), dummy));

    /* create convolution primitive and add it to net */
    conv_bwd_data_.reset(new convolution_backward_data(*bwd_data_pd_, *diff_dst_mem_,
                                      *weights_mem_, *diff_src_mem_));

    bwd_data_primitives_.push_back(*conv_bwd_data_);
    return;
}

template<typename T>
void Convolution2DBwdData<T>::execute(void* diff_src, void* w, void* diff_dst)
{
//    LOG(INFO) << "Convolution forward without bias";
//    LOG(INFO) << conv_fwd_;

    diff_src_mem_->set_data_handle(diff_src);
    weights_mem_->set_data_handle(w);
    diff_dst_mem_->set_data_handle(diff_dst);
    //conv_fwd_->execute();
    bwd_data_stream_->submit(bwd_data_primitives_);

    //set back data handke
    diff_src_mem_->set_data_handle(dummy);
    weights_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);

    return;
}

template class Convolution2DBwdData<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
