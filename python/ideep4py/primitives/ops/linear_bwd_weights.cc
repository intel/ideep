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
#include "linear_bwd_weights.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
LinearBwdWeights<T>::LinearBwdWeights(
        mkldnn::memory::dims src_d, mkldnn::memory::dims diff_w_d,
        mkldnn::memory::dims diff_b_d, mkldnn::memory::dims diff_dst_d)
{
    bwd_weights_stream_.reset(new stream(stream::kind::eager));
    //create linear primitive
    if (linear_bwd_weights_ == NULL) {
        setup(src_d, diff_w_d, diff_b_d, diff_dst_d);
    }
}

template<typename T>
LinearBwdWeights<T>::~LinearBwdWeights()
{
}

template<typename T>
void LinearBwdWeights<T>::setup(mkldnn::memory::dims src_d, mkldnn::memory::dims diff_w_d,
        mkldnn::memory::dims diff_b_d, mkldnn::memory::dims diff_dst_d)
{
    assert(src_d != NULL);
    assert(diff_w_d != NULL);
    assert(diff_b_d != NULL);
    assert(diff_dst_d != NULL);

    src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(), memory::format::any));
    diff_weights_md_.reset(new memory::desc({diff_w_d}, memory_data_type<T>(), memory::format::any));
    diff_dst_md_.reset(new memory::desc({diff_dst_d}, memory_data_type<T>(), memory::format::any));
    //LOG(INFO) << "src_d"<<src_d[0] << "," << src_d[1];
    //LOG(INFO) << "weights" << diff_w_d[0] << "," << diff_w_d[1];
    //LOG(INFO) << "dst_d" << diff_dst_d[0] << "," << diff_dst_d[1];
    //create linear
    if (!diff_b_d.empty()) {
        diff_bias_md_.reset(new memory::desc({diff_b_d}, memory_data_type<T>(), memory::format::any));
        bwd_weights_desc_.reset(new inner_product_backward_weights::desc(*src_md_, *diff_weights_md_, 
                    *diff_bias_md_, *diff_dst_md_));
    } else {
        bwd_weights_desc_.reset(new inner_product_backward_weights::desc(*src_md_, *diff_weights_md_,
                    *diff_dst_md_));
    }

    //FIXME
    //jiangzho, Current linear bwd need a fwd pd as hint, will remove in future
    fwd_desc_.reset(new inner_product_forward::desc(prop_kind::forward, *src_md_,
                *diff_weights_md_, *diff_dst_md_));
    fwd_pd_.reset(new inner_product_forward::primitive_desc(*fwd_desc_, cpu_engine));
    bwd_weights_pd_.reset(new inner_product_backward_weights::primitive_desc(*bwd_weights_desc_, cpu_engine, *fwd_pd_));
    
    //store the expected memory format
    src_fmt_ = static_cast<mkldnn::memory::format>(bwd_weights_pd_.get()->src_primitive_desc().desc().data.format);
    diff_weights_fmt_ = static_cast<mkldnn::memory::format>(bwd_weights_pd_.get()->diff_weights_primitive_desc().desc().data.format);
    diff_dst_fmt_ = static_cast<mkldnn::memory::format>(bwd_weights_pd_.get()->diff_dst_primitive_desc().desc().data.format); 

    //create linear primitive and add it to net
    src_mem_.reset(new memory(bwd_weights_pd_.get()->src_primitive_desc(), dummy));
    diff_weights_mem_.reset(new memory(bwd_weights_pd_.get()->diff_weights_primitive_desc(), dummy));  
    diff_dst_mem_.reset(new memory(bwd_weights_pd_.get()->diff_dst_primitive_desc(), dummy));
    //create linear primitive and add it to net
    if (!diff_b_d.empty()) {
        diff_bias_mem_.reset(new memory({{{diff_b_d}, memory_data_type<T>(), memory::format::x}, cpu_engine}, dummy));
        linear_bwd_weights_.reset(new inner_product_backward_weights(*bwd_weights_pd_, *src_mem_, *diff_dst_mem_, 
                    *diff_weights_mem_, *diff_bias_mem_));
    } else {
        linear_bwd_weights_.reset(new inner_product_backward_weights(*bwd_weights_pd_, *src_mem_, *diff_dst_mem_,
                    *diff_weights_mem_));
    }
    bwd_weights_primitives_.push_back(*linear_bwd_weights_);
    return;
}

template<typename T>
void LinearBwdWeights<T>::execute(void* src, void* diff_w, void* diff_b, void* diff_dst)
{
    //LOG(INFO) << "linear backward weights";
    src_mem_->set_data_handle(src);
    diff_weights_mem_->set_data_handle(diff_w);
    diff_bias_mem_->set_data_handle(diff_b);
    diff_dst_mem_->set_data_handle(diff_dst);
    bwd_weights_stream_->submit(bwd_weights_primitives_);
    src_mem_->set_data_handle(dummy);
    diff_weights_mem_->set_data_handle(dummy);
    diff_bias_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    return;
}

template<typename T>
void LinearBwdWeights<T>::execute(void* src, void* diff_w, void* diff_dst)
{
    // LOG(INFO) << "linear  without bias";
    src_mem_->set_data_handle(src);
    diff_weights_mem_->set_data_handle(diff_w);
    diff_dst_mem_->set_data_handle(diff_dst);
    bwd_weights_stream_->submit(bwd_weights_primitives_);
    src_mem_->set_data_handle(dummy);
    diff_weights_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    return;
}

template class LinearBwdWeights<float>;

