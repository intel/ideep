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
#include "mkldnn.hpp"
#include "linear_bwd_data.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
LinearBwdData<T>::LinearBwdData(
        mkldnn::memory::dims diff_src_d,
        mkldnn::memory::dims w_d,
        mkldnn::memory::dims diff_dst_d
        )
{
    bwd_data_stream_.reset(new stream(stream::kind::eager));
    //create linear primitive
    if (linear_bwd_data_ == NULL) {
        setup(diff_src_d, w_d, diff_dst_d);
    }
}
template<typename T>
LinearBwdData<T>::~LinearBwdData()
{
}

template<typename T>
void LinearBwdData<T>::setup(
        mkldnn::memory::dims diff_src_d,
        mkldnn::memory::dims w_d,
        mkldnn::memory::dims diff_dst_d
        ) 
{
    assert(diff_src_d != NULL);
    assert(w_d != NULL);
    assert(diff_dst_d != NULL);

    diff_src_md_.reset(new memory::desc({diff_src_d}, memory_data_type<T>(), memory::format::any));
    weights_md_.reset(new memory::desc({w_d}, memory_data_type<T>(), memory::format::any));
    diff_dst_md_.reset(new memory::desc({diff_dst_d}, memory_data_type<T>(), memory::format::any));
    //LOG(INFO) << "diff_src_d" << diff_src_d[0]<<", "<<diff_src_d[1];
    //LOG(INFO) << "w_d" << w_d[0] << "," << w_d[1];
    //LOG(INFO) << "diff_dst_d" << diff_dst_d[0] << "," << diff_dst_d[1];
    /*create a linear descriptor*/
    bwd_data_desc_.reset(new inner_product_backward_data::desc(*diff_src_md_, *weights_md_, *diff_dst_md_));

    //jiangzho: Current linear bwd need a fwd pd as hint, will remove in future
    fwd_desc_.reset(new inner_product_forward::desc(prop_kind::forward, *diff_src_md_, *weights_md_, *diff_dst_md_));
    fwd_pd_.reset(new inner_product_forward::primitive_desc(*fwd_desc_, cpu_engine));

    /* create backward linear prim desc*/
    bwd_data_pd_.reset(new inner_product_backward_data::primitive_desc(*bwd_data_desc_, cpu_engine, *fwd_pd_));
    
    //store the expected memory format
    diff_src_fmt_ = static_cast<mkldnn::memory::format>(bwd_data_pd_.get()->diff_src_primitive_desc().desc().data.format);
    weights_fmt_ = static_cast<mkldnn::memory::format>(bwd_data_pd_.get()->weights_primitive_desc().desc().data.format);
    diff_dst_fmt_ = static_cast<mkldnn::memory::format>(bwd_data_pd_.get()->diff_dst_primitive_desc().desc().data.format); 

    // create memory primitive based on dummy data
    diff_src_mem_.reset(new memory(bwd_data_pd_.get()->diff_src_primitive_desc(), dummy));
    weights_mem_.reset(new memory(bwd_data_pd_.get()->weights_primitive_desc(), dummy)); 
    diff_dst_mem_.reset(new memory(bwd_data_pd_.get()->diff_dst_primitive_desc(), dummy)); 
    
    //create linear primitive and add it to net
    linear_bwd_data_.reset(new inner_product_backward_data(*bwd_data_pd_, *diff_dst_mem_, *weights_mem_, *diff_src_mem_));
    bwd_data_primitives_.push_back(*linear_bwd_data_);
    return;
}

template<typename T>
void LinearBwdData<T>::execute(void* diff_src, void* w, void* diff_dst)
{
    //LOG(INFO) << "linear fwd without bias"
    diff_src_mem_->set_data_handle(diff_src);
    weights_mem_->set_data_handle(w);
    diff_dst_mem_->set_data_handle(diff_dst);
    //linear_bwd->execute();
    bwd_data_stream_->submit(bwd_data_primitives_);
    diff_src_mem_->set_data_handle(dummy);
    weights_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    return;
}

template class LinearBwdData<float>;


