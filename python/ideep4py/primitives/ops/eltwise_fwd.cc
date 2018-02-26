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
#include "eltwise_fwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T1, typename T2>
EltwiseFwd<T1, T2>::EltwiseFwd(mkldnn::memory::dims src_d, mkldnn::algorithm alg_kind, mkldnn::memory::format src_fmt, T2 alpha, T2 beta)
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    // create eltwise primitive
    if (eltwise_fwd_ == nullptr) {
        setup(src_d, alg_kind, src_fmt, alpha, beta);
    }
}

template<typename T1, typename T2>
EltwiseFwd<T1, T2>::~EltwiseFwd()
{
}

template<typename T1, typename T2>
void EltwiseFwd<T1, T2>::setup(mkldnn::memory::dims src_d, mkldnn::algorithm alg_kind, mkldnn::memory::format src_fmt, T2 alpha, T2 beta)
{
    //LOG(INFO) << "Eltwise forward_setup";
    assert(src_d != nullptr);

    /* create memory descriptors for eltwise data w/ no specified format */
    src_md_.reset(new memory::desc({src_d}, memory_data_type<T1>(),
                                   src_fmt));
    src_mpd_.reset(new memory::primitive_desc(*src_md_, cpu_engine));
    /* create a eltwise*/
    fwd_desc_.reset(new eltwise_forward::desc(prop_kind::forward, alg_kind,
                                             *src_md_, alpha, beta));

    fwd_pd_.reset(new eltwise_forward::primitive_desc(*fwd_desc_, cpu_engine));

    //store the expected memory format
    src_fmt_ = src_fmt;
    dst_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->dst_primitive_desc().desc().data.format);
    
    // create memory primitive based on dummy data
    src_mem_.reset(new memory(*src_mpd_, dummy));
    dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc(), dummy));

    /* create eltwise primitive and add it to net */
    eltwise_fwd_.reset(new eltwise_forward(*fwd_pd_, *src_mem_, *dst_mem_));

    fwd_primitives_.push_back(*eltwise_fwd_);
    return;
}

template<typename T1, typename T2>
void EltwiseFwd<T1, T2>::execute(void* src, void* dst)
{
    //LOG(INFO) << "Eltwise forward";

    src_mem_->set_data_handle(src);
    dst_mem_->set_data_handle(dst);
    fwd_stream_->submit(fwd_primitives_);
    
    //after exec, set data handle back
    src_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);
    
    return;
}

template class EltwiseFwd<float, float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
