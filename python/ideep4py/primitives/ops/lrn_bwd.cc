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
#include "lrn_bwd.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template <typename T>
LocalResponseNormalizationBwd<T>::LocalResponseNormalizationBwd(
    mkldnn::memory::dims src_d,
    mkldnn::memory::dims diff_dst_d,
    mkldnn::memory::dims ws_d,
    mkldnn::memory::data_type ws_dt,
    int n, double k, double alpha, double beta,
    mkldnn::algorithm alg_kind):alg_kind_(mkldnn::algorithm::lrn_across_channels)
{
    bwd_stream_.reset(new stream(stream::kind::eager));
    // setup
    if ( bwd_ == NULL){
        setup(src_d, diff_dst_d, ws_d, ws_dt, n, k, alpha, beta, alg_kind_);
    }
}

template <typename T>
LocalResponseNormalizationBwd<T>::~LocalResponseNormalizationBwd(){}

template <typename T>
void LocalResponseNormalizationBwd<T>::setup(
    mkldnn::memory::dims src_d, 
    mkldnn::memory::dims diff_dst_d,
    mkldnn::memory::dims ws_d,
    mkldnn::memory::data_type ws_dt,
    int n, double k, double alpha, double beta,
    mkldnn::algorithm alg_kind)
{
    //LOG(INFO) << "lrn backward_setup";

    //LOG(INFO) << "src_d[0]=" << src_d[0] << "; src_d[1]" << src_d[1] << "; src_d[2]=" << src_d[2] << "; src_d[3]=" << src_d[3];
   // LOG(INFO) << "diff_dst_d[0]=" << diff_dst_d[0] << "; diff_dst_d[1]" << diff_dst_d[1] << "; diff_dst_d[2]=" << diff_dst_d[2] << "; diff_dst_d[3]=" << diff_dst_d[3];
   // LOG(INFO) << "ws_d[0]=" << ws_d[0] << "; ws_d[1]" << ws_d[1] << "; ws_d[2]=" << ws_d[2] << "; ws_d[3]=" << ws_d[3];
    
    alg_kind_ = alg_kind;

    // create memory desc
    src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(),
    get_desired_format(src_d[1])));

    diff_dst_md_.reset(new memory::desc({diff_dst_d}, memory_data_type<T>(),
    get_desired_format(diff_dst_d[1]))); // use diff dst chanel to decide fmt

    //Need a forward hint to create backward, will be removed in future
    // create a lrn descriptor
    fwd_desc_.reset(new lrn_forward::desc(prop_kind::forward_training, alg_kind_,
        *diff_dst_md_, n, alpha, beta, k));
    fwd_pd_.reset(new lrn_forward::primitive_desc( *fwd_desc_, cpu_engine));

    bwd_desc_.reset(new lrn_backward::desc(alg_kind_,
        *src_md_, *diff_dst_md_,n, alpha, beta, k));
    bwd_pd_.reset(new lrn_backward::primitive_desc(*bwd_desc_, cpu_engine,
        *fwd_pd_));

    // store expected primitive format
    diff_src_fmt_ = static_cast<mkldnn::memory::format>(bwd_pd_.get()->diff_src_primitive_desc().desc().data.format);
    diff_dst_fmt_ = get_desired_format(diff_dst_d[1]);
    src_fmt_ = get_desired_format(diff_dst_d[1]);

    // create MKL-DNN internal memory object with dummy data
    src_mem_.reset(new memory({{{src_d}, memory_data_type<T>(), src_fmt_}, cpu_engine}, dummy));
    diff_src_mem_.reset(new memory(bwd_pd_.get()->diff_src_primitive_desc(), dummy));
    diff_dst_mem_.reset(new memory({{{diff_dst_d}, memory_data_type<T>(), diff_dst_fmt_}, cpu_engine}, dummy));

    // store workspace's dims and fmt to create ws tensor
    ws_fmt_ = get_desired_format(ws_d[1]);
    ws_mem_.reset(new memory({{{ws_d}, ws_dt, ws_fmt_}, cpu_engine}, dummy)); // use ws dims's channel to decide format
    
    bwd_.reset(new lrn_backward(
            *bwd_pd_, *src_mem_, *diff_dst_mem_, *ws_mem_, *diff_src_mem_));

    bwd_primitives_.push_back(*bwd_);
    return;
}

template<typename T>
void LocalResponseNormalizationBwd<T>::execute(void*src, void *diff_src, void *diff_dst, void *ws)
{
    //LOG(INFO) << "lrn backward";
    
    diff_src_mem_->set_data_handle(diff_src); //
    diff_dst_mem_->set_data_handle(diff_dst); //
    src_mem_->set_data_handle(src);
   
    assert(ws!=NULL);
    ws_mem_->set_data_handle(ws); // output workspace

        
    bwd_stream_->submit(bwd_primitives_);

    // set back data handle
    diff_src_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    src_mem_->set_data_handle(dummy);
    assert(ws!=NULL);
    ws_mem_->set_data_handle(dummy);
    
    //LOG(INFO) << "lrn backward finish";
    return;
}

template class LocalResponseNormalizationBwd<float>;
