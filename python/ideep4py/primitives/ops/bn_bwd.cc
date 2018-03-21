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


#include "mkldnn.hpp"
#include "bn_bwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
void batch_normalization_bwd<T>::setup(mkldnn::memory::dims src_d,
                                       mkldnn::memory::dims diff_dst_d,
                                       float eps, bool scale_shift) {
    flags_ |= scale_shift ? use_scale_shift : 0;

    // memory desc
    auto  src_md = memory::desc({src_d}, memory_data_type<T>(),
                                get_desired_format(bn_size_));
    auto  diff_dst_md = memory::desc({diff_dst_d}, memory_data_type<T>(),
                                     get_desired_format(bn_size_));

    // fwd desc & primitive desc
    auto fwd_desc = batch_normalization_forward::desc(prop_kind::forward_training, src_md, eps, flags_);
    auto fwd_pd = batch_normalization_forward::primitive_desc(fwd_desc, cpu_engine);

    // bwd desc & primitive desc
    auto bwd_desc = batch_normalization_backward::desc(
                    scale_shift ? prop_kind::backward : prop_kind::backward_data,
                    diff_dst_md, src_md, eps, flags_);
    auto bwd_pd = batch_normalization_backward::primitive_desc(
                  bwd_desc, cpu_engine, fwd_pd);

    // memory primitive
    src_mem_.reset(new memory({src_md, cpu_engine}, dummy));
    diff_dst_mem_.reset(new memory({diff_dst_md, cpu_engine}, dummy));
    mean_mem_.reset(new memory(bwd_pd.mean_primitive_desc(), dummy));
    var_mem_.reset(new memory(bwd_pd.variance_primitive_desc(), dummy));
    diff_src_mem_.reset(new memory({src_md, cpu_engine}, dummy));

    // bn bwd primitive
    if ((flags_ & use_scale_shift) && mkldnn_use_scaleshift) {
        w_mem_.reset(new memory(bwd_pd.weights_primitive_desc(), dummy));
        diff_w_mem_.reset(new memory(bwd_pd.diff_weights_primitive_desc(), dummy));

        bn_bwd_.reset(new batch_normalization_backward(bwd_pd, *src_mem_, *mean_mem_,
                      *var_mem_, *diff_dst_mem_, *w_mem_, *diff_src_mem_, *diff_w_mem_));
    } else {
        bn_bwd_.reset(new batch_normalization_backward(bwd_pd, *src_mem_, *mean_mem_,
                      *var_mem_, *diff_dst_mem_, *diff_src_mem_));
    }

    bwd_primitives_.push_back(*bn_bwd_);

    return;
}

template<typename T>
void batch_normalization_bwd<T>::execute(void *src, void *diff_dst,
                                         void *mean, void *var,
                                         void *w, void *diff_src,
                                         void *diff_w) {
    // couple with buffer
    src_mem_->set_data_handle(src);
    diff_dst_mem_->set_data_handle(diff_dst);
    mean_mem_->set_data_handle(mean);
    var_mem_->set_data_handle(var);

    if (flags_ & use_scale_shift) {
        w_mem_->set_data_handle(w);
        diff_w_mem_->set_data_handle(diff_w);
    }

    diff_src_mem_->set_data_handle(diff_src);

    // exec
    bwd_stream_->submit(bwd_primitives_);

    // decouple
    src_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    mean_mem_->set_data_handle(dummy);
    var_mem_->set_data_handle(dummy);

    if (flags_ & use_scale_shift) {
        w_mem_->set_data_handle(dummy);
        diff_w_mem_->set_data_handle(dummy);
    }

    diff_src_mem_->set_data_handle(dummy);

    return;
}

template class batch_normalization_bwd<float>;
