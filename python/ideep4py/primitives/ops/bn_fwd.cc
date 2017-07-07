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
#include "bn_fwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
void batch_normalization_fwd<T>::setup(mkldnn::memory::dims src_d,
                                       float eps, bool scale_shift,
                                       bool global_stats, bool training) {

    flags_ |= scale_shift ? use_scale_shift : 0;
    flags_ |= global_stats ? use_global_stats : 0;

    pkind_ = training ?
             prop_kind::forward_training :
             prop_kind::forward_scoring;

    // memory desc
    auto src_md = memory::desc({src_d}, memory_data_type<T>(), get_desired_format(src_d[1]));

    // fwd desc & primitive desc
    auto fwd_desc = batch_normalization_forward::desc(pkind_, src_md, eps, flags_);
    auto fwd_pd = batch_normalization_forward::primitive_desc(fwd_desc, cpu_engine);

    // memory primitive
    src_mem_.reset(new memory({src_md, cpu_engine}, dummy));
    dst_mem_.reset(new memory(fwd_pd.dst_primitive_desc(), dummy));

    if (flags_ & use_scale_shift)
        w_mem_.reset(new memory(fwd_pd.weights_primitive_desc(), dummy));

    if (training || (flags_ & use_global_stats)) {
        mean_mem_.reset(new memory(fwd_pd.mean_primitive_desc(), dummy));
        var_mem_.reset(new memory(fwd_pd.variance_primitive_desc(), dummy));
    }

    // bn fwd primitive
    if (!training && !(flags_ & use_global_stats)) {
        if ((flags_ & use_scale_shift) && mkldnn_use_scaleshift) {
            bn_fwd_.reset(new batch_normalization_forward(
                          fwd_pd, *src_mem_, *w_mem_, *dst_mem_));
        } else {
            bn_fwd_.reset(new batch_normalization_forward(
                          fwd_pd, *src_mem_, *dst_mem_));
        }
    } else if (flags_ & use_global_stats) {
        if ((flags_ & use_scale_shift) && mkldnn_use_scaleshift) {
            bn_fwd_.reset(new batch_normalization_forward(
                          fwd_pd, *src_mem_, (const primitive::at)*mean_mem_,
                          (const primitive::at)*var_mem_, *w_mem_, *dst_mem_));
        } else {
            bn_fwd_.reset(new batch_normalization_forward(
                          fwd_pd, *src_mem_, (const primitive::at)*mean_mem_,
                          (const primitive::at)*var_mem_, *dst_mem_));
        }
    } else {
        if ((flags_ & use_scale_shift) && mkldnn_use_scaleshift) {
            bn_fwd_.reset(new batch_normalization_forward(
                          fwd_pd, *src_mem_, *w_mem_, *dst_mem_, *mean_mem_, *var_mem_));
        } else {
            bn_fwd_.reset(new batch_normalization_forward(
                          fwd_pd, *src_mem_, *dst_mem_, *mean_mem_, *var_mem_));
        }
    }

    fwd_primitives_.push_back(*bn_fwd_);

    return;
}

template<typename T>
void batch_normalization_fwd<T>::execute(void *src, void *w, void *dst,
                                         void *mean, void *var) {
    // couple with buffer
    src_mem_->set_data_handle(src);
    dst_mem_->set_data_handle(dst);

    if (flags_ & use_scale_shift)
        w_mem_->set_data_handle(w);

    if ((pkind_ == prop_kind::forward_training) ||
        (flags_ & use_global_stats)) {
        mean_mem_->set_data_handle(mean);
        var_mem_->set_data_handle(var);
    }

    // exec
    fwd_stream_->submit(fwd_primitives_);

    // decouple
    src_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);

    if (flags_ & use_scale_shift)
        w_mem_->set_data_handle(dummy);

    if ((pkind_ == prop_kind::forward_training) ||
        (flags_ & use_global_stats)) {
        mean_mem_->set_data_handle(dummy);
        var_mem_->set_data_handle(dummy);
    }

    return;
}

template class batch_normalization_fwd<float>;
