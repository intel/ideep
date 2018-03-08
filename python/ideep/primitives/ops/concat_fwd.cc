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
#include "concat_fwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
ConcatFwd<T>::ConcatFwd( std::vector<mkldnn::memory::dims> src_ds,
                         mkldnn::memory::dims dst_d, int axis)
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    // create concat primitive
    if (concat_fwd_ == NULL) {
        setup(src_ds, dst_d, axis);
    }
}

template<typename T>
ConcatFwd<T>::~ConcatFwd()
{
}

template<typename T>
void ConcatFwd<T>::setup( std::vector<mkldnn::memory::dims> src_ds, 
                          mkldnn::memory::dims dst_d,
                          int axis)
{
    //LOG(INFO) << "Concat forward_setup";
    
    assert(src_ds.size() > 0);
    axis_ = axis;

    //LOG(INFO) << "dst dims: [" << dst_d[0] << "," << dst_d[1] 
        //<< "," << dst_d[2] << "," << dst_d[3] << "]";

    //FIXME
    // Currently, concat's src fms is hard set
    memory::format src_fmt = get_desired_format(src_ds[0][1]); //

    for (int i = 0; i < src_ds.size(); i++) {
        //FIXME
        //Currently, concat's src fmt hard set as nchw, need to pay attention in future for performance issue
        memory::dims src_tz = src_ds[i];
        
        auto src_mpd = memory::primitive_desc(
                {{src_tz}, memory_data_type<T>(), src_fmt}, cpu_engine);
        auto src_mem = memory({src_mpd}, dummy);

        src_mpds_.push_back(src_mpd);
        src_mems_.push_back(src_mem);

        // concat only accept mkldnn::primitive::at parameter
        src_prim_at_.push_back(primitive::at(src_mem));


        // store src fmt
        src_fmts_.push_back(src_fmt);
    }

    // FIXME
    // here, if set format as any, will create memory fail?????
    dst_md_.reset(new memory::desc(dst_d, memory_data_type<T>(), src_fmt));
    dst_mem_.reset(new memory({{{dst_d}, memory_data_type<T>(), src_fmt}, cpu_engine}, dummy));
    //dst_md_.reset(new memory::desc(dst_d, memory_data_type<T>(), mkldnn::memory::format::any));
    //dst_mem_.reset(new memory({{{dst_d}, memory_data_type<T>(), mkldnn::memory::format::any}, cpu_engine}, dummy));

    // create concat pd/primitive
    concat_pd_.reset(new concat::primitive_desc(*dst_md_, axis_, src_mpds_));
    concat_fwd_.reset(new concat(*concat_pd_, src_prim_at_, *dst_mem_));

    // store dst fmr
    dst_fmt_ = static_cast<mkldnn::memory::format>(concat_pd_.get()->dst_primitive_desc().desc().data.format);

    return;
}

template<typename T>
void ConcatFwd<T>::execute(std::vector<void*> src, void *dst)
{
    //LOG(INFO) << "Concat forward";
    assert(src.size() == src_mems_.size());

    for (int i = 0; i < src_mems_.size(); i++) {
        src_mems_[i].set_data_handle(src[i]);
    }
    dst_mem_->set_data_handle(dst);

    fwd_stream_->submit({*concat_fwd_});

    //after exec, set data handle back
    for (int i = 0; i < src_mems_.size(); i++) {
        src_mems_[i].set_data_handle(dummy);
    }
    dst_mem_->set_data_handle(dummy);

    return;
}

template class ConcatFwd<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
