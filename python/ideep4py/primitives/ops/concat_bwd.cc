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
#include "concat_bwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
ConcatBwd<T>::ConcatBwd( std::vector<mkldnn::memory::dims> diff_src_ds,
                         mkldnn::memory::dims diff_dst_d,
                         int axis)
{
    bwd_stream_.reset(new stream(stream::kind::eager));
    // create concat primitive
    setup(diff_src_ds, diff_dst_d,  axis);
}

template<typename T>
ConcatBwd<T>::~ConcatBwd()
{
}

template<typename T>
void ConcatBwd<T>::setup( std::vector<mkldnn::memory::dims> diff_src_ds, 
                          mkldnn::memory::dims diff_dst_d,
                          int axis)
{
    //LOG(INFO) << "Concat backward_setup";
    
    assert(diff_src_ds.size() > 0);
    axis_ = axis;

    /* init the offset */
    memory::dims offsets = {0, 0, 0, 0};

    //LOG(INFO) << "diff dst dims: [" << diff_dst_d[0] << "," << diff_dst_d[1] 
      //  << "," << diff_dst_d[2] << "," << diff_dst_d[3] << "]";

    //FIXME
    // Currently, concat backward's diff_dst fmt is hard set, and store it
    memory::format diff_dst_fmt = get_desired_format(diff_dst_d[1]); //
    diff_dst_fmt_ = diff_dst_fmt;

    // create diff dst md/mpt/mem
    diff_dst_mpd_.reset(new memory::primitive_desc(
                {{diff_dst_d}, memory_data_type<T>(), diff_dst_fmt}, cpu_engine));
    diff_dst_mem_.reset(new memory(
                {{{diff_dst_d}, memory_data_type<T>(), diff_dst_fmt}, cpu_engine}, dummy));

    for (int i = 0; i < diff_src_ds.size(); i++) {
        //FIXME
        //Currently, concat's diff src fmt hard set as diff_dst fmt, need to pay attention in future for performance issue
        memory::dims diff_src_tz = diff_src_ds[i];
        //LOG(INFO) << "diff src dims: [" << diff_src_tz[0] << "," << diff_src_tz[1] 
        //    << "," << diff_src_tz[2] << "," << diff_src_tz[3] << "]";
        
        auto diff_src_mpd = memory::primitive_desc(
                {{diff_src_tz}, memory_data_type<T>(), diff_dst_fmt}, cpu_engine);
        auto diff_src_mem = memory({diff_src_mpd}, dummy);

        // store diff src fmt, same as diff dst
        diff_src_fmts_.push_back(diff_dst_fmt);

        diff_src_mems_.push_back(diff_src_mem);

        // create view from gy to gxs[i]
        std::shared_ptr<view::primitive_desc> view_pd;
        view_pd.reset(new view::primitive_desc(*diff_dst_mpd_, diff_src_tz, offsets));
        // create reorder primitive from gy to gxs[i]
        std::shared_ptr<reorder::primitive_desc> reorder_pd;
        reorder_pd.reset(new reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), diff_src_mpd));

        std::shared_ptr<mkldnn::reorder> reorder_prim;
        reorder_prim.reset(new reorder(*reorder_pd, *diff_dst_mem_, diff_src_mems_[i]));
    
        bwd_primitives_.push_back(*reorder_prim);
    
        offsets[axis_] += diff_src_tz[axis_];
    }

    return;
}

template<typename T>
void ConcatBwd<T>::execute(std::vector<void*> diff_src, void *diff_dst)
{
    //LOG(INFO) << "Concat backward";
    assert(diff_src.size() == diff_src_mems_.size());

    for (int i = 0; i < diff_src_mems_.size(); i++) {
        diff_src_mems_[i].set_data_handle(diff_src[i]);
    }
    diff_dst_mem_->set_data_handle(diff_dst);

    bwd_stream_->submit(bwd_primitives_);

    //after exec, set data handle back
    for (int i = 0; i < diff_src_mems_.size(); i++) {
        diff_src_mems_[i].set_data_handle(dummy);
    }
    diff_dst_mem_->set_data_handle(dummy);

    return;
}

template class ConcatBwd<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
