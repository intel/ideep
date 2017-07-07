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
#include "reorder_op.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
ReorderOp<T>::ReorderOp( mkldnn::memory::dims dims, mkldnn::memory::format src_fmt, mkldnn::memory::format dst_fmt)
{
    reorder_stream_.reset(new stream(stream::kind::eager));
    // create primitive
    if (reorder_prim_ == NULL) {
        setup(dims, src_fmt, dst_fmt);
    }
}

template<typename T>
ReorderOp<T>::~ReorderOp()
{
}

template<typename T>
void ReorderOp<T>::setup(mkldnn::memory::dims dims, 
                         mkldnn::memory::format src_fmt,
                         mkldnn::memory::format dst_fmt)
{
    //LOG(INFO) << "Reorder setup";
    
    assert(src_fmt != dst_mfmt);

    src_md_.reset(new memory::desc(dims, memory_data_type<T>(), src_fmt));
    dst_md_.reset(new memory::desc(dims, memory_data_type<T>(), dst_fmt));
    
    src_mem_.reset(new memory({*src_md_, cpu_engine},dummy));
    dst_mem_.reset(new memory({*dst_md_, cpu_engine},dummy));

    reorder_prim_ = std::make_shared<mkldnn::reorder>(reorder(*src_mem_, *dst_mem_));

    return;
}

template<typename T>
void ReorderOp<T>::execute(void* src, void* dst)
{
    //LOG(INFO) << "Reorder execute";
    src_mem_->set_data_handle(src);
    dst_mem_->set_data_handle(dst);
    reorder_stream_->submit({*reorder_prim_});

    //after exec, set data handle back
    src_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);
    return;
}

template class ReorderOp<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
