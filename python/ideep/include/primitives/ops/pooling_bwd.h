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


#pragma once
#ifndef _POOLING_BWD_H_
#define _POOLING_BWD_H_

#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "logging.h"
#include "op.h"

template <typename T>
class Pooling2DBwd: public Op<T>{
public:
    Pooling2DBwd(mkldnn::memory::dims diff_src_d,
                 mkldnn::memory::dims diff_dst_d,
                 mkldnn::memory::dims ws_d,
                 mkldnn::memory::data_type ws_dt,
                 int ker_h, int ker_w,
                 int sy, int sx,
                 int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                 mkldnn::algorithm alg_kind); // alg_kind = pooling_max
                                            // or pooling_avg
    ~Pooling2DBwd();
    
    /*
     * Pooling backward primitive setup
     * Params:
     * diff_src_d: diff src
     * diff_dst_d: diff dst
     */
    void setup(mkldnn::memory::dims diff_src_d, 
               mkldnn::memory::dims diff_dst_d,
               mkldnn::memory::dims ws_d,
               mkldnn::memory::data_type ws_dt,
               int ker_h, int ker_w,
               int sy, int sx,
               int pad_lh, int pad_lw, int pad_rh, int pad_rw,
               mkldnn::algorithm alg_kind); // alg_kind = pooling_max
                                            // or pooling_avg

    /*
     * Pooling backward execute 
     * params:
     * diff_src: diff_src
     * diff_dst: diff_dst
     * ws: workspace
     */
    void execute(void *diff_src, void *diff_dst, void *ws=NULL);

public:
    // expected memory format
    mkldnn::memory::format diff_src_fmt_;
    mkldnn::memory::format diff_dst_fmt_;
    mkldnn::memory::format ws_fmt_;

    // algo
    mkldnn::algorithm alg_kind_;
private:
    // pooling primitive
    std::shared_ptr<mkldnn::pooling_backward> bwd_;
    std::shared_ptr<mkldnn::stream> bwd_stream_;
    
    // MKL-DNN memory, just dummy data
    std::shared_ptr<mkldnn::memory> ws_mem_;
    std::shared_ptr<mkldnn::memory> diff_src_mem_;
    std::shared_ptr<mkldnn::memory> diff_dst_mem_;
    std::shared_ptr<mkldnn::memory::desc> diff_src_md_;
    std::shared_ptr<mkldnn::memory::desc> diff_dst_md_;

    // fwd hint
    std::shared_ptr<mkldnn::pooling_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_;
    
    std::shared_ptr<mkldnn::pooling_backward::desc> bwd_desc_;
    std::shared_ptr<mkldnn::pooling_backward::primitive_desc> bwd_pd_;
    
    std::vector<mkldnn::primitive> bwd_primitives_;
};

#endif // _POOLING_BWD_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
