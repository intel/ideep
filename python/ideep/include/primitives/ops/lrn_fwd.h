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
#ifndef _LRN_FWD_H_
#define _LRN_FWD_H_

#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "logging.h"
#include "op.h"

template <typename T>
class LocalResponseNormalizationFwd: public Op<T>{
public:
    LocalResponseNormalizationFwd(mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt,
            int n, double k, double alpha, double beta,
            mkldnn::algorithm alg_kind); // alg_kind = mkldnn::algorithm::lrn_across_channels

    ~LocalResponseNormalizationFwd();

    /*
     * lrn forward primitive setup
     * Params:
     * src_d: input
     * dst_d: out_put
     */
    void setup(mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt,
            int n, double k, double alpha, double beta,
            mkldnn::algorithm alg_kind); // alg_kind = mkldnn::algorithm::lrn_across_channels                                      

    /*
     * lrn forward execute 
     * params:
     * src: input
     * dst: output
     * ws: workspace
     */
    void execute(void *src, void *dst, void *ws=NULL);

public:
    // expected memory format
    mkldnn::memory::format src_fmt_;
    mkldnn::memory::format dst_fmt_;
    mkldnn::memory::format ws_fmt_;
    //workspace size
    mkldnn::memory::dims ws_dims_;
    mkldnn::memory::data_type ws_dt_;
    size_t ws_size_;

    // algo
    mkldnn::algorithm alg_kind_;
    // int local_size_;
private:
    // lrn primitive
    std::shared_ptr<mkldnn::lrn_forward> fwd_;
    std::shared_ptr<mkldnn::stream> fwd_stream_;
    
    // MKL-DNN memory, just dummy data
    std::shared_ptr<mkldnn::memory> ws_mem_;
    std::shared_ptr<mkldnn::memory> src_mem_;
    std::shared_ptr<mkldnn::memory> dst_mem_;
    std::shared_ptr<mkldnn::memory::desc> src_md_;
    std::shared_ptr<mkldnn::memory::desc> dst_md_;

    std::shared_ptr<mkldnn::lrn_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::lrn_forward::primitive_desc> fwd_pd_;
    
    std::vector<mkldnn::primitive> fwd_primitives_;
};

#endif // _LRN_FWD_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
