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

#include <mkldnn.hpp>
#include <vector>
#include "op.h"

template <typename...> class EltwiseBwd;
template <typename T1, typename T2>
class EltwiseBwd<T1, T2> : public Op<T1>
{
public:
    EltwiseBwd(mkldnn::memory::dims src_d, mkldnn::algorithm alg_kind, mkldnn::memory::format dst_diff_fmt, T2 alpha, T2 beta);
    ~EltwiseBwd();

    /*
     * Eltwise backward primitive setup
     * Params:
     * src_d: input, (n,c,h,w)
     * dst_d: output, (n, out_c, out_h, out_w)
     */
    void setup(mkldnn::memory::dims src_d, mkldnn::algorithm alg_kind, mkldnn::memory::format dst_diff_fmt, T2 alpha, T2 beta);

    /*
     * Eltwise backward execute
     */
    void execute(void* src, void* dst_diff, void *src_diff);

public:
    // expected memory format for this primitive instance
    // backward
    mkldnn::memory::format src_diff_fmt_;
    
    // Eltwise primitive
    std::shared_ptr<mkldnn::primitive> eltwise_bwd_;

private:
    //MKLDNN memory
    //backward
    std::shared_ptr<mkldnn::memory> src_mem_; // x
    std::shared_ptr<mkldnn::memory> dst_diff_mem_; //gy
    std::shared_ptr<mkldnn::memory> src_diff_mem_; //gx

    std::shared_ptr<mkldnn::stream> bwd_stream_;
    std::vector<mkldnn::primitive> bwd_primitives_;

    //desc & prmitive desc
    //backward
    std::shared_ptr<mkldnn::eltwise_backward::desc> bwd_desc_;
    std::shared_ptr<mkldnn::eltwise_backward::primitive_desc> bwd_pd_;

    //memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md_; //x 
    std::shared_ptr<mkldnn::memory::desc> dst_diff_md_; // gy 

    //memory primitive desc
    std::shared_ptr<mkldnn::memory::primitive_desc> src_mpd_; //x 
    std::shared_ptr<mkldnn::memory::primitive_desc> dst_diff_mpd_; //gy 

    // fwd primitive desc
    std::shared_ptr<mkldnn::eltwise_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::eltwise_forward::primitive_desc> fwd_pd_;
};


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
