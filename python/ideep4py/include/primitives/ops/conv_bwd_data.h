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


#ifndef _CONV_BWD_DATA_H_
#define _CONV_BWD_DATA_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "op.h"

template <typename T>
class Convolution2DBwdData : public Op<T>
{
public:
    Convolution2DBwdData(mkldnn::memory::dims diff_src_d,
                         mkldnn::memory::dims w_d,
                         mkldnn::memory::dims diff_dst_d,
                         int dilate_y, int dilate_x,
                         int sy, int sx,
                         int pad_lh, int pad_lw, int pad_rh, int pad_rw);
    ~Convolution2DBwdData();

    /*
     * Convolution backward data primitive setup
     * Params:
     * diff_src_d: input, (n,c,h,w)
     * w_d: diff weight, (out_c, in_c, h, w)
     * diff_dst_d: output, (n, out_c, out_h, out_w)
     */
    void setup(mkldnn::memory::dims diff_src_d, 
               mkldnn::memory::dims w_d,
               mkldnn::memory::dims diff_dst_d,
               int dilate_y, int dilate_x,
               int sy, int sx,
               int pad_lh, int pad_lw,
               int pad_rh, int pad_rw);

    /*
     * Convolution backward weights without bias
     */
    void execute(void* diff_src, void* w, void* diff_dst);

public:
    // expected memory format for this primitive instance
    // forward
    mkldnn::memory::format diff_src_fmt_;
    mkldnn::memory::format weights_fmt_;
    mkldnn::memory::format diff_dst_fmt_;
    
    // convolution primitive
    std::shared_ptr<mkldnn::primitive> conv_bwd_data_;

private:
    //MKLDNN memory
    //backward weights
    std::shared_ptr<mkldnn::memory> diff_src_mem_; // gx
    std::shared_ptr<mkldnn::memory> weights_mem_;// W
    std::shared_ptr<mkldnn::memory> diff_dst_mem_; //gy
    
    //
    std::shared_ptr<mkldnn::stream> bwd_data_stream_;
    std::vector<mkldnn::primitive> bwd_data_primitives_;

    //desc & prmitive desc
    //backward weights
    std::shared_ptr<mkldnn::convolution_backward_data::desc> bwd_data_desc_;
    std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> bwd_data_pd_;

    // FIXME
    // forward hint, will be remove in future
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_pd_;

    //memory dims
    mkldnn::memory::dims dilates_;
    mkldnn::memory::dims strides_;
    mkldnn::memory::dims padding_l_;
    mkldnn::memory::dims padding_r_;

    //memory desc
    //forward & backward can share same mem desc
    std::shared_ptr<mkldnn::memory::desc> diff_src_md_; //gx
    std::shared_ptr<mkldnn::memory::desc> weights_md_;// W
    std::shared_ptr<mkldnn::memory::desc> diff_dst_md_; // gy
};

#endif // _CONV_BWD_DATA_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
