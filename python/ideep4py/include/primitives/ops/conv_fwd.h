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


#ifndef _CONV_FWD_H_
#define _CONV_FWD_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "op.h"

template <typename T>
class Convolution2DFwd : public Op<T>
{
public:
    Convolution2DFwd(mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
                     mkldnn::memory::dims b_d, mkldnn::memory::dims dst_d,
                     int dilate_y, int dilate_x,
                     int sy, int sx,
                     int pad_lh, int pad_lw, int pad_rh, int pad_rw);
    ~Convolution2DFwd();

    /*
     * Convolution forward primitive setup
     * Params:
     * src_d: input, (n,c,h,w)
     * W_d: weight, (out_c, in_c, h, w)
     * b_d: bias, if no bias, expected b_d as None dims ({}), not NULL
     * dst_d: output, (n, out_c, out_h, out_w)
     */
    void setup(mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
               mkldnn::memory::dims b_d, mkldnn::memory::dims dst_d,
               int dilate_y, int dilate_x,
               int s1, int s2,
               int pl1, int pl2,
               int pr1, int pr2);

    /*
     * Convolution forward execute with bias
     */
    void execute(void* src, void* w, void* b, void* dst);

    /*
     * Convolution forward execute without bias
     */
    void execute(void* src, void* w, void* dst);

public:
    // expected memory format for this primitive instance
    // forward
    mkldnn::memory::format src_fmt_;
    mkldnn::memory::format weights_fmt_;
    mkldnn::memory::format dst_fmt_;
    
    // convolution primitive
    std::shared_ptr<mkldnn::primitive> conv_fwd_;

private:
    //MKLDNN memory
    //forward
    std::shared_ptr<mkldnn::memory> src_mem_; // x
    std::shared_ptr<mkldnn::memory> weights_mem_;// W
    std::shared_ptr<mkldnn::memory> bias_mem_;// b
    std::shared_ptr<mkldnn::memory> dst_mem_; //y

    std::shared_ptr<mkldnn::stream> fwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_;

    //desc & prmitive desc
    //forward
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_pd_;

    //memory dims
    mkldnn::memory::dims dilates_;
    mkldnn::memory::dims strides_;
    mkldnn::memory::dims padding_l_;
    mkldnn::memory::dims padding_r_;

    //memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md_; //x 
    std::shared_ptr<mkldnn::memory::desc> weights_md_;// W
    std::shared_ptr<mkldnn::memory::desc> bias_md_; // b
    std::shared_ptr<mkldnn::memory::desc> dst_md_; // y 
};

#endif // _CONV_FWD_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
