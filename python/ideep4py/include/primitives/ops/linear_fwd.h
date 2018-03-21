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


#ifndef _LINEAR_FWD_H_
#define _LINEAR_FWD_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "op.h"

template <typename T>
class LinearFwd : public Op<T>
{
public:
    LinearFwd(mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
              mkldnn::memory::dims b_d, mkldnn::memory::dims dst_d);
    ~LinearFwd();
    /*Linear forward primitive setup
     * Params:
     * src_d: input, (n, c, h, w)
     * W_d: weight, (out_c, in_c, h, w)
     * b_d: bias, if no bias, expected b_d as None dims({}), not NULL
     * dst_d: output, (n, out_c, out_h, out_w)
     */
    void setup(mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
               mkldnn::memory::dims b_d, mkldnn::memory::dims dst_d);
    /*
     * Linear forward execute with bias
     */
    void execute(void *src, void* w, void* b, void* dst);
    /*
     * Linear forward execute without bias
     */
    void execute(void *src, void* w, void* dst);
public:
    //expected memory format for this primitive instance
    //forward
    mkldnn::memory::format src_fmt_;
    mkldnn::memory::format weights_fmt_;
    mkldnn::memory::format dst_fmt_;
    //linear primitive
    std::shared_ptr<mkldnn::primitive> linear_fwd_;
private:
    //MKLDNN memory
    //forward
    std::shared_ptr<mkldnn::memory> src_mem_;// x
    std::shared_ptr<mkldnn::memory> weights_mem_;// W
    std::shared_ptr<mkldnn::memory> bias_mem_;// b
    std::shared_ptr<mkldnn::memory> dst_mem_; // y
    
    std::shared_ptr<mkldnn::stream> fwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_;

    //desc & primitive desc
    //forward
    std::shared_ptr<mkldnn::inner_product_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwd_pd_;
    //memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md_;//x
    std::shared_ptr<mkldnn::memory::desc> weights_md_;//W
    std::shared_ptr<mkldnn::memory::desc> bias_md_;//b
    std::shared_ptr<mkldnn::memory::desc> dst_md_;// y
};
#endif //__LINEAR_FWD_H_






























