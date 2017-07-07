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


#ifndef _LINEAR_BWD_WEIGHTS_H_
#define _LINEAR_BWD_WEIGHTS_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "op.h"

template<typename T>
class LinearBwdWeights : public Op<T>
{
public:
    LinearBwdWeights(mkldnn::memory::dims src_d, mkldnn::memory::dims diff_w_d,
                     mkldnn::memory::dims diff_b_d, mkldnn::memory::dims diff_dst_d);
    ~LinearBwdWeights();
    /*
     * Linear backward weights primitive setup
     * Params:
     * src_d: input, (n,c,h,w)
     * diff_w_d: diff weight, (out_c, in_c, h, w)
     * diff_b_d: diff_bias
     * diff_dst_d: output, (n, out_c, out_h, out_w)
     */
    void setup(mkldnn::memory::dims src_d, mkldnn::memory::dims diff_w_d,
               mkldnn::memory::dims diff_b_d, mkldnn::memory::dims diff_dst_d);
    /*
     * Linear backward wieghts with bias
     */
    void execute(void* src, void* diff_w, void* diff_b, void* diff_dst);
    /*
     * Linear backward weights without bias
     */
    void execute(void* src, void* diff_w, void* diff_dst);
public:
    //expected memory format for this primitive instance
    // forward
    mkldnn::memory::format src_fmt_;
    mkldnn::memory::format diff_weights_fmt_;
    mkldnn::memory::format diff_dst_fmt_;
    //linear primitive
    std::shared_ptr<mkldnn::primitive> linear_bwd_weights_;
private:
    //MKLDNN memory
    //backward weights
    std::shared_ptr<mkldnn::memory> src_mem_;//x
    std::shared_ptr<mkldnn::memory> diff_weights_mem_; // gw
    std::shared_ptr<mkldnn::memory> diff_bias_mem_; //gb
    std::shared_ptr<mkldnn::memory> diff_dst_mem_; // gy
    //
    std::shared_ptr<mkldnn::stream> bwd_weights_stream_;
    std::vector<mkldnn::primitive> bwd_weights_primitives_;
    //desc & primitive desc
    //backward weights
    std::shared_ptr<mkldnn::inner_product_backward_weights::desc> bwd_weights_desc_;
    std::shared_ptr<mkldnn::inner_product_backward_weights::primitive_desc> bwd_weights_pd_;
    //FIXME
    //forward hint, will be removed in future
    std::shared_ptr<mkldnn::inner_product_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwd_pd_;

    //memory desc
    //forward & backward can share the same mem desc
    std::shared_ptr<mkldnn::memory::desc> src_md_; //x
    std::shared_ptr<mkldnn::memory::desc> diff_weights_md_;//gW
    std::shared_ptr<mkldnn::memory::desc> diff_bias_md_;//gb
    std::shared_ptr<mkldnn::memory::desc> diff_dst_md_;//gy
};

#endif //_CONV_BWD_WEIGHTS_H























