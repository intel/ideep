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


#ifndef _BN_BWD_H_
#define _BN_BWD_H_

#include <mkldnn.hpp>
#include <vector>
#include "op.h"

template <typename T>
class batch_normalization_bwd : public Op<T> {
public:
    batch_normalization_bwd(mkldnn::memory::dims src_d,
                            mkldnn::memory::dims diff_dst_d,
                            float eps, bool scale_shift) :
                            flags_(0), bn_size_(src_d[1]),
                            bn_bwd_(nullptr), src_mem_(nullptr),
                            diff_dst_mem_(nullptr), mean_mem_(nullptr),
                            var_mem_(nullptr), w_mem_(nullptr),
                            diff_src_mem_(nullptr), diff_w_mem_(nullptr),
                            bwd_stream_(new mkldnn::stream(mkldnn::stream::kind::eager)) {
        setup(src_d, diff_dst_d, eps, scale_shift);
    }

    ~batch_normalization_bwd() {}

    void setup(mkldnn::memory::dims src_d,
               mkldnn::memory::dims diff_dst_d,
               float eps, bool scale_shift);

    void execute(void *src, void *diff_dst, void *mean,
                 void *var, void *w, void *diff_src, void *diff_w);

public:
    mkldnn_memory_format_t get_src_fmt() {
        return (*src_mem_).get_primitive_desc().desc().data.format;
    }

    mkldnn_memory_format_t get_diff_dst_fmt() {
        return (*diff_dst_mem_).get_primitive_desc().desc().data.format;
    }

    mkldnn_memory_format_t get_diff_src_fmt() {
        return (*diff_src_mem_).get_primitive_desc().desc().data.format;
    }

    mkldnn_memory_format_t get_diff_w_fmt() {
        return (*diff_w_mem_).get_primitive_desc().desc().data.format;
    }

private:
    unsigned long flags_;
    int bn_size_;

    std::shared_ptr<mkldnn::primitive> bn_bwd_;

    std::shared_ptr<mkldnn::memory> src_mem_;
    std::shared_ptr<mkldnn::memory> diff_dst_mem_;
    std::shared_ptr<mkldnn::memory> mean_mem_;
    std::shared_ptr<mkldnn::memory> var_mem_;
    std::shared_ptr<mkldnn::memory> w_mem_;
    std::shared_ptr<mkldnn::memory> diff_src_mem_;
    std::shared_ptr<mkldnn::memory> diff_w_mem_;

    std::vector<mkldnn::primitive> bwd_primitives_;
    std::shared_ptr<mkldnn::stream> bwd_stream_;

    mkldnn::memory::desc get_desc_data(mkldnn::memory m) {
        return m.get_primitive_desc().desc().data;
    }
};

#endif // _BN_BWD_H_
