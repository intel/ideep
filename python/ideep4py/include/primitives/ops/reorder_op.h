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


#ifndef _REORDER_OP_H_
#define _REORDER_OP_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "op.h"

template <typename T>
class ReorderOp : public Op<T>
{
public:
    ReorderOp(mkldnn::memory::dims dims, mkldnn::memory::format src_fmt, mkldnn::memory::format dst_fmt);
    ~ReorderOp();

    /*
     * Reorder primitive setup
     * Params:
     * dims:
     * src_fmt: 
     * dst_fmt:
     */
    void setup(mkldnn::memory::dims dims, mkldnn::memory::format src_fmt, mkldnn::memory::format dst_fmt);

    /*
     * reorder execute
     */
    void execute(void* src, void* dst);

public:
    // expected memory format for this primitive instance
    mkldnn::memory::format src_fmt_;
    mkldnn::memory::format dst_fmt_;
    
    // reorder primitive
    std::shared_ptr<mkldnn::reorder> reorder_prim_;

private:
    //MKLDNN memory
    //forward
    std::shared_ptr<mkldnn::memory> src_mem_; // x
    std::shared_ptr<mkldnn::memory> dst_mem_; //y

    std::shared_ptr<mkldnn::stream> reorder_stream_;

    //memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md_; //x 
    std::shared_ptr<mkldnn::memory::desc> dst_md_; // y 
};

#endif // _REORDER_OP_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
