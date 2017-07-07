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


#ifndef _CONCAT_FWD_H_
#define _CONCAT_FWD_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "op.h"

template <typename T>
class ConcatFwd : public Op<T>
{
public:
    ConcatFwd(std::vector<mkldnn::memory::dims> src_ds,
              mkldnn::memory::dims dst_d, int axis);
    ~ConcatFwd();

    /*
     * Concat forward primitive setup
     * Params:
     * src_ds: inputs
     * dst_d: output, (n, out_c, out_h, out_w)
     * axis: axis to concat
     */
    void setup(std::vector<mkldnn::memory::dims> src_d, 
               mkldnn::memory::dims dst_d,
               int axis);

    /*
     * Concat forward execute with bias
     */
    void execute(std::vector<void*> src, void *dst);

public:
    // expected memory format for this primitive instance
    // forward
    std::vector<mkldnn::memory::format> src_fmts_;
    mkldnn::memory::format dst_fmt_;
    
private:
    int axis_;

    //MKLDNN memory
    //forward
    //memory desc
    std::vector<mkldnn::memory::primitive_desc> src_mpds_; //xs
    std::vector<mkldnn::memory> src_mems_;
    std::vector<mkldnn::primitive::at> src_prim_at_; // xs
    
    std::shared_ptr<mkldnn::memory::desc> dst_md_; // y 
    std::shared_ptr<mkldnn::memory> dst_mem_; //y

    std::shared_ptr<mkldnn::stream> fwd_stream_;

    //desc & prmitive desc
    std::shared_ptr<mkldnn::concat::primitive_desc> concat_pd_;
    std::shared_ptr<mkldnn::concat> concat_fwd_;
};

#endif // _CONCAT_FWD_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
