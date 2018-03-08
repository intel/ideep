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


#ifndef _CONV_BWD_DATA_FACTORY_
#define _CONV_BWD_DATA_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "conv_bwd_data.h"

template <typename T>
class Convolution2DBwdDataFactory : public OpFactory<T>
{
private:
    Convolution2DBwdDataFactory() {}
    ~Convolution2DBwdDataFactory() {}

public:
    static Convolution2DBwdData<T>* get(mkldnn::memory::dims diff_src,
                                        mkldnn::memory::dims w,
                                        mkldnn::memory::dims diff_dst,
                                        int dilate_y, int dilate_x,
                                        int sy, int sx,
                                        int pad_lh, int pad_lw, int pad_rh, int pad_rw) {
        Convolution2DBwdData<T>* conv2d_backward_data = NULL;

        //try to find a suitable one in pool
        conv2d_backward_data = dynamic_cast<Convolution2DBwdData<T>*> (
                            Convolution2DBwdDataFactory<T>::get_instance().get_conv2d_bwd_data( diff_src, w, diff_dst, dilate_y, dilate_x, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw));

        if (conv2d_backward_data == NULL) {
            //LOG(INFO) << "create a new one for conv2d bwd data";
            conv2d_backward_data = new Convolution2DBwdData<T>( diff_src, w, diff_dst, dilate_y, dilate_x, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw);
            Convolution2DBwdDataFactory<T>::get_instance().set_conv2d_bwd_data( diff_src, w, diff_dst, dilate_y, dilate_x, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, conv2d_backward_data);
        } else {
            //LOG(INFO) << "reuse a existed one for conv2d bwd data";
        }
        return conv2d_backward_data;
    }

    static Convolution2DBwdDataFactory& get_instance() {
        static Convolution2DBwdDataFactory instance_;
        return instance_;
    }

private:
#define CONVOLUTION2D_BWD_DATA_PREFIX "conv2d_bwd_data_"
    Op<T>* get_conv2d_bwd_data(mkldnn::memory::dims diff_src,
                               mkldnn::memory::dims w,
                               mkldnn::memory::dims diff_dst,
                               int dilate_y, int dilate_x,
                               int sy, int sx,
                               int pad_lh, int pad_lw, int pad_rh, int pad_rw) {
        std::string key = CONVOLUTION2D_BWD_DATA_PREFIX;

        key += dims_to_string(diff_src);
        key += dims_to_string(w);
        key += dims_to_string(diff_dst);
        key += int_to_string(dilate_y);
        key += int_to_string(dilate_x);
        key += int_to_string(sy);
        key += int_to_string(sx);
        key += int_to_string(pad_lh);
        key += int_to_string(pad_lw);
        key += int_to_string(pad_rh);
        key += int_to_string(pad_rw);

        return this->get_op(key);
    }

    void set_conv2d_bwd_data(mkldnn::memory::dims diff_src,
                             mkldnn::memory::dims w,
                             mkldnn::memory::dims diff_dst,
                             int dilate_y, int dilate_x,
                             int sy, int sx,
                             int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                             Op<T> *op) {
        std::string key = CONVOLUTION2D_BWD_DATA_PREFIX;

        key += dims_to_string(diff_src);
        key += dims_to_string(w);
        key += dims_to_string(diff_dst);
        key += int_to_string(dilate_y);
        key += int_to_string(dilate_x);
        key += int_to_string(sy);
        key += int_to_string(sx);
        key += int_to_string(pad_lh);
        key += int_to_string(pad_lw);
        key += int_to_string(pad_rh);
        key += int_to_string(pad_rw);

        this->set_op(key, op);
    }
};

#endif // _CONV_BWD_DATA_FACTORY_
