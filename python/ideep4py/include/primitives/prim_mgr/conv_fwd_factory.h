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


#ifndef _CONV_FWD_FACTORY_
#define _CONV_FWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "conv_fwd.h"

template <typename T>
class Convolution2DFwdFactory : public OpFactory<T>
{
private:
    Convolution2DFwdFactory() {}
    ~Convolution2DFwdFactory() {}

public:
    static Convolution2DFwd<T>* get( mkldnn::memory::dims x, mkldnn::memory::dims w,
                         mkldnn::memory::dims b, mkldnn::memory::dims y,
                         int dilate_y, int dilate_x,
                         int sy, int sx,
                         int pad_lh, int pad_lw, int pad_rh, int pad_rw) {
        Convolution2DFwd<T>* conv2d_forward = NULL;

        //try to find a suitable one in pool
        conv2d_forward = dynamic_cast<Convolution2DFwd<T>*> (
                            Convolution2DFwdFactory<T>::get_instance().get_conv2d_fwd( x, w, b, y, dilate_y, dilate_x, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw));

        if (conv2d_forward == NULL) {
            //LOG(INFO) << "create a new one for conv2d fwd";
            conv2d_forward = new Convolution2DFwd<T>( x, w, b, y, dilate_y, dilate_x, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw);
            Convolution2DFwdFactory<T>::get_instance().set_conv2d_fwd( x, w, b, y, dilate_y, dilate_x, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, conv2d_forward);
        } else {
            //LOG(INFO) << "reuse exist one for conv2d fwd";
        }
        return conv2d_forward;
    }

    static Convolution2DFwdFactory& get_instance() {
        static Convolution2DFwdFactory instance_;
        return instance_;
    }

private:
#define CONVOLUTION2D_FWD_PREFIX "conv2d_fwd_"
    Op<T>* get_conv2d_fwd( mkldnn::memory::dims x, mkldnn::memory::dims w,
                           mkldnn::memory::dims b, mkldnn::memory::dims y,
                           int dilate_y, int dilate_x,
                           int sy, int sx,
                           int pad_lh, int pad_lw, int pad_rh, int pad_rw) {
        std::string key = CONVOLUTION2D_FWD_PREFIX;

        key += dims_to_string(x);
        key += dims_to_string(w);
        key += dims_to_string(b);
        key += dims_to_string(y);
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

    void set_conv2d_fwd( mkldnn::memory::dims x, mkldnn::memory::dims w,
                         mkldnn::memory::dims b, mkldnn::memory::dims y,
                         int dilate_y, int dilate_x,
                         int sy, int sx,
                         int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                         Op<T> *op) {
        std::string key = CONVOLUTION2D_FWD_PREFIX;

        key += dims_to_string(x);
        key += dims_to_string(w);
        key += dims_to_string(b);
        key += dims_to_string(y);
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

#endif // _CONV_FWD_FACTORY_
