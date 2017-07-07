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


#ifndef _POOLING_FWD_FACTORY_
#define _POOLING_FWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "pooling_fwd.h"

template <typename T>
class Pooling2DFwdFactory : public OpFactory<T>
{
private:
    Pooling2DFwdFactory() {}
    ~Pooling2DFwdFactory() {}

public:
    static Pooling2DFwd<T>* get(mkldnn::memory::dims src_d,
                                mkldnn::memory::dims dst_d,
                                int ker_h, int ker_w,
                                int sy, int sx,
                                int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                                mkldnn::algorithm alg_kind) {
        Pooling2DFwd<T>* pooling2d_forward = NULL;

        //try to find a suitable one in pool
        pooling2d_forward = dynamic_cast<Pooling2DFwd<T>*> (
                            Pooling2DFwdFactory<T>::get_instance().get_pooling2d_fwd( src_d, dst_d, ker_h, ker_w, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, alg_kind));

        if (pooling2d_forward == NULL) {
            //LOG(INFO) << "create a new one for pooling fwd: " << alg_kind;
            pooling2d_forward = new Pooling2DFwd<T>( src_d, dst_d, ker_h, ker_w, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, alg_kind);
            Pooling2DFwdFactory<T>::get_instance().set_pooling2d_fwd( src_d, dst_d, ker_h, ker_w, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, alg_kind, pooling2d_forward);
        } else {
            //LOG(INFO) << "reuse exist one for pooling fwd: " << alg_kind;
        }
        return pooling2d_forward;
    }

    static Pooling2DFwdFactory& get_instance() {
        static Pooling2DFwdFactory instance_;
        return instance_;
    }

private:
#define POOLING2D_FWD_PREFIX "pooling2d_fwd_"
    Op<T>* get_pooling2d_fwd(mkldnn::memory::dims src_d,
                             mkldnn::memory::dims dst_d,
                             int ker_h, int ker_w,
                             int sy, int sx,
                             int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                             mkldnn::algorithm alg_kind) {
        std::string key = POOLING2D_FWD_PREFIX;

        key += dims_to_string(src_d);
        key += dims_to_string(dst_d);
        key += int_to_string(ker_h);
        key += int_to_string(ker_w);
        key += int_to_string(sy);
        key += int_to_string(sx);
        key += int_to_string(pad_lh);
        key += int_to_string(pad_lw);
        key += int_to_string(pad_rh);
        key += int_to_string(pad_rw);
        key += int_to_string(alg_kind);

        return this->get_op(key);
    }

    void set_pooling2d_fwd(mkldnn::memory::dims src_d,
                           mkldnn::memory::dims dst_d,
                           int ker_h, int ker_w,
                           int sy, int sx,
                           int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                           mkldnn::algorithm alg_kind,
                           Op<T> *op) {
        std::string key = POOLING2D_FWD_PREFIX;

        key += dims_to_string(src_d);
        key += dims_to_string(dst_d);
        key += int_to_string(ker_h);
        key += int_to_string(ker_w);
        key += int_to_string(sy);
        key += int_to_string(sx);
        key += int_to_string(pad_lh);
        key += int_to_string(pad_lw);
        key += int_to_string(pad_rh);
        key += int_to_string(pad_rw);
        key += int_to_string(alg_kind);

        this->set_op(key, op);
    }
};

#endif // _POOLING_FWD_FACTORY_
