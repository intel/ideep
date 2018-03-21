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


#ifndef _POOLING_BWD_FACTORY_
#define _POOLING_BWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "pooling_bwd.h"

template <typename T>
class Pooling2DBwdFactory : public OpFactory<T>
{
private:
    Pooling2DBwdFactory() {}
    ~Pooling2DBwdFactory() {}

public:
    static Pooling2DBwd<T>* get(mkldnn::memory::dims src_d,
                                mkldnn::memory::dims dst_d,
                                mkldnn::memory::dims ws_d,
                                mkldnn::memory::data_type ws_dt,
                                int ker_h, int ker_w,
                                int sy, int sx,
                                int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                                mkldnn::algorithm alg_kind) {
        Pooling2DBwd<T>* pooling2d_backward = NULL;

        //try to find a suitable one in pool
        pooling2d_backward = dynamic_cast<Pooling2DBwd<T>*> (
                             Pooling2DBwdFactory<T>::get_instance().get_pooling2d_bwd( src_d, dst_d, ws_d, ws_dt, ker_h, ker_w, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, alg_kind));

        if (pooling2d_backward == NULL) {
            //LOG(INFO) << "create a new one for pooling bwd: " << alg_kind;
            pooling2d_backward = new Pooling2DBwd<T>( src_d, dst_d, ws_d, ws_dt, ker_h, ker_w, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, alg_kind);
            Pooling2DBwdFactory<T>::get_instance().set_pooling2d_bwd( src_d, dst_d, ws_d, ws_dt, ker_h, ker_w, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, alg_kind, pooling2d_backward);
        } else {
            //LOG(INFO) << "reuse exist one for pooling bwd: " << alg_kind;
        }
        return pooling2d_backward;
    }

    static Pooling2DBwdFactory& get_instance() {
        static Pooling2DBwdFactory instance_;
        return instance_;
    }

private:
#define POOLING2D_BWD_PREFIX "pooling2d_bwd_"
    Op<T>* get_pooling2d_bwd(mkldnn::memory::dims src_d,
                             mkldnn::memory::dims dst_d,
                             mkldnn::memory::dims ws_d,
                             mkldnn::memory::data_type ws_dt,
                             int ker_h, int ker_w,
                             int sy, int sx,
                             int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                             mkldnn::algorithm alg_kind) {
        std::string key = POOLING2D_BWD_PREFIX;

        key += dims_to_string(src_d);
        key += dims_to_string(dst_d);
        key += dims_to_string(ws_d);
        key += int_to_string(ws_dt);
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
    };

    void set_pooling2d_bwd(mkldnn::memory::dims src_d,
                           mkldnn::memory::dims dst_d,
                           mkldnn::memory::dims ws_d,
                           mkldnn::memory::data_type ws_dt,
                           int ker_h, int ker_w,
                           int sy, int sx,
                           int pad_lh, int pad_lw, int pad_rh, int pad_rw,
                           mkldnn::algorithm alg_kind,
                           Op<T> *op) {
        std::string key = POOLING2D_BWD_PREFIX;

        key += dims_to_string(src_d);
        key += dims_to_string(dst_d);
        key += dims_to_string(ws_d);
        key += int_to_string(ws_dt);
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

#endif // _POOLING_BWD_FACTORY_
