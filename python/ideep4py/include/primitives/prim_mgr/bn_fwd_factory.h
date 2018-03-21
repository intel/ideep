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


#ifndef _BN_FWD_FACTORY_
#define _BN_FWD_FACTORY_

#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "bn_fwd.h"

template <typename T>
class batch_normalization_fwd_factory : public OpFactory<T> {

private:
    batch_normalization_fwd_factory() {}
    ~batch_normalization_fwd_factory() {}

public:
    static batch_normalization_fwd<T> * get(
            mkldnn::memory::dims src_d, float eps,
            bool scale_shift, bool global_stats, bool training) {

        auto bn_fwd = dynamic_cast<batch_normalization_fwd<T>*>(
                      batch_normalization_fwd_factory<T>::get_instance().get_bn_fwd(
                      src_d, eps, scale_shift, global_stats, training));

        if (bn_fwd == nullptr) {
            bn_fwd = new batch_normalization_fwd<T>(
                         src_d, eps, scale_shift, global_stats, training);
            batch_normalization_fwd_factory<T>::get_instance().set_bn_fwd(
                    src_d, eps, scale_shift, global_stats, training, bn_fwd);
        }

        return bn_fwd;
    }

    static batch_normalization_fwd_factory & get_instance() {
        static batch_normalization_fwd_factory instance_;
        return instance_;
    }

private:
#define BN_FWD_PREFIX "bn_fwd_"
    Op<T> * get_bn_fwd(mkldnn::memory::dims src_d, float eps, bool scale_shift,
                       bool global_stats, bool training) {

        std::string key = BN_FWD_PREFIX;

        key += dims_to_string(src_d);
        key += float_to_string(eps);
        key += bool_to_string(scale_shift);
        key += bool_to_string(global_stats);
        key += bool_to_string(training);

        return this->get_op(key);
    }

    void set_bn_fwd(mkldnn::memory::dims src_d, float eps, bool scale_shift,
                    bool global_stats, bool training, Op<T> *op) {

        std::string key = BN_FWD_PREFIX;

        key += dims_to_string(src_d);
        key += float_to_string(eps);
        key += bool_to_string(scale_shift);
        key += bool_to_string(global_stats);
        key += bool_to_string(training);

        this->set_op(key, op);
    }
};

#endif // _BN_FWD_FACTORY_
