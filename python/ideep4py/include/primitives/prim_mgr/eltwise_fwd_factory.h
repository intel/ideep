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


#pragma once

#include <mkldnn.hpp>
#include <string>
#include <typeinfo>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "eltwise_fwd.h"

template <typename T1, typename T2>
class EltwiseFwdFactory : public OpFactory<T1>
{
private:
    EltwiseFwdFactory() {}
    ~EltwiseFwdFactory() {}

public:
    static EltwiseFwd<T1, T2>* get(mkldnn::memory::dims x, mkldnn::algorithm alg_kind, mkldnn::memory::format src_fmt, T2 alpha, T2 beta) {
        EltwiseFwd<T1, T2>* eltwise_forward = nullptr;

        //try to find a suitable one in pool
        eltwise_forward = dynamic_cast<EltwiseFwd<T1, T2>*> (
                            EltwiseFwdFactory<T1, T2>::get_instance().get_eltwise_fwd(x, alg_kind, src_fmt, alpha, beta));

        if (eltwise_forward == nullptr) {
            //LOG(INFO) << "create a new one for eltwise fwd";
            eltwise_forward = new EltwiseFwd<T1, T2>(x, alg_kind, src_fmt, alpha, beta);
            EltwiseFwdFactory<T1, T2>::get_instance().set_eltwise_fwd(x, alg_kind, src_fmt, alpha, beta, eltwise_forward);
        } else {
            //LOG(INFO) << "reuse exist one for eltwise fwd";
        }
        return eltwise_forward;
    }

    static EltwiseFwdFactory& get_instance() {
        static EltwiseFwdFactory instance_;
        return instance_;
    }

private:
#define ELTWISE_FWD_PREFIX "eltwise_fwd_"
    Op<T1>* get_eltwise_fwd(mkldnn::memory::dims x, mkldnn::algorithm alg_kind, mkldnn::memory::format src_fmt, T2 alpha, T2 beta) {
        std::string key = ELTWISE_FWD_PREFIX;

        key += dims_to_string(x);
        key += int_to_string((int)alg_kind);
        // key += typeid(alpha).name();
        key + float_to_string((float)alpha);
        key + float_to_string((float)beta);
        key += int_to_string(src_fmt);

        return this->get_op(key);
    }

    void set_eltwise_fwd(mkldnn::memory::dims x, mkldnn::algorithm alg_kind, mkldnn::memory::format src_fmt, T2 alpha, T2 beta, Op<T1>* op) {
        std::string key = ELTWISE_FWD_PREFIX;

        key += dims_to_string(x);
        key += int_to_string((int)alg_kind);
        // key += typeid(alpha).name();
        key + float_to_string((float)alpha);
        key + float_to_string((float)beta);
        key += int_to_string(src_fmt);

        this->set_op(key, op);
    }
};
