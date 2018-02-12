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


#ifndef _LRN_FWD_FACTORY_
#define _LRN_FWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "lrn_fwd.h"

template <typename T>
class LocalResponseNormalizationFwdFactory : public OpFactory<T>
{
private:
    LocalResponseNormalizationFwdFactory() {}
    ~LocalResponseNormalizationFwdFactory() {}

public:
    static LocalResponseNormalizationFwd<T>* get(
            mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt,
            int n, double k, double alpha, double beta,
            mkldnn::algorithm alg_kind)
    {
        LocalResponseNormalizationFwd<T>* lrn_forward = NULL;

        //try to find a suitable one in pool
        lrn_forward = dynamic_cast<LocalResponseNormalizationFwd<T>*> (
            LocalResponseNormalizationFwdFactory<T>::get_instance().get_lrn_fwd(src_d, src_fmt, n, k, alpha, beta, alg_kind));

        if (lrn_forward == NULL) {
            //LOG(INFO) << "create a new one for lrn fwd: " << alg_kind;
            lrn_forward = new LocalResponseNormalizationFwd<T>(src_d, src_fmt, n, k, alpha, beta, alg_kind);
            LocalResponseNormalizationFwdFactory<T>::get_instance().set_lrn_fwd( src_d, src_fmt, n, k, alpha, beta, alg_kind, lrn_forward);
        } else {
            //LOG(INFO) << "reuse exist one for lrn fwd: " << alg_kind;
        }
        return lrn_forward;
    }

    static LocalResponseNormalizationFwdFactory& get_instance() {
        static LocalResponseNormalizationFwdFactory instance_;
        return instance_;
    }

private:
#define LRN_FWD_PREFIX "lrn_fwd_"
    Op<T>* get_lrn_fwd(mkldnn::memory::dims src_d,
                        mkldnn::memory::format src_fmt,
                        int n, double k, double alpha, double beta,
                        mkldnn::algorithm alg_kind) {
        std::string key = LRN_FWD_PREFIX;

        key += dims_to_string(src_d);
        key += int_to_string(src_fmt);
        key += int_to_string(n);
        key += double_to_string(k);
        key += double_to_string(alpha);
        key += double_to_string(beta);
        key += int_to_string(alg_kind);

        return this->get_op(key);
    }

    void set_lrn_fwd(mkldnn::memory::dims src_d,
            mkldnn::memory::format src_fmt,
            int n, double k, double alpha, double beta,
            mkldnn::algorithm alg_kind, Op<T> *op) {
        std::string key = LRN_FWD_PREFIX;

        key += dims_to_string(src_d);
        key += int_to_string(src_fmt);
        key += int_to_string(n);
        key += double_to_string(k);
        key += double_to_string(alpha);
        key += double_to_string(beta);
        key += int_to_string(alg_kind);

        this->set_op(key, op);
    }
};

#endif // _LRN_FWD_FACTORY_
