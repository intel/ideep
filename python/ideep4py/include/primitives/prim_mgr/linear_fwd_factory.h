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


#ifndef _LINEAR_FWD_FACTORY_
#define _LINEAR_FWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "linear_fwd.h"

template <typename T>
class LinearFwdFactory : public OpFactory<T>
{
private:
    LinearFwdFactory() {}
    ~LinearFwdFactory() {}

public:
    static LinearFwd<T>* get(mkldnn::memory::dims x, mkldnn::memory::dims w,
            mkldnn::memory::dims b, mkldnn::memory::dims y) {
        LinearFwd<T>* linear_forward = NULL;
        //try to find a suitable one in pool
        linear_forward = dynamic_cast<LinearFwd<T>*> (
                LinearFwdFactory<T>::get_instance().get_linear_fwd(x, w, b, y));
        if (linear_forward == NULL) {
            //LOG(INFO) << "create a new one for linear fwd";
            linear_forward = new LinearFwd<T>(x, w, b, y);
            LinearFwdFactory<T>::get_instance().set_linear_fwd(x, w, b, y, linear_forward);
        } else {
            //LOG(INFO) << "reuse exist one linear fwd";
        }
        return linear_forward;
    }
    static LinearFwdFactory& get_instance() {
        static LinearFwdFactory instance_;
        return instance_;
    }

private:
#define LINEAR_FWD_PREFIX "linear_fwd_"
    Op<T>* get_linear_fwd(mkldnn::memory::dims x,
                          mkldnn::memory::dims w,
                          mkldnn::memory::dims b,
                          mkldnn::memory::dims y) {
        std::string key = LINEAR_FWD_PREFIX;

        key += dims_to_string(x);
        key += dims_to_string(w);
        key += dims_to_string(b);
        key += dims_to_string(y);

        return this->get_op(key);
    }

    void set_linear_fwd(mkldnn::memory::dims x,
                        mkldnn::memory::dims w,
                        mkldnn::memory::dims b,
                        mkldnn::memory::dims y,
                        Op<T>* op) {
        std::string key = LINEAR_FWD_PREFIX;

        key += dims_to_string(x);
        key += dims_to_string(w);
        key += dims_to_string(b);
        key += dims_to_string(y);

        return;
    }
};

#endif //_LINEAR_FWD_FACTORY
