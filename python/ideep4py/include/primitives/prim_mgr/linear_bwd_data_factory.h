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


#ifndef _LINEAR_BWD_DATA_FACTORY_
#define _LINEAR_BWD_DATA_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "linear_bwd_data.h"

template <typename T>
class LinearBwdDataFactory : public OpFactory<T>
{
private:
    LinearBwdDataFactory() {}
    ~LinearBwdDataFactory() {}

public:
    static LinearBwdData<T>* get(mkldnn::memory::dims diff_src,
            mkldnn::memory::dims w, mkldnn::memory::dims diff_dst) {
        LinearBwdData<T>* linear_backward_data = NULL;
        //try to find a suitable one in pool
        linear_backward_data = dynamic_cast<LinearBwdData<T>*>(
                LinearBwdDataFactory<T>::get_instance().get_linear_bwd_data(diff_src, w, diff_dst));
        if (linear_backward_data == NULL) {
            //LOG(INFO) << "create a new one for linear bwd data";
            linear_backward_data = new LinearBwdData<T>(diff_src, w, diff_dst);
            LinearBwdDataFactory<T>::get_instance().set_linear_bwd_data(diff_src, w, diff_dst, linear_backward_data);
        } else {
            //LOG(INFO) << "reuse a exited one for linear bwd data";
        }
        return linear_backward_data;
    }

    static LinearBwdDataFactory& get_instance() {
        static LinearBwdDataFactory instance_;
        return instance_;
    }

private:
#define LINEAR_BWD_DATA_PREFIX "linear_bwd_data_"
    Op<T>* get_linear_bwd_data(mkldnn::memory::dims diff_src,
                               mkldnn::memory::dims w,
                               mkldnn::memory::dims diff_dst) {
        std::string key = LINEAR_BWD_DATA_PREFIX;

        key += dims_to_string(diff_src);
        key += dims_to_string(w);
        key += dims_to_string(diff_dst);

        return this->get_op(key);
    }

    void set_linear_bwd_data(mkldnn::memory::dims diff_src,
                             mkldnn::memory::dims w,
                             mkldnn::memory::dims diff_dst,
                             Op<T> *op) {
        std::string key = LINEAR_BWD_DATA_PREFIX;

        key += dims_to_string(diff_src);
        key += dims_to_string(w);
        key += dims_to_string(diff_dst);

        this->set_op(key, op);
    }
};

#endif //_LINEAR_BWD_DATA_FACTORY_
