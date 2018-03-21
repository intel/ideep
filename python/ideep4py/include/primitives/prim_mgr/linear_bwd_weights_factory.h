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


#ifndef _LINEAR_BWD_WEIGHTS_FACTORY_
#define _LINEAR_BWD_WEIGHTS_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "linear_bwd_weights.h"

template <typename T>
class LinearBwdWeightsFactory : public OpFactory<T>
{
private:
    LinearBwdWeightsFactory() {}
    ~LinearBwdWeightsFactory() {}

public:
    static LinearBwdWeights<T>* get(mkldnn::memory::dims x, mkldnn::memory::dims diff_w,
            mkldnn::memory::dims diff_b, mkldnn::memory::dims diff_y) {
        LinearBwdWeights<T>* linear_backward_weights = NULL;
        //try to find a suit one in pool
        linear_backward_weights = dynamic_cast<LinearBwdWeights<T>*>(
                LinearBwdWeightsFactory<T>::get_instance().get_linear_bwd_weights(x, diff_w, diff_b, diff_y));
        if (linear_backward_weights == NULL) {
            //LOG(INFO) << "create a new one for linear bwd weights";
            linear_backward_weights = new LinearBwdWeights<T>(x, diff_w, diff_b, diff_y);
            LinearBwdWeightsFactory<T>::get_instance().set_linear_bwd_weights(x, diff_w, diff_b, diff_y, linear_backward_weights);
        } else {
            //LOG(INFO) << "reuse existed one for linear bwd weights";
        }
        return linear_backward_weights;
    }

    static LinearBwdWeightsFactory& get_instance() {
        static LinearBwdWeightsFactory instance_;
        return instance_;
    }

private:
#define LINEAR_BWD_WEIGHTS_PREFIX "linear_bwd_weights_"
    Op<T>* get_linear_bwd_weights(mkldnn::memory::dims x,
                                  mkldnn::memory::dims diff_w,
                                  mkldnn::memory::dims diff_b,
                                  mkldnn::memory::dims diff_y) {
        std::string key = LINEAR_BWD_WEIGHTS_PREFIX;

        key += dims_to_string(x);
        key += dims_to_string(diff_w);
        key += dims_to_string(diff_b);
        key += dims_to_string(diff_y);

        return this->get_op(key);
    }

    void set_linear_bwd_weights(mkldnn::memory::dims x,
                                mkldnn::memory::dims diff_w,
                                mkldnn::memory::dims diff_b,
                                mkldnn::memory::dims diff_y,
                                Op<T> *op) {
        std::string key = LINEAR_BWD_WEIGHTS_PREFIX;

        key += dims_to_string(x);
        key += dims_to_string(diff_w);
        key += dims_to_string(diff_b);
        key += dims_to_string(diff_y);

        this->set_op(key, op);
    }
};

#endif//_LINEAR_BWD_WEIGHTS_FACTORY_
