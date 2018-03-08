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


#ifndef _CONCAT_BWD_FACTORY_
#define _CONCAT_BWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "concat_bwd.h"

template <typename T>
class ConcatBwdFactory : public OpFactory<T>
{
private:
    ConcatBwdFactory() {}
    ~ConcatBwdFactory() {}

public:
    static ConcatBwd<T>* get( std::vector<mkldnn::memory::dims> diff_src,
                         mkldnn::memory::dims diff_dst,
                         int axis) {
        ConcatBwd<T>* concat_backward = NULL;

        //try to find a suitable one in pool
        concat_backward = dynamic_cast<ConcatBwd<T>*> (
                            ConcatBwdFactory<T>::get_instance().get_concat_bwd(diff_src, diff_dst, axis));

        if (concat_backward == NULL) {
            //LOG(INFO) << "create a new one for concat bwd";
            concat_backward = new ConcatBwd<T>( diff_src, diff_dst, axis);
            ConcatBwdFactory<T>::get_instance().set_concat_bwd( diff_src, diff_dst, axis, concat_backward);
        } else {
            //LOG(INFO) << "reuse exist one for concat bwd";
        }
        return concat_backward;
    }

    static ConcatBwdFactory& get_instance() {
        static ConcatBwdFactory instance_;
        return instance_;
    }

private:
#define CONCAT_BWD_PREFIX "concat_bwd_"
    Op<T>* get_concat_bwd( std::vector<mkldnn::memory::dims> diff_src, 
			   mkldnn::memory::dims diff_dst,
                           int axis) {
        std::string key = CONCAT_BWD_PREFIX;
 
        for (int i = 0; i < diff_src.size(); i++) {
            key += dims_to_string(diff_src[i]);
        }
        key += dims_to_string(diff_dst);
        key += int_to_string(axis);

        return this->get_op(key);
    }

    void set_concat_bwd( std::vector<mkldnn::memory::dims> diff_src, 
                          mkldnn::memory::dims diff_dst,
                          int axis,
                          Op<T> *op) {
        std::string key = CONCAT_BWD_PREFIX;
 
        for (int i = 0; i < diff_src.size(); i++) {
            key += dims_to_string(diff_src[i]);
        }
        key += dims_to_string(diff_dst);
        key += int_to_string(axis);

        this->set_op(key, op);
    }
};

#endif // _CONCAT_BWD_FACTORY_
