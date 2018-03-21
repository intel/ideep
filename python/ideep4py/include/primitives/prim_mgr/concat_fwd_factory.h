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


#ifndef _CONCAT_FWD_FACTORY_
#define _CONCAT_FWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "concat_fwd.h"

template <typename T>
class ConcatFwdFactory : public OpFactory<T>
{
private:
    ConcatFwdFactory() {}
    ~ConcatFwdFactory() {}

public:
    static ConcatFwd<T>* get( std::vector<mkldnn::memory::dims> src,
                         mkldnn::memory::dims dst,
                         int axis) {
        ConcatFwd<T>* concat_forward = NULL;

        //try to find a suitable one in pool
        concat_forward = dynamic_cast<ConcatFwd<T>*> (
                            ConcatFwdFactory<T>::get_instance().get_concat_fwd(src, dst, axis));

        if (concat_forward == NULL) {
            //LOG(INFO) << "create a new one for concat fwd";
            concat_forward = new ConcatFwd<T>( src, dst, axis);
            ConcatFwdFactory<T>::get_instance().set_concat_fwd( src, dst, axis, concat_forward);
        } else {
            //LOG(INFO) << "reuse exist one for concat fwd";
        }
        return concat_forward;
    }

    static ConcatFwdFactory& get_instance() {
        static ConcatFwdFactory instance_;
        return instance_;
    }

private:
#define CONCAT_FWD_PREFIX "concat_fwd_"
    Op<T>* get_concat_fwd( std::vector<mkldnn::memory::dims> src, 
			   mkldnn::memory::dims dst,
                           int axis) {
        std::string key = CONCAT_FWD_PREFIX;
 
        for (int i = 0; i < src.size(); i++) {
            key += dims_to_string(src[i]);
        }
        key += dims_to_string(dst);
        key += int_to_string(axis);

        return this->get_op(key);
    }

    void set_concat_fwd( std::vector<mkldnn::memory::dims> src, 
                          mkldnn::memory::dims dst,
                          int axis,
                          Op<T> *op) {
        std::string key = CONCAT_FWD_PREFIX;
 
        for (int i = 0; i < src.size(); i++) {
            key += dims_to_string(src[i]);
        }
        key += dims_to_string(dst);
        key += int_to_string(axis);

        this->set_op(key, op);
    }
};

#endif // _CONCAT_FWD_FACTORY_
