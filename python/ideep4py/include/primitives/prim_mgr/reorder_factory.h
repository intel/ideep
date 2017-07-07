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


#ifndef _REORDER_FACTORY_
#define _REORDER_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "reorder_op.h"

template <typename T>
class ReorderFactory : public OpFactory<T>
{
private:
    ReorderFactory() {}
    ~ReorderFactory() {}

public:
    static ReorderOp<T>* get(mkldnn::memory::dims dims, mkldnn::memory::format src_fmt, mkldnn::memory::format dst_fmt) {
        ReorderOp<T>* reorder_op = NULL;

        //try to find a suitable one in pool
        reorder_op = dynamic_cast<ReorderOp<T>*> (
                            ReorderFactory<T>::get_instance().get_reorder(dims, src_fmt, dst_fmt));

        if (reorder_op == NULL) {
            //LOG(INFO) << "create a new one for reorder";
            reorder_op = new ReorderOp<T>( dims, src_fmt, dst_fmt);
            ReorderFactory<T>::get_instance().set_reorder( dims, src_fmt, dst_fmt, reorder_op);
        } else {
            //LOG(INFO) << "reuse exist one for reorder";
        }
        return reorder_op;
    }

    static ReorderFactory& get_instance() {
        static ReorderFactory instance_;
        return instance_;
    }

private:
#define REORDER_PREFIX "reorder_"
    Op<T>* get_reorder(mkldnn::memory::dims dims,
                       mkldnn::memory::format src_fmt,
                       mkldnn::memory::format dst_fmt) {
        std::string key = REORDER_PREFIX;

        key += dims_to_string(dims);
        key += int_to_string((int)src_fmt);
        key += int_to_string((int)dst_fmt);

        return this->get_op(key);
    }

    void set_reorder(mkldnn::memory::dims dims,
                     mkldnn::memory::format src_fmt,
                     mkldnn::memory::format dst_fmt,
                     Op<T> *op) {
        std::string key = REORDER_PREFIX;

        key += dims_to_string(dims);
        key += int_to_string((int)src_fmt);
        key += int_to_string((int)dst_fmt);

        this->set_op(key, op);
    }
};

#endif // _REORDER_FACTORY_
