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


#ifndef _OP_FACTORY_
#define _OP_FACTORY_

#include <unordered_map>
#include <string>
#include "op.h"
#include "config.h"

extern bool enable_prim_reuse;

template <typename T>
class OpFactory {
public:
    OpFactory() {};
    ~OpFactory() {};
    // virtual Op<T>* get() {return NULL;}

    Op<T>* get_op(std::string key) {
        // if not enable primitive reuse
        // just return NULL
        if (!enable_prim_reuse)
            return NULL;

        auto stream_iter = map_.find(key);
        if (stream_iter == map_.end()) {
            return NULL;
        } else {
            return stream_iter->second;
        }
    };

    void set_op(std::string key, Op<T>* op) {
        // if not enable primitive reuse
        // just return
        if (!enable_prim_reuse)
            return;

        auto stream_iter = map_.find(key);
        if (stream_iter == map_.end()) {
            map_[key]=op;
        } else {
            throw new std::invalid_argument("cannot set same key to a new stream");
        }
    };

public:
    std::unordered_map<std::string, Op<T>*> map_;
};

#endif // _OP_FACTORY_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
