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


#ifndef _UTILS_H_
#define _UTILS_H_

#include <mkldnn.hpp>
#include <iostream>
#include <sstream>
#include <cassert>
#include "param.h"
using namespace mkldnn;

static inline mkldnn::algorithm pooling_algo_convert(pooling_param_t::algorithm input) {
    switch(input) {
        case pooling_param_t::algorithm::pooling_max:
            return mkldnn::pooling_max;
        case pooling_param_t::algorithm::pooling_avg:
            return mkldnn::pooling_avg;
        case pooling_param_t::algorithm::pooling_avg_include_padding:
            return mkldnn::pooling_avg_include_padding;
        case pooling_param_t::algorithm::pooling_avg_exclude_padding:
            return mkldnn::pooling_avg_exclude_padding;
        default:
            return mkldnn::pooling_max;
    }
}

static inline mkldnn::algorithm lrn_algo_convert(lrn_param_t::algorithm input) {
    switch(input) {
        case lrn_param_t::algorithm::lrn_across_channels:
            return mkldnn::lrn_across_channels;
        case lrn_param_t::algorithm::lrn_within_channel:
            return mkldnn::lrn_within_channel;
        default:
            return mkldnn::lrn_across_channels;
    }
}

#endif // _UTILS_H_
