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
#include "omp.h"
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

template<typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return(a + b - 1) / b;
}
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

inline void fast_memcpy(char* data_o, char *data_i, size_t len)
{
    size_t nelems_float = len / 4;
    size_t nelems_char = len % 4;
    const int block_size = 16;
    const auto num_blocks_float = nelems_float / block_size;
    const auto rem_elems_float =  nelems_float % block_size;
    float* output_f = (float*)data_o;
    float* input_f = (float*) data_i;
    char* output_c = (char*) data_o;
    char* input_c = (char*) data_i;
#   pragma omp parallel
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        balance211(num_blocks_float, nthr, ithr, start, end);
        start = start * block_size;
        end = end * block_size;
#       pragma omp simd
        for (size_t e = start; e < end; ++e) {
            output_f[e] = input_f[e];
        }
        if (rem_elems_float != 0 && ithr ==  nthr -1 )  {
            for (auto e = nelems_float - rem_elems_float; e < nelems_float; ++e) {
                output_f[e] = input_f[e];
            }
        }
        if (nelems_char != 0 && ithr ==  nthr -1){
            for (auto e = nelems_float*4; e < len; ++e) {
                output_c[e] = input_c[e];
            }
        }
    }
    return;
}

#endif // _UTILS_H_
