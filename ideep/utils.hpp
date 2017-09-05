#ifndef _DNET_UTILS_HPP_
#define _DNET_UTILS_HPP_

#include <mkldnn.hpp>
using namespace mkldnn;

memory::format get_desired_format(int channel);

template<typename T>
void eltwise_multiply(T* x1, T* x2, T* y, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        y[i] = x1[i] * x2[i];
    }
}

#endif
