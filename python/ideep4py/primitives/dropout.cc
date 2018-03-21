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


#include <ctime>
#include <memory>
#include <mkldnn.hpp>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpu_info.h"
#include "dropout.h"
#include "layer.h"
#include "mkl_vsl.h"
#include "prim_factory.h"
#include "reorder_op.h"
#include "tensor.h"

static void bernoulli_generate(const long n, const double p, int* r) {
    std::srand(std::time(0));
    const int seed = 17 + std::rand() % 4096;

#ifdef _OPENMP
    int nthr = omp_get_max_threads();
    const int threshold = nthr * OpenMpManager::getProcessorSpeedMHz() / 3;
    const bool run_parallel = (omp_in_parallel() == 0) && (n >= threshold);
    if (!run_parallel) {
        nthr = 1;
    }

# pragma omp parallel num_threads(nthr)
    {
        const int ithr = omp_get_thread_num();
        const long avg_amount = (n + nthr - 1) / nthr;
        const long my_offset = ithr * avg_amount;
        const long my_amount = std::min(my_offset + avg_amount, n) - my_offset;
#else
    {
        const long my_amount = n;
        const long my_offset = 0;
#endif
        if (my_amount > 0) {
            VSLStreamStatePtr stream;
            vslNewStream(&stream, VSL_BRNG_MCG31, seed);
            vslSkipAheadStream(stream, my_offset);
            viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
            vslDeleteStream(&stream);
        }
    }
}

template<typename T>
std::vector<Tensor*> Dropout<T>::Forward(Tensor* x, float ratio) {
    const auto scale = 1.0 / (1.0 - ratio);
    const auto x_buf = static_cast<T*>(x->data());
    const auto size = x->size();
    const auto mask = new Tensor(x->ndims(), x->dims(), x->format(), x->type());
    const auto y = new Tensor(x->ndims(), x->dims(), x->format(), x->type());

    // Init the mask
    std::unique_ptr<int[]> bernouli_nums(new int[size]);
    bernoulli_generate(size, 1.0 - ratio, bernouli_nums.get());

    const auto mask_buf = static_cast<T*>(mask->data());
    const auto y_buf = static_cast<T*>(y->data());

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        mask_buf[i] = bernouli_nums[i] * scale;
        y_buf[i] = mask_buf[i] * x_buf[i];
    }

    return std::vector<Tensor*>{mask, y};
}

template<typename T>
Tensor* Dropout<T>::Backward(Tensor* mask, Tensor* gy) {
    assert(mask->size() == gy->size());

    // Reorder mask if needed
    auto gy_fmt = gy->cxx_format();
    auto mask_fmt = mask->cxx_format();
    void* mask_data = mask->data();
    shared_ptr<avx::byte> mask_reorder;

    if (gy_fmt == mask_fmt) {
        //LOG(INFO) << "mask fmt matched";
    } else {
       // LOG(INFO) << "mask fmt not match, need to reorder";
       // LOG(INFO) << "mask_fmt=" << mask_fmt <<", gy_fmt=" << gy_fmt;
        auto reorder_op = ReorderFactory<T>::get(mask->dims(), mask_fmt, gy_fmt);
        mask_reorder = Allocator::malloc(mask->len(), MPOOL_REORDER);
        //mask_reorder = new avx::byte[mask->len()];
        reorder_op->execute(mask->data(), mask_reorder.get());
        mask_data = mask_reorder.get();
    }

    const auto size = mask->size();
    const auto gx = new Tensor(gy->ndims(), gy->dims(), gy->format(), gy->type());

    //const auto mask_buf = static_cast<T*>(mask_reorder ? mask_reorder : mask->data());
    const auto mask_buf = static_cast<T*>(mask_data);
    const auto gy_buf = static_cast<T*>(gy->data());
    const auto gx_buf = static_cast<T*>(gx->data());

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        gx_buf[i] = mask_buf[i] * gy_buf[i];
    }

    return gx;
}

template class Dropout<float>;
