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


#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include <omp.h>
#include "mkl_vml_functions.h"
#include "layer.h"
#include "tensor.h"
#include "bn.h"
#include "bn_fwd.h"
#include "bn_bwd.h"
#include "prim_factory.h"
#include "reorder_op.h"

template<typename T>
void batch_normalization_inv(T *var, float eps, int size, T *inv) {
    int blk_nthr = omp_get_max_threads(),
        blk_num = blk_nthr,
        blk_len = size / blk_num,
        blk_len_ex = size % blk_num;

    if (!blk_len)
        blk_nthr = size;

    T *var_eps = reinterpret_cast<T *>(new avx::byte[size * sizeof(T)]);

    # pragma omp parallel num_threads(blk_nthr)
    {
        int ithr = omp_get_thread_num();
        int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
        int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
                     blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
        int bend = bstart + blen;

        for (int b = bstart; b < bend; b++)
            var_eps[b] = var[b] + eps;
    }

    vsPowx(size, var_eps, -0.5, inv);
    delete(reinterpret_cast<avx::byte *>(var_eps));
    return;
}

template<typename T>
std::vector<Tensor *> batch_normalization<T>::Forward(
    Tensor *src, Tensor *w, Tensor *mean, Tensor *var, float eps) {

    assert(memory_data_type<T>() == src.cxx_data_type());

    bool scale_shift = w ? true : false;
    bool global_stats = mean ? true : false;
    bool training = mean ? false : true;

    auto bn_fwd = batch_normalization_fwd_factory<T>::get(
            (mkldnn::memory::dims)src->dims(),
            eps, scale_shift, global_stats, training);

    void *src_data = src->data();
    shared_ptr<avx::byte> src_itnl;
    if (src->cxx_format() != bn_fwd->get_src_fmt()) {
        auto reorder = ReorderFactory<T>::get(
            (mkldnn::memory::dims)src->dims(),
            (mkldnn::memory::format)src->cxx_format(),
            (mkldnn::memory::format)bn_fwd->get_src_fmt());
        src_itnl= Allocator::malloc(src->len(), MPOOL_REORDER);
        //src_itnl = new avx::byte[src->len()];
        reorder->execute(src_data, src_itnl.get());
        src_data = src_itnl.get();
    }

#if 0
    auto dst = new Tensor(src->ndims(), src->dims(),
                          (mkldnn_memory_format_t)bn_fwd->get_dst_fmt(),
                          src->type());
    mean = training ?
           new Tensor(bn_fwd->get_mean_ndims(), bn_fwd->get_mean_dims(),
                      (mkldnn_memory_format_t)bn_fwd->get_mean_fmt(),
                      src->type()) : mean;
    var = training ?
          new Tensor(bn_fwd->get_var_ndims(), bn_fwd->get_var_dims(),
                     (mkldnn_memory_format_t)bn_fwd->get_var_fmt(),
                     src->type()) : var;
#else
    auto data = Allocator::malloc(src->dims(), type2size(src->type()), MPOOL_BN_FWD);
    auto dst = new Tensor(src->ndims(), src->dims(), data,
            (mkldnn_memory_format_t)bn_fwd->get_dst_fmt(),
            src->type());

    Tensor *inv;
    if (training) {
        auto data_mean = Allocator::malloc(bn_fwd->get_mean_dims(), type2size(src->type()), MPOOL_BN_FWD);
        mean = new Tensor(bn_fwd->get_mean_ndims(), bn_fwd->get_mean_dims(), data_mean,
                      (mkldnn_memory_format_t)bn_fwd->get_mean_fmt(),
                      src->type());
        auto data_var = Allocator::malloc(bn_fwd->get_var_dims(), type2size(src->type()), MPOOL_BN_FWD);
        var = new Tensor(bn_fwd->get_var_ndims(), bn_fwd->get_var_dims(), data_var,
                     (mkldnn_memory_format_t)bn_fwd->get_var_fmt(),
                     src->type());
        auto data_inv = Allocator::malloc(bn_fwd->get_var_dims(), type2size(src->type()), MPOOL_BN_FWD);
        inv = new Tensor(bn_fwd->get_var_ndims(), bn_fwd->get_var_dims(), data_inv,
                     (mkldnn_memory_format_t)bn_fwd->get_var_fmt(),
                     src->type());
    }
#endif

    bn_fwd->execute(src_data, (w ? w->data() : nullptr),
                    dst->data(), (mean ? mean->data() : nullptr),
                    (var ? var->data() : nullptr));

    std::vector<Tensor *> outs;
    outs.push_back(dst);
    if (training) {
        outs.push_back(mean);
        outs.push_back(var);

        batch_normalization_inv(reinterpret_cast<T *>(var->data()), eps,
                                var->desc().data.dims[0],
                                reinterpret_cast<T *>(inv->data()));
        outs.push_back(inv);
    }

    return outs;
}

template<typename T>
std::vector<Tensor *> batch_normalization<T>::Backward(
            Tensor *src, Tensor *diff_dst, Tensor *mean,
            Tensor *var, Tensor *w, float eps) {

    assert(memory_data_type<T>() == src.cxx_data_type());

    bool scale_shift = w ? true : false;

    auto bn_bwd = batch_normalization_bwd_factory<T>::get(
            (mkldnn::memory::dims)src->dims(),
            (mkldnn::memory::dims)diff_dst->dims(),
            eps, scale_shift);

    void *src_data = src->data();
    shared_ptr<avx::byte> src_itnl;
    if (src->cxx_format() != bn_bwd->get_src_fmt()) {
        auto reorder = ReorderFactory<T>::get(
            (mkldnn::memory::dims)src->dims(),
            (mkldnn::memory::format)src->cxx_format(),
            (mkldnn::memory::format)bn_bwd->get_src_fmt());
        //src_itnl = new avx::byte[src->len()];
        src_itnl= Allocator::malloc(src->len(), MPOOL_REORDER);
        reorder->execute(src_data, src_itnl.get());
        src_data = src_itnl.get();
    }

    void *diff_dst_data = diff_dst->data();
    shared_ptr<avx::byte> diff_dst_itnl;
    if (diff_dst->cxx_format() != bn_bwd->get_diff_dst_fmt()) {
        auto reorder = ReorderFactory<T>::get(
            (mkldnn::memory::dims)diff_dst->dims(),
            (mkldnn::memory::format)diff_dst->cxx_format(),
            (mkldnn::memory::format)bn_bwd->get_diff_dst_fmt());
        diff_dst_itnl = Allocator::malloc(diff_dst->len(), MPOOL_REORDER);
        //diff_dst_itnl = new avx::byte[diff_dst->len()];
        reorder->execute(diff_dst_data, diff_dst_itnl.get());
        diff_dst_data = diff_dst_itnl.get();
    }

#if 0
    auto diff_src = new Tensor(src->ndims(), src->dims(),
                    (mkldnn_memory_format_t)bn_bwd->get_diff_src_fmt(),
                    src->type());
    auto diff_w = scale_shift ?
                  new Tensor(w->ndims(), w->dims(),
                  (mkldnn_memory_format_t)bn_bwd->get_diff_w_fmt(),
                  w->type()) : (Tensor *)(nullptr);
#else
    auto data = Allocator::malloc(src->dims(), type2size(src->type()), MPOOL_BN_BWD);
    auto diff_src = new Tensor(src->ndims(), src->dims(), data,
                    (mkldnn_memory_format_t)bn_bwd->get_diff_src_fmt(),
                    src->type());
    Tensor *diff_w = nullptr;
    if (scale_shift) {
        auto data_diff_w = Allocator::malloc(w->dims(), type2size(src->type()), MPOOL_BN_BWD);
        diff_w = new Tensor(w->ndims(), w->dims(), data_diff_w,
                (mkldnn_memory_format_t)bn_bwd->get_diff_w_fmt(),
                w->type());
    }
#endif

    bn_bwd->execute(src_data, diff_dst_data, mean->data(), var->data(),
                    (w ? w->data() : nullptr), diff_src->data(),
                    (diff_w ? diff_w->data() : nullptr));

    std::vector<Tensor *> outs;
    outs.push_back(diff_src);
    if (scale_shift)
        outs.push_back(diff_w);

    return outs;
}

template class batch_normalization<float>;
