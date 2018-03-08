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
#include <cstdlib>
#include <memory>
#include <omp.h>
#include "tensor.h"
#include "sum.h"

using namespace mkldnn;

static inline bool optimized_format(Tensor *t) {
    switch(t->format()) {
    case mkldnn_nChw16c:
    case mkldnn_nChw8c:
    case mkldnn_OIhw8i8o:
    case mkldnn_OIhw16i16o:
    case mkldnn_OIhw8i16o2i:
    case mkldnn_OIhw8o16i2o:
    case mkldnn_OIhw8o8i:
    case mkldnn_OIhw16o16i:
    case mkldnn_Oihw8o:
    case mkldnn_Oihw16o:
        return true;
    default:
        return false;
    }
}

template<typename T>
static T * sum_nChwXC_along_channel(T *src, mkldnn_memory_format_t format,
                                    mkldnn_dims_t dims, vector<int> axis, T *dst) {
    int mb = dims[0],
        ic = dims[1],
        ih = dims[2],
        iw = dims[3];
    const int cg = format == mkldnn_nChw16c ? 16 : 8;
    int cn = ic / cg;

    int blk_nthr = omp_get_max_threads(),
        blk_num = blk_nthr,
        blk_len = mb / blk_num,
        blk_len_ex = mb % blk_num;

    if (!blk_len)
        blk_nthr = mb;

    T *buf = reinterpret_cast<T *>(new avx::byte[ic * blk_nthr * sizeof(T)]);

    # pragma omp parallel num_threads(blk_nthr)
    {
        int ithr = omp_get_thread_num();
        int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
        int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
                     blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
        int bend = bstart + blen;

        T *loc_src = src + bstart * ic * ih * iw;
        if ((cg == 16) && (((unsigned long)buf & 0xf) == 0) && (((unsigned long)loc_src & 0xf) == 0)) {
            for (int b = bstart; b < bend; b++) {
                T *loc_buf = buf + ithr * ic;
                for (int c = 0; c < cn; c++) {
                    if (b == bstart)
                        for (int o = 0; o < cg; o++)
                            loc_buf[o] = 0;
                    for (int hw = 0; hw < ih * iw; hw++) {
                        __asm__(
                                "mov %0, %%rax\n"
                                "mov %1, %%rbx\n"
                                ".byte 0x62, 0xf1, 0x7c, 0x48, 0x10, 0x00\n" //vmovups (%%rax), %%zmm0
                                ".byte 0x62, 0xf1, 0x7c, 0x48, 0x58, 0x03\n" //vaddps (%%rbx), %%zmm0, %%zmm0
                                ".byte 0x62, 0xf1, 0x7c, 0x48, 0x11, 0x00\n" //vmovups %%zmm0, (%%rax)
                                :"+r"(loc_buf)
                                :"r"(loc_src)
                                :"rax", "rbx"
                                );
                        loc_src += cg;
                    }

                    loc_buf += cg;
                }
            }
        } else if ((cg == 8) && (((unsigned long)buf & 0x7) == 0) && (((unsigned long)loc_src & 0x7) == 0)) {
             for (int b = bstart; b < bend; b++) {
                T *loc_buf = buf + ithr * ic;
                for (int c = 0; c < cn; c++) {
                    if (b == bstart)
                        for (int o = 0; o < cg; o++)
                            loc_buf[o] = 0;
                    for (int hw = 0; hw < ih * iw; hw++) {
                        __asm__(
                                "mov %0, %%rax\n"
                                "mov %1, %%rbx\n"
                                ".byte 0xc5, 0xfc, 0x10, 0x00\n" //vmovups (%%rax), %%ymm0
                                ".byte 0xc5, 0xfc, 0x58, 0x03\n" //vaddps (%%rbx), %%ymm0, %%ymm0
                                ".byte 0xc5, 0xfc, 0x11, 0x00\n" //vmovups %%ymm0, (%rax)
                                :"+r"(loc_buf)
                                :"r"(loc_src)
                                :"rax", "rbx"
                                );
                        loc_src += cg;
                    }

                    loc_buf += cg;
                }
            }
        } else {
            for (int b = bstart; b < bend; b++) {
                T *loc_buf = buf + ithr * ic;
                for (int c = 0; c < cn; c++) {
                    if (b == bstart)
                        for (int o = 0; o < cg; o++)
                            loc_buf[o] = 0;

                    for (int hw = 0; hw < ih * iw; hw++) {
                        for (int o = 0; o < cg; o++)
                            loc_buf[o] += loc_src[o];
                        loc_src += cg;
                    }

                    loc_buf += cg;
                }
            }
        }

    }

    // Allreduce
    int c_nthr = omp_get_max_threads(),
        c_num = c_nthr,
        c_len = ic / c_num,
        c_len_ex = ic % c_num;

    if (!c_len)
        c_nthr = ic;

    # pragma omp parallel num_threads(c_nthr)
    {
        int ithr = omp_get_thread_num();
        int clen = ithr < c_len_ex ? c_len + 1 : c_len;
        int cstart = ithr <= c_len_ex ? (c_len + 1) * ithr :
                     c_len_ex * (c_len + 1) + (ithr - c_len_ex) * c_len;
        int cend = cstart + clen;

        for (int c = cstart; c < cend; c++)
            dst[c] = 0;

        for (int i = 0; i < blk_nthr; i++) {
            T *loc_buf = buf + i * ic;
            for (int c = cstart; c < cend; c++)
                dst[c] += loc_buf[c];
        }
    }

    delete(reinterpret_cast<avx::byte *>(buf));

    return dst;
}

// 4 dimensions(NCHW/OIHW) opitimzation for mkldnn backend only.
Tensor * sum_opt_along_axis(Tensor *src, vector<int> axis) {
    int axises = axis.size();
    vector<int> valid_axis_4dim = {0, 2, 3};

    if (src->ndims() != 4 || axises != 3) {
        return nullptr;
    }

    auto valid_axis = [](int axises,
                         vector<int> axis,
                         vector<int> valid_axis) -> bool {
        for (int i = 0; i < axises; i++) {
            if (valid_axis[i] != axis[i])
                return false;
        }
        return true;
    };

    try {
        switch (src->format()) {
        case mkldnn_nChw8c:
            if (!valid_axis(axises, axis, valid_axis_4dim))
                throw std::runtime_error(
                    "Invalid axis in tensor sum along axis <mkldnn_nChw8c>");
            break;
        case mkldnn_nChw16c:
            if (!valid_axis(axises, axis, valid_axis_4dim))
                throw std::runtime_error(
                    "Invalid axis in tensor sum along axis <mkldnn_nChw16c>");
            break;
        default:
            throw std::runtime_error(
                "Invalid format in tensor sum along axis");
            break;
        }
    } catch (std::runtime_error &e) {
        (void)e;
        return nullptr;
    }

    Tensor *dst = nullptr;
    try {
        switch (src->type()) {
        case FLOAT32:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<float *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<float *>(dst->data()));
            break;
        case SINT32:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<int32_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<int32_t *>(dst->data()));
            break;
        case SINT16:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<int16_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<int16_t *>(dst->data()));
            break;
        case SINT8:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<int8_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<int8_t *>(dst->data()));
            break;
        case UINT8:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<uint8_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<uint8_t *>(dst->data()));
            break;
        default:
            throw std::runtime_error(
            "Invalid dtype in tensor opt sum along axis");
            break;
        }
    } catch (std::runtime_error &e) {
        (void)e;
        return nullptr;
    }

    return dst;
}

// Less optimization gained in case of first dimension in small size
template<typename T>
static T * sum_along_axis(T *src, int src_ndims, mkldnn_dims_t src_dims,
                          vector<int> axis, vector<int> dst_dims, T *dst) {
    int tail = 1;
    for (int d = 1; d < src_ndims; d++)
        tail *= src_dims[d];

    bool along_mb = false;
    for (int a = 0; a < axis.size(); a++) {
        if (axis[a] == 0) {
            along_mb = true;
            break;
        }
    }

    int gbl_ws_size = 1;
    for (int d = 1; d < src_ndims; d++) {
        int a = 0;
        for (; a < axis.size(); a++)
            if (d == axis[a])
                break;

        if (a >= axis.size())
            gbl_ws_size *= src_dims[d];
    }

    int mb = src_dims[0];
    int blk_nthr = omp_get_max_threads(),
        blk_num = blk_nthr,
        blk_len = mb / blk_num,
        blk_len_ex = mb % blk_num;

    if (!blk_len)
        blk_nthr = mb;

    T *gbl_ws[blk_nthr];
    # pragma omp parallel num_threads(blk_nthr)
    {
        int ithr = omp_get_thread_num();
        int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
        int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
                     blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
        int bend = bstart + blen;

        T *loc_ws[blen];
        for (int b = bstart; b < bend; b++) {
            T *loc_src = src + b * tail;
            T *cur_src = loc_src;

            // Intialize for new blk
            vector<int> cur_dims;
            for (int d = 0; d < src_ndims; d++)
                cur_dims.push_back(src_dims[d]);

            vector<int> cur_axis;
            for (int a = 0; a < axis.size(); a++)
                if (axis[a] != 0)
                    cur_axis.insert(cur_axis.begin(), axis[a]);

            // Sum along axis[a]
            for (int a = 0; a < cur_axis.size(); a++) {

                int cur_fore = 1;
                for (int d = 1; d < cur_axis[a]; d++)
                    cur_fore *= cur_dims[d];

                int cur_tail = 1;
                for (int d = cur_axis[a] + 1; d < cur_dims.size(); d++)
                    cur_tail *= cur_dims[d];

                int cur_ws_size = cur_fore * cur_tail;
                T *ws = reinterpret_cast<T *>(new avx::byte[cur_ws_size * sizeof(T)]);
                for (int o = 0; o < cur_ws_size; o++) ws[o] = 0;

                // kernel
                for (int base = 0, off = 0, w = 0; w < cur_ws_size;) {
                    for (int t = 0; t < cur_dims[cur_axis[a]]; t++) {
                        ws[w] += cur_src[off + t * cur_tail];
                    }
                    w++; if (0 == w % cur_tail) {
                        off = base + cur_tail * cur_dims[cur_axis[a]];
                        base = off;
                    } else {
                        off += 1;
                    }
                }

                // adjust dims and cur_axis for sum in next axis
                cur_dims.erase(cur_dims.begin() + cur_axis[a]);
                for (int _a = a + 1; _a < cur_axis.size(); _a++) {
                    if (cur_axis[_a] > cur_axis[a])
                        cur_axis[_a] -= 1;
                }

                // refresh buffer
                if (cur_src != loc_src) delete(reinterpret_cast<avx::byte *>(cur_src));
                if (a == cur_axis.size() - 1) loc_ws[b - bstart] = ws;

                cur_src = ws;
            }
        }

        if (along_mb) {
            // local allreduce
            if (src_ndims == 2 && axis.size() == 1 && axis[0] == 0) {
                loc_ws[0] = reinterpret_cast<T *>(new avx::byte[tail * sizeof(T)]);
                for (int o = 0; o < tail; o++)
                    loc_ws[0][o] = 0;
                for (int b = bstart; b < bend; b++) {
                    T *loc_src = src + b * tail;
                    for (int o = 0; o < tail; o++)
                        loc_ws[0][o] += loc_src[o];
                }
            } else {
                for (int b = 1; b < blen; b++) {
                    for (int o = 0; o < gbl_ws_size; o++)
                        loc_ws[0][o] += loc_ws[b][o];
                    delete(reinterpret_cast<avx::byte *>(loc_ws[b]));
                }
            }

            gbl_ws[ithr] = loc_ws[0];
        } else {
            // cpy to dst
            for (int b = bstart; b < bend; b++) {
                for (int o = 0; o < gbl_ws_size; o++)
                    dst[b * gbl_ws_size + o] = loc_ws[b - bstart][o];
                delete(reinterpret_cast<avx::byte *>(loc_ws[b - bstart]));
            }
        }
    }

    if (along_mb) {
        // global allreduce
        int c_nthr = omp_get_max_threads(),
            c_num = c_nthr,
            c_len = gbl_ws_size / c_num,
            c_len_ex = gbl_ws_size % c_num;

        if (!c_len)
            c_nthr = gbl_ws_size;

        # pragma omp parallel num_threads(c_nthr)
        {
            int ithr = omp_get_thread_num();
            int clen = ithr < c_len_ex ? c_len + 1 : c_len;
            int cstart = ithr <= c_len_ex ? (c_len + 1) * ithr :
                         c_len_ex * (c_len + 1) + (ithr - c_len_ex) * c_len;
            int cend = cstart + clen;

            for (int c = cstart; c < cend; c++)
                dst[c] = 0;

            for (int i = 0; i < blk_nthr; i++) {
                T *loc_buf = gbl_ws[i];
                for (int c = cstart; c < cend; c++)
                    dst[c] += loc_buf[c];
            }
        }

        for (int i = 0; i < blk_nthr; i++)
            delete(reinterpret_cast<avx::byte *>(gbl_ws[i]));
    }

    return dst;
}

Tensor * sum_common_along_axis(Tensor *src, vector<int> axis) {
    auto dims = src->desc().data.dims;
    vector<int> o_dims;
    int o_ndims = src->ndims() - axis.size();

    // TODO: Support sum all
    if ((o_ndims != 1 && o_ndims != 2 && o_ndims != 4) ||
        axis.size() == 0)
        return nullptr;

    for (int d = 0; d < src->ndims(); d++) {
        unsigned a = 0; for (; a < axis.size(); a++) {
            if (d == axis[a])
                break;
        }

        if (a >= axis.size())
            o_dims.push_back(dims[d]);
    }

    Tensor *dst = nullptr;
    try {
        switch (src->type()) {
        case FLOAT32:
            dst = new Tensor(o_ndims, o_dims, src->type());
            sum_along_axis(static_cast<float *>(src->data()),
                           src->ndims(), src->desc().data.dims, axis,
                           o_dims, static_cast<float *>(dst->data()));
            break;
        case SINT32:
            dst = new Tensor(o_ndims, o_dims, src->type());
            sum_along_axis(static_cast<int32_t *>(src->data()),
                           src->ndims(), src->desc().data.dims, axis,
                           o_dims, static_cast<int32_t *>(dst->data()));
            break;
        case SINT16:
            dst = new Tensor(o_ndims, o_dims, src->type());
            sum_along_axis(static_cast<int16_t *>(src->data()),
                           src->ndims(), src->desc().data.dims, axis,
                           o_dims, static_cast<int16_t *>(dst->data()));
            break;
        case SINT8:
            dst = new Tensor(o_ndims, o_dims, src->type());
            sum_along_axis(static_cast<int8_t *>(src->data()),
                           src->ndims(), src->desc().data.dims, axis,
                           o_dims, static_cast<int8_t *>(dst->data()));
            break;
        case UINT8:
            dst = new Tensor(o_ndims, o_dims, src->type());
            sum_along_axis(static_cast<uint8_t *>(src->data()),
                           src->ndims(), src->desc().data.dims, axis,
                           o_dims, static_cast<uint8_t *>(dst->data()));
            break;
        default:
            throw std::runtime_error(
                  "Invalid dtype in tensor sum common along axis");
            break;
        }
    } catch (std::runtime_error &e) {
        (void)e;
        return nullptr;
    }

    return dst;
}

Tensor * blas_sum(Tensor *src, vector<int> axis) {
    if (optimized_format(src))
        return sum_opt_along_axis(src, axis);
    else
        return sum_common_along_axis(src, axis);
}
