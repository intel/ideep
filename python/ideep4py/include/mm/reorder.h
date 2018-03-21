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


#pragma once

#include "mkldnn.hpp"
#include "tensor.h"

extern engine cpu_engine;
static constexpr int MAX_NDIM = 12; //XXX: For now

class Reorderer {
//protected:
public:
    bool non_trivial_;
    mkldnn::memory dst_;
    std::shared_ptr<avx::byte> data_;

    int ndims_;
    int size_;
    char format_[4];
    Py_ssize_t itemsize_;
    Py_ssize_t strides_[MAX_NDIM];
    Py_ssize_t shape_[MAX_NDIM];

    void _collect_buffer_info() {
        auto md = dst_.get_primitive_desc().desc();
        int ndims = md.data.ndims;

        ndims_ = ndims;
        switch(static_cast<mkldnn::memory::data_type>(md.data.data_type)) {
            case mkldnn::memory::f32:
                strcpy(format_, "f");
                itemsize_ = 4;
                break;
            case mkldnn::memory::s32:
                strcpy(format_, "i");
                itemsize_ = 4;
                break;
            case mkldnn::memory::s16:
                strcpy(format_, "h");
                itemsize_ = 2;
                break;
            case mkldnn::memory::s8:
                strcpy(format_, "b");
                itemsize_ = 1;
                break;
            case mkldnn::memory::u8:
                strcpy(format_, "B");
                itemsize_ = 1;
                break;
            default:
                break;
        }

        for (int i = 0; i < ndims; i ++) {
            shape_[i] = md.data.dims[i];
        }

        Py_ssize_t sd = itemsize_;

        for (int i = ndims -1; i >= 0; --i) {
            strides_[i] = sd;
            sd *= shape_[i];
        }
    }

    inline avx::byte *data() const { return data_.get(); }

public:
#if 0
    Reorderer(const py_handle in)
        :Reorderer(in.get()) {}
#endif
    Reorderer(const Tensor *src)
        : non_trivial_(src->incompatible()), dst_([src] () {
                if (src->incompatible()) {
                    auto md_data = src->desc().data;

                    mkldnn::memory::dims adims(md_data.dims
                        , md_data.dims + md_data.ndims);

                    mkldnn::memory::primitive_desc pd ({adims
                        , static_cast<mkldnn::memory::data_type>(md_data.data_type)
                        , static_cast<mkldnn::memory::format>(::public_format(md_data.format))}
                        , src->get_engine());

                    // XXX: magic number 4 is a hack
                    return mkldnn::memory(pd, reinterpret_cast<void *>(4));
                } else {
                    return src->mkldnn_memory();
                }} ()), size_(src->size()) {
            if (src->incompatible()) {
                auto pd = dst_.get_primitive_desc();

                data_ = std::shared_ptr<avx::byte>(new avx::byte [pd.get_size()]
                        , [](avx::byte *p) {delete [] p;});

                dst_.set_data_handle(data_.get());

            } else {
                data_ = src->share_data();
            }

            _collect_buffer_info();
        }

    mkldnn::reorder fire(const Tensor *src) {
        mkldnn::reorder reorder(src->mkldnn_memory(), dst_);
        mkldnn::stream s(mkldnn::stream::eager);

        s.submit({reorder}).wait();
        return reorder;
    }

    mkldnn::reorder sync(const Tensor *src) {
        mkldnn::reorder reorder(dst_, src->mkldnn_memory());
        mkldnn::stream s(mkldnn::stream::eager);

        s.submit({reorder}).wait();
        return reorder;
    }

    inline bool non_trivial() const {
        return non_trivial_;
    }

#if 0
    // PEP 3118 interface
    int build_view(Py_buffer *view, int flags) {
        view->buf = data_.get();
        view->itemsize = itemsize_;
        view->readonly = 0;
        view->internal = nullptr;
        view->len = size_ * itemsize_;

        if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
            view->format = format_;
        } else {
            view->format = nullptr;
        }

        if ((flags & PyBUF_ND) == PyBUF_ND) {
            view->ndim = ndims_;
            view->shape = shape_;
        } else {
            view->ndim = 0;
            view->shape = nullptr;
        }

        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
            view->strides = strides_;
        } else {
            view->strides = nullptr;
        }

        view->suboffsets = nullptr;

        return 0;
    }

    // Array protocol
    PyArrayInterface *build_array_struct(void) {
        auto arrstr = new PyArrayInterface();

        arrstr->two = 2;
        arrstr->nd = ndims_;
        arrstr->typekind = *((char *)format_);
        arrstr->itemsize = itemsize_;
        arrstr->flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_NOTSWAPPED |
            NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
        arrstr->flags &= ~(NPY_ARRAY_UPDATEIFCOPY | NPY_ARRAY_OWNDATA);
        arrstr->shape = shape_;
        arrstr->strides = strides_;
        arrstr->data = data_.get();
        arrstr->descr = nullptr;

        return arrstr;
    }
#endif
};
