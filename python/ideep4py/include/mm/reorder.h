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
#include "ideep.hpp"

static constexpr int MAX_NDIM = 12; //XXX: For now

class reorderer {
//protected:
public:
  using tensor = ideep::tensor;
  using data_type_t = mkldnn::memory::data_type;
  using format_t = ideep::format;
  using reorder = ideep::reorder;
  using descriptor = tensor::descriptor;
  using scratch_allocator = ideep::utils::scratch_allocator;

  bool non_trivial_;
  tensor dst_;
  void *data_;

  int ndims_;
  int size_;
  char format_[4];
  ssize_t itemsize_;
  ssize_t strides_[MAX_NDIM];
  ssize_t shape_[MAX_NDIM];

  void _collect_buffer_info() {
    ndims_ = dst_.ndims();

    switch(dst_.get_data_type()) {
    case data_type_t::f32:
      strcpy(format_, "f");
      itemsize_ = 4;
      break;
    case data_type_t::s32:
      strcpy(format_, "i");
      itemsize_ = 4;
      break;
    case data_type_t::s16:
      strcpy(format_, "h");
      itemsize_ = 2;
      break;
    case data_type_t::s8:
      strcpy(format_, "b");
      itemsize_ = 1;
      break;
    case data_type_t::u8:
      strcpy(format_, "B");
      itemsize_ = 1;
      break;
    default:
      break;
    }

    auto _dims = dst_.get_dims();
    for (int i = 0; i < ndims_; i ++) {
      shape_[i] = _dims[i];
    }

    ssize_t sd = itemsize_;

    for (int i = ndims_ - 1; i >= 0; --i) {
      strides_[i] = sd;
      sd *= shape_[i];
    }
  }

  inline void *data() const { return data_; }

public:
  reorderer(const tensor &src) :
      non_trivial_(!src.is_public_format()),
      dst_([&] () {
        if (non_trivial()) {
          tensor dst;
          dst.init<scratch_allocator, reorder>({src.get_dims(), src.get_data_type(),
              descriptor::public_compatible_format(src.get_descriptor())});
          return dst;
        } else {
          return src;
      }} ()),
      size_(src.get_size()) {
    if (non_trivial()) {
      data_ = dst_.get_data_handle();
    } else {
      data_ = src.get_data_handle();
    }

    _collect_buffer_info();
  }

  void fire(const tensor &src) {
    // TODO : src -> dst
    if (non_trivial())
      reorder::compute(src, dst_);
  }

  void sync(const tensor &src) {
    // TODO : dst -> src
    if (non_trivial())
      reorder::compute(dst_, src);
  }

  inline bool non_trivial() const {
    return non_trivial_;
  }
};
