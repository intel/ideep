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
#include <cstring>
#include <Python.h>
#include "mdarray.h"
#include "ideep.hpp"

using tensor = ideep::tensor;

class basic {
public:
  using tensor = ideep::tensor;
  using sum = ideep::sum;

  static PyObject *copyto(mdarray *dst, mdarray *src) {
    tensor dst_ = *dst->get();
    tensor src_ = *src->get();

    if (src_.get_data_type() != dst_.get_data_type() ||
        src_.get_dims() != dst_.get_dims()) {
      throw error(mkldnn_invalid_arguments,
            std::string("mismatch src and dst mdarray"));
      return nullptr;
    }

    fast_memcpy((char *)dst_.get_data_handle(),
                (char *)src_.get_data_handle(), src_.get_size());

    dst_.init(src_.get_descriptor(), dst_.get_data_handle());

    Py_RETURN_NONE;
  }

  static PyObject *copyto(mdarray *dst, Py_buffer *view) {
    tensor dst_ = *dst->get();

    if (dst_.get_size() != (unsigned)view->len) {
      throw error(mkldnn_invalid_arguments,
            std::string("mismatch src and dst mdarray"));
      return nullptr;
    }

    fast_memcpy((char *)dst_.get_data_handle(),
                (char *)view->buf, view->len);

    Py_RETURN_NONE;
  }

  static mdarray acc_sum(std::vector<mdarray> arrays) {
    std::vector<float> scales;
    std::vector<tensor> inputs;
    for (unsigned a = 0; a < arrays.size(); a++) {
      scales.push_back(1.0);
      inputs.push_back(*arrays[a]);
    }

    tensor output;
    sum::compute(scales, inputs, output);
    return mdarray(output);
  }
};
