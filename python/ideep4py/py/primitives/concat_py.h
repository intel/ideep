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


#ifndef _CONCAT_PY_H_
#define _CONCAT_PY_H_

#include <vector>
#include <memory>
#include "mdarray.h"
#include "ideep.hpp"

class Concat
{
public:
  using scratch_allocator = ideep::utils::scratch_allocator;
  using tensor = ideep::tensor;
  using concat = ideep::concat;
  using spliter = ideep::reorder;

  static mdarray Forward(std::vector<mdarray> inputs, int axis) {
    std::vector<tensor> inputs_;
    for (mdarray elems : inputs) {
      inputs_.push_back(*elems.get());
    }

    tensor dst;
    concat::compute<scratch_allocator>(inputs_, axis, dst);
    auto out = mdarray(dst);

    return out;
  }


  static std::vector<mdarray> Backward(mdarray *grady,
                                       std::vector<int> offsets,
                                       int axis) {
    std::vector<mdarray> gxs;
    std::vector<int> axis_len;
    tensor::dims offset_dims(grady->get()->ndims(), 0);
    tensor::dims grady_dims = grady->get()->get_dims();
    tensor::dims gradx_dims(grady_dims);

    // FIXME
    // For split function usage. if not support, fallback to numpy
    bool ret = is_valid_offsets(grady_dims, offsets, axis_len, axis);
    if (!ret)
      return gxs;

    for (unsigned i = 0; i < axis_len.size(); i++) {
      gradx_dims[axis] = axis_len[i];

      auto gradx = spliter::compute<scratch_allocator>(*grady->get(),
                   gradx_dims, offset_dims);

      gxs.push_back(mdarray(gradx));
      offset_dims[axis] += axis_len[i];
    }

    return gxs;
  }


private:
  static bool is_valid_offsets(tensor::dims grady_dims, std::vector<int> &offsets,
        std::vector<int> &axis_len, int axis) {
    int min_value = -1;
    std::vector<int> valid_offsets;
    for (unsigned i = 0; i < offsets.size(); i++) {
      if (offsets[i] < 0)
          offsets[i] += grady_dims[axis];

      if (offsets[i] == 0) // mkldnn can't handle zero dim
          return false;
      else if (offsets[i] > min_value) {
        min_value = offsets[i];

        // larger than max value in corresponding dims
        if (offsets[i] >= grady_dims[axis])
          return false;
        else
          valid_offsets.push_back(offsets[i]);
      } else { // out of order
        return false;
      }
    }

    if (valid_offsets.empty())
      return false;

    // push dim len along axis
    for (unsigned i = 0; i < valid_offsets.size(); i++) {
      if (i == 0)
        axis_len.push_back(valid_offsets[i]);
      else
        axis_len.push_back(valid_offsets[i] - valid_offsets[i - 1]);
    }

    // push last dim len
    axis_len.push_back(grady_dims[axis] - valid_offsets.back());

    return true;
  }
};

#endif // _CONCAT_PY_H_
