/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

%module (package="mkldnn.api") lrn_backward
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include <mkldnn.hpp>
  using mkldnn::handle_traits;

  #include "mdarray.h"
%}

%init %{
  import_array();
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import memory.i
%import mdarray.i
%import lrn_forward.i

namespace mkldnn {

%rename (desc) lrn_backward::desc;
%rename (primitive_desc) lrn_backward::primitive_desc;

%exception lrn_backward::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct lrn_backward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, double alpha, double beta);
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, double alpha, double beta, double k);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const lrn_forward::primitive_desc &hint_fwd_primitive_desc);

        memory::primitive_desc diff_src_primitive_desc() const;

        memory::primitive_desc workspace_primitive_desc() const;

        memory::primitive_desc diff_dst_primitive_desc() const;
    };

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src);

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src);
};

}
