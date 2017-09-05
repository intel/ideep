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

%module (package="mkldnn.api") lrn_forward
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

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

%rename (desc) lrn_forward::desc;
%rename (primitive_desc) lrn_forward::primitive_desc;

%exception lrn_forward::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct lrn_forward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, double alpha, double beta);
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, double alpha, double beta, double k);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc src_primitive_desc() const;

        memory::primitive_desc workspace_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &workspace,
            const memory &dst);

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst);
};

} // namespace mkldnn
