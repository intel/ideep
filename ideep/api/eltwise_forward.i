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

%module (package="mkldnn.api") eltwise_forward
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include <mkldnn.hpp>
  using mkldnn::handle_traits;
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

%rename (desc) eltwise_forward::desc;
%rename (primitive_desc) eltwise_forward::primitive_desc;

%exception eltwise_forward::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct eltwise_forward : public primitive {
    struct desc {
        mkldnn_relu_desc_t data;
        // template <typename T>
        desc(prop_kind aprop_kind, algorithm alg_kind
            , const memory::desc &src_desc, double alpha, double beta);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);
        memory::primitive_desc dst_primitive_desc() const;
    };

    eltwise_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst);
};

} // namespace mkldnn
