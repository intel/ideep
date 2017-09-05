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

%module (package="mkldnn.api") bn_backward
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
%import bn_forward.i

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

%exception batch_normalization_backward::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

%rename (desc) batch_normalization_backward::desc;
%rename (primitive_desc) batch_normalization_backward::primitive_desc;
struct batch_normalization_backward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        //template <typename T>
        //desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
        //        const memory::desc &data_desc, T epsilon, unsigned flags);
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, double epsilon, unsigned flags);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const batch_normalization_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc diff_weights_primitive_desc() const;

        memory::primitive_desc mean_primitive_desc() const;

        memory::primitive_desc variance_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    // Prop_kind == backward
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const primitive::at &weights, const memory &diff_src,
            const memory &diff_weights);

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance,const primitive::at &diff_dst,
            const primitive::at &weights,  const memory &diff_src);

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const memory &diff_src);
};

} // namespace mkldnn
