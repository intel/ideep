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

%module (package="mkldnn.api") convolution_backward_weights
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
%include pep_3118.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import memory.i
%import mdarray.i
%import convolution_forward.i

namespace mkldnn {

%rename (desc) convolution_backward_weights::desc;
%rename (primitive_desc) convolution_backward_weights::primitive_desc;

%exception convolution_backward_weights::desc::desc {
  try {
    $action
  } catch (mkldnn::error &e) {
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct convolution_backward_weights : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);

        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc src_primitive_desc() const;
        memory::primitive_desc diff_weights_primitive_desc() const;
        memory::primitive_desc diff_bias_primitive_desc() const;
        memory::primitive_desc diff_dst_primitive_desc() const;
    };

    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights, const memory &diff_bias);

    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights);
};

}

%extend_ro_attr_and_own(bwb_op<mkldnn::convolution_backward_weights>
                , mdarray, extra, extra_get)

%buffer_protocol_producer (bwb_op<mkldnn::convolution_backward_weights>)
%buffer_protocol_producer (bw_op<mkldnn::convolution_backward_weights>)

%template (conv_bwb_op) bwb_op<mkldnn::convolution_backward_weights>;
%template (conv_bw_op) bw_op<mkldnn::convolution_backward_weights>;
