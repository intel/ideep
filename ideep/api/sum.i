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

%module (package="mkldnn.api") sum
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include "mkldnn.hpp"
  using mkldnn::handle_traits;
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import memory.i

%at_sequence_typemap(std::vector<mkldnn::primitive::at>);

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

%exception sum::primitive_desc::primitive_desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

%rename (primitive_desc) sum::primitive_desc;
struct sum : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
                std::vector<memory::primitive_desc> inputs);

        primitive_desc(const memory::desc &output, std::vector<double> scale,
                std::vector<memory::primitive_desc> inputs);

        primitive_desc(std::vector<double> scale,
                std::vector<memory::primitive_desc> inputs);

        memory::primitive_desc dst_primitive_desc() const;

    };

    sum(const primitive_desc sum_pd,
            std::vector<primitive::at> inputs, const memory output);
};

} // namespace mkldnn
