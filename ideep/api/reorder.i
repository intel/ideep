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

%module (package="mkldnn.api") reorder
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

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

%rename (primitive_desc) reorder::primitive_desc;

%exception reorder::primitive_desc {
  try {
    $action
  } catch (mkldnn::error &e) {
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

%exception reorder::reorder {
  try {
    $action
  } catch (mkldnn::error &e) {
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct reorder : public primitive {
    struct primitive_desc {
        primitive_desc(const memory::primitive_desc &input,
                       const memory::primitive_desc &output);
    };

    reorder(const primitive_desc &aprimitive_desc,
            const primitive::at &input, const memory &output);

    reorder(const primitive::at &input, const memory &output);
};

} // namespace mkldnn
