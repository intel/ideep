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

%module (package="mkldnn.api") view
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include "mkldnn.hpp"
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

%rename (primitive_desc) view::primitive_desc;
struct view : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const memory::primitive_desc &input, memory::dims dims,
                memory::dims offsets);

        memory::primitive_desc dst_primitive_desc() const;
    };

    view(const primitive_desc &view_pd, primitive::at input);

    view(memory input, memory::dims dims, memory::dims offsets);
};

} // namespace mkldnn
