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

%module (package="mkldnn.api") dropout
%{
  #define SWIG_FILE_WITH_INIT
  #include "dropout.h"
%}

%init %{
  import_array();
%}

%include stl.i
%include exception.i
%include pep_3118.i
%include "dropout.h"

%feature("flatnested");
%feature("nodefaultctor");

%template(dropout_f32) dropout<float>;
