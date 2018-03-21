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


%define %int_sequence_typemap(integer_sequence_compitable_type)

%typemap(typecheck) (integer_sequence_compitable_type) {
  $1 = PySequence_Check($input);
}

%typemap(in) (integer_sequence_compitable_type) (int count) {
  count = PySequence_Size($input);

  for (int i =0; i < count; i ++) {
    PyObject *o = PySequence_GetItem($input, i);
    $1.push_back(PyLong_AsLong(o));
  }
}
%enddef

%define %at_sequence_typemap(at_sequence_compitable_type)

%typemap(typecheck) (at_sequence_compitable_type) {
  $1 = PySequence_Check($input);
}

%typemap(in) (at_sequence_compitable_type) (int count,
    at_sequence_compitable_type ins) {
  count = PySequence_Size($input);
  for (int i =0; i < count; i ++) {
    PyObject *o = PySequence_GetItem($input, i);
    mkldnn::primitive::at *tmp;
    int res1 = SWIG_ConvertPtr(o, reinterpret_cast<void **>(&tmp)
        , $descriptor(mkldnn::primitive::at *), 0);

    if (!SWIG_IsOK(res1)) {
      SWIG_exception_fail(SWIG_ArgError(res1)
          , "typemap 'mkldnn::primitive::at' sequence type failed");
    }
    if (tmp == nullptr) {
      SWIG_exception_fail(SWIG_ArgError(res1)
          , "Input is not a sequential type of 'mkldnn::primitive::at'");
    }
    ins.emplace_back(*tmp);
  }

  $1 = std::move(ins);
}
%enddef
