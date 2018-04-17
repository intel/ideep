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


%{
  template <class T>
  struct buffer_traits {
    #define GET_SELF_OBJ(self, that) \
      do { \
        int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0); \
        if (!SWIG_IsOK(res1)) { \
          PyErr_SetString(PyExc_ValueError, "Wrong self object in getbuffer wrapper"); \
          return -1; \
        } \
      } while (0)

    static int getbuffer(PyObject *self, Py_buffer *view, int flags) {
      void *that;

      GET_SELF_OBJ(self, that);

      // TODO: support smart pointer and raw at same time
      return (*reinterpret_cast<T *>(that))->getbuffer(self, view, flags);
    }
  };
%}

%define %buffer_protocol_producer(type...)
  %feature("python:bf_getbuffer") type "buffer_traits<" %str(type) ">::getbuffer";

#if defined(NEWBUFFER_ON)
  %feature("python:tp_flags") type "Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_HAVE_NEWBUFFER";
#endif

%enddef

%define %buffer_protocol_typemap(VIEW)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) (VIEW) {
  $1 = PyObject_CheckBuffer($input);
}

%typemap(in) (VIEW) (int res, Py_buffer *view = nullptr
  , int flags = PyBUF_C_CONTIGUOUS | PyBUF_RECORDS_RO) {
  view = new Py_buffer;
  res = PyObject_GetBuffer($input, view, flags);
  if (res != 0) {
    $1 = NULL;
    goto fail;
  } else {
    $1 = ($1_ltype) view;
  }
  // TODO: IF WE CONFRONT A F_CONTINGUOUS ONE???
}
%enddef
