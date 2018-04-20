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
  struct map_traits {
    static Py_ssize_t mp_length(PyObject *self) {
      void *that;

      int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);
      if (!SWIG_IsOK(res1)) {
        PyErr_SetString(PyExc_ValueError, "Wrong self object in mp_length");
        return 0;
      }

      return (*reinterpret_cast<T *>(that))->mp_length(self);
    }

    static PyObject *mp_subscript(PyObject *self, PyObject *op) {
      void *that;

      int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);
      if (!SWIG_IsOK(res1)) {
        PyErr_SetString(PyExc_ValueError, "Wrong self object in mp_subscript");
        return nullptr;
      }

      return (*reinterpret_cast<T *>(that))->mp_subscript(self, op);
    }

    static int mp_ass_subscript(PyObject *self, PyObject *ind, PyObject *op) {
      void *that;

      int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);
      if (!SWIG_IsOK(res1)) {
        PyErr_SetString(PyExc_ValueError, "Wrong self object in mp_subscript");
        return -1;
      }

      return (*reinterpret_cast<T *>(that))->mp_ass_subscript(self, ind, op);
    }
  };
%}

%define %map_slot(name, type)
  %feature("python:mp_" %str(name)) type "map_traits<" %str(type) ">::mp_" %str(name);
%enddef

%define %map_protocol(type...)
  %map_slot(length, type)
  %map_slot(subscript, type)
  %map_slot(ass_subscript, type)
%enddef
