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
  struct tp_traits {
    static PyObject *tp_richcompare(PyObject *self, PyObject *other, int cmp_op) {
      PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
                                            , NPY_ARRAY_ELEMENTSTRIDES, nullptr);
      if (surrogate == nullptr)
        return nullptr;

      PyObject *res = PyObject_RichCompare(surrogate, other, cmp_op);
      Py_DECREF(surrogate);
      return res;
    }

    static PyObject *tp_iter(PyObject *self) {
      PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
                                            , NPY_ARRAY_ELEMENTSTRIDES, nullptr);
      if (surrogate == nullptr)
          return nullptr;

      PyObject *res = PyObject_GetIter(surrogate);
      Py_DECREF(surrogate);
      return res;
    }
  };
%}

%define %tp_slot(name, type)
  %feature("python:tp_" %str(name)) type "tp_traits<" %str(type) ">::tp_" %str(name);
%enddef

%define %tp_protocol(type...)
  %tp_slot(richcompare, type)
  %tp_slot(iter, type)
%enddef
