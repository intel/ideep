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
  struct getattr_traits {
    static PyObject *getattro_hook(PyObject *self, PyObject *name) {

      // Call python default first.
      PyObject *res = PyObject_GenericGetAttr(self, name);

      // notify our hook if we find nothing from outside.
      if (res == nullptr && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();

        void *that;
        int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);

        if (!SWIG_IsOK(res1)) {
          PyErr_SetString(PyExc_ValueError, "Wrong self object in getattro wrapper");
          res = nullptr;
        }

        // XXX: should we bump up reference counter?
        // TODO: Support both raw and smart pointer
        res = reinterpret_cast<T *>(that)->get()->getattro(self, name);
      }

      return res;
    }
  };
%}

%define %getattr_wrapper(type...)
  %feature("python:tp_getattro") type "getattr_traits<" %str(type) ">::getattro_hook";
%enddef
