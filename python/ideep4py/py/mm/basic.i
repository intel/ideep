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
  #include "basic.h"
%}

%typemap(in) (std::vector<mdarray *> arrays) {
    int i;
    int argc;
    std::vector<mdarray *> varr;
    if (!PyTuple_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expected a tuple");
        return nullptr;
    }
    argc = PyTuple_Size($input);
    for (i = 0; i < argc; i++) {
        PyObject *obj = PyTuple_GET_ITEM($input, i);
        if (!implementation::mdarray::is_mdarray(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expected a mdarray in acc_sum");
            return nullptr;
        }
#if 0
        if (!PyArray_Check(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expected a array");
            return nullptr;
        }
#endif
        void *that;
        int res1 = SWIG_ConvertPtr(obj, &that, nullptr, 0);
        if (!SWIG_IsOK(res1)) {
            PyErr_SetString(PyExc_ValueError, "Can't convert mdarray pyobject");
            return nullptr;
        }
        varr.push_back((mdarray *)that);
    }
    $1 = varr;
}

class basic {
public:
    static PyObject *copyto(mdarray *dst, mdarray *src);
    static PyObject *copyto(mdarray *dst, Py_buffer *view);
    static mdarray acc_sum(std::vector<mdarray *> arrays);
};

