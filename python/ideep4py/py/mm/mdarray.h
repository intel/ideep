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


#ifndef _MDARRAY_H_
#define _MDARRAY_H_
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <forward_list>
#include <stdexcept>
#include <type_traits>
#include <swigpyrun.h>
#include "ideep.hpp"
#include "reorder.h"

namespace implementation {
  class mdarray;
}

using py_handle = std::shared_ptr<implementation::mdarray>;

namespace implementation {

#if PY_VERSION_HEX >= 0x03000000
  int g_init();
#else
  void g_init();
#endif

#define NPY_ARRAY_SURROGATE_ENTRY(mdarray) \
  PyObject *surrogate = PyArray_FromAny(mdarray, nullptr, 0, 0 \
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr)   \

#define NPY_ARRAY_SURROGATE_EXIT()

#define nb_unary_map_impl(method) \
  PyObject * m_ ## method ## _map_impl(PyObject *self) { \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate); \
    Py_DECREF(surrogate);   \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  } \

#define nb_unary_map(method) \
  nb_unary_map_impl(method) \
  PyObject * m_ ## method (PyObject *self) {    \
    return m_ ## method ## _map_impl(self); \
  } \

#define nb_binary_map_impl(method) \
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o) {   \
    PyObject *left = self, *right = o;                                  \
    if (is_mdarray(left)) {                                             \
      left = PyArray_FromAny(left, nullptr, 0, 0                        \
        , NPY_ARRAY_ELEMENTSTRIDES, nullptr);                           \
    }                                                                   \
    if (is_mdarray(right)) {                                            \
      right = PyArray_FromAny(right, nullptr, 0, 0                      \
        , NPY_ARRAY_ELEMENTSTRIDES, nullptr);                           \
    }                                                                   \
    PyObject *res = PyNumber_ ## method(left, right);                   \
    if (left != self)                                                   \
      Py_DECREF(left);                                                  \
    if (right != o)                                                     \
      Py_DECREF(right);                                                 \
    return res;                                                         \
  }

#define nb_binary_map_impl_with_target_func(method, tfunc) \
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o) {    \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## tfunc(surrogate, o); \
    Py_DECREF(surrogate);   \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  }

#define nb_binary_map(method) \
  nb_binary_map_impl(method) \
  PyObject * m_ ## method (PyObject *self, PyObject *o) {    \
    return m_ ## method ## _map_impl(self, o); \
  } \

#define nb_ternary_map_impl(method) \
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o1, PyObject *o2) {    \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate, o1, o2); \
    Py_DECREF(surrogate); \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  }

#define nb_ternary_map(method) \
  nb_ternary_map_impl(method) \
  PyObject * m_ ## method (PyObject *self, PyObject *o1, PyObject *o2) {    \
    return m_ ## method ## _map_impl(self, o1, o2); \
  } \

class mdarray : public ideep::tensor {
public:
  // It is exposed to python
  //
  static constexpr int MAX_NDIM = 12; //XXX: For now

  // class Reorder_buffer : Reorderer {
  // public:
  //   Reorder_buffer(const py_handle in)
  //       :Reorderer(in.get()->tensor()) {}
  // };

public:
  using tensor = ideep::tensor;
  using descriptor = ideep::tensor::descriptor;
  using data_type_t = mkldnn::memory::data_type;
  using dims_t = mkldnn::memory::dims;
  using error = mkldnn::error;

  typedef size_t size_type;

  mdarray();
  virtual ~mdarray() = default;

  mdarray(const mdarray &m) : tensor(m) {}

  mdarray(const tensor &t) : tensor(t) {}

  mdarray(dims_t &dims, data_type_t dt) : tensor(descriptor(dims, dt)) {}

  inline dims_t get_dims_from_view(Py_buffer *view) {
    return dims_t(view->shape, view->shape + view->ndim);
  };

  inline data_type_t get_dtype_from_view(Py_buffer *view) {
    data_type_t dt;
    std::string format(view->format);
    if (std::string::npos != format.find_last_of('f')) {
      dt = data_type_t::f32;
    } else if (std::string::npos != format.find_last_of('i')) {
      dt = data_type_t::s32;
    } else if (std::string::npos != format.find_last_of('h')) {
      dt = data_type_t::s16;
    } else if (std::string::npos != format.find_last_of('b')) {
      dt = data_type_t::s8;
    } else if (std::string::npos != format.find_last_of('B')) {
      dt = data_type_t::u8;
    } else {
      throw error(mkldnn_invalid_arguments,
          std::string("mdarray does not support data type: ") +
          format);
    }
    return dt;
  };

  inline void *get_buff_from_view(Py_buffer *view) {
    // TODO: re-alignment
    return view->buf;
  }

  mdarray(Py_buffer *view, char input_type='d') :
      tensor({ get_dims_from_view(view), get_dtype_from_view(view) },
             get_buff_from_view(view)) { /* TODO: input_type */ }

  static bool is_mdarray(PyObject *o);

  //FIXME
  inline void unpickled_data(void *pdata) {
    //data_.reset(reinterpret_cast<avx::byte *>(pdata));
    //m_.set_data_handle(pdata);
    return;
  }

  // PEP 3118 interface
  int build_view(Py_buffer *view, int flags, const reorderer &reorder) {
      view->buf = reorder.data_;
      view->itemsize = reorder.itemsize_;
      view->readonly = 0;
      view->internal = nullptr;
      view->len = reorder.size_ * reorder.itemsize_;

      if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
          view->format = const_cast<char *>(reorder.format_);
      } else {
          view->format = nullptr;
      }

      if ((flags & PyBUF_ND) == PyBUF_ND) {
          view->ndim = reorder.ndims_;
          view->shape = const_cast<Py_ssize_t *>(reorder.shape_);
      } else {
          view->ndim = 0;
          view->shape = nullptr;
      }

      if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          view->strides = const_cast<Py_ssize_t *>(reorder.strides_);
      } else {
          view->strides = nullptr;
      }

      view->suboffsets = nullptr;

      return 0;
  }

  // PyObject *__getstate__(void) const;

  // void __setstate__(PyObject *state);

  PyObject *py_mdarray_from(PyObject *o) const;

  /// d = a * x + b * y, using x's format
  template<class T>
  static void axpby(mdarray *dst, T a, mdarray *x, T b, mdarray *y);

  /// Interface to directly contact python
  template<class T>
  PyObject *axpby(T a, T b, PyObject *o);

  template<class T>
  PyObject *inplace_axpby(T a, PyObject *self, T b, PyObject *o);

  PyObject *flat(void);

  PyObject *reshape(py_handle *self, std::vector<int> dims);

  PyObject *m_mult_div(PyObject *self, PyObject *o, int mult_or_div, bool inplace);

  // PyObject *sum(std::vector<int> axis, bool keepdims);

  // PEP: 3118 Buffer Protocol Producer
  virtual int getbuffer(PyObject *obj, Py_buffer *view, int flags);

  PyObject *getattro(PyObject *self, PyObject *name);

  PyObject *m_Add(PyObject *self, PyObject *o);
  nb_binary_map_impl(Add);
  PyObject *m_InPlaceAdd(PyObject *self, PyObject *o);
  nb_binary_map_impl(InPlaceAdd);
  PyObject *m_Subtract(PyObject *self, PyObject *o);
  nb_binary_map_impl(Subtract);
  PyObject *m_InPlaceSubtract(PyObject *self, PyObject *o);
  nb_binary_map_impl(InPlaceSubtract);
  PyObject *m_Multiply(PyObject *self, PyObject *o);
  nb_binary_map_impl(Multiply);
  PyObject *m_InPlaceMultiply(PyObject *self, PyObject *o);
  nb_binary_map_impl(InPlaceMultiply);
  // SWIG: nb_true_divide (no slot) <= nb_divide
  PyObject *m_Divide(PyObject *self, PyObject *o);
#if PY_VERSION_HEX < 0x03000000
  nb_binary_map_impl(Divide);
#else
  nb_binary_map_impl_with_target_func(Divide, TrueDivide);
#endif
  PyObject *m_InPlaceDivide(PyObject *self, PyObject *o);
#if PY_VERSION_HEX < 0x03000000
  nb_binary_map_impl(InPlaceDivide);
#else
  nb_binary_map_impl_with_target_func(InPlaceDivide, InPlaceTrueDivide);
#endif

  nb_binary_map(Remainder);
  nb_binary_map(Divmod);
  nb_unary_map(Negative);
  nb_unary_map(Positive);
  nb_unary_map(Absolute);
  nb_unary_map(Invert);
  nb_binary_map(Lshift);
  nb_binary_map(Rshift);
  nb_binary_map(And);
  nb_binary_map(Xor);
  nb_binary_map(Or);
  nb_binary_map(InPlaceRemainder);
  nb_ternary_map(InPlacePower);
  nb_binary_map(InPlaceLshift);
  nb_binary_map(InPlaceRshift);
  nb_binary_map(InPlaceAnd);
  nb_binary_map(InPlaceXor);
  nb_binary_map(InPlaceOr);
  nb_binary_map(FloorDivide);
  nb_binary_map(InPlaceFloorDivide);
#if (PY_VERSION_HEX >= 0x03000000)
  nb_binary_map(MatrixMultiply);
  nb_binary_map(InPlaceMatrixMultiply);
#endif

  Py_ssize_t mp_length(PyObject *self);
  PyObject *mp_subscript(PyObject *self, PyObject *op);
  int mp_ass_subscript(PyObject *self, PyObject *ind, PyObject *op);

  inline tensor &get_tensor() { return *this; }

  inline void reset_tensor(tensor &dst) {
      init(dst.get_descriptor(), dst.get_data_handle()); }

private:
  struct WeDontManageIt {
    void operator() (const Py_buffer *view) {
      PyBuffer_Release(const_cast<Py_buffer *>(view));
      delete view;
    }
  };

  std::unique_ptr<const Py_buffer, WeDontManageIt> view_;

protected:
  reorderer *sync_reorder_;
};
}

class mdarray : public py_handle {
public:
  using tensor = ideep::tensor;
  using data_type_t = mkldnn::memory::data_type;

  mdarray() {};

  mdarray(tensor &tensor) :
      py_handle(std::make_shared<implementation::mdarray>(tensor)) {}

  mdarray(mkldnn::memory::dims &dims, mkldnn::memory::data_type dt) :
      py_handle(std::make_shared<implementation::mdarray>(dims, dt)) {}

  mdarray(Py_buffer *view, char input_type='d') :
      py_handle(std::make_shared<implementation::mdarray>(view, input_type)) {}

  static PyObject *mdarray_shape_get(mdarray *self) {
    implementation::mdarray *m = self->get();
    auto dims = m->get_dims();
    auto ndims = m->ndims();
    PyObject *intTuple = PyTuple_New(ndims);

    if (!intTuple)
      goto fail;

    for (int i = 0; i < ndims; i++) {
      PyObject *o = PyLong_FromLong(dims[i]);

      if (!o) {
        Py_DECREF(intTuple);
        intTuple = NULL;
        goto fail;
      }

      PyTuple_SET_ITEM(intTuple, i, o);
    }

    fail:
      return intTuple;
  }

  static PyObject *mdarray_dtype_get(mdarray *self) {
    implementation::mdarray *m = self->get();
    PyArray_Descr *pd;

    // Translate our data_type to numpy one
    switch (m->get_data_type()) {
      case data_type_t::f32:
        pd = PyArray_DescrFromType(NPY_FLOAT);
        break;
      case data_type_t::s32:
        pd= PyArray_DescrFromType(NPY_INT);
        break;
      case data_type_t::s16:
        pd= PyArray_DescrFromType(NPY_INT16);
        break;
      case data_type_t::s8:
        pd= PyArray_DescrFromType(NPY_INT8);
        break;
      case data_type_t::u8:
        pd= PyArray_DescrFromType(NPY_UINT8);
        break;
      default:
        PyErr_SetString(PyExc_ValueError, "Bad mdarray data_type");
        return nullptr;
    }

    return reinterpret_cast<PyObject *>(pd);
  }

  static long mdarray_size_get(mdarray *self) {
    return self->get()->get_size();
  }

  static long mdarray_ndim_get(mdarray *self) {
    return self->get()->ndims();
  }

  static bool mdarray_is_mdarray_get(mdarray *self) {
    return true;
  }
};

// using reorder_buffer = implementation::mdarray::Reorder_buffer;

#endif // _MDARRAY_H_
