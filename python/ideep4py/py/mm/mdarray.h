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
#include <mkldnn.hpp>
#include <type_traits>
#include <swigpyrun.h>
#include "mem.h"
#include "tensor.h"
#include "reorder.h"

// FIXME
// use global engine to init mdarray
using namespace mkldnn;
extern engine cpu_engine;

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
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o) {    \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate, o); \
    Py_DECREF(surrogate);   \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
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


//class mdarray : public Tensor {
class mdarray {
public:
  // It is exposed to python
  //
  static constexpr int MAX_NDIM = 12; //XXX: For now

  class Reorder_buffer : Reorderer {
  public:
    Reorder_buffer(const py_handle in)
        :Reorderer(in.get()->tensor()) {}
  };

public:
  typedef size_t size_type;
  // Generated on demand
  //FIXME 
  //yli135: add default constructor so that we can pass vector<mdarray> form native
  mdarray();
  virtual ~mdarray() = default;

  mdarray(Tensor *tensor) : tensor_(tensor) {}

  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , const mkldnn::engine &engine)
    : tensor_(new Tensor(dims, dt, format, engine)) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : tensor_(new Tensor(pd)) {}

#if 0
  mdarray(int ndims, vector<int> dims, void *data,
          mkldnn_memory_format_t mm_fmt, data_type_t type=FLOAT32)
    : tensor_(new Tensor(ndims, dims, data, mm_fmt, type)) {}
#endif

  mdarray(Py_buffer *view, char input_type='d') {// input_type : 'd'-->data, 'w'-->weight
    data_type_t dt;
    std::string format(view->format);
    if (std::string::npos != format.find_last_of('f')) {
      dt = FLOAT32;
    } else if (std::string::npos != format.find_last_of('i')) {
      dt = SINT32;
    } else if (std::string::npos != format.find_last_of('h')) {
      dt = SINT16;
    } else if (std::string::npos != format.find_last_of('b')) {
      dt = SINT8;
    } else if (std::string::npos != format.find_last_of('B')) {
      dt = UINT8;
    } else {
      throw mkldnn::error(mkldnn_invalid_arguments
          , std::string("MKLDNN does not support data type: ")
          + format);
    }
    vector<int> dims(view->shape, view->shape + view->ndim);
    //std::unique_ptr<Tensor> tensor(new Tensor(view->ndim, dims, view->buf, dt)); 
    tensor_.reset(new Tensor(view->ndim, dims, view->buf, dt, input_type)); 

    PyBuffer_Release(view);

#if 0
    ndims_ = view->ndim;
    dims_.assign(view->shape, view->shape + view->ndim);
    size_ = view->len / view->itemsize;
    type_ = dt;
    data_ = std::shared_ptr<avx::byte>(new avx::byte [view->len]
                    , [] (avx::byte *p) {delete [] p;});
    memcpy(data_.get(), view->buf, view->len);
    mm_fmt_ = ndims2format(ndims_);
    memory::data_type type = to_mkldnn_type();
    mem_.reset(new mkldnn::memory(
                { { { dims_ }, type, static_cast<memory::format>(mm_fmt_) }
                , cpu_engine }, data_.get()));
#endif
  }

  static bool is_mdarray(PyObject *o);
  
  //FIXME
  inline void unpickled_data(void *pdata) {
    //data_.reset(reinterpret_cast<avx::byte *>(pdata));
    //m_.set_data_handle(pdata);
    return;
  }

  // PEP 3118 interface
  int build_view(Py_buffer *view, int flags, const Reorderer &reorder) {
      view->buf = reorder.data_.get();
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

#if 0
  // Array protocol
  PyArrayInterface *build_array_struct(void) {
      auto arrstr = new PyArrayInterface();

      arrstr->two = 2;
      arrstr->nd = ndims_;
      arrstr->typekind = *((char *)format_);
      arrstr->itemsize = itemsize_;
      arrstr->flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_NOTSWAPPED |
          NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
      arrstr->flags &= ~(NPY_ARRAY_UPDATEIFCOPY | NPY_ARRAY_OWNDATA);
      arrstr->shape = shape_;
      arrstr->strides = strides_;
      arrstr->data = data_.get();
      arrstr->descr = nullptr;

      return arrstr;
  }
#endif

  PyObject *__getstate__(void) const;

  void __setstate__(PyObject *state);

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

  PyObject *reshape(py_handle *self, vector<int> dims);

  PyObject *m_mult_div(PyObject *self, PyObject *o, int mult_or_div, bool inplace);

  PyObject *sum(std::vector<int> axis, bool keepdims);

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

  inline Tensor* tensor() {
      return tensor_.get();
  }
  inline Tensor &tensor2() {
      return *(tensor_.get());
  }
  inline int ndims() const {
      return tensor_->ndims();
  }
  inline memory::desc desc() const {
      return tensor_->desc();
  }
  inline size_type size() const {
      return tensor_->size();
  }
  inline void *data() const {
      return tensor_->data();
  }
  inline mkldnn::engine get_engine() const {
      return tensor_->get_engine();
  }
  inline mkldnn::memory mkldnn_memory() const {
      return tensor_->mkldnn_memory();
  }
private:
  struct WeDontManageIt {
    void operator() (const Py_buffer *view) {
      PyBuffer_Release(const_cast<Py_buffer *>(view));
      delete view;
    }
  };

  std::unique_ptr<const Py_buffer, WeDontManageIt> view_;

protected:
  std::unique_ptr<Tensor> tensor_;
  Reorderer *sync_reorder_;

#if 0
private:
  static mkldnn::memory::desc _d_from_view(const Py_buffer *view
      , mkldnn::memory::format order) {
    mkldnn::memory::dims dims (view->ndim);

    for( int i=0; i < view->ndim; i++)
      dims[i] = view->shape[i];

    std::string format(view->format);
    mkldnn::memory::data_type dt;

    if (view->itemsize == 4) {
      if (std::string::npos != format.find_last_of('f')) {
        dt = mkldnn::memory::f32;
      } else if (std::string::npos != format.find_last_of('i')) {
        dt = mkldnn::memory::s32;
      } else
        throw mkldnn::error(mkldnn_invalid_arguments
            , std::string("MKLDNN does not support data type: ")
            + format);
    } else
      throw mkldnn::error(mkldnn_invalid_arguments
          , "MKLDNN does not support itemsize other than 4");

    return mkldnn::memory::desc(dims, dt, order);
  }
#endif
};

}

//
// Actual interface for python
// DO NOT add field
//
class mdarray : public py_handle {
public:
  //FIXME 
  //yli135: add default constructor so that we can pass vector<mdarray> form native
  mdarray() {};

  mdarray(Tensor *tensor)
    : py_handle(std::make_shared<implementation::mdarray>(tensor)) {}

  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &engine)
    : py_handle(std::make_shared<implementation::mdarray>
        (dims, dt, format, engine)) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : py_handle(std::make_shared<implementation::mdarray>(pd)) {}

  mdarray(Py_buffer *view, char input_type='d')
    : py_handle(std::make_shared<implementation::mdarray>(view, input_type)) {}

#if 0
  mdarray(int ndims, vector<int> dims, void *data,
          mkldnn_memory_format_t mm_fmt, data_type_t type=FLOAT32)
    : py_handle(std::make_shared<implementation::mdarray>(ndims, dims, data, mm_fmt, type)) {}
#endif

  static PyObject *mdarray_shape_get(mdarray *arg) {
    implementation::mdarray *self = arg->get();
    int ndim = self->ndims();
    PyObject *intTuple = PyTuple_New(ndim);
    auto data = self->desc().data;

    if (!intTuple)
      goto fail;

    for (int i = 0; i<ndim; i++) {
      PyObject *o = PyLong_FromLong(data.dims[i]);

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
    switch (static_cast<mkldnn::memory::data_type>(m->desc().data.data_type)) {
      case mkldnn::memory::f32:
        pd = PyArray_DescrFromType(NPY_FLOAT);
        break;
      case mkldnn::memory::s32:
        pd= PyArray_DescrFromType(NPY_INT);
        break;
      case mkldnn::memory::s16:
        pd= PyArray_DescrFromType(NPY_INT16);
        break;
      case mkldnn::memory::s8:
        pd= PyArray_DescrFromType(NPY_INT8);
        break;
      case mkldnn::memory::u8:
        pd= PyArray_DescrFromType(NPY_UINT8);
        break;
      default:
        PyErr_SetString(PyExc_ValueError, "Bad mdarray data_type");
        return nullptr;
    }

    return reinterpret_cast<PyObject *>(pd);
  }

  static long mdarray_size_get(mdarray *self) {
    return self->get()->size();
  }

  static long mdarray_ndim_get(mdarray *self) {
    return self->get()->desc().data.ndims;
  }

  static bool mdarray_is_mdarray_get(mdarray *self) {
    return true;
  }
};

using reorder_buffer = implementation::mdarray::Reorder_buffer;

#endif // _MDARRAY_H_
