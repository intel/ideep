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


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#if defined(OPENMP_AFFINITY)
#include "cpu_info.h"
#endif
#include "mdarray.h"
#include <mkl_vml_functions.h>
#include "mkldnn_ex.h"
// #include "dlcp_py.h"

namespace implementation {

static PyObject *PyType_reorder_buffer = nullptr;

static swig_type_info *SwigTy_mdarray = nullptr;
//static swig_type_info *SwigTy_engine = nullptr;
static PyObject *PyType_mdarray = nullptr;

// get mdarray from PyObject
static inline mdarray *get_mdarray_from_PyObject(PyObject *self) {
    void *oprd_self;
    int res = SWIG_ConvertPtr(self, &oprd_self, nullptr, 0);
    if (!SWIG_IsOK(res)) {
        // PyErr_SetString(PyExc_ValueError, "Error self PyObject");
        return NULL;
    }
    return (reinterpret_cast<py_handle *>(oprd_self))->get();
}

//check whether mdarray support this operation
static inline bool is_mdarray_supported(PyObject *self, PyObject *o) {
    // get self mdarray
    mdarray *self_mdarray = get_mdarray_from_PyObject(self);
    if (!self_mdarray)
        return false;

    // o is ndarray
    // if size not equal, mean array broadcast
    if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type) {
        if ((size_t)PyArray_SIZE(reinterpret_cast<PyArrayObject *>(o))
                != self_mdarray->size() ||
                !PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o))) {
            return false;
        }
        return true;
    }

    // o is mdarray
    if (reinterpret_cast<PyTypeObject *>(o->ob_type)
            == reinterpret_cast<PyTypeObject *>(PyType_mdarray)) {
        // if o is mdarray, try to get mdarray
        mdarray *o_mdarray = get_mdarray_from_PyObject(o);
        if (!o_mdarray)
            return false;

        // not support different size's mdarray's operations
        if (o_mdarray->size() != self_mdarray->size())
            return false;

        return true;
    }

    return false;
}

PyObject *queryPyTypeObject(const char *name) {
  swig_type_info *info = SWIG_TypeQuery(name);
  if (info != nullptr) {
    SwigPyClientData *cd
      = (SwigPyClientData *)info->clientdata;
    return reinterpret_cast<PyObject *>(cd->pytype);
  }

  throw mkldnn::error(mkldnn_invalid_arguments
      , "Failed to find reorderer object");
}

// We brought this to global scope to mitigate it consumption
#if PY_VERSION_HEX >= 0x03000000
int g_init() {
#else
void g_init() {
#endif
  PyType_reorder_buffer = queryPyTypeObject("_p_reorder_buffer");
  SwigTy_mdarray = SWIG_TypeQuery("_p_mdarray");
  PyType_mdarray = queryPyTypeObject("_p_mdarray");
  //SwigTy_engine = SWIG_TypeQuery("_p_mkldnn__engine");

#if PY_VERSION_HEX < 0x03000000
  if ((reinterpret_cast<PyTypeObject *>(PyType_mdarray)->tp_flags
    & Py_TPFLAGS_HAVE_NEWBUFFER) != Py_TPFLAGS_HAVE_NEWBUFFER)
    throw mkldnn::error(mkldnn_invalid_arguments
    , "Python2 should have new buffer flag on!");
#endif

  // XXX: I don't quite understand it, and its repercussions :)
  SwigPyObject_stype = SWIG_MangledTypeQuery("_p_SwigPyObject");

  if (SwigPyObject_stype == nullptr)
    throw mkldnn::error(mkldnn_invalid_arguments
        , "Failed to find SwigPyObject object");

  // Initiate static variables imported from numpy include
  import_array();

#if defined(OPENMP_AFFINITY)
  // google::SetStderrLogging(1);
  // google::InitGoogleLogging("mkldnn");
  OpenMpManager::bindOpenMpThreads();
  OpenMpManager::printVerboseInformation();
#endif

  // dlCompression::init();

#if PY_VERSION_HEX >= 0x03000000
  return 0;
#else
  return;
#endif
}

//FIXME: macro SWIG_as_voidptr is copied from mdarray_wrap.cpp
#define SWIG_as_voidptr(a) const_cast< void * >(static_cast< const void * >(a))

// Pickle
PyObject *mdarray::__getstate__() const {
  auto md = desc();
  void *raw_data = data();
  int ndims = md.data.ndims;
  mkldnn::memory::dims dims;
  mkldnn::memory::data_type dtype = static_cast<mkldnn::memory::data_type>(md.data.data_type);
  mkldnn::memory::format format = static_cast<mkldnn::memory::format>(md.data.format);
  static mkldnn::engine engine = get_engine();

  PyObject *py_dims = PyTuple_New(ndims);
  for (int i = 0; i < ndims; i++) {
    PyObject *py_dim = PyLong_FromLong(md.data.dims[i]);
    PyTuple_SetItem(py_dims, i, py_dim);
  }

  PyObject *py_dtype = PyLong_FromLong((long)dtype);
  PyObject *py_format = PyLong_FromLong((long)format);
  PyObject *py_engine = PyLong_FromVoidPtr((void *)&engine);
  PyObject *py_rdata = PyLong_FromVoidPtr((void *)raw_data);

  PyObject *state = PyTuple_New(5);
  PyTuple_SetItem(state, 0, py_dims);
  PyTuple_SetItem(state, 1, py_dtype);
  PyTuple_SetItem(state, 2, py_format);
  PyTuple_SetItem(state, 3, py_engine);
  PyTuple_SetItem(state, 4, py_rdata);

  return state;
}

// Unpickle.
void mdarray::__setstate__(PyObject *state) {
  return;
}

PyObject *mdarray::py_mdarray_from(PyObject *o) const {
  PyObject *argList = Py_BuildValue("(O)", o);

  if (argList == nullptr) {
    PyErr_SetString(PyExc_SystemError, "Can not create argument list");
    return nullptr;
  }

  o = PyObject_CallObject(PyType_mdarray, argList);

  Py_DECREF(argList);

  if (o == nullptr) {
    PyErr_SetString(PyExc_BufferError, "Cannot create mdarray from input");
    return nullptr;
  }

  return o;
}

template<class T>
void mdarray::axpby(mdarray *dst, T a, mdarray *x, T b, mdarray *y) {
    ::axpby(dst->tensor(), a, x->tensor(), b, y->tensor());
}

template<class T>
PyObject *mdarray::axpby(T a, T b, PyObject *o) {
  /// Resource manager, for GCC do not accept lambda
  struct py_decref {
    void operator () (PyObject *p) {
      Py_DECREF(p);
    }
  };

  std::unique_ptr<PyObject, py_decref> op(nullptr);

  /// Create mdarray from buffer provider
  if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type) {
    o = py_mdarray_from(o);
    op.reset(o);
  }

  void *oprd2;
  int res = SWIG_ConvertPtr(o, &oprd2, nullptr, 0);

  if (!SWIG_IsOK(res)) {
    PyErr_SetString(PyExc_ValueError, "Wrong operand object in add wrapper");
    return nullptr;
  }

  auto x = (reinterpret_cast<py_handle *>(oprd2))->get();
  py_handle *output = new py_handle(new mdarray(x->mkldnn_memory().get_primitive_desc()));

  /// Switch position for format consistency
  axpby(output->get(), b, x, a, this);

  PyObject *resultobj = SWIG_Python_NewPointerObj(nullptr
      , SWIG_as_voidptr(output), SwigTy_mdarray, SWIG_POINTER_OWN |  0 );

  return resultobj;
}

template<class T>
PyObject *mdarray::inplace_axpby(T a, PyObject *self, T b, PyObject *o) {
  // Resource manager, for GCC do not accept lambda
  struct py_decref {
    void operator () (PyObject *p) {
      Py_DECREF(p);
    }
  };

  std::unique_ptr<PyObject, py_decref> op(nullptr);

  // Create mdarray from buffer provider
  if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type) {
    o = py_mdarray_from(o);
    op.reset(o);
  }

  void *oprd2;
  int res = SWIG_ConvertPtr(o, &oprd2, nullptr, 0);

  if (!SWIG_IsOK(res)) {
    PyErr_SetString(PyExc_ValueError, "Wrong operand object in add wrapper");
    return nullptr;
  }

  auto y = (reinterpret_cast<py_handle *>(oprd2))->get();
  axpby(this, a, this, b, y);
  Py_INCREF(self);

  return self;
}

PyObject *mdarray::m_Add(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (!is_mdarray_supported(self, o)) {
    return m_Add_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    // Make compatibility with Non-C-Contiguous array.
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_Add_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return axpby(1.0f, 1.0f, o);
  }
}

PyObject *mdarray::m_Subtract(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (!is_mdarray_supported(self, o)) {
    return m_Subtract_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_Subtract_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return axpby(1.0f, -1.0f, o);
  }
}

PyObject *mdarray::m_InPlaceAdd(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (!is_mdarray_supported(self, o)) {
    return m_InPlaceAdd_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_InPlaceAdd_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return inplace_axpby(1.0f, self, 1.0f, o);
  }
}

PyObject *mdarray::m_InPlaceSubtract(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (!is_mdarray_supported(self, o)) {
    return m_InPlaceSubtract_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_InPlaceSubtract_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return inplace_axpby(1.0f, self, -1.0f, o);
  }
}

template <typename T>
void plain_mult(const T *a, const T *b, T *o, int size) {
  for (int idx = 0; idx < size; idx++)
    o[idx] = a[idx] * b[idx];
}

template <typename T>
void plain_div(const T *a, const T *b, T *o, int size) {
  for (int idx = 0; idx < size; idx++)
    o[idx] = a[idx] / b[idx];
}

enum {mmult, mdiv};
PyObject *mdarray::m_mult_div(PyObject *self, PyObject *o, int mult_or_div, bool inplace) {
  struct py_decref {
    void operator () (PyObject *p) {
      Py_DECREF(p);
    }
  };

  std::unique_ptr<PyObject, py_decref> op(nullptr);

  enum mult_type_t { MULT_UNKNOWN, MULT_ELTWISE, MULT_SCALAR };

  PyTypeObject *oprd2_type = reinterpret_cast<PyTypeObject *>(o->ob_type);
  int mult_type = static_cast<int>(MULT_UNKNOWN);
  if (oprd2_type == &PyArray_Type) {
    mult_type = MULT_ELTWISE;
    o = py_mdarray_from(o);
    op.reset(o);
  } else if (PyObject_HasAttrString(o, "is_mdarray")) {
    mult_type = MULT_ELTWISE;
  } else if (PyFloat_Check(o) || PyInt_Check(o) || PyNumber_Check(o)) {
    mult_type = MULT_SCALAR;
  }

  PyObject *resultobj = nullptr;

  switch (static_cast<enum mult_type_t>(mult_type)) {
  case MULT_ELTWISE: {
    void *oprd2;
    int res = SWIG_ConvertPtr(o, &oprd2, nullptr, 0);
    if (!SWIG_IsOK(res)) {
      PyErr_SetString(PyExc_ValueError, "Error oprd2 %matrix element multiply");
      break;
    }

    auto oprd1_mdarr = this;
    auto oprd2_mdarr = (reinterpret_cast<py_handle *>(oprd2))->get();

    if (oprd1_mdarr->size() != oprd2_mdarr->size()) {
      PyErr_SetString(PyExc_SystemError, "Abnormal matrix size %matrix element multiply");
      break;
    }

    std::vector<mkldnn::primitive> prims;
    std::unique_ptr<mkldnn::memory> mreorder;

    auto oprd2_internal_m = reorder_if_must(oprd2_mdarr->mkldnn_memory(),
                               oprd1_mdarr->mkldnn_memory().get_primitive_desc(),
                               mreorder,
                               &prims);
    mkldnn::stream s(mkldnn::stream::kind::eager);
    s.submit(prims).wait();

    mkldnn::memory::desc res_desc = oprd1_mdarr->desc();
    mkldnn::memory::dims res_tz;
    mkldnn::memory::data_type res_dtype =
          static_cast<mkldnn::memory::data_type>(res_desc.data.data_type);
    mkldnn::memory::format res_fmt =
          static_cast<mkldnn::memory::format>(res_desc.data.format);
    mkldnn::engine res_engine = oprd1_mdarr->get_engine();

    assert(oprd1_mdarr->ndims() == 2 || oprd1_mdarr->ndims() == 4);
    for (int ndim = 0; ndim < static_cast<int>(oprd1_mdarr->ndims()); ndim++)
      res_tz.push_back(res_desc.data.dims[ndim]);

    mdarray *res_mdarr;
    if (!inplace) {
      res_mdarr = new mdarray(res_tz, res_dtype, res_fmt, res_engine);
    } else {
      res_mdarr = oprd1_mdarr;
    }

    assert(mkldnn::memory::f32 == res_dtype ||
           mkldnn::memory::s32 == res_dtype ||
           mkldnn::memory::s16 == res_dtype ||
           mkldnn::memory::s8 == res_dtype ||
           mkldnn::memory::u8 == res_dtype );
    assert(mmult == mult_or_div ||
           mdiv == mult_or_div);
    if (mkldnn::memory::f32 == res_dtype) {
      switch (mult_or_div) {
      case mmult:
        vsMul(oprd1_mdarr->size(),
              reinterpret_cast<const float *>(oprd1_mdarr->data()),
              reinterpret_cast<const float *>(oprd2_internal_m.get_data_handle()),
              reinterpret_cast<float *>(res_mdarr->data()));
        break;

      case mdiv:
        plain_div(reinterpret_cast<const float *>(oprd1_mdarr->data()),
                  reinterpret_cast<const float *>(oprd2_internal_m.get_data_handle()),
                  reinterpret_cast<float *>(res_mdarr->data()),
                  static_cast<int>(oprd1_mdarr->size()));
        break;
      }
    } else if (mkldnn::memory::s32 == res_dtype) {
      switch (mult_or_div) {
      case mmult:
        plain_mult(reinterpret_cast<const int *>(oprd1_mdarr->data()),
                   reinterpret_cast<const int *>(oprd2_internal_m.get_data_handle()),
                   reinterpret_cast<int *>(res_mdarr->data()),
                   static_cast<int>(oprd1_mdarr->size()));
        break;

      case mdiv:
        plain_div(reinterpret_cast<const int *>(oprd1_mdarr->data()),
                  reinterpret_cast<const int *>(oprd2_internal_m.get_data_handle()),
                  reinterpret_cast<int *>(res_mdarr->data()),
                  static_cast<int>(oprd1_mdarr->size()));
        break;
      }
    } else if (mkldnn::memory::s16 == res_dtype) {
      switch (mult_or_div) {
      case mmult:
        plain_mult(reinterpret_cast<const int16_t *>(oprd1_mdarr->data()),
                   reinterpret_cast<const int16_t *>(oprd2_internal_m.get_data_handle()),
                   reinterpret_cast<int16_t *>(res_mdarr->data()),
                   static_cast<int>(oprd1_mdarr->size()));
        break;

      case mdiv:
        plain_div(reinterpret_cast<const int16_t *>(oprd1_mdarr->data()),
                  reinterpret_cast<const int16_t *>(oprd2_internal_m.get_data_handle()),
                  reinterpret_cast<int16_t *>(res_mdarr->data()),
                  static_cast<int>(oprd1_mdarr->size()));
        break;
      }
    } else if (mkldnn::memory::s8 == res_dtype) {
      switch (mult_or_div) {
      case mmult:
        plain_mult(reinterpret_cast<const int8_t *>(oprd1_mdarr->data()),
                   reinterpret_cast<const int8_t *>(oprd2_internal_m.get_data_handle()),
                   reinterpret_cast<int8_t *>(res_mdarr->data()),
                   static_cast<int>(oprd1_mdarr->size()));
        break;

      case mdiv:
        plain_div(reinterpret_cast<const int8_t *>(oprd1_mdarr->data()),
                  reinterpret_cast<const int8_t *>(oprd2_internal_m.get_data_handle()),
                  reinterpret_cast<int8_t *>(res_mdarr->data()),
                  static_cast<int>(oprd1_mdarr->size()));
        break;
      }
    } else if (mkldnn::memory::u8 == res_dtype) {
      switch (mult_or_div) {
      case mmult:
        plain_mult(reinterpret_cast<const uint8_t *>(oprd1_mdarr->data()),
                   reinterpret_cast<const uint8_t *>(oprd2_internal_m.get_data_handle()),
                   reinterpret_cast<uint8_t *>(res_mdarr->data()),
                   static_cast<int>(oprd1_mdarr->size()));
        break;

      case mdiv:
        plain_div(reinterpret_cast<const uint8_t *>(oprd1_mdarr->data()),
                  reinterpret_cast<const uint8_t *>(oprd2_internal_m.get_data_handle()),
                  reinterpret_cast<uint8_t *>(res_mdarr->data()),
                  static_cast<int>(oprd1_mdarr->size()));
        break;
      }
    }

    if (!inplace) {
      auto res_py_handle = new py_handle(res_mdarr);
      resultobj = SWIG_Python_NewPointerObj(nullptr,
                       SWIG_as_voidptr(res_py_handle),
                       SwigTy_mdarray,
                       SWIG_POINTER_OWN | 0);
    } else {
      resultobj = self;
      Py_INCREF(self);
    }

    break;
  }

  case MULT_SCALAR: {
    double a = PyInt_Check(o) ?
               static_cast<double>(PyInt_AsLong(o)) :
               PyFloat_AsDouble(o),
           b = 0.0;

    a = (mmult == mult_or_div) ? a : (1 / a);

    if (!inplace) {
      resultobj = axpby(a, b, self);
    } else {
      resultobj = inplace_axpby(a, self, b, self);;
    }
    break;
  }

  case MULT_UNKNOWN:
  default:
    PyErr_SetString(PyExc_SystemError, "Abnormal type % matrix * scalar");
    break;
  }

  return resultobj;
}

PyObject *mdarray::m_Multiply(PyObject *self, PyObject *o) {
  if (!is_mdarray_supported(self, o)) {
    return m_Multiply_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_Multiply_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return m_mult_div(self, o, mmult, false);
  }
}

PyObject *mdarray::m_InPlaceMultiply(PyObject *self, PyObject *o) {
  if (!is_mdarray_supported(self, o)) {
    return m_InPlaceMultiply_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_InPlaceMultiply_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return m_mult_div(self, o, mmult, true);
  }
}

PyObject *mdarray::m_Divide(PyObject *self, PyObject *o) {
  if (!is_mdarray_supported(self, o)) {
    return m_Divide_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_Divide_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return m_mult_div(self, o, mdiv, false);
  }
}

PyObject *mdarray::m_InPlaceDivide(PyObject *self, PyObject *o) {
  if (!is_mdarray_supported(self, o)) {
    return m_InPlaceDivide_map_impl(self, o);
  } else if (PyArray_Check(o) &&
      !PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject *>(o))) {
    PyObject *_o = o;
#if PY_VERSION_HEX < 0x03000000
    _o = reinterpret_cast<PyObject *>(PyArray_ContiguousFromAny(
      o, PyArray_ISFLOAT(reinterpret_cast<PyArrayObject *>(o)) ? NPY_FLOAT : NPY_INT, 0, 0));
#endif
    PyObject *ret = m_InPlaceDivide_map_impl(self, _o);
#if PY_VERSION_HEX < 0x03000000
    Py_DECREF(_o);
#endif
    return ret;
  } else {
    return m_mult_div(self, o, mdiv, true);
  }
}

int mdarray::getbuffer(PyObject *self, Py_buffer *view, int flags) {
  if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
    PyErr_SetString(PyExc_ValueError, "carray is not Fortran contiguous");
    return -1;
  }

  if (view == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  // reorderer type object
  if (PyType_reorder_buffer == nullptr) {
    PyErr_SetString(PyExc_NameError, "name 'reorderer' is not defined");
    return -1;
  }

  // Wrote some python in C++ :)
  PyObject *argList = Py_BuildValue("(O)", self);
  if (argList == nullptr) {
    return -1;
  }

  // TODO: Do we need to cache this thing?
  PyObject *rbobj = PyObject_CallObject(PyType_reorder_buffer, argList);
  Py_DECREF(argList);

  if (rbobj == nullptr) {
    return -1;
  }

  Reorderer *rb;
  int res = SWIG_ConvertPtr(rbobj, reinterpret_cast<void **>(&rb), nullptr, 0);

  if (!SWIG_IsOK(res)) {
    PyErr_SetString(PyExc_RuntimeError, "Can't get C++ object from python object");
    return -1;
  }

  if (rb->non_trivial())
    rb->fire(this->tensor());

  if (build_view(view, flags, *rb)) {
    PyErr_SetString(PyExc_RuntimeError, "Can't build Py_buffer!");
    return -1;
  }

  // Stolen reference
  view->obj = rbobj;
  sync_reorder_ = rb;

  // reset self mdarray's tensor, keep buffer consistency.
  if (rb->non_trivial()) {
    mdarray *src_mdarray = get_mdarray_from_PyObject(self);
    if (!src_mdarray) {
      PyErr_SetString(PyExc_RuntimeError, "Can't get src mdarray from python object!");
      return -1;
    }

    Tensor *src_tensor = src_mdarray->tensor();
    mkldnn::memory::dims src_dims = (mkldnn::memory::dims)src_tensor->dims();
    mkldnn_memory_format_t dst_fmt = public_format(src_tensor->format());

    Tensor *dst_tensor = new Tensor(src_dims.size(), src_dims, rb->data_,
                                    dst_fmt, src_tensor->type());
    src_mdarray->reset_tensor(dst_tensor);
  }
  return 0;
}

PyObject *mdarray::getattro(PyObject *self, PyObject *name) {
  // XXX: Recursive alarm !!! XXX
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  if (surrogate == nullptr)
    return nullptr;

  // Watch the reference count of surrogate if more compicated
  // looking up method involved
  PyObject * attr = PyObject_GetAttr(surrogate, name);

  // The surrogate will be destroyed after attribute is done
  Py_DECREF(surrogate);

  if (attr == nullptr && PyErr_ExceptionMatches(PyExc_AttributeError)) {
    PyErr_Clear();

    // Switch to our exception message if things gone wrong
    PyTypeObject *tp = Py_TYPE(self);
    PyErr_Format(PyExc_AttributeError
        , "mdarray '%.50s' object has no attribute '%p'", tp->tp_name, name);
  }

  return attr;
}

Py_ssize_t mdarray::mp_length(PyObject *self) {
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  if (surrogate == nullptr)
    return -1;

  Py_ssize_t len = PyMapping_Length(surrogate);
  Py_DECREF(surrogate);

  // TODO: Exception localize
  return len;
}

PyObject *mdarray::mp_subscript(PyObject *self, PyObject *op) {
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  if (surrogate == nullptr)
    return nullptr;

  PyObject *ret = PyObject_GetItem(surrogate, op);
  Py_DECREF(surrogate);

  // TODO: Exception localize
  return ret;
}

int mdarray::mp_ass_subscript(PyObject *self, PyObject *ind, PyObject *op) {
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  int ret;

  if (surrogate == nullptr)
    return -1;

  if (op == nullptr)
    ret = PyObject_DelItem(surrogate, ind);
  else
    ret = PyObject_SetItem(surrogate, ind, op);

  if (sync_reorder_ && sync_reorder_->non_trivial()) {
    sync_reorder_->sync(this->tensor());
  }

  Py_DECREF(surrogate);

  // TODO: Exception localize
  return ret;
}

PyObject *mdarray::flat() {
  long int dims[1] = {static_cast<long int>(this->size())};

  int typenum = NPY_NOTYPE;
  switch(static_cast<mkldnn::memory::data_type>(this->mkldnn_memory().get_primitive_desc().desc().data.data_type)) {
    case mkldnn::memory::f32:
      typenum = NPY_FLOAT32;
      break;
    case mkldnn::memory::s32:
      typenum = NPY_INT;
      break;
    case mkldnn::memory::s16:
      typenum = NPY_INT16;
      break;
    case mkldnn::memory::s8:
      typenum = NPY_INT8;
      break;
    case mkldnn::memory::u8:
      typenum = NPY_UINT8;
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "Bad mdarray data_type");
      break;
  }

  PyObject *plain_arr = nullptr;
  plain_arr = PyArray_SimpleNewFromData(1, dims, typenum, this->data());
  if (!plain_arr)
    PyErr_SetString(PyExc_ValueError, "Can't create plain array with format from mdarray");

  return plain_arr;
}

PyObject *mdarray::reshape(py_handle *self, vector<int> dims)
{
    if (dims.size() != 4 && dims.size() != 2) {
        PyErr_SetString(PyExc_ValueError,"Only support reshape to 2 dimension");
        return nullptr;
    }
    int idx_unknown = -1;
    size_t size = 1;
    for (unsigned int i = 0; i < dims.size(); i++) {
        if (dims[i] < 0) {
            if (idx_unknown == -1) {
                idx_unknown = i;
            } else {
                PyErr_SetString(PyExc_ValueError,"Only support 1 unkown dimension");
                return nullptr;
            }
        } else {
            size *= dims[i];
        }
    }
    if (idx_unknown == -1) {
        if (size != this->size()) {
            PyErr_SetString(PyExc_ValueError,"Wrong dimension to reshape");
            return nullptr;
        }
    } else if (this->size() % size) {
        PyErr_SetString(PyExc_ValueError,"Wrong dimension to reshape");
        return nullptr;
    } else {
        dims[idx_unknown] = this->size() / size;
    }
    Tensor *tensor = tensor_->reshape(dims);
    if (tensor == nullptr) {
        PyErr_SetString(PyExc_ValueError,"The dimension is not valid in reshape");
        return nullptr;
    } else {
        //mdarray *new_array = new ::mdarray(tensor);
        py_handle *output = new py_handle(new mdarray(tensor));
        PyObject *resultobj = SWIG_Python_NewPointerObj(nullptr
                , SWIG_as_voidptr(output), SwigTy_mdarray, SWIG_POINTER_OWN |  0 );
        return resultobj;
    }
}

PyObject *mdarray::sum(vector<int> axis, bool keepdims)
{
    auto tensor = tensor_->sum(axis);
    if (tensor) {
        if (keepdims) {
            vector<int> expected_shape;
            for (int v = 0; v < this->ndims(); v++)
                expected_shape.push_back(this->desc().data.dims[v]);

            for (unsigned a = 0; a < axis.size(); a++)
                expected_shape[axis[a]] = 1;

            auto _tensor = tensor->reshape(expected_shape);
            delete tensor;
            tensor = _tensor;
        }

        auto output = new py_handle(new mdarray(tensor));
        auto resultobj = SWIG_Python_NewPointerObj(nullptr,
                             SWIG_as_voidptr(output), SwigTy_mdarray,
                             SWIG_POINTER_OWN | 0);
        return resultobj;
    } else {
        return nullptr;
    }
}

bool mdarray::is_mdarray(PyObject *o)
{
    return (reinterpret_cast<PyTypeObject *>(o->ob_type)
            == reinterpret_cast<PyTypeObject *>(PyType_mdarray));
}

}
