#include <glog/logging.h>
#if defined(OPENMP_AFFINITY)
#include "cpu_info.h"
#endif
#include "mdarray.h"
#include <mkl_vml_functions.h>

namespace implementation {

static PyObject *PyType_reorder_buffer = nullptr;

static swig_type_info *SwigTy_mdarray = nullptr;
static swig_type_info *SwigTy_engine = nullptr;
static PyObject *PyType_mdarray = nullptr;

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
  SwigTy_engine = SWIG_TypeQuery("_p_mkldnn__engine");

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
  google::SetStderrLogging(1);
  google::InitGoogleLogging("mkldnn");
  OpenMpManager::bindOpenMpThreads();
  OpenMpManager::printVerboseInformation();
#endif

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
  mkldnn::engine p_e = get_engine();

  PyObject *Py_p_engine = SWIG_Python_NewPointerObj(nullptr
      , SWIG_as_voidptr(&p_e), SwigTy_engine, 0);

  if (Py_p_engine == nullptr) {
    PyErr_SetString(PyExc_SystemError, "Can not create mkldnn cpu engine pyobject");
    return nullptr;
  }

  PyObject *argList = Py_BuildValue("(OiO)", o
      , reorderer::public_format(
          static_cast<mkldnn::memory::format>(desc().data.format)
        ), Py_p_engine);

  if (argList == nullptr) {
    PyErr_SetString(PyExc_SystemError, "Can not create argument list");
    return nullptr;
  }

  o = PyObject_CallObject(PyType_mdarray, argList);

  Py_DECREF(argList);
  Py_DECREF(Py_p_engine);

  if (o == nullptr) {
    PyErr_SetString(PyExc_BufferError, "Cannot create mdarray from input");
    return nullptr;
  }

  return o;
}

template<class T>
void mdarray::axpby(mdarray *dst, T a, mdarray *x, T b, mdarray *y) {
  std::vector<mkldnn::primitive> prims;
  std::unique_ptr<mkldnn::memory> mreorder;

  /// Reorder to x's format
  auto mid = reorder_if_must(y->m_, x->m_.get_primitive_desc()
      , mreorder, &prims);

  mkldnn::sum::primitive_desc sum_pd({a, b}
      , {x->m_.get_primitive_desc(), mid.get_primitive_desc()});

  std::vector<mkldnn::memory::primitive::at> inputs_at {x->m_, mid};

  mkldnn::sum sum_prim(sum_pd, inputs_at, dst->m_);
  prims.push_back(sum_prim);

  mkldnn::stream s(mkldnn::stream::kind::eager);
  s.submit(prims).wait();
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
  py_handle *output = new py_handle(new mdarray(x->m_.get_primitive_desc()));

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
  if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type &&
      PyArray_SIZE(reinterpret_cast<PyArrayObject *>(o)) !=
      static_cast<int>(this->size())) {
    return m_Add_map_impl(self, o);
  } else {
    return axpby(1.0, 1.0, o);
  }
}

PyObject *mdarray::m_Subtract(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type &&
      PyArray_SIZE(reinterpret_cast<PyArrayObject *>(o)) !=
      static_cast<int>(this->size())) {
    return m_Subtract_map_impl(self, o);
  } else {
    return axpby(1.0, -1.0, o);
  }
}

PyObject *mdarray::m_InPlaceAdd(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type &&
      PyArray_SIZE(reinterpret_cast<PyArrayObject *>(o)) !=
      static_cast<int>(this->size())) {
    return m_InPlaceAdd_map_impl(self, o);
  } else {
    return inplace_axpby(1.0, self, 1.0, o);
  }
}

PyObject *mdarray::m_InPlaceSubtract(PyObject *self, PyObject *o) {
  // Array Broadcast
  if (reinterpret_cast<PyTypeObject *>(o->ob_type) == &PyArray_Type &&
      PyArray_SIZE(reinterpret_cast<PyArrayObject *>(o)) !=
      static_cast<int>(this->size())) {
    return m_InPlaceSubtract_map_impl(self, o);
  } else {
    return inplace_axpby(1.0, self, -1.0, o);
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

    auto oprd2_internal_m = reorder_if_must(oprd2_mdarr->m_,
                               oprd1_mdarr->m_.get_primitive_desc(),
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
           mkldnn::memory::s32 == res_dtype);
    assert(MMULT == mult_or_div ||
           MDIV == mult_or_div);
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
  return m_mult_div(self, o, mmult, false);
}

PyObject *mdarray::m_InPlaceMultiply(PyObject *self, PyObject *o) {
  return m_mult_div(self, o, mmult, true);
}

PyObject *mdarray::m_Divide(PyObject *self, PyObject *o) {
  return m_mult_div(self, o, mdiv, false);
}

PyObject *mdarray::m_InPlaceDivide(PyObject *self, PyObject *o) {
  return m_mult_div(self, o, mdiv, true);
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

  reorderer *rb;
  int res = SWIG_ConvertPtr(rbobj, reinterpret_cast<void **>(&rb), nullptr, 0);

  if (!SWIG_IsOK(res)) {
    PyErr_SetString(PyExc_RuntimeError, "Can't get C++ object from python object");
    return -1;
  }

  if (rb->non_trivial())
    rb->fire(this);

  if (rb->build_view(view, flags)) {
    PyErr_SetString(PyExc_RuntimeError, "Can't build Py_buffer!");
    return -1;
  }

  // Stolen reference
  view->obj = rbobj;
  sync_reorder_ = rb;

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
        , "'%.50s' object has no attribute '%U'", tp->tp_name, name);
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
    sync_reorder_->sync(this);
  }

  Py_DECREF(surrogate);

  // TODO: Exception localize
  return ret;
}

PyObject *mdarray::flat() {
  long int dims[1] = {static_cast<long int>(this->size())};
  int typenum = (this->memory().get_primitive_desc().desc().data.data_type == mkldnn::memory::f32) ? NPY_FLOAT32 : NPY_INT32;

  PyObject *plain_arr = nullptr;
  plain_arr = PyArray_SimpleNewFromData(1, dims, typenum, this->data());
  if (!plain_arr)
    PyErr_SetString(PyExc_ValueError, "Can't create plain array with format from mdarray");

  return plain_arr;
}

int s_op::getbuffer(PyObject *self, Py_buffer *view, int flags) {
  if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
    PyErr_SetString(PyExc_ValueError, "carray is not Fortran contiguous");
    return -1;
  }

  if (view == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  // Only for the first, framework do it for us next time
  if (reorder_ == nullptr) {
    reorder_.reset(new reorderer(this));
  }
  if (reorder_->non_trivial() && (reorder_->is_reordered() == false)) {
    mkldnn::reorder rb_p = reorder_->fire(this);
    reorder_->set_reordered();
  }

  if ( reorder_->build_view(view, flags) ) {
    PyErr_SetString(PyExc_RuntimeError, "Can't build Py_buffer!");
    return -1;
  }

  view->obj = self;
  sync_reorder_ = reorder_.get();
  Py_INCREF(self);

  return 0;
}

}
