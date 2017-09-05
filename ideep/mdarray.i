%module (package="mkldnn") mdarray
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstring>
  #include <iostream>
  #include <vector>
  #include <numeric>
  #include <memory>
  #include <stdexcept>
  #include <mkldnn.hpp>
#define SWIG_INLINE
  #include "mdarray.h"
%}

%init %{
  import_array();
  implementation::g_init();
%}

%include exception.i
%include pep_3118.i
%include getattro.i
%include asnumber.i
%include asmap.i
%include attribute.i

%import support.i
%import memory.i

%buffer_protocol_producer(mdarray)
%buffer_protocol_typemap(Py_buffer *view)
%getattr_wrapper(mdarray)
%number_protocol(mdarray)
%map_protocol(mdarray)

%define %codegen(Class, ret_type, attrib, getter)
%{
  inline ret_type %mangle(Class) ##_## attrib ## _get(Class *self_) {
    return (ret_type) Class::getter(self_);
  }
%}
%enddef

%define %extend_ro_attr(Class, ret_type, attrib, getter)
  %immutable Class::attrib;
  %extend Class {
    ret_type attrib;
  }
  %codegen(Class, ret_type, attrib, getter)
%enddef

%define %extend_ro_attr_and_own(Class, ret_type, attrib, getter)
  %immutable Class::attrib;
  %newobject Class::attrib;

  %extend Class {
    ret_type attrib;
  }

  %codegen(Class, ret_type *, attrib, getter)
%enddef

%extend_ro_attr(mdarray, PyObject *, dtype, mdarray_dtype_get)
%extend_ro_attr(mdarray, PyObject *, shape, mdarray_shape_get)
%extend_ro_attr(mdarray, long, size, mdarray_size_get)
%extend_ro_attr(mdarray, long, ndim, mdarray_ndim_get)
%extend_ro_attr_and_own(mdarray, mkldnn::memory, memory, mdarray_memory_get)
%extend_ro_attr(mdarray, bool, is_mdarray, mdarray_is_mdarray_get)

%extend mdarray {
  int setbuffer(Py_buffer *view) {
    return (*$self)->setbuffer(view);
  }

  void reset_buf_order() {
    (*$self)->reset_buf_order();
  }

  PyObject *axpby(double a, double b, PyObject *y) {
    return (*$self)->axpby(a, b, y);
  }

  PyObject *inplace_axpby(double a, double b, PyObject *y) {
    /// Second param y is a harmless dummy
    return (*$self)->inplace_axpby(a, y, b, y);
  }

  PyObject *flat() {
    return (*self)->flat();
  }
}

%extend mdarray {
  PyObject *__getstate__() {
    return (*$self)->__getstate__();
  }

  %typemap(default) (PyObject *state) {
    PyObject *state;

    if (!PyArg_UnpackTuple(args, (char *)"mdarray___setstate__", 0, 1, &state)) SWIG_fail;

    if (!PyTuple_Check(state)) SWIG_fail;

    PyObject *py_dims = PyTuple_GetItem(state, 0);
    PyObject *py_dtype = PyTuple_GetItem(state, 1);
    PyObject *py_format = PyTuple_GetItem(state, 2);
    PyObject *py_engine = PyTuple_GetItem(state, 3);
    PyObject *py_rdata = PyTuple_GetItem(state, 4);

    int ndims;
    mkldnn::memory::dims dims;
    if (!PyTuple_Check(py_dims)) {
      SWIG_fail;
    } else {
      ndims = PyTuple_Size(py_dims);
      for (int i = 0; i < ndims; i++) {
        PyObject *_dim = PyTuple_GetItem(py_dims, i);
        dims.push_back(PyLong_AsLong(_dim));
      }
    }

    mkldnn::memory::data_type dtype =
        static_cast<mkldnn::memory::data_type>(PyLong_AsLong(py_dtype));
    mkldnn::memory::format format =
        static_cast<mkldnn::memory::format>(PyLong_AsLong(py_format));
    mkldnn::engine engine =
        *static_cast<mkldnn::engine *>(PyLong_AsVoidPtr(py_engine));
    void *rdata = PyLong_AsVoidPtr(py_rdata);

    mdarray *unpickled_mdarr = new mdarray(dims, dtype, format, engine);
    (*unpickled_mdarr)->unpickled_data(rdata);
    SwigPyObject *sobj = SWIG_Python_GetSwigThis(self);
    if (sobj) {
      sobj->ptr = reinterpret_cast<void *>(unpickled_mdarr);
      sobj->ty = SWIGTYPE_p_mdarray;
      sobj->own = 0;
      sobj->next = 0;
    } else {
      SWIG_fail;
    }
  }

  void __setstate__(PyObject *state) {
    (*$self)->__setstate__(state);
  }
}

%exception mdarray::mdarray {
  try {
    $action
  } catch (mkldnn::error &e) {
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

class mdarray: public py_handle {
public:
  // It is deliberately NOT matching prototypes!
  mdarray(mkldnn::memory::dims dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &e);

  mdarray(mkldnn::memory::primitive_desc pd);
  mdarray(mkldnn::memory::primitive_desc pd, mkldnn::memory mp);
  mdarray(Py_buffer *view
      , mkldnn::memory::format, mkldnn::engine &);

  virtual ~mdarray();
};

template <class p_t
, typename pd_t = typename p_t::primitive_desc>
class f_s_op: public mdarray {
public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
    , std::vector<mkldnn::primitive> *dag);
  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<mkldnn::primitive> *dag);
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public mdarray {
public:
  bd_op(pd_t &op, py_handle gy, py_handle W
  , std::vector<mkldnn::primitive> *dag);
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public mdarray {
public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
  , std::vector<mkldnn::primitive> *dag);
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public mdarray {
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
  , std::vector<mkldnn::primitive> *dag);
};

class reorder_buffer {
public:
  reorder_buffer(mdarray in);
};
