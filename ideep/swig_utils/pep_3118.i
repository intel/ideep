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
%typemap(typecheck) (VIEW) {
  $1 = PyObject_CheckBuffer($input);
}

%typemap(in) (VIEW) (int res, Py_buffer *view = nullptr
  , int flags = PyBUF_C_CONTIGUOUS | PyBUF_RECORDS) {
  view = new Py_buffer;
  res = PyObject_GetBuffer($input, view, flags);
  $1 = ($1_ltype) view;
  // TODO: IF WE CONFRONT A F_CONTINGUOUS ONE???
}
%enddef
