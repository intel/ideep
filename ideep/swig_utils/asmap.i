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
