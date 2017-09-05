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
