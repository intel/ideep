%{
  // TODO: Support both raw or smart pointer type
  #define nb_unary(op, m) \
    static PyObject * nb_ ## op (PyObject *self) {    \
      void *that;                                                 \
      int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);        \
      if (!SWIG_IsOK(res1)) {                                     \
        PyErr_SetString(PyExc_ValueError, "Wrong self object in nb_unary wrapper");  \
        return nullptr;                                                \
      }                                                           \
      return (*reinterpret_cast<T *>(that))->m_ ## m(self);  \
    }

  #define nb_binary(op, m) \
    static PyObject * nb_ ## op (PyObject *left, PyObject *right) {    \
      void *that;                                                 \
      int res1 = SWIG_ConvertPtr(left, &that, nullptr, 0);        \
      if (SWIG_IsOK(res1)) {                                      \
        return (*reinterpret_cast<T *>(that))->m_ ## m(left, right);  \
      } else {                                                    \
        res1 = SWIG_ConvertPtr(right, &that, nullptr, 0);         \
        if (!SWIG_IsOK(res1)) {                                   \
          PyErr_SetString(PyExc_ValueError, "Wrong self object in nb_binary wrapper");  \
          return nullptr;                                             \
        }                                                         \
        return (*reinterpret_cast<T *>(that))->m_ ## m(right, left);  \
      }                                                           \
    }

  #define nb_ternary(op, m) \
    static PyObject * nb_ ## op (PyObject *self, PyObject *o1, PyObject *o2) {    \
      void *that;                                                 \
      int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);        \
      if (!SWIG_IsOK(res1)) {                                     \
        PyErr_SetString(PyExc_ValueError, "Wrong self object in nb_ternary wrapper");  \
        return nullptr;                                                \
      }                                                           \
      return (*reinterpret_cast<T *>(that))->m_ ## m(self, o1, o2);  \
    }


  template <class T>
  struct number_traits {
    nb_binary(add, Add);
    nb_binary(subtract, Subtract);
    nb_binary(multiply, Multiply);
    nb_binary(divide, Divide);
    nb_binary(remainder, Remainder);
    nb_binary(divmod, Divmod);
    nb_ternary(power, Power);
    nb_unary(negative, Negative);
    nb_unary(positive, Positive);
    nb_unary(absolute, Absolute);
    nb_unary(invert, Invert);
    nb_binary(lshift, Lshift);
    nb_binary(rshift, Rshift);
    nb_binary(and, And);
    nb_binary(xor, Xor);
    nb_binary(or, Or);
    nb_binary(inplace_add, InPlaceAdd);
    nb_binary(inplace_subtract, InPlaceSubtract);
    nb_binary(inplace_multiply, InPlaceMultiply);
    nb_binary(inplace_divide, InPlaceDivide);
    nb_binary(inplace_remainder, InPlaceRemainder);
    nb_ternary(inplace_power, InPlacePower);
    nb_binary(inplace_lshift, InPlaceLshift);
    nb_binary(inplace_rshift, InPlaceRshift);
    nb_binary(inplace_and, InPlaceAnd);
    nb_binary(inplace_xor, InPlaceXor);
    nb_binary(inplace_or, InPlaceOr);
    nb_binary(floor_divide, FloorDivide);
    nb_binary(true_divide, TrueDivide);
    nb_binary(inplace_floor_divide, InPlaceFloorDivide);
    nb_binary(inplace_true_divide, InPlaceTrueDivide);
    nb_binary(matrix_multiply, MatrixMultiply);
    nb_binary(inplace_matrix_multiply, InPlaceMatrixMultiply);
  };
%}

%define %nb_slot(name, type)
  %feature("python:nb_" %str(name)) type "number_traits<" %str(type) ">::nb_" %str(name);
%enddef

%define %number_protocol(type...)
  %nb_slot(add, type);
  %nb_slot(subtract, type);
  %nb_slot(multiply, type);
  %nb_slot(divide, type)
  %nb_slot(divmod, type);
  %nb_slot(negative, type);
  %nb_slot(positive, type);
  %nb_slot(absolute, type);
  %nb_slot(invert, type);
  %nb_slot(lshift, type);
  %nb_slot(rshift, type);
  %nb_slot(and, type);
  %nb_slot(xor, type);
  %nb_slot(or, type);
  %nb_slot(inplace_add, type);
  %nb_slot(inplace_subtract, type);
  %nb_slot(inplace_multiply, type);
  %nb_slot(inplace_divide, type)
  %nb_slot(inplace_remainder, type);
  %nb_slot(inplace_power, type);
  %nb_slot(inplace_lshift, type);
  %nb_slot(inplace_rshift, type);
  %nb_slot(inplace_and, type);
  %nb_slot(inplace_xor, type);
  %nb_slot(inplace_or, type);
  %nb_slot(floor_divide, type);
  %nb_slot(inplace_floor_divide, type);
  %nb_slot(matrix_multiply, type);
  %nb_slot(inplace_matrix_multiply, type);
%enddef
