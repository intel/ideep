%define %int_sequence_typemap(integer_sequence_compitable_type)

%typemap(typecheck) (integer_sequence_compitable_type) {
  $1 = PySequence_Check($input);
}

%typemap(in) (integer_sequence_compitable_type) (int count) {
  count = PySequence_Size($input);

  for (int i =0; i < count; i ++) {
    PyObject *o = PySequence_GetItem($input, i);
    $1.push_back(PyLong_AsLong(o));
  }
}
%enddef

%define %at_sequence_typemap(at_sequence_compitable_type)

%typemap(typecheck) (at_sequence_compitable_type) {
  $1 = PySequence_Check($input);
}

%typemap(in) (at_sequence_compitable_type) (int count,
    at_sequence_compitable_type ins) {
  count = PySequence_Size($input);
  for (int i =0; i < count; i ++) {
    PyObject *o = PySequence_GetItem($input, i);
    mkldnn::primitive::at *tmp;
    int res1 = SWIG_ConvertPtr(o, reinterpret_cast<void **>(&tmp)
        , $descriptor(mkldnn::primitive::at *), 0);

    if (!SWIG_IsOK(res1)) {
      SWIG_exception_fail(SWIG_ArgError(res1)
          , "typemap 'mkldnn::primitive::at' sequence type failed"); 
    }
    if (tmp == nullptr) {
      SWIG_exception_fail(SWIG_ArgError(res1)
          , "Input is not a sequential type of 'mkldnn::primitive::at'");
    }
    ins.emplace_back(*tmp);
  }

  $1 = std::move(ins);
}
%enddef
