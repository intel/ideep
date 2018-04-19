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
#include "utils.h"

namespace implementation {
  class mdarray;
}

class reorderer;

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

// FIXME: Redundant interceptions in lambda []
class mdarray : public ideep::tensor {
public:
  using tensor = ideep::tensor;
  using data_type_t = mkldnn::memory::data_type;
  using dims_t = mkldnn::memory::dims;
  using format_t = ideep::format;
  using error = mkldnn::error;
  using scratch_allocator = ideep::utils::scratch_allocator;
  using reorder = ideep::reorder;
  using convolution_forward = ideep::convolution_forward;

  static constexpr int MAX_NDIM = 12; //XXX: For now

  typedef size_t size_type;

  mdarray() = default;
  virtual ~mdarray();

  // Create an memory entity from tensor
  // mdarray must be an owner of memory. In the case of the ctor,
  // * It is guaranteed that input tensor is a memory owner. If tensor
  //   is not a memory owner, please use ctor `mdarray(const mdarray &m)`.
  // * It is guaranteed that input tensor is a memory entity not a view.
  //   ALLOWED: tensor(entity) -> mdarray
  //   NOT-ALLOWED: mdarray(entity) -> mdarray/tensor(view) -> mdarray
  //   If tensor is a view, please use ctor `mdarray(const mdarray &m)`.
  mdarray(const tensor &t) :
      tensor(t),
      buff_([t]() {
            if (t.get_tensor_buffer().get() != nullptr) {
              return t.get_tensor_buffer();
            } else {
              throw error(mkldnn_invalid_arguments, std::string(
                  "mdarray ctor does not support view input"));
              return std::shared_ptr<char>(nullptr);
            }
          } ()),
      view_(nullptr) {}

  // Share from a mdarray
  // * If src mdarray is a memory entity, this mdarray shares the buffer.
  // * If src mdarray is a memory view, this mdarray shares the view.
  // this_mdarray->buff_ (src is entity)
  // this_mdarray->view->rb(other)->data_ (src is view)
  mdarray(const mdarray &m) :
      tensor(m),
      buff_(m.get_shared_buff()),
      view_(nullptr) {
    Py_buffer *view = nullptr;
    if (m.view_.get()) {
      // m is view
      view = new Py_buffer;
      // No need to modify attributes in view to keep consistence
      // between view and `this`(array). View in consumer(this) is just a
      // record of its producer. Hold sharing memory entity `view->obj`
      // only here. When `this` is a producer, a new view will be created
      // according to current `this`(array). Refer to `getbuffer`.
      memcpy((void *)(view), (void *)(m.view_.get()), sizeof(Py_buffer));
      Py_INCREF(m.view_->obj);
    } else {
      // m is entity
    }
    view_.reset(view);
  }

  // Memory entity created by array attributes
  mdarray(dims_t dims, data_type_t dt) :
      tensor({dims, dt, [dims]() {
            return ndims2format(dims.size());
          } ()}, [&]() {
            return reinterpret_cast<void *>(
                new scratch_allocator::byte<tensor>[dims2size(dims, dt)]);
          } ()),
      buff_(std::shared_ptr<char>((char *)get_data_handle(), [](char *p) {
            auto _p = reinterpret_cast<scratch_allocator::byte<tensor> *>(p);
            delete [] _p;
          })),
      view_(nullptr) {}

  // Memory view created by producer's view
  mdarray(Py_buffer *view, char input_type='d') :
      tensor({[&]() {
            return dims_t(view->shape, view->shape + view->ndim);
          } (), [&]() {
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
                  std::string("mdarray does not support data type: ") + format);
            }
            return dt;
          } (), [&]() {
            return ndims2format(view->ndim, input_type);
          } ()}, [&]() {
            void *buf = view->buf;
            if ((uint64_t)buf & (_TENSOR_MEM_ALIGNMENT_ - 1)) {
              buf = reinterpret_cast<void *>(
                  new scratch_allocator::byte<tensor>[view->len]);
              fast_memcpy((char *)buf, (char *)view->buf, view->len);
            }
            return buf;
          } ()),
      buff_([&] () {
            if (get_data_handle() != view->buf) {
              return std::shared_ptr<char>((char *)get_data_handle(),
                  [](char *p) {
                    auto _p =
                        reinterpret_cast<scratch_allocator::byte<tensor> *>(p);
                    delete [] _p;
                  });
            } else {
              // Im not the owner of the memory
              return std::shared_ptr<char>((char *)view->buf, [](char *p) {});
            }
          } ()), view_(view) {
    // Init weight array in prefered format for CNN convolution
    if (input_type == 'w' && ndims() == 4) {
      auto desc_in = convolution_forward::
          expected_weights_descriptor(get_dims(), get_data_type());
      if (get_descriptor() != desc_in) {
        auto buf = reinterpret_cast<void *>(
            new scratch_allocator::byte<tensor>[get_size()]);
        tensor wgt_in = tensor(desc_in, buf);
        reorder::compute(*this, wgt_in);

        init(wgt_in.get_descriptor(), wgt_in.get_data_handle());

        buff_.reset();
        buff_ = std::shared_ptr<char>((char *)buf, [](char *p) {
              auto _p = reinterpret_cast<scratch_allocator::byte<tensor> *>(p);
              delete [] _p;
            });

        view_.reset();
      }
    }

    if (view_.get() && get_data_handle() != view->buf) {
      view_.reset();
    }
  }

  static bool is_mdarray(PyObject *o);

  //FIXME
  inline void unpickled_data(void *pdata) {
    //data_.reset(reinterpret_cast<avx::byte *>(pdata));
    //m_.set_data_handle(pdata);
    return;
  }

  // PEP 3118 interface
  int build_view(Py_buffer *view, int flags, const reorderer &reorder);

  // PyObject *__getstate__(void) const;

  // void __setstate__(PyObject *state);

  PyObject *py_mdarray_from(PyObject *o) const;

  /// d = a * x + b * y, using x's format
  static void axpby(tensor &dst, float a, const tensor &x, float b, const tensor &y);

  /// Interface to directly contact python
  PyObject *axpby(float a, float b, PyObject *o);

  PyObject *inplace_axpby(float a, PyObject *self, float b, PyObject *o);

  PyObject *flat(void);

  PyObject *reshape(py_handle *self, std::vector<int> dims);

  PyObject *m_mult_div(PyObject *self, PyObject *o, int mult_or_div, bool inplace);

  PyObject *sum(std::vector<int> axis, bool keepdims);

  // PEP: 3118 Buffer Protocol Producer
  virtual int getbuffer(PyObject *self, Py_buffer *view, int flags);

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

  inline std::shared_ptr<char> get_shared_buff() const { return buff_; }

private:
  static inline size_t dims2size(dims_t &dims, data_type_t dt) {
    size_t itemsize;
    switch(dt) {
    case data_type_t::f32:
    case data_type_t::s32:
      itemsize = 4;
      break;
    case data_type_t::s16:
      itemsize = 2;
      break;
    case data_type_t::u8:
    case data_type_t::s8:
      itemsize = 1;
      break;
    default:
      throw error(mkldnn_invalid_arguments, std::string(
          "mdarray does not support data type: ") + std::to_string(dt));
    }

    size_t nelems = 1;
    for (unsigned d = 0; d < dims.size(); d++)
      nelems *= dims[d];

    return nelems * itemsize;
  }

  static inline
  format_t ndims2format(int ndims, char input_type = 'd')
  {
    switch (ndims) {
    case 1:
      return format_t::x;
    case 2:
      return (input_type == 'd') ? format_t::nc : format_t::oi;
    case 4:
      return (input_type == 'd') ? format_t::nchw : format_t::oihw;
    default:
      throw error(mkldnn_invalid_arguments, std::string(
          "MKLDNN does not support dimensions") + std::to_string(ndims));
      return format_t::format_undef;
    }
  }

  inline ssize_t *get_view_shape() {
    static ssize_t shape[MAX_NDIM];
    auto dims = get_dims();
    for (int d = 0; d < ndims(); d++)
      shape[d] = dims[d];

    return shape;
  }

  inline ssize_t *get_view_strides(ssize_t itemsize) {
    static ssize_t strides[MAX_NDIM];
    ssize_t sd = itemsize;
    for (int d = ndims() - 1; d >= 0; --d) {
      strides[d] = sd;
      sd *= get_dims()[d];
    }

    return strides;
  }

  inline ssize_t get_view_itemsize() {
    ssize_t itemsize;
    switch(get_data_type()) {
    case data_type_t::f32: itemsize = 4; break;
    case data_type_t::s32: itemsize = 4; break;
    case data_type_t::s16: itemsize = 2; break;
    case data_type_t::s8: itemsize = 1; break;
    case data_type_t::u8: itemsize = 1; break;
    default:
      throw error(mkldnn_invalid_arguments,
          std::string("get_view_itemsize, unsupport data type"));
      break;
    }
    return itemsize;
  }

  inline char *get_view_format() {
    static char format[4];
    switch(get_data_type()) {
    case data_type_t::f32: strcpy(format, "f"); break;
    case data_type_t::s32: strcpy(format, "i"); break;
    case data_type_t::s16: strcpy(format, "h"); break;
    case data_type_t::s8: strcpy(format, "b"); break;
    case data_type_t::u8: strcpy(format, "B"); break;
    default:
      throw error(mkldnn_invalid_arguments,
          std::string("get_view_format, unsupport data type"));
      break;
    }
    return format;
  }

  struct view_manager {
    void operator() (const Py_buffer *view) const {
      PyBuffer_Release(const_cast<Py_buffer *>(view));
      delete view;
    }

  };

  std::shared_ptr<char> buff_;
  std::unique_ptr<const Py_buffer, view_manager> view_;
};
}

// `reorderer` is designed from iDeep internal format.
// `reorderer` also is considered as a memory holder, when memory shareing
// request from python buffer protocol. `reorderer` will be descreased or
// deleted by protocol consumer, when related view releases. Memory entity
// mdarray always creates new `reorderer` to consumer, and memory view
// mdarray always shares the `reorderer` in view to consumer.
class reorderer {
public:
  using tensor = ideep::tensor;
  using data_type_t = mkldnn::memory::data_type;
  using format_t = ideep::format;
  using reorder = ideep::reorder;
  using descriptor = tensor::descriptor;
  using scratch_allocator = ideep::utils::scratch_allocator;
  using mdarray = implementation::mdarray;

  bool non_trivial_;
  std::shared_ptr<char> data_;

  inline void *data() const {
    return reinterpret_cast<void *>(data_.get());
  }

public:
  reorderer(const mdarray &src) :
      non_trivial_(!src.is_public_format()) {
    if (non_trivial()) {
      data_ = std::shared_ptr<char>(reinterpret_cast<char *>(
          new scratch_allocator::byte<tensor>[src.get_size()]),
          [](char *p) {
            auto _p = reinterpret_cast<scratch_allocator::byte<tensor> *>(p);
            delete [] _p;
          });
    } else {
      data_ = src.get_shared_buff();
    }
  }

  void fire(const mdarray &src) {
    if (non_trivial()) {
      tensor dst;
      dst.init({src.get_dims(), src.get_data_type(),
          descriptor::public_compatible_format(src.get_descriptor())},
          (void *)data_.get());
      reorder::compute(src, dst);
    }
  }

  inline bool non_trivial() const {
    return non_trivial_;
  }
};

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
    return self->get()->get_nelems();
  }

  static long mdarray_ndim_get(mdarray *self) {
    return self->get()->ndims();
  }

  static bool mdarray_is_mdarray_get(mdarray *self) {
    return true;
  }
};

class reorder_buffer : reorderer {
public:
  reorder_buffer(const py_handle in) :
    reorderer(*in.get()) {}
};

#endif // _MDARRAY_H_
