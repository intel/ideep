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

// Just grab it from MKL-DNN
namespace avx {
  inline void* malloc(size_t size, int alignment) {
      void *ptr;
      int rc = ::posix_memalign(&ptr, alignment, size);
      return (rc == 0) ? ptr : 0;
  }
  inline void free(void* p) { ::free(p); }

  struct compatible {
      enum { default_alignment = 64 };
      static void* operator new(size_t sz) {
          return malloc(sz, default_alignment);
      }
      static void* operator new(size_t sz, void* p) { (void)sz; return p; }
      static void* operator new[](size_t sz) {
          return malloc(sz, default_alignment);
      }
      static void operator delete(void* p) { free(p); }
      static void operator delete[](void* p) { free(p); }
  };

  struct byte: public compatible {
    char q;
  };
}

namespace implementation {
  class mdarray;
}

using py_handle = std::shared_ptr<implementation::mdarray>;

template <class to>
static bool isa(const py_handle &t) {
  return to::classof(t.get());
}

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

class mdarray {
public:
  // It is exposed to python
  //
  static constexpr int MAX_NDIM = 12; //XXX: For now

  class reorderer {
  protected:
    bool non_trivial_;
    bool reordered_;
    mkldnn::memory dst_;
    std::shared_ptr<avx::byte> data_;

    int ndims_;
    int size_;
    char format_[4];
    Py_ssize_t itemsize_;
    Py_ssize_t strides_[MAX_NDIM];
    Py_ssize_t shape_[MAX_NDIM];

    void _collect_buffer_info() {
      auto md = dst_.get_primitive_desc().desc();
      int ndims = md.data.ndims;

      ndims_ = ndims;
      switch(static_cast<mkldnn::memory::data_type>(md.data.data_type)) {
        case mkldnn::memory::f32:
          strcpy(format_, "f");
          itemsize_ = 4;
          break;
        case mkldnn::memory::s32:
          strcpy(format_, "i");
          itemsize_ = 4;
          break;
        default:
          break;
      }

      for (int i = 0; i < ndims; i ++) {
        shape_[i] = md.data.dims[i];
      }

      Py_ssize_t sd = itemsize_;

      for (int i = ndims -1; i >= 0; --i) {
        strides_[i] = sd;
        sd *= shape_[i];
      }
    }

    inline avx::byte *data() const { return data_.get(); }

  public:
    reorderer(const py_handle in)
      :reorderer(in.get()) {}

    reorderer(const mdarray *src)
      : non_trivial_(src->incompatible()), reordered_(false), dst_([src] () {
          if (src->incompatible()) {
            auto md_data = src->desc().data;

            mkldnn::memory::dims adims(md_data.dims
                , md_data.dims + md_data.ndims);

            mkldnn::memory::primitive_desc pd ({adims
                , static_cast<mkldnn::memory::data_type>(md_data.data_type)
                , public_format(
                    static_cast<mkldnn::memory::format>(md_data.format))}
                , src->get_engine());

            // XXX: magic number 4 is a hack
            return mkldnn::memory(pd, reinterpret_cast<void *>(4));
          } else {
            return src->memory();
          }} ()), size_(src->size()) {
        if (src->incompatible()) {
          auto pd = dst_.get_primitive_desc();

          data_ = std::shared_ptr<avx::byte>(new avx::byte [pd.get_size()]
              , [](avx::byte *p) {delete [] p;});

          dst_.set_data_handle(data_.get());

        } else {
          data_ = src->share_data();
        }

        _collect_buffer_info();
      }

    mkldnn::reorder fire(const mdarray *src) {
      mkldnn::reorder reorder(src->memory(), dst_);
      mkldnn::stream s(mkldnn::stream::eager);

      s.submit({reorder}).wait();
      return reorder;
    }

    mkldnn::reorder sync(const mdarray *src) {
      mkldnn::reorder reorder(dst_, src->memory());
      mkldnn::stream s(mkldnn::stream::eager);

      s.submit({reorder}).wait();
      return reorder;
    }

    inline bool non_trivial() const {
      return non_trivial_;
    }

    inline void set_reordered() {
      reordered_ = true;
    }

    inline void reset_reorder() {
      reordered_ = false;
    }

    inline bool is_reordered() const {
      return reordered_;
    }

    static mkldnn::memory::format public_format(
        mkldnn::memory::format origin) {
      mkldnn::memory::format ret;

      // review this relations carefully
      switch(origin) {
      case mkldnn::memory::nchw:
      case mkldnn::memory::nhwc:
      case mkldnn::memory::chwn:
      case mkldnn::memory::nChw8c:
      case mkldnn::memory::nChw16c:
        ret = mkldnn::memory::nchw;
        break;
      case mkldnn::memory::oihw:
      case mkldnn::memory::ihwo:
      case mkldnn::memory::hwio:
      case mkldnn::memory::OIhw8i8o:
      case mkldnn::memory::OIhw16i16o:
      case mkldnn::memory::OIhw8o8i:
      case mkldnn::memory::OIhw16o16i:
      case mkldnn::memory::OIhw8i16o2i:
      case mkldnn::memory::OIhw8o16i2o:
      case mkldnn::memory::Oihw8o:
      case mkldnn::memory::Oihw16o:
      case mkldnn::memory::Ohwi8o:
      case mkldnn::memory::Ohwi16o:
      case mkldnn::memory::OhIw16o4i:
        ret = mkldnn::memory::oihw;
        break;
      default:
        ret = origin;
        break;
      }

      return ret;
    }

    // PEP 3118 interface
    int build_view(Py_buffer *view, int flags) {
      view->buf = data_.get();
      view->itemsize = itemsize_;
      view->readonly = 0;
      view->internal = nullptr;
      view->len = size_ * itemsize_;

      if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        view->format = format_;
      } else {
        view->format = nullptr;
      }

      if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = ndims_;
        view->shape = shape_;
      } else {
        view->ndim = 0;
        view->shape = nullptr;
      }

      if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->strides = strides_;
      } else {
        view->strides = nullptr;
      }

      view->suboffsets = nullptr;

      return 0;
    }

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
  };


  // Array Protocol interface
  class reorder_array : public reorderer {
  public:
    reorder_array(const py_handle in) : reorderer(in) {}
    reorder_array(const mdarray *src) : reorderer(src) {}

  };

public:
  typedef size_t size_type;
  // Generated on demand
  virtual ~mdarray() = default;

  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , const mkldnn::engine &engine)
    : mdarray({{std::move(dims), dt, format}, engine}) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : size_([&pd] () {
                    auto md = pd.desc().data;
                    return std::accumulate(md.dims, md.dims + md.ndims, 1
                        , std::multiplies<int>());
                  }())
              // Use primitive desc's reference
              , data_(new avx::byte [pd.get_size()]
                  , [](avx::byte *p) {delete [] p;})
              , m_(pd, data_.get())
              , view_(nullptr), rtti(raw)
              , internal_order_([&pd] () {
                  auto md = pd.desc().data;
                    return reorderer::public_format(
                        static_cast<mkldnn::memory::format>(md.format)
                        ) != md.format;
                  } ()), purpose_(sink) {}

  mdarray(mkldnn::memory::primitive_desc pd, mkldnn::memory mp)
    : size_([&pd] () {
                    auto md = pd.desc().data;
                    return std::accumulate(md.dims, md.dims + md.ndims, 1
                        , std::multiplies<int>());
                  }())
              // Use primitive desc's reference
              , data_(std::shared_ptr<avx::byte>(
                   reinterpret_cast<avx::byte *>(mp.get_data_handle())
                   , [] (avx::byte *p) {}))
              , m_(pd, data_.get())
              , view_(nullptr), rtti(raw)
              , internal_order_([&pd] () {
                  auto md = pd.desc().data;
                    return reorderer::public_format(
                        static_cast<mkldnn::memory::format>(md.format)
                        ) != md.format;
                  } ()), purpose_(sink) {}

  mdarray(Py_buffer *view
      , mkldnn::memory::format format
      , const mkldnn::engine &e)
    : size_(view->len/view->itemsize)
          , data_ ([view]() {
             unsigned long adrs = reinterpret_cast<unsigned long>(view->buf);
             if (adrs % 16 != 0) {
               return std::shared_ptr<avx::byte>(new avx::byte [view->len]
                   , [] (avx::byte *p) {delete [] p;});
             } else
               return std::shared_ptr<avx::byte>(
                   reinterpret_cast<avx::byte *>(view->buf)
                   , [] (avx::byte *p) {});
           } ())
          , m_({_d_from_view(view, format), e}, data_.get())
          , view_(view), rtti(raw), internal_order_(false), purpose_(source) {

    assert(m_.get_primitive_desc().get_size()
        == static_cast<decltype(
          m_.get_primitive_desc().get_size())>(view->len));

    if (data_.get() != view->buf) {
      // XXX: Add OpenMP thing?
      memcpy(data_.get(), view->buf, view->len);
      view_.reset(nullptr);
    }
  }

  // TODO: for view case, shared buffer won't expand life in this case
  // because mdarray will destroy it when out of service.
  //
  int setbuffer(const Py_buffer *view) {
    if (purpose_ == sink)
      // TODO: not support by provided buffer to numpy
      goto fail;
    else {
      // TODO: Guard this section with asserts
      view_.reset(view);

      unsigned long adrs = reinterpret_cast<unsigned long>(view->buf);

      if (adrs % 16 != 0) {
        data_.reset(new avx::byte [view->len]
            , [] (avx::byte *p) {delete [] p;});
        memcpy(data_.get(), view->buf, view->len);
        view_.reset(nullptr);
      } else
        data_.reset(reinterpret_cast<avx::byte *>(view->buf)
            , [] (avx::byte *p) {});

      assert(m_.get_primitive_desc().get_size()
          == static_cast<decltype(
            m_.get_primitive_desc().get_size())>(view->len));

      m_.set_data_handle(data());
    }

    return 0;
  fail:
    return -1;
  }

  inline void unpickled_data(void *pdata) {
    data_.reset(reinterpret_cast<avx::byte *>(pdata));
    m_.set_data_handle(pdata);
    return;
  }

  inline void *data() const { return data_.get(); }
  inline size_type size() const { return size_; }
  inline size_type len() const { return m_.get_primitive_desc().get_size(); }
  inline mkldnn::engine get_engine() const {
    return m_.get_primitive_desc().get_engine();
  }

  inline int ndims() const {
    auto md = m_.get_primitive_desc().desc();
    return md.data.ndims;
  }

  inline mkldnn::memory memory() const {
    return m_;
  }

  inline mkldnn::memory::desc desc() const {
    return m_.get_primitive_desc().desc();
  }

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

  PyObject *m_mult_div(PyObject *self, PyObject *o, int mult_or_div, bool inplace);

  // PEP: 3118 Buffer Protocol Producer
  virtual int getbuffer(PyObject *obj, Py_buffer *view, int flags);

  virtual void reset_buf_order() {}

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
  PyObject *m_InPlaceDivide(PyObject *self, PyObject *o);

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

private:
  struct WeDontManageIt {
    void operator() (const Py_buffer *view) {
      PyBuffer_Release(const_cast<Py_buffer *>(view));
      delete view;
    }
  };

  // Attributes
  size_type size_;
  std::shared_ptr<avx::byte> data_;
  mkldnn::memory m_;
  std::unique_ptr<const Py_buffer, WeDontManageIt> view_;

protected:
  enum mdarray_ty{
    raw, dual_out
  };
  mdarray_ty rtti;
  bool internal_order_;
  reorderer *sync_reorder_;

  enum purpose {
    source, sink
  } purpose_;

public:
  inline bool incompatible() const { return internal_order_; }
  std::shared_ptr<avx::byte> share_data() const {
    return data_;
  }

  static bool classof(const mdarray *p) {
    return p->get_kind() == raw;
  }

  mdarray_ty get_kind() const { return rtti; }

  static mkldnn::memory reorder_if_must(mkldnn::memory user
      , mkldnn::memory::primitive_desc expect
      , std::unique_ptr<mkldnn::memory> &mreorder
      , std::vector<mkldnn::primitive> *dag) {

    if (user.get_primitive_desc() != expect) {
      mkldnn::memory interm(expect);
#if 0
      auto user_mpd = user.get_primitive_desc();
      mkldnn::memory::format user_fmt = static_cast<mkldnn::memory::format>(
          user_mpd.desc().data.format);
      mkldnn::memory::format mkl_fmt = static_cast<mkldnn::memory::format>(
          expect.desc().data.format);
      mkldnn::memory::data_type dtype = static_cast<mkldnn::memory::data_type>(
          expect.desc().data.data_type);

      if ((user_fmt == mkldnn::memory::format::nChw16c &&
           mkl_fmt == mkldnn::memory::format::nChw8c) ||
          (mkl_fmt == mkldnn::memory::format::nChw16c &&
           user_fmt == mkldnn::memory::format::nChw8c)) {
          auto m = expect.desc().data;
          int n = m.dims[0], c = m.dims[1], h = m.dims[2], w = m.dims[3];
          mkldnn::memory::dims tz = {n, c, h, w};
          mreorder.reset(new mkldnn::memory({{{ tz }, dtype, mkldnn::memory::format::nchw }, expect.get_engine()}));
          //auto mreorder = new mkldnn::memory({{{ tz }, dtype, mkldnn::memory::format::nchw }, expect.get_engine()});
          auto rep1 = mkldnn::reorder(user, *mreorder);
          auto rep2 = mkldnn::reorder(*mreorder, interm);
          dag->push_back(rep1);
          dag->push_back(rep2);
          //static int spl_nr = 0;
          //printf("\n   %d *Reorder(split) iutput from:%d, to:%d\n", spl_nr++, user_fmt, mkl_fmt);
      } else {
          dag->push_back(mkldnn::reorder(user, interm));
      }
#else
      dag->push_back(mkldnn::reorder(user, interm));
#endif
      return interm;
    }

    return user;
  }

protected:
  // Private helpers
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
};

// XXX: solve dual outputs problem
// Type system should be rework
// TODO: review polymophic relationship
// TODO: rework the names

class s_op: public mdarray {
public:
  using mdarray::reorderer;

  s_op(mkldnn::memory::primitive_desc dst
      , std::vector<mkldnn::primitive> *dag)
    : mdarray(dst), dag_(dag), reorder_(nullptr),
      mreorder_(nullptr) {
  }

  virtual int getbuffer(PyObject *self
      , Py_buffer *view, int flags) override;

  virtual void reset_buf_order() override {
    if (reorder_.get()) {
        reorder_->reset_reorder();
    }
  }
protected:
  std::vector<mkldnn::primitive> *dag_;
  std::unique_ptr<reorderer> reorder_;
  std::unique_ptr<mkldnn::memory> mreorder_;
};

class d_op : public s_op {
public:
  // XXX: Tricky part, how extra managed
  d_op(mkldnn::memory::primitive_desc major
      , mkldnn::memory::primitive_desc extra
      , std::vector<mkldnn::primitive> *dag):
    s_op(major, dag), extra(std::make_shared<s_op>(extra, dag))
    , dag_(dag) {
    rtti = dual_out;
  }

  static py_handle extra_get(const d_op *that) {
    return that->extra;
  }

  static bool classof(const mdarray *p) {
    return p->get_kind() == dual_out;
  }
protected:
  // This seems unique, but it will share in python
  // Ugly. XXX
  py_handle extra;
  std::vector<mkldnn::primitive> *dag_;
};

class t_op : public s_op {
public:
  // XXX: Tricky part, how extra managed
  t_op(mkldnn::memory::primitive_desc major
      , mkldnn::memory::primitive_desc b_
      , mkldnn::memory::primitive_desc w_
      , std::vector<mkldnn::primitive> *dag)
    : s_op(major, dag), b_(std::make_shared<s_op>(b_, dag))
    , w_(std::make_shared<s_op>(w_, dag)), dag_(dag) {
    rtti = dual_out;
  }

  static py_handle bias_get(const t_op *that) {
    return that->b_;
  }

  static py_handle wrks_get(const t_op *that) {
    return that->w_;
  }

  static bool classof(const mdarray *p) {
    return p->get_kind() == dual_out;
  }
protected:
  // This seems unique, but it will share in python
  // Ugly. XXX
  py_handle b_, w_;
  std::vector<mkldnn::primitive> *dag_;
};

using namespace mkldnn;

//
// Active primitive
//
template <class p_t
, typename pd_t = typename p_t::primitive_desc>
class f_s_op: public s_op {
private:
  f_s_op(pd_t &op, mdarray *x, mdarray *W
      , std::vector<primitive> *dag)
    : s_op(op.dst_primitive_desc(), dag)
      , x_reordered_(reorder_if_must(x->memory(), op.src_primitive_desc()
            , mreorder_, dag_))
      , W_reordered_(reorder_if_must(W->memory(), op.weights_primitive_desc()
      , mreorder_, dag_)) {}

public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
      , std::vector<primitive> *dag)
    : f_s_op(op, x.get(), W.get(), dag) {
      deps_ = {x, W, b};
      dag_->push_back(p_t(op, x_reordered_, W_reordered_, b->memory()
            , this->memory()));
    }

  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<primitive> *dag)
    : f_s_op(op, x.get(), W.get(), dag){
      deps_= {x, W};
      dag_->push_back(p_t(op, x_reordered_, W_reordered_, this->memory()));
    }

private:
  mkldnn::memory x_reordered_, W_reordered_;
  std::vector<py_handle> deps_;
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public s_op {
private:
  bd_op(pd_t &op
      , mdarray *gy, mdarray *W, std::vector<primitive> *dag)
    : s_op(op.diff_src_primitive_desc(), dag)
      , gy_reordered_(reorder_if_must(gy->memory()
            , op.diff_dst_primitive_desc(), mreorder_, dag_))
      , W_reordered_(reorder_if_must(W->memory()
            , op.weights_primitive_desc(), mreorder_, dag_)) {}

public:
  bd_op(pd_t &op, py_handle gy, py_handle W
      , std::vector<primitive> *dag)
    : bd_op(op, gy.get(), W.get(), dag) {
      deps_= {gy, W};
      dag_->push_back(p_t(op, gy_reordered_, W_reordered_, this->memory()));
    }

private:
  mkldnn::memory gy_reordered_, W_reordered_;
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public d_op {
public:
  bwb_op(pd_t &op
      , mdarray *x, mdarray *gy, std::vector<primitive> *dag)
    : d_op(op.diff_weights_primitive_desc(), op.diff_bias_primitive_desc()
        , dag)
      , x_reordered_(reorder_if_must(x->memory(), op.src_primitive_desc()
            , mreorder_, dag_))
      , gy_reordered_(reorder_if_must(gy->memory()
          , op.diff_dst_primitive_desc(), mreorder_, dag_)) {}

public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : bwb_op(op, x.get(), gy.get(), dag) {
      deps_ = {x, gy};
      dag_->push_back(p_t(op, x_reordered_, gy_reordered_, memory()
            , extra->memory()));
    }

private:
  mkldnn::memory x_reordered_, gy_reordered_;
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public s_op {
public:
  bw_op(pd_t &op
      , mdarray *x, mdarray *gy, std::vector<primitive> *dag)
    : s_op(op.diff_weights_primitive_desc(), dag)
      , x_reordered_(reorder_if_must(x->memory(), op.src_primitive_desc()
            , mreorder_, dag_))
      , gy_reordered_(reorder_if_must(gy->memory()
      , op.diff_dst_primitive_desc(), mreorder_, dag_)) {}

public:
  bw_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : bw_op(op, x.get(), gy.get(), dag) {
      deps_ = {x, gy};
      dag_ ->push_back(p_t(op, x_reordered_, gy_reordered_, memory()));
    }

private:
  mkldnn::memory x_reordered_, gy_reordered_;
  std::vector<py_handle> deps_;
};

//
// Passive primitive
//
template<class p_t, typename pd_t = typename p_t::primitive_desc>
class passive_f_op: public s_op {
public:
  passive_f_op(pd_t &op, std::vector<primitive> *dag)
    : s_op(op.dst_primitive_desc(), dag) {}

public:
  passive_f_op(pd_t &op, py_handle x
      , std::vector<primitive> *dag)
    : passive_f_op(op, dag) {
      deps_ = {x};
      dag_ ->push_back(p_t(op, x->memory(), memory()));
    }

private:
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class passive_bd_op: public s_op {
public:
  passive_bd_op(pd_t &op, std::vector<primitive> *dag)
    : s_op(op.dst_primitive_desc(), dag) {}

public:
  passive_bd_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : passive_bd_op(op, dag) {
      deps_ = {x, gy};
      dag_ ->push_back(p_t(op, x->memory(), gy->memory(), memory()));
    }

private:
  std::vector<py_handle> deps_;
};
}

//
// Actual interface for python
// DO NOT add field
//
class mdarray : public py_handle {
public:
  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &engine)
    : py_handle(std::make_shared<implementation::mdarray>
        (dims, dt, format, engine)) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : py_handle(std::make_shared<implementation::mdarray>(pd)) {}

  mdarray(mkldnn::memory::primitive_desc pd, mkldnn::memory mp)
    : py_handle(std::make_shared<implementation::mdarray>(pd, mp)) {}

  mdarray(Py_buffer *view
      , mkldnn::memory::format format
      , mkldnn::engine &e)
    : py_handle(std::make_shared<implementation::mdarray>(view, format, e)) {}

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

  static mkldnn::memory *mdarray_memory_get(mdarray *self) {
    return new mkldnn::memory((*self)->memory());
  }

  static bool mdarray_is_mdarray_get(mdarray *self) {
    return true;
  }
};

using namespace mkldnn;

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class f_s_op : public py_handle {
public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
      , std::vector<primitive> *dag)
    : py_handle(std::make_shared< implementation::f_s_op<p_t, pd_t> >
       (op, x, W, b, dag)){}

  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<primitive> *dag)
    : py_handle(std::make_shared< implementation::f_s_op<p_t, pd_t> >
       (op, x, W, dag)) {}
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op : public py_handle {
public:
  bd_op(pd_t &op, py_handle gy, py_handle W
      , std::vector<primitive> *dag)
    : py_handle (std::make_shared< implementation::bd_op<p_t, pd_t> >
        (op, gy, W, dag)) {}
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public py_handle {
public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : py_handle (std::make_shared< implementation::bwb_op<p_t, pd_t> >
        (op, x, gy, dag)) {}

  static py_handle *extra_get(const py_handle *in) {
    if (isa<implementation::d_op>(*in)){
        return new py_handle(implementation::d_op::extra_get
        (static_cast<implementation::d_op *>(in->get())));
    }

    // Raise exception?
    return nullptr;
  }
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public py_handle {
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : py_handle (std::make_shared< implementation::bw_op<p_t, pd_t> >
        (op, x, gy, dag)) {}
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class passive_f_op: public py_handle {
public:
  passive_f_op(pd_t &op, py_handle x
      , std::vector<primitive> *dag)
    : py_handle (std::make_shared< implementation::passive_f_op<p_t, pd_t> >
        (op, x, dag)) {}
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class passive_bd_op: public py_handle {
public:
  passive_bd_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : py_handle (std::make_shared< implementation::passive_bd_op<p_t, pd_t> >
        (op, x, gy, dag)) {}
};

using reorder_buffer = implementation::mdarray::reorderer;

#endif
