#ifndef _ABSTRACT_TYPES_HPP_
#define _ABSTRACT_TYPES_HPP_

#include <string>
#include <mkldnn.h>
#include <mkldnn.hpp>

namespace ideep {

using error = mkldnn::error;

#define IDEEP_ENFORCE(condition, message) \
  do {  \
    error::wrap_c_api((condition) \
        ? mkldnn_success : mkldnn_invalid_arguments, (message));  \
  } while(false) \

#define IDEEP_STD_EQUAL(v, i) \
  std::all_of(v.begin(), v.end(), [](decltype(v)::value_type k){return k == i;})

/// Same class for resource management, except public default constructor
/// Movable support for better performance
template <typename T, typename traits = mkldnn::handle_traits<T>>
class c_wrapper{
protected:
  std::shared_ptr<typename std::remove_pointer<T>::type> _data;
public:
  /// Constructs a C handle wrapper.
  /// @param t The C handle to wrap.
  /// @param weak A flag to specify whether to construct a weak wrapper.
  c_wrapper(T t = nullptr, bool weak = false): _data(t, [weak]() {
    auto dummy = [](T) {
      return decltype(traits::destructor(0))(0);
    };
    return weak? dummy : traits::destructor;
  }()) {}

  bool operator==(const T other) const { return other == _data.get(); }
  bool operator!=(const T other) const { return !(*this == other); }

  c_wrapper(const c_wrapper& other): _data(other._data) {}
  c_wrapper(c_wrapper&& movable) : _data(std::move(movable._data)) {}

  c_wrapper &operator=(c_wrapper&& other) {
    _data = std::move(other._data);
    return *this;
  }

  c_wrapper &operator=(const c_wrapper& other) {
    _data = other._data;
    return *this;
  }

  /// Resets the value of a C handle.
  /// @param t The new value of the C handle.
  /// @param weak A flag to specify whether the wrapper should be weak.
  void reset(T t, bool weak = false) {
    auto dummy_destructor = [](T) {
      return decltype(traits::destructor(0))(0);
    };
    _data.reset(t, weak ? dummy_destructor : traits::destructor);
  }

  /// Returns the value of the underlying C handle.
  T get() const { return _data.get(); }

  bool operator==(const c_wrapper &other) const {
    return other._data.get() == _data.get();
  }
  bool operator!=(const c_wrapper &other) const {
    return !(*this == other);
  }
};

/// C wrappers which form a functioning complex, in case multiple
/// Primitives needed to finish certain task.
template <typename T>
class c_wrapper_complex : public c_wrapper<T> {
public:
  using size_type = typename std::vector<c_wrapper<T>>::size_type;
  constexpr static int max_reorder_needed = 3;

  c_wrapper_complex() {}

  inline bool need_reorder_input(int pos) const {
    if (pos < max_reorder_needed/* auxiliaries_.size()*/)
      return auxiliaries_[pos] != nullptr;
    return false;
  }
protected:
  c_wrapper<T> auxiliaries_[max_reorder_needed];
};

using batch_normalization_flag = mkldnn::batch_normalization_flag;
using query = mkldnn::query;
using round_mode = mkldnn::round_mode;

/// hide other formats
enum format {
  format_undef = mkldnn_format_undef,
  any = mkldnn_any,
  blocked = mkldnn_blocked,
  x = mkldnn_x,
  nc = mkldnn_nc,
  io = mkldnn_io,
  oi = mkldnn_oi,
  nchw = mkldnn_nchw,
  nhwc = mkldnn_nhwc,
  chwn = mkldnn_chwn,
  ncdhw = mkldnn_ncdhw,
  ndhwc = mkldnn_ndhwc,
  oihw = mkldnn_oihw,
  ihwo = mkldnn_ihwo,
  hwio = mkldnn_hwio,
  oidhw = mkldnn_oidhw,
  goihw = mkldnn_goihw,
  hwigo = mkldnn_hwigo,
  ntc = mkldnn_ntc,
  tnc = mkldnn_tnc,
  format_last = mkldnn_format_last
};

/// cpu execution engine only.
struct engine: public mkldnn::engine {
  explicit engine(const mkldnn_engine_t& aengine) = delete;
  engine(engine const &) = delete;
  void operator =(engine const &) = delete;

  /// Singleton CPU engine for all primitives
  static engine &cpu_engine();

  /// Put this global engine in only one library
  #define INIT_GLOBAL_ENGINE \
  ideep::engine &ideep::engine::cpu_engine() { \
    static engine cpu_engine; \
    return cpu_engine; \
  }

  inline static format default_format(int ndims) {
    switch(ndims) {
    case 1:
      return format::x;
    case 2:
      return format::nc;
    case 3:
      return format::blocked;
    case 4:
      return format::nchw;
    case 5:
      return format::ncdhw;
    default:
      return format::format_undef;
    }
  }

private:
  /// Constructs an engine.
  ///
  /// @param akind The kind of engine to construct.
  /// @param dformat The default data type of the engine.

  engine(kind akind = kind::cpu)
    :mkldnn::engine(akind, 0) {
  }
};

/// A default stream
struct stream: public mkldnn::stream {
  using mkldnn::stream::stream;
  static stream default_stream() {
    return stream(mkldnn::stream::kind::eager);
  }
};

using kind = mkldnn::primitive::kind;
using prop_kind = mkldnn::prop_kind;
using algorithm = mkldnn::algorithm;
using padding_kind = mkldnn::padding_kind;
}

#endif
