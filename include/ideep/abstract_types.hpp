#ifndef _ABSTRACT_TYPES_HPP_
#define _ABSTRACT_TYPES_HPP_

#include <string>
#include <mkldnn.h>
#include <mkldnn.hpp>

namespace ideep {

using error = mkldnn::error;

/// Same class for resource management, except public default constructor
template <typename T> class c_wrapper: public mkldnn::handle<T>{
  public:
    using mkldnn::handle<T>::handle;
    using mkldnn::handle<T>::operator ==;
    using mkldnn::handle<T>::operator !=;
};

/// C wrappers which form a functioning complex, in case multiple
/// Primitives needed to finish certain task.
template <typename T>
class c_wrapper_complex : public c_wrapper<T> {
public:
  using size_type = typename std::vector<c_wrapper<T>>::size_type;

  c_wrapper_complex(): auxiliaries_(3) {}

  c_wrapper_complex(size_type num_of_aux)
    : auxiliaries_(num_of_aux) {}

  inline bool need_reorder_input(int pos) const {
    if (pos < auxiliaries_.size())
      return auxiliaries_[pos] != nullptr;
    return false;
  }

protected:
  std::vector<c_wrapper<T>> auxiliaries_;
};

using batch_normalization_flag = mkldnn::batch_normalization_flag;
using query = mkldnn::query;

/// hide other formats
enum format {
  format_undef = mkldnn_format_undef,
  any = mkldnn_any,
  x = mkldnn_x,
  nc = mkldnn_nc,
  io = mkldnn_io,
  oi = mkldnn_oi,
  nchw = mkldnn_nchw,
  nhwc = mkldnn_nhwc,
  chwn = mkldnn_chwn,
  oihw = mkldnn_oihw,
  ihwo = mkldnn_ihwo,
  hwio = mkldnn_hwio,
  goihw = mkldnn_goihw,
  blocked = mkldnn_blocked,
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
      return format::goihw;
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
    stream def(mkldnn::stream::kind::eager);
    return def;
  }
};

using kind = mkldnn::primitive::kind;
using prop_kind = mkldnn::prop_kind;
using algorithm = mkldnn::algorithm;
using padding_kind = mkldnn::padding_kind;
}

#endif
