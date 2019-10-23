#ifndef IDEEP_ABSTRACT_TYPES_HPP
#define IDEEP_ABSTRACT_TYPES_HPP

#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cstdlib>
#include <mkldnn.h>
#include <mkldnn.hpp>

namespace ideep {

#ifdef _WIN32
#define IDEEP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define IDEEP_EXPORT __attribute__((__visibility__("default")))
#else
#define IDEEP_EXPORT
#endif

#ifndef NDEBUG
#define IDEEP_ENFORCE(condition, message) \
  do {  \
    error::wrap_c_api((condition) \
        ? mkldnn_success : mkldnn_invalid_arguments, (message));  \
  } while(false)
#else
#define IDEEP_ENFORCE(condition, message)
#endif

#define IDEEP_STD_ANY_LE(v, i) \
  std::any_of(v.begin(), v.end(), []( \
        std::remove_reference<decltype(v)>::type::value_type k){return k <= i;})

#define IDEEP_STD_EACH_SUB(v, i) \
  for (auto it = v.begin(); it != v.end(); it++) {*it -= i;}

// For convolution with grouped weights, the ndims must be 5 (goihw) or 6 (goidhw)
#define IDEEP_IS_GROUPED(id, wd) (((id == 4 && (wd).size() == 5) \
    || (id == 5 && (wd).size() == 6)) ? 1 : 0)

#define IDEEP_MOD_PTR(ptr, bytes) (((uintptr_t)(ptr)) & ((bytes) - 1))
#define IDEEP_IS_ALIGNED_PTR(ptr, bytes) ((IDEEP_MOD_PTR(ptr, bytes)) == 0)

struct error: public std::exception {
    mkldnn_status_t status;
    const char* message;

    error(mkldnn_status_t astatus, const char* amessage)
        : status(astatus), message(amessage) {}

    static void wrap_c_api(mkldnn_status_t status, const char* message) {
      if (status != mkldnn_success) {
        throw error(status, message);
      }
    }
};

/// Same class for resource management, except public default constructor
/// Movable support for better performance
template <typename T, typename traits = mkldnn::handle_traits<T>>
class c_wrapper :
  public std::shared_ptr<typename std::remove_pointer<T>::type> {
  using super = std::shared_ptr<typename std::remove_pointer<T>::type>;
public:
  c_wrapper(T t = nullptr, bool weak = false)
    : super(t, [weak]() {
        auto dummy = [](T) { return decltype(traits::destructor(0))(0); };
        return weak? dummy : traits::destructor; }()) {}

  using super::super;
  /// Resets the value of a C handle.
  void reset(T t, bool weak = false) {
    auto dummy_destructor = [](T) { return decltype(traits::destructor(0))(0); };
    super::reset(t, weak ? dummy_destructor : traits::destructor);
  }
};

using key_t = std::string;
using scale_t = std::vector<float>;

using query = mkldnn::query;
using kind = mkldnn::primitive::kind;
using prop_kind = mkldnn::prop_kind;
using algorithm = mkldnn::algorithm;
using padding_kind = mkldnn::padding_kind;
using batch_normalization_flag = mkldnn::batch_normalization_flag;
using query = mkldnn::query;
using round_mode = mkldnn::round_mode;
using rnn_direction = mkldnn::rnn_direction;

#define IDEEP_OP_SCALE_MASK(scale_size) (((scale_size) > 1) ? 2 : 0)
#define IDEEP_TENSOR_SCALE_MASK(scale_size, grouped) \
  (((scale_size) > 1) ? ((grouped) ? 3 : 1) : 0)

const scale_t IDEEP_DEF_SCALE {1.0f};

constexpr int IDEEP_U8_MAX = 0xFF;
constexpr int IDEEP_S8_MAX = 0x7F;
constexpr int IDEEP_S32_MAX = 0x7FFFFFFF;
const std::map<mkldnn::memory::data_type, int> dt_max_map
{
  {mkldnn::memory::data_type::s32, IDEEP_S32_MAX},
  {mkldnn::memory::data_type::s8, IDEEP_S8_MAX},
  {mkldnn::memory::data_type::u8, IDEEP_U8_MAX}
};

enum lowp_kind {
  LOWP_U8S8 = 0,
  LOWP_S8S8 = 1
};

enum rnn_kind {
  RNN_RELU = 0,
  RNN_TANH = 1,
  LSTM = 2,
  GRU = 3
};

/// hide other formats
enum format {
  format_undef = mkldnn_format_undef,
  any = mkldnn_any,
  blocked = mkldnn_blocked,
  x = mkldnn_x,
  nc = mkldnn_nc,
  io = mkldnn_io,
  oi = mkldnn_oi,
  ncw = mkldnn_ncw,
  nwc = mkldnn_nwc,
  oiw = mkldnn_oiw,
  wio = mkldnn_wio,
  nchw = mkldnn_nchw,
  nhwc = mkldnn_nhwc,
  chwn = mkldnn_chwn,
  ncdhw = mkldnn_ncdhw,
  ndhwc = mkldnn_ndhwc,
  oihw = mkldnn_oihw,
  ihwo = mkldnn_ihwo,
  hwio = mkldnn_hwio,
  oidhw = mkldnn_oidhw,
  dhwio = mkldnn_dhwio,
  goihw = mkldnn_goihw,
  goidhw = mkldnn_goidhw,
  hwigo = mkldnn_hwigo,
  ntc = mkldnn_ntc,
  tnc = mkldnn_tnc,
  ldigo = mkldnn_ldigo,
  ldgoi = mkldnn_ldgoi,
  ldgo = mkldnn_ldgo,
  ldsnc = mkldnn_ldsnc,
  rnn_packed = mkldnn_rnn_packed,
  iohw = mkldnn_format_last + 1,
  format_last = iohw + 1
};

/// cpu execution engine only.
struct engine: public mkldnn::engine {
  explicit engine(const mkldnn_engine_t& aengine) = delete;
  engine(engine const&) = delete;
  void operator =(engine const&) = delete;

  /// Singleton CPU engine for all primitives
  static IDEEP_EXPORT engine& cpu_engine();

  inline static format default_format(int ndims) {
    switch(ndims) {
    case 1:
      return format::x;
    case 2:
      return format::nc;
    case 3:
      return format::ncw;
    case 4:
      return format::nchw;
    case 5:
      return format::ncdhw;
    case 6:
      return format::goidhw;
    default:
      return format::format_undef;
    }
  }

private:
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

}

#endif
