#ifndef IDEEP_ABSTRACT_TYPES_HPP
#define IDEEP_ABSTRACT_TYPES_HPP

#include <dnnl.h>
#include <dnnl.hpp>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "allocators.hpp"

namespace ideep {

#ifdef _WIN32
#define IDEEP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define IDEEP_EXPORT __attribute__((__visibility__("default")))
#else
#define IDEEP_EXPORT
#endif

using error = dnnl::error;
using memory = dnnl::memory;
using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = dnnl::query;
using kind = dnnl::primitive::kind;
using prop_kind = dnnl::prop_kind;
using algorithm = dnnl::algorithm;
using batch_normalization_flag = dnnl::normalization_flags;
using scale_t = std::vector<float>;
using zero_point_t = std::vector<int32_t>;
using exec_args = std::unordered_map<int, memory>;
using rnn_direction = dnnl::rnn_direction;

// for computation cache
using key_t = std::string;

#ifndef NDEBUG
#define IDEEP_ENFORCE(condition, message)                                \
  do {                                                                   \
    error::wrap_c_api(                                                   \
        (condition) ? dnnl_success : dnnl_invalid_arguments, (message)); \
  } while (false)
#else
#define IDEEP_ENFORCE(condition, message)
#endif

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define IDEEP_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define IDEEP_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define IDEEP_LIKELY(expr) (expr)
#define IDEEP_UNLIKELY(expr) (expr)
#endif

#define IDEEP_CHECK(condition, message)                                  \
  if (IDEEP_UNLIKELY(!(condition))) {                                    \
    error::wrap_c_api(dnnl_invalid_arguments, (message));                \
  }

const scale_t IDEEP_DEF_SCALE{1.0f};
const zero_point_t IDEEP_DEF_ZP{0};
const scale_t IDEEP_EMPTY_SCALE;
const zero_point_t IDEEP_EMPTY_ZP;

enum lowp_kind {
  u8s8 = 0,
  s8s8 = 1,
  LOWP_U8S8 = u8s8,
  LOWP_S8S8 = s8s8,
};

enum rnn_kind { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

static bool has_bf16_type_support() {
  // for v1.8
  // static bool support_bf16 = isa >= dnnl::cpu_isa::avx512_core
  //                           && isa != dnnl::cpu_isa::avx2_vnni;
  static bool support_bf16 =
      dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx2_vnni_2;
  return support_bf16;
}

static bool check_isa_is_avx2_vnni_2() {
  static bool is_avx2_vnni_2 =
      dnnl::get_effective_cpu_isa() == dnnl::cpu_isa::avx2_vnni_2;
  return is_avx2_vnni_2;
}

static bool has_fp16_type_support() {
  static bool support_fp16 =
      dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_fp16 ||
      dnnl::get_effective_cpu_isa() == dnnl::cpu_isa::avx2_vnni_2;
  return support_fp16;
}

static bool has_amx_fp16_support() {
  static bool support_amx_fp16 =
      dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_amx_fp16;
  return support_amx_fp16;
}

/// cpu execution engine only.
struct engine : public dnnl::engine {
  friend class tensor;

  /// Singleton CPU engine for all primitives
  static IDEEP_EXPORT engine& cpu_engine();

  /// Singleton GPU engine for all primitives
  static IDEEP_EXPORT engine& gpu_engine();

  engine(kind akind = kind::cpu, size_t index = 0)
      : dnnl::engine(akind, index),
        malloc(utils::allocator::malloc),
        free(utils::allocator::free) {}

  void set_allocator(
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    this->malloc = malloc;
    this->free = free;
  }

 private:
  std::function<void*(size_t)> malloc;
  std::function<void(void*)> free;
};

/// A default stream
struct stream : public dnnl::stream {
  static dnnl::stream& default_stream() {
    static dnnl::stream s(engine::cpu_engine());
    return s;
  }
};
} // namespace ideep

#endif
