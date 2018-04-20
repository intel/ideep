#ifndef _FAST_MATH_HPP_
#define _FAST_MATH_HPP_
#include <assert.h>
#include <string>
#include <type_traits>
#include <immintrin.h>
#include "abstract_types.hpp"
namespace ideep {
namespace utils {

typedef enum {
    isa_any,
    sse42,
    avx2,
    avx512_common,
    avx512_core,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;

template <cpu_isa_t T> struct TypeMap {};
#define MAP_T(v, F, I) \
  template<> struct TypeMap<v> { using tF = F; using tI = I;};
MAP_T(avx2, __m256, __m256i)
#undef MAP_T
#define TF  typename TypeMap<isa>::tF
#define TI  typename TypeMap<isa>::tI

template<cpu_isa_t isa = avx2>
class fast_math {
  static constexpr int thread_hold = 1024;
public:

  template<typename T>
  static inline unsigned get_vec_sz() {
    switch (isa) {
    case avx2:
      return 256/8/sizeof(T);
    case avx512_common:
    case avx512_core:
      return 512/8/sizeof(T);
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return 0;
    }
  }

  // Move this to utils
  template<typename T>
  static inline TI size_to_mask(unsigned nres) {
    constexpr int on = -1;
    constexpr int off = 0;
    switch (isa) {
    case avx2:
      assert(nres < 8 && nres > 0);
      switch(nres) {
      case 1:
        return _mm256_set_epi32(off, off, off, off, off, off, off, on);
      case 2:
        return _mm256_set_epi32(off, off, off, off, off, off, on, on);
      case 3:
        return _mm256_set_epi32(off, off, off, off, off, on, on, on);
      case 4:
        return _mm256_set_epi32(off, off, off, off, on, on, on, on);
      case 5:
        return _mm256_set_epi32(off, off, off, on, on, on, on, on);
      case 6:
        return _mm256_set_epi32(off, off, on, on, on, on, on, on);
      case 7:
        return _mm256_set_epi32(off, on, on, on, on, on, on, on);
      default:
        return _mm256_set_epi32(off, off, off, off, off, off, off, off);
      }
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }

#define BIN_OPS(name) \
  template<typename T> \
  static TF name##_ps (TF v1, TF v2) { \
    switch (isa) {  \
    case avx2:  \
      return _mm256_##name##_ps(v1, v2); \
    default: \
      throw error(mkldnn_unimplemented, "Not implemented!"); \
      return set1_ps(0.f); \
    } \
  }

  BIN_OPS(add);
  BIN_OPS(mul);
  BIN_OPS(div);
#undef BIN_OPS

  template<typename T>
  static TF set1_ps (const T v) {
    switch (isa) {
    case avx2:
      return _mm256_set1_ps(v);
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return set1_ps(0.f);
    }
  }

  template<typename T>
  static TF sqrt_ps (TF v) {
    switch (isa) {
    case avx2:
      return _mm256_sqrt_ps(v);
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return set1_ps(0.f);
    }
  }

  template<typename T>
  static TF load_ps (const T *src) {
    switch (isa) {
    case avx2:
      return _mm256_load_ps(src);
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return set1_ps(0.f);
    }
  }

  template<typename T>
  static TF maskload_ps (const T *src, TI mask) {
    switch (isa) {
    case avx2:
      return _mm256_maskload_ps(src, mask);
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return set1_ps(0.f);
    }
  }

  template<typename T>
  static void store_ps (T *dst, TF v) {
    switch (isa) {
    case avx2:
      _mm256_store_ps(dst, v);
      return;
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return;
    }
  }

  template<typename T>
  static void maskstore_ps (T *dst, TI mask, TF v) {
    switch (isa) {
    case avx2:
      _mm256_maskstore_ps(dst, mask, v);
      return;
    default:
      throw error(mkldnn_unimplemented, "Not implemented!");
      return;
    }
  }

  // Unary ops
  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void single_thread_vecwise_unary_op(
      T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    auto vec_sz = get_vec_sz<T>();
    auto nvec = nelems / vec_sz;
    auto nres = nelems % vec_sz;
    for (unsigned vec = 0; vec < nvec; vec ++, src+=vec_sz, dst+=vec_sz) {
      TF vmm1 = load_ps(src);
      vmm1 = op(vmm1);
      store_ps(dst, vmm1);
    }

    if (nres != 0) {
      TI mask = size_to_mask<T>(nres);
      TF vmm1 = maskload_ps(src, mask);
      vmm1 = op_mask(vmm1, mask);
      maskstore_ps(dst, mask, vmm1);
    }
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void vecwise_unary_op (T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    if (nelems < thread_hold)
      single_thread_vecwise_unary_op(dst, src, nelems, op, op_mask);
  }

  template<class elem_t = float>
  static void inv_square_var(float epsilon,
      const elem_t* inv_sqrt_var, elem_t* variance, unsigned nelems) {
    if (isa == avx2) {
      if (std::is_same<elem_t, float>::value) {
        const float *src = reinterpret_cast<const float *>(inv_sqrt_var);
        float *dst = reinterpret_cast<float *>(variance);

        TF ones = set1_ps(1.f);
        TF epsilones = set1_ps(epsilon);
        auto vec_inv_square = [ones, epsilones] (TF vmm1) {
          vmm1 = mul_ps(vmm1, vmm1);
          vmm1 = add_ps(vmm1, epsilones);
          vmm1 = div_ps(ones, vmm1);
          return vmm1;
        };
        auto mask_vec_inv_square =
          [ones, epsilones] (TF vmm1, TI) {
            vmm1 = mul_ps(vmm1, vmm1);
            vmm1 = add_ps(vmm1, epsilones);
            vmm1 = div_ps(ones, vmm1);
            return vmm1;
        };
        vecwise_unary_op(dst, src, nelems, vec_inv_square, mask_vec_inv_square);
      } else {
        throw error(mkldnn_unimplemented, "Not implemented!");
      }
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }

  template<class elem_t = float>
  static void inv_sqrt_var(float epsilon,
      const void* variance, void* inv_sqrt_var, unsigned nelems) {
    if (isa == avx2) {
      if (std::is_same<elem_t, float>::value) {
        const float *src =
          reinterpret_cast<const float *>(variance);
        float *dst =
          reinterpret_cast<float *>(inv_sqrt_var);

        unsigned nvec = nelems / 8;
        unsigned nres = nelems % 8;
        TF ones = set1_ps(1.f);
        TF epsilones = set1_ps(epsilon);
        for (unsigned vec = 0; vec < nvec; vec ++, src+=8, dst+=8) {
          TF vmm1 = load_ps(src);
          vmm1 = add_ps(vmm1, epsilones);
          vmm1 = sqrt_ps(vmm1);
          vmm1 = div_ps(ones, vmm1);
          store_ps(dst, vmm1);
        }

        if (nres != 0) {
          TI mask = size_to_mask<elem_t>(nres);
          TF vmm1 = maskload_ps(src, mask);
          vmm1 = add_ps(vmm1, epsilones);
          vmm1 = sqrt_ps(vmm1);
          vmm1 = div_ps(ones, vmm1);
          maskstore_ps(dst, mask, vmm1);
        }
      } else {
        throw error(mkldnn_unimplemented, "Not implemented!");
      }
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }

  // binary ops
  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void single_thread_vecwise_binary_op(
      T *dst, const T *src1, const T *src2, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    auto vec_sz = get_vec_sz<T>();
    auto nvec = nelems / vec_sz;
    auto nres = nelems % vec_sz;
    for (unsigned vec = 0; vec < nvec;
        vec ++, src1+=vec_sz, src2+=vec_sz, dst+=vec_sz) {
      TF vmm1 = load_ps(src1);
      TF vmm2 = load_ps(src2);
      vmm2 = op(vmm1, vmm2);
      store_ps(dst, vmm2);
    }

    if (nres != 0) {
      TI mask = size_to_mask<T>(nres);
      TF vmm1 = maskload_ps(src1, mask);
      TF vmm2 = maskload_ps(src2, mask);
      vmm2 = op_mask(vmm1, vmm2);
      maskstore_ps(dst, mask, vmm2);
    }
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void vecwise_binary_op (T *dst, const T *src1, const T *src2,
      size_t nelems, vec_op op, vec_op_mask op_mask) {
    if (nelems < thread_hold)
      single_thread_vecwise_binary_op(dst, src1, src2, nelems, op, op_mask);
  }

  template<class elem_t = float>
  static void add(elem_t *dst, const elem_t *src1, const elem_t *src2,
      unsigned nelems) {
    if (std::is_same<elem_t, float>::value) {
      auto op = [] (TF vmm1, TF vmm2) {
        vmm1 = add_ps<elem_t>(vmm1, vmm2);
        return vmm1;
      };
      vecwise_binary_op(dst, src1, src2, nelems, op, op);
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }
};
}
}
#endif
