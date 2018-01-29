#ifndef _FAST_MATH_HPP_
#define _FAST_MATH_HPP_
#include <assert.h>
#include <string>
#include <type_traits>
#include <immintrin.h>
#include <ideep/abstract_types.hpp>
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

template<cpu_isa_t isa = avx2>
class fast_math {
  static constexpr int thread_hold = 1024;
public:
  // Move this to utils
  static inline __m256i size_to_mask(unsigned nres) {
    assert(nres < 8 && nres > 0);
    constexpr int on = -1;
    constexpr int off = 0;
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
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void single_thread_vecwise_unary_op(
      T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    constexpr unsigned vec_sz = 256/8/sizeof(float);
    auto nvec = nelems / vec_sz;
    auto nres = nelems % vec_sz;
    for (unsigned vec = 0; vec < nvec; vec ++, src+=vec_sz, dst+=vec_sz) {
      __m256 vmm1 = _mm256_load_ps(src);
      vmm1 = op(vmm1);
      _mm256_store_ps(dst, vmm1);
    }

    if (nres != 0) {
      __m256i mask = size_to_mask(nres);
      __m256 vmm1 = _mm256_maskload_ps(src, mask);
      vmm1 = op_mask(vmm1, mask);
      _mm256_maskstore_ps(dst, mask, vmm1);
    }
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void vecwise_unary_op (T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    if (nelems < thread_hold)
      single_thread_vecwise_unary_op(dst, src, nelems, op, op_mask);
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void single_thread_vecwise_binary_op(
      T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    constexpr unsigned vec_sz = 256/8/sizeof(float);
    auto nvec = nelems / vec_sz;
    auto nres = nelems % vec_sz;
    for (unsigned vec = 0; vec < nvec; vec ++, src+=vec_sz, dst+=vec_sz) {
      __m256 vmm1 = _mm256_load_ps(src);
      __m256 vmm2 = _mm256_load_ps(dst);
      vmm2 = op(vmm1, vmm2);
      _mm256_store_ps(dst, vmm2);
    }

    if (nres != 0) {
      __m256i mask = size_to_mask(nres);
      __m256 vmm1 = _mm256_maskload_ps(src, mask);
      __m256 vmm2 = _mm256_maskload_ps(src, mask);
      vmm2 = op_mask(vmm1, vmm2, mask);
      _mm256_maskstore_ps(dst, mask, vmm2);
    }
  }

  template<typename vec_op, typename vec_op_mask, typename T = float>
  static inline void vecwise_binary_op (T *dst, const T *src, size_t nelems,
      vec_op op, vec_op_mask op_mask) {
    if (nelems < thread_hold)
      single_thread_vecwise_binary_op(dst, src, nelems, op, op_mask);
  }

  template<class elem_t = float>
  static void inv_square_var(float epsilon,
      const elem_t* inv_sqrt_var, elem_t* variance, unsigned nelems) {
    if (isa == avx2) {
      if (std::is_same<elem_t, float>::value) {
        const float *src = reinterpret_cast<const float *>(inv_sqrt_var);
        float *dst = reinterpret_cast<float *>(variance);

        __m256 ones = _mm256_set1_ps(1.f);
        __m256 epsilones = _mm256_set1_ps(epsilon);
        auto vec_inv_square = [ones, epsilones] (__m256 vmm1) {
          vmm1 = _mm256_mul_ps(vmm1, vmm1);
          vmm1 = _mm256_add_ps(vmm1, epsilones);
          vmm1 = _mm256_div_ps(ones, vmm1);
          return vmm1;
        };
        auto mask_vec_inv_square = [ones, epsilones] (__m256 vmm1, __m256i) {
          vmm1 = _mm256_mul_ps(vmm1, vmm1);
          vmm1 = _mm256_add_ps(vmm1, epsilones);
          vmm1 = _mm256_div_ps(ones, vmm1);
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
        __m256 ones = _mm256_set1_ps(1.f);
        __m256 epsilones = _mm256_set1_ps(epsilon);
        for (unsigned vec = 0; vec < nvec; vec ++, src+=8, dst+=8) {
          __m256 vmm1 = _mm256_load_ps(src);
          vmm1 = _mm256_add_ps(vmm1, epsilones);
          vmm1 = _mm256_sqrt_ps(vmm1);
          vmm1 = _mm256_div_ps(ones, vmm1);
          _mm256_store_ps(dst, vmm1);
        }

        if (nres != 0) {
          __m256i mask = size_to_mask(nres);
          __m256 vmm1 = _mm256_maskload_ps(src, mask);
          vmm1 = _mm256_add_ps(vmm1, epsilones);
          vmm1 = _mm256_sqrt_ps(vmm1);
          vmm1 = _mm256_div_ps(ones, vmm1);
          _mm256_maskstore_ps(dst, mask, vmm1);
        }
      } else {
        throw error(mkldnn_unimplemented, "Not implemented!");
      }
    } else {
      throw error(mkldnn_unimplemented, "Not implemented!");
    }
  }
};
}
}
#endif
