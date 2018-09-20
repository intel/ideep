#ifndef _UTILS_CPP
#define _UTILS_CPP

#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num()  0
#define omp_in_parallel()     0
#endif

namespace ideep {
namespace utils {

// Fast alternative to heavy string method
using bytestring = std::string;


inline bytestring to_bytes(const int arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
#ifndef __AVX__
  if (arg == 0)
    return bytestring();

  auto len = sizeof(arg) - (__builtin_clz(arg) / 8);
#else
  unsigned int lz;
  asm volatile ("lzcntl %1, %0": "=r" (lz): "r" (arg));
  auto len = sizeof(int) - lz / 8;
#endif

  return bytestring(as_cstring, len);
}

inline bytestring to_bytes(const float arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  return bytestring(as_cstring, sizeof(float));
}

inline bytestring to_bytes(const uint64_t arg) {
  auto as_cstring = reinterpret_cast<const char *>(&arg);
  return bytestring(as_cstring, sizeof(uint64_t));
}

template <typename T>
inline bytestring to_bytes(const std::vector<T> arg) {
  bytestring bytes;
  bytes.reserve(arg.size() * sizeof(T));

  for (T elems : arg) {
    bytes.append(to_bytes(elems));
    bytes.append(1, 'x');
  }

  bytes.pop_back();

  return bytes;
}

template <typename T, typename =
  typename std::enable_if<std::is_enum<T>::value>::type>
inline bytestring to_bytes(T arg) {
  return std::to_string(static_cast<int>(arg));
}

template <typename T, typename =
  typename std::enable_if< std::is_class<T>::value>::type, typename = void>
inline bytestring to_bytes(const T arg) {
  return arg.to_bytes();
}

enum algorithm {
  F_UNDEF = 0,
  F_CONV_FWD,
  F_RELU_FWD,
  F_BN_FWD,
  F_SUM,
  F_LAST,
};

enum fusion_type {
  FUSION_UNKNOWN = 0,
  FUSION_CONV_RELU,
  FUSION_CONV_SUM,
  FUSION_MAX,
};

template<typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return(a + b - 1) / b;
}
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

inline void fast_memcpy(char* data_o, char *data_i, size_t len)
{
    size_t nelems_float = len / 4;
    size_t nelems_char = len % 4;
    const int block_size = 16;
    const auto num_blocks_float = nelems_float / block_size;
    const auto rem_elems_float =  nelems_float % block_size;
    float* output_f = (float*)data_o;
    float* input_f = (float*) data_i;
    char* output_c = (char*) data_o;
    char* input_c = (char*) data_i;
#   pragma omp parallel
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        balance211(num_blocks_float, nthr, ithr, start, end);
        start = start * block_size;
        end = end * block_size;
#       pragma omp simd
        for (size_t e = start; e < end; ++e) {
            output_f[e] = input_f[e];
        }
        if (rem_elems_float != 0 && ithr ==  nthr -1 )  {
            for (auto e = nelems_float - rem_elems_float; e < nelems_float; ++e) {
                output_f[e] = input_f[e];
            }
        }
        if (nelems_char != 0 && ithr ==  nthr -1){
            for (auto e = nelems_float*4; e < len; ++e) {
                output_c[e] = input_c[e];
            }
        }
    }
    return;
}

template<typename data_type_t>
inline void fast_memset(data_type_t *data_o, data_type_t val, size_t len)
{
    size_t nelems_float = len;
    const int block_size = 16;
    const auto num_blocks_float = nelems_float / block_size;
    const auto rem_elems_float =  nelems_float % block_size;
    float *output_f = (float *)data_o;
#   pragma omp parallel
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        balance211(num_blocks_float, nthr, ithr, start, end);
        start = start * block_size;
        end = end * block_size;
#       pragma omp simd
        for (size_t e = start; e < end; ++e) {
            output_f[e] = val;
        }
        if (rem_elems_float != 0 && ithr ==  nthr -1 )  {
            for (auto e = nelems_float - rem_elems_float; e < nelems_float; ++e) {
                output_f[e] = val;
            }
        }
    }
    return;
}

}
}
#endif
