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


#ifndef IDEEP_ALLOCATOR_HPP
#define IDEEP_ALLOCATOR_HPP

namespace ideep {

struct computation;
struct convolution_forward;
struct convolution_backward_data;
struct convolution_backward_weights;
struct lrn_forward;
struct lrn_backward;
struct pooling_forward;
struct pooling_backward;
struct eltwise_forward;
struct eltwise_backward;
struct sum;
struct concat;
struct softmax_forward;
struct softmax_backward;
struct batch_normalization_forward_inference;
struct batch_normalization_forward_training;
struct batch_normalization_backward;
struct inner_product_forward;
struct inner_product_backward_data;
struct inner_product_backward_weights;
struct eltwise_binary;
struct reorder;

namespace utils {

class allocator {
public:
  allocator() = default;

  template<class computation_t = computation>
  static char *malloc(size_t size, size_t alignment) {
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ((ptr)? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif /* _WIN32 */
    return (rc == 0) ? (char*)ptr : nullptr;
  }

  template<class computation_t = computation>
  static void free(void *p) {
#ifdef _WIN32
    _aligned_free((void*)p);
#else
    ::free((void*)p);
#endif /* _WIN32 */
  }
};

// Default SA implementation (by computation)
class scratch_allocator {
public:
  scratch_allocator() = default;

  #define NO_SCRATCH "no scratch"

  template<typename key_t>
  static char *malloc(size_t size, size_t alignment, key_t key) {
    // TODO: Scratch Allocator Implementation
    return nullptr;
  }

  // To instance for compuatations
  template<class computation_t>
  static char *malloc(size_t size, size_t alignment) {
    // Route default computation type to default allocator
    return allocator::template malloc<>(size, alignment);
  }

  template<class computation_t>
  static void free(void *p) {
    // Route default computation type to default allocator
    return allocator::template free<>(p);
  }
};

#define SCRATCH_ALLOCATOR_INSTANCE(COMPUTATION, COMPUTATION_NAME) \
template<> \
char *scratch_allocator::malloc<COMPUTATION>(size_t size, size_t aligment) { \
  return scratch_allocator::template malloc<sa_mem_pool_t>( \
      size, aligment, COMPUTATION_NAME); \
}

typedef enum {
    SA_MPOOL_ANON,
    SA_MPOOL_CONV_FWD,
    SA_MPOOL_CONV_BWD_DATA,
    SA_MPOOL_CONV_BWD_WEIGHTS,
    SA_MPOOL_LRN_FWD,
    SA_MPOOL_LRN_BWD,
    SA_MPOOL_POOLING_FWD,
    SA_MPOOL_POOLING_BWD,
    SA_MPOOL_ELTWISE_FWD,
    SA_MPOOL_ELTWISE_BWD,
    SA_MPOOL_SUM,
    SA_MPOOL_CONCAT,
    SA_MPOOL_SOFTMAX_FWD,
    SA_MPOOL_SOFTMAX_BWD,
    SA_MPOOL_BN_FWD_INF,
    SA_MPOOL_BN_FWD_TRN,
    SA_MPOOL_BN_BWD,
    SA_MPOOL_IP_FWD,
    SA_MPOOL_IP_BWD_DATA,
    SA_MPOOL_IP_BWD_WEIGHTS,
    SA_MPOOL_ELTWISE_BIN,
    SA_MPOOL_REORDER,
} sa_mem_pool_t;

SCRATCH_ALLOCATOR_INSTANCE(
    convolution_forward,                   SA_MPOOL_CONV_FWD          )
SCRATCH_ALLOCATOR_INSTANCE(
    convolution_backward_data,             SA_MPOOL_CONV_BWD_DATA     )
SCRATCH_ALLOCATOR_INSTANCE(
    convolution_backward_weights,          SA_MPOOL_CONV_BWD_WEIGHTS  )
SCRATCH_ALLOCATOR_INSTANCE(
    lrn_forward,                           SA_MPOOL_LRN_FWD           )
SCRATCH_ALLOCATOR_INSTANCE(
    lrn_backward,                          SA_MPOOL_LRN_BWD           )
SCRATCH_ALLOCATOR_INSTANCE(
    pooling_forward,                       SA_MPOOL_POOLING_FWD       )
SCRATCH_ALLOCATOR_INSTANCE(
    pooling_backward,                      SA_MPOOL_POOLING_BWD       )
SCRATCH_ALLOCATOR_INSTANCE(
    eltwise_forward,                       SA_MPOOL_ELTWISE_FWD       )
SCRATCH_ALLOCATOR_INSTANCE(
    eltwise_backward,                      SA_MPOOL_ELTWISE_BWD       )
SCRATCH_ALLOCATOR_INSTANCE(
    sum,                                   SA_MPOOL_SUM               )
SCRATCH_ALLOCATOR_INSTANCE(
    concat,                                SA_MPOOL_CONCAT            )
SCRATCH_ALLOCATOR_INSTANCE(
    softmax_forward,                       SA_MPOOL_SOFTMAX_FWD       )
SCRATCH_ALLOCATOR_INSTANCE(
    softmax_backward,                      SA_MPOOL_SOFTMAX_BWD       )
SCRATCH_ALLOCATOR_INSTANCE(
    batch_normalization_forward_inference, SA_MPOOL_BN_FWD_INF        )
SCRATCH_ALLOCATOR_INSTANCE(
    batch_normalization_forward_training,  SA_MPOOL_BN_FWD_TRN        )
SCRATCH_ALLOCATOR_INSTANCE(
    batch_normalization_backward,          SA_MPOOL_BN_BWD            )
SCRATCH_ALLOCATOR_INSTANCE(
    inner_product_forward,                 SA_MPOOL_IP_FWD            )
SCRATCH_ALLOCATOR_INSTANCE(
    inner_product_backward_data,           SA_MPOOL_IP_BWD_DATA       )
SCRATCH_ALLOCATOR_INSTANCE(
    inner_product_backward_weights,        SA_MPOOL_IP_BWD_WEIGHTS    )
SCRATCH_ALLOCATOR_INSTANCE(
    eltwise_binary,                        SA_MPOOL_ELTWISE_BIN       )
SCRATCH_ALLOCATOR_INSTANCE(
    reorder,                               SA_MPOOL_REORDER           )

}
}

#endif
