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

#include <mutex>
#include <list>
#include <sstream>

namespace ideep {

#define DEFAULT_ALIGNMENT 64

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
  static char *malloc(size_t size) {
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, DEFAULT_ALIGNMENT);
    int rc = ((ptr)? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, DEFAULT_ALIGNMENT, size);
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
  #define GET_PTR(t, p, offset) \
      (reinterpret_cast<t*>(reinterpret_cast<size_t>(p) + \
      static_cast<size_t>(offset)))

  class memory {
  public:
    memory(const char *owner) : alloc_size_(0), free_size_(0),
        alignment_(DEFAULT_ALIGNMENT), seq_(0), owner_(owner) {}

    void *malloc(size_t size) {
      std::lock_guard<std::mutex> lock(mutex_);
      void *ptr;
      int idx = to_index(size);

      if (!free_hashline_[idx].empty()) {
        header_t *head = nullptr;
        std::list<header_t *> &list = free_hashline_[idx];
        typename std::list<header_t *>::iterator it;
        for(it = list.begin(); it != list.end(); ++it) {
          if((*it)->size_ == size) {
            head = *it;
            break;
          }
        }
        if (head) {
          list.erase(it);
          void *ptr = static_cast<void *>(head);
          free_size_ -= size;
          return GET_PTR(void, ptr, alignment_);
        }
      }

      // No cached memory
      size_t len = size + alignment_;
      int rc = ::posix_memalign(&ptr, alignment_, len);
      if (rc != 0)
        throw std::invalid_argument("Out of memory");
      header_t *head = static_cast<header_t *>(ptr);
      head->size_ = size;
      head->seq_ = seq_++;
      alloc_size_ += size;
      return GET_PTR(void, ptr, alignment_);
    }

    void free(void *ptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      header_t *head = GET_PTR(header_t, ptr, -alignment_);
      int idx = to_index(head->size_);
      free_hashline_[idx].push_back(head);
      free_size_ += head->size_;
    }
  private:
    inline int to_index(size_t size) {
      std::ostringstream os;
      os << std::hex << "L" << size << "_";
      size_t hash = std::hash<std::string>{}(os.str());
      return hash % MAX_ENTRY;
    }

    typedef struct {
      size_t size_;
      int seq_;
    } header_t;

    static constexpr int MAX_ENTRY = 512;

    size_t alloc_size_;
    size_t free_size_;
    const size_t alignment_;
    std::list<header_t *> free_hashline_[MAX_ENTRY];
    std::mutex mutex_;
    int seq_;
    std::string owner_;
  };

  scratch_allocator() = default;

  // To instance for compuatations
  template<class computation_t = computation>
  static char *malloc(size_t size) {
    // Route default computation type to default allocator
    return allocator::template malloc<computation_t>(size);
  }

  template<class computation_t>
  static void free(void *p) {
    // Route default computation type to default allocator
    return allocator::template free<computation_t>(p);
  }
};

#define SCRATCH_ALLOCATOR_INSTANCE(computation_t) \
static scratch_allocator::memory computation_t##_mpool(#computation_t); \
\
template<> \
char *scratch_allocator::malloc<computation_t>(size_t size) { \
  return static_cast<char *>(computation_t##_mpool.malloc(size)); \
} \
\
template<> \
void scratch_allocator::free<computation_t>(void *ptr) { \
  return computation_t##_mpool.free(ptr); \
}

SCRATCH_ALLOCATOR_INSTANCE(convolution_forward)
SCRATCH_ALLOCATOR_INSTANCE(convolution_backward_data)
SCRATCH_ALLOCATOR_INSTANCE(convolution_backward_weights)
SCRATCH_ALLOCATOR_INSTANCE(lrn_forward)
SCRATCH_ALLOCATOR_INSTANCE(lrn_backward)
SCRATCH_ALLOCATOR_INSTANCE(pooling_forward)
SCRATCH_ALLOCATOR_INSTANCE(pooling_backward)
SCRATCH_ALLOCATOR_INSTANCE(eltwise_forward)
SCRATCH_ALLOCATOR_INSTANCE(eltwise_backward)
SCRATCH_ALLOCATOR_INSTANCE(sum)
SCRATCH_ALLOCATOR_INSTANCE(concat)
SCRATCH_ALLOCATOR_INSTANCE(softmax_forward)
SCRATCH_ALLOCATOR_INSTANCE(softmax_backward)
SCRATCH_ALLOCATOR_INSTANCE(batch_normalization_forward_inference)
SCRATCH_ALLOCATOR_INSTANCE(batch_normalization_forward_training)
SCRATCH_ALLOCATOR_INSTANCE(batch_normalization_backward)
SCRATCH_ALLOCATOR_INSTANCE(inner_product_forward)
SCRATCH_ALLOCATOR_INSTANCE(inner_product_backward_data)
SCRATCH_ALLOCATOR_INSTANCE(inner_product_backward_weights)
SCRATCH_ALLOCATOR_INSTANCE(eltwise_binary)
SCRATCH_ALLOCATOR_INSTANCE(reorder)

}
}

#endif
