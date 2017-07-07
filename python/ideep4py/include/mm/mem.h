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


#pragma once

#include <stdlib.h>
#include <mutex>
#include <list>
#include <functional>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include "utils.h"

using namespace std;
static constexpr int DEFAULT_ALIGNMENT = 64;

typedef enum {
    MPOOL_ANON,
    MPOOL_REORDER,
    MPOOL_ELTWISE_FWD,
    MPOOL_ELTWISE_BWD,
    MPOOL_BN_FWD,
    MPOOL_BN_BWD,
    MPOOL_LRN_FWD,
    MPOOL_LRN_BWD,
    MPOOL_CONV_FWD,
    MPOOL_CONV_BWD,
    MPOOL_POOLING_FWD,
    MPOOL_POOLING_BWD,
    MPOOL_IP_FWD,
    MPOOL_IP_BWD,
    MPOOL_CONCAT_FWD,
    MPOOL_CONCAT_BWD,
} mem_pool_t;

template <std::size_t ALIGNMENT>
class Memory {
public:
    Memory() : alloc_size_(0), free_size_(0), seq_(0) {}
    Memory(const char *name) : alloc_size_(0), free_size_(0)
             , seq_(0), name_(name) {}
    virtual ~Memory() {
        //std::cout << name_ << " alloc size " << alloc_size_ << " free size "
        //    << free_size_ << std::endl;
    }

    void* malloc(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        void *ptr;
        int idx = to_index(size);

        if (!free_hashline_[idx].empty()) {
            block_t *block = nullptr; 
            std::list<block_t *> &list = free_hashline_[idx];
            typename std::list<Memory<ALIGNMENT>::block_t*>::iterator it;
            for(it=list.begin(); it != list.end(); ++it) {
                if((*it)->header_.size_ == size) {
                    block = *it; 
                    break;
                }
            }
            if (block) {
                list.erase(it);
                void *ptr = static_cast<void *>(block);
                free_size_ -= size;
                //std::cout << name_ << " cache alloc seq " << block->header_.seq_ << " size " << block->header_.size_ << std::endl;
                return GET_PTR(void, ptr, ALIGNMENT);
            }
        }
        // No cached memory
        size_t len = size + ALIGNMENT;
        int rc = ::posix_memalign(&ptr, ALIGNMENT, len);
        if (rc != 0) {
            throw std::invalid_argument("Out of memory");
        }
        block_t *block = static_cast<block_t *>(ptr); 
        block->header_.size_ = size;
        alloc_size_ += size;
        //std::cout << name_ << " system alloc seq " << seq_ << " size " << size << std::endl;
        block->header_.seq_ = seq_++;
        return GET_PTR(void, ptr, ALIGNMENT);
    }

    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        //block_t *block = static_cast<block_t *>(ptr - ALIGNMENT);
        block_t *block = GET_PTR(block_t, ptr, -ALIGNMENT);
        int idx = to_index(block->header_.size_);
        free_hashline_[idx].push_back(block);
        free_size_ += block->header_.size_;
        //std::cout << name_ << " free seq " << block->header_.seq_ << " size " << block->header_.size_ << std::endl;
    }

    void epoch() {
    }
private:
    int to_index(std::size_t size) {
        std::string str = long_to_string(size);
        std::size_t hash = std::hash<std::string>{}(str);
        int idx = hash % MAX_ENTRY;
        return idx;
    }

    typedef union _header_str {
        struct {
            std::size_t size_;
            int seq_;
        };
        char pad_[ALIGNMENT];
    } header_t;

    typedef struct _block_str {
        header_t header_;
        char data_[];
    } block_t;

    static constexpr int MAX_ENTRY = 512;

    std::size_t alloc_size_;
    std::size_t free_size_;
    std::list<block_t *> free_hashline_[MAX_ENTRY];
    std::mutex mutex_;
    int seq_;
    std::string name_;
};

void* dnn_malloc(size_t size, mem_pool_t pool=MPOOL_ANON);
void dnn_free(void *p, mem_pool_t pool=MPOOL_ANON);

// Just grab it from MKL-DNN
namespace avx {
#if 1
    inline void* malloc(size_t size, int alignment) {
        return ::dnn_malloc(size);
    }
    inline void free(void* p) { ::dnn_free(p); }
#else
    inline void* malloc(size_t size, int alignment) {
        void *ptr;
        int rc = ::posix_memalign(&ptr, alignment, size);
        return (rc == 0) ? ptr : 0;
    }
    inline void free(void* p) { ::free(p); }
#endif

    struct compatible {
        enum { default_alignment = DEFAULT_ALIGNMENT };
        static void* operator new(size_t sz) {
            return malloc(sz, default_alignment);
        }
        static void* operator new(size_t sz, void* p) { (void)sz; return p; }
        static void* operator new[](size_t sz) {
            return malloc(sz, default_alignment);
        }
        static void operator delete(void* p) {
            free(p); }
        static void operator delete[](void* p) {
            free(p); }
    };

    struct byte: public compatible {
        char q;
    };
}

class Allocator {
    public:
        static std::shared_ptr<avx::byte> malloc(size_t len, mem_pool_t mpool);
        static std::shared_ptr<avx::byte> malloc(vector<int> dims, int element_sz, mem_pool_t mpool);
};
