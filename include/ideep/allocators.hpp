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

#include <sstream>

namespace ideep {
namespace utils {

class allocator {
public:
  constexpr static size_t tensor_memalignment = 4096;
  allocator() = default;

  template<class computation_t = void>
  static char *malloc(size_t size) {
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, tensor_memalignment);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, tensor_memalignment, size);
#endif /* _WIN32 */
    return (rc == 0) ? (char*)ptr : nullptr;
  }

  template<class computation_t = void>
  static void free(void *p) {
#ifdef _WIN32
    _aligned_free((void*)p);
#else
    ::free((void*)p);
#endif /* _WIN32 */
  }
};

}
}
#endif
