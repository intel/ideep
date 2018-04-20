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


#ifndef DL_COMPRESSION_IMPL_HPP
#define DL_COMPRESSION_IMPL_HPP


#include <stdint.h>
#include <stdio.h>

// Disable the copy and assignment operator for a class

#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
    classname(const classname&);\
    classname& operator=(const classname&)

#define DL_COMP_BLOCK_NUM 256

#define DL_COMP_HEAD_MAGIC 0xdeadbeef

typedef struct __attribute__((__packed__))
{
    int magic;
    int payloadLen;
    int exponent;
} dl_comp_head;

bool dl_comp_check_avx512_supported(void);

void dl_comp_float_vector_add(const float *invec, float *inoutvec, size_t count);

void dl_comp_avx512_float_vector_add(const float *invec, float *inoutvec, size_t count);

void dl_comp_int8_vector_add(const int8_t *invec, int8_t *inoutvec, size_t count);

void dl_comp_avx512_int8_vector_add(const int8_t *invec, int8_t *inoutvec, size_t count);

dl_comp_return_t compress_helper(float *src, int8_t *dst, float *diff, dl_comp_method_t method, size_t count);

dl_comp_return_t decompress_helper(const int8_t *src, float *dst, dl_comp_method_t method);

/*
 * Abstract base class for quantization
 */
class DLCompressBase {

public:
    DLCompressBase() = default;
    // Compress with error feedback
    virtual dl_comp_return_t compress_buffer(float *src, int8_t *dst, float *diff, size_t count, bool inPlace = false) = 0;
    // Compress  without error feedback
    virtual dl_comp_return_t compress_buffer(float *src, int8_t *dst, size_t count, bool inPlace = false) = 0;
    virtual dl_comp_return_t decompress_buffer(const int8_t *src, float *dst, size_t blockCount) = 0;
    // virtual size_t get_dataCount_in_compressed_buffer(const int8_t *src, size_t blockCount) = 0;
    // virtual dl_comp_return_t compress_sum(const int8_t *invec, int8_t *inoutvec, size_t blockCount) = 0;
    virtual dl_comp_return_t compress_sum2(const int8_t *invec, int8_t *inoutvec, size_t blockCount) = 0;
    // virtual void dump_compressed_buffer(const int8_t *src, size_t blockCount) = 0;
    // virtual bool check_compressed_buffer(const float *comp1, const int8_t *comp2, const float *diff, size_t blockCount) = 0;
    virtual ~DLCompressBase(void) {};

public:
    static DLCompressBase* get_compression_instance(dl_comp_method_t method);

    DISABLE_COPY_AND_ASSIGN(DLCompressBase);
};


class DLCompressDFP : public DLCompressBase {

    friend class DLCompressBase;
public:
    virtual ~DLCompressDFP(void) {};
    virtual dl_comp_return_t compress_buffer(float *src, int8_t *dst, float *diff, size_t count, bool inPlace = false);
    virtual dl_comp_return_t compress_buffer(float *src, int8_t *dst, size_t count, bool inPlace = false);
    virtual dl_comp_return_t decompress_buffer(const int8_t *src, float *dst, size_t blockCount);
    // virtual size_t get_dataCount_in_compressed_buffer(const int8_t *src, size_t blockCount);
    // virtual dl_comp_return_t compress_sum(const int8_t *invec, int8_t *inoutvec, size_t blockCount);
    virtual dl_comp_return_t compress_sum2(const int8_t *invec, int8_t *inoutvec, size_t blockCount);
    // virtual void dump_compressed_buffer(const int8_t *src, size_t blockCount);
    // virtual bool check_compressed_buffer(const float *comp1, const int8_t *comp2, const float *diff, size_t blockCount);

public:
    static dl_comp_return_t compress_block(float *src, int8_t *dst, float *diff, size_t count, int *scale);
    static dl_comp_return_t decompress_block(const int8_t *src, float *dst, size_t count, int scale);
    static dl_comp_return_t compress_block_sum(const int8_t *invec, int8_t *inoutvec);
    static dl_comp_return_t avx512_decompress_block(const int8_t *src, float *dst, size_t count, int scale);
    static dl_comp_return_t avx512_compress_block(float *src, int8_t *dst, float *diff, size_t count, int *scale);
    static dl_comp_return_t compress_block_sum2(const int8_t *invec, int8_t *inoutvec);

private:
    DLCompressDFP(): avx512_enabled_(dl_comp_check_avx512_supported()) {};

private:
    bool avx512_enabled_;

DISABLE_COPY_AND_ASSIGN(DLCompressDFP);
};


#endif /* DL_COMPRESSION_IMPL_HPP */
