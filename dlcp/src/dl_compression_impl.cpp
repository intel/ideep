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


#include <cmath>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <string.h>

#include "dl_compression.h"
#include "dl_compression_impl.hpp"
#include "dl_compression_util.hpp"

bool g_avx512_supported = dl_comp_check_avx512_supported();

bool dl_comp_check_avx512_supported()
{
    const unsigned long avx512_features = (_FEATURE_AVX512F | _FEATURE_AVX512CD | _FEATURE_AVX512VL | _FEATURE_AVX512BW);
    return _may_i_use_cpu_feature( avx512_features );
}

DLCompressBase* DLCompressBase::get_compression_instance(dl_comp_method_t method)
{
    DLCompressBase *pInstance = NULL;
    static DLCompressDFP dfpInstance;

    switch(method) {
        case DL_COMP_DFP:
            pInstance = &dfpInstance;
            break;

        case DL_COMP_NONE:

        default:
            pInstance = NULL;
            DLCP_LOG(INFO, "Unsupported Compression Method");
    }

    return pInstance;
}

dl_comp_return_t DLCompressDFP::compress_block(float *src, int8_t *dst, float *diff, size_t count, int *scale)
{
    // Do quantization
    // only handle float buffer as src and int8_t as dst
    float max_abs = 0.;
    float max_abs_log2 = 0.;
    float d_value;
    int8_t decomp_value = 0;

    if (NULL != diff) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < count; ++i) {
            src[i] += diff[i];
        }
    }

    for (size_t i = 0; i < count; ++i) {
        max_abs = std::max(max_abs, std::abs(src[i]));
    }

    max_abs_log2 = std::log2f(max_abs);
    // If max_log2 is equal to -inf, this means max_abs is 0.
    // In this case, we set scale as 0.
    if (max_abs_log2 * (-1.0) == std::numeric_limits<float>::infinity()) {
        *scale = 0;
    } else {
        *scale = 8*sizeof(int8_t) - ((int)std::ceil(max_abs_log2) + 1);
    }
     
    float pow2_scale = std::pow(2, *scale);

    for (size_t i = 0; i < count; ++i) {
        // It's corner case that the result value of src[i]*pow2_scale will be
        // bigger than 127.5f. The value will rounded up to 128. This is out of range
        // of int8_t. (-128 - 127) So we set it as 127.
        
        float round_value = std::round(src[i]*pow2_scale);
        if (round_value <= 127.0f) {
            decomp_value = (int8_t)round_value;
        } else {
            decomp_value = 127;
        }
        if (NULL != diff) {
            d_value = ((float)decomp_value) / pow2_scale;
            diff[i] = src[i] - d_value;
        }
        dst[i] = decomp_value;
    }

    return DL_COMP_OK;
}

dl_comp_return_t DLCompressDFP::avx512_compress_block(float *src, int8_t *dst, float *diff, size_t count, int *scale)
{
    // If count is smaller than 16 we use non-avx512 implementation
    // 16 is the element number which one avx512 register can hold
    if (count < DL_COMP_BLOCK_NUM) {
        return compress_block(src, dst, diff, count, scale);
    }
   

    DLCP_ASSERT(count % 16 == 0, "count can't be divided by 16!");

    // Do quantization
    // Error FeedBack
    if (NULL != diff) {
        dl_comp_avx512_float_vector_add(diff, src, count);
    }

    float max_abs = 0.;
    float max_abs_log2 = 0.;
    size_t group_size = 16;
    __m512 max_vec = _mm512_set1_ps(0.0f);

    for (size_t idx = 0; idx < count; idx += group_size) {
        __m512 float_vec     = _mm512_loadu_ps(src+idx);
        __m512 float_abs_vec = _mm512_abs_ps(float_vec);
        __mmask16 cmp_mask = _mm512_cmp_ps_mask(max_vec, float_abs_vec, _CMP_GE_OS);
        max_vec = _mm512_mask_mov_ps(float_abs_vec, cmp_mask, max_vec);
    }

    max_abs = _mm512_reduce_max_ps(max_vec);

    max_abs_log2 = std::log2f(max_abs);
    // If max_log2 is equal to -inf, this means max_abs is 0.
    // In this case, we set scale as 0.
    if (max_abs_log2 * (-1.0) == std::numeric_limits<float>::infinity()) {
        *scale = 0;
    } else {
        *scale = 8*sizeof(int8_t) - ((int)std::ceil(max_abs_log2) + 1);
    }

    float pow2_scale = std::pow(2, *scale);

    float pow2_scale_inv = 1.0f / std::pow(2, *scale);
    __m512 pow2_scale_v = _mm512_set1_ps(pow2_scale);
    __m512 pow2_scale_inv_v = _mm512_set1_ps(pow2_scale_inv);
    __mmask16 mask = _mm512_int2mask(0xFFFF);
    float *f32_diff;
    for (size_t idx = 0; idx < count; idx += group_size) {
        float *f32_src      = src + idx;
        int8_t *i8_dst      = dst + idx;
        __m512 f32_src_v    = _mm512_loadu_ps(f32_src);
        __m512 f32_result_v = _mm512_mul_ps(f32_src_v, pow2_scale_v);
        __m512i i32_round_v = _mm512_cvt_roundps_epi32(f32_result_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // satruation has already been considered in cvt instruction
        _mm512_mask_cvtsepi32_storeu_epi8(i8_dst, mask, i32_round_v);
        if (NULL != diff) {
            f32_diff     = diff + idx;
            __m512 f32_round_v  = _mm512_cvt_roundepi32_ps(i32_round_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 f32_dequant_v = _mm512_mul_ps(f32_round_v, pow2_scale_inv_v);
            __m512 f32_diff_v    = _mm512_sub_ps(f32_src_v, f32_dequant_v);
             _mm512_storeu_ps(f32_diff, f32_diff_v);
        }
    }
    return DL_COMP_OK;
}

dl_comp_return_t DLCompressDFP::compress_buffer(float *src, int8_t *dst, float *diff, size_t count, bool inPlace)
{
    dl_comp_return_t ret    = DL_COMP_FAIL;
    dl_comp_head *compHead    = NULL;
    int scale               = 0;
    for (size_t i = 0, comp_block = 0; i < count; i += DL_COMP_BLOCK_NUM) {
        comp_block = (i + DL_COMP_BLOCK_NUM) < count ? DL_COMP_BLOCK_NUM : (count - i);
        compHead = (dl_comp_head *)dst;
        if (!inPlace) {
            dst += sizeof(dl_comp_head);
        }
        if (!avx512_enabled_ || comp_block < DL_COMP_BLOCK_NUM) {
            ret = compress_block(src, dst, diff, comp_block, &scale);
        } else {
           ret = avx512_compress_block(src, dst, diff, comp_block, &scale);
        }
        if (ret == DL_COMP_FAIL) {
            return ret;
        }
        if (inPlace) {
            memmove(dst+sizeof(dl_comp_head), dst, comp_block);
            dst += sizeof(dl_comp_head);
        }
        compHead->magic = DL_COMP_HEAD_MAGIC;
        compHead->exponent = scale;
        compHead->payloadLen = comp_block;
        dst += comp_block;
        src += comp_block;
        if (NULL != diff) {
            diff += comp_block;
        }
    }
    
    return DL_COMP_OK;
}

dl_comp_return_t DLCompressDFP::compress_buffer(float *src, int8_t *dst, size_t count, bool inPlace)
{
    dl_comp_return_t ret = compress_buffer(src, dst, NULL, count, inPlace);
    return ret;
}

dl_comp_return_t DLCompressDFP::decompress_buffer(const int8_t *src, float *dst, size_t blockCount)
{
    dl_comp_head *compHead   = NULL;
    dl_comp_return_t ret;
    const int8_t *origSrc = src;
    float *origDst = dst;
    int8_t decomp_block[DL_COMP_BLOCK_NUM];


    if (blockCount == 0) {
        return DL_COMP_OK;
    }
        
    do {
        src = origSrc + (blockCount - 1) * (sizeof(dl_comp_head) + DL_COMP_BLOCK_NUM);
        dst = origDst + (blockCount - 1) * DL_COMP_BLOCK_NUM;
        compHead = (dl_comp_head *)src;
        if (compHead->magic != DL_COMP_HEAD_MAGIC) {
            return DL_COMP_FAIL_INVALID_COMPRESSED_FORMAT;
        }
        size_t count = compHead->payloadLen;
        int scale = compHead->exponent; 
        if (blockCount == 1) {
            memcpy(decomp_block, src + sizeof(dl_comp_head), count);
        }
        if (!avx512_enabled_) {
            if (blockCount != 1) {
                ret = decompress_block(src + sizeof(dl_comp_head), dst, count, scale);
            } else {
                ret = decompress_block(decomp_block, dst, count, scale);
            }
        } else {
            if (blockCount != 1) {
                ret = avx512_decompress_block(src + sizeof(dl_comp_head), dst, count, scale);
            } else {
                ret = avx512_decompress_block(decomp_block, dst, count, scale);
            }
        }
        if (ret != DL_COMP_OK) {
            return ret;
        }
        blockCount--;
    } while (blockCount > 0);

    return ret;
}

dl_comp_return_t DLCompressDFP::avx512_decompress_block(const int8_t *src, float *dst, size_t count, int scale)
{
    // If count is smaller than 16 we use non-avx512 implementation
    //16 is the element number which one avx512 register can hold
    if (count < DL_COMP_BLOCK_NUM) {
        return decompress_block(src, dst, count, scale);
    }

    DLCP_ASSERT(count % 16 == 0, "count can't be divided by 16!");

    // Do de-quantization
    float pow2_scale_inv = 1.0f / std::pow(2, scale);
    size_t group_size = 16;
    //size_t num_group = count / group_size;
    __m512 scale_factor = _mm512_set1_ps(pow2_scale_inv);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < count; idx += group_size) {
        __m512 float_vec    = _mm512_set_ps((float)src[idx + 15], (float)src[idx + 14],
                                            (float)src[idx + 13], (float)src[idx + 12],
                                            (float)src[idx + 11], (float)src[idx + 10],
                                            (float)src[idx + 9], (float)src[idx + 8],
                                            (float)src[idx + 7], (float)src[idx + 6],
                                            (float)src[idx + 5], (float)src[idx + 4],
                                            (float)src[idx + 3], (float)src[idx + 2],
                                            (float)src[idx + 1], (float)src[idx]);
        __m512 result_vec   = _mm512_mul_ps(float_vec, scale_factor);
        _mm512_storeu_ps(dst+idx, result_vec);
    }
    return DL_COMP_OK;
}

dl_comp_return_t DLCompressDFP::decompress_block(const int8_t *src, float *dst, size_t count, int scale)
{
    // Do de-quantization
    // only handle int8_t as src and float as dst
    float pow2_scale_inv = 1.0f / std::pow(2, scale);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < count; ++i) {
        dst[i] = (float)src[i];
        dst[i] *= pow2_scale_inv;
    }

    return DL_COMP_OK;
}

#if 0
size_t DLCompressDFP::get_dataCount_in_compressed_buffer(const int8_t *src, size_t blockCount) {
    size_t sum = 0;
    dl_comp_head *compHead = NULL;

    if (blockCount == 0) {
        return sum;
    }

    do {
        compHead = (dl_comp_head *)src;
        DLCP_ASSERT(compHead->magic == DL_COMP_HEAD_MAGIC, "Invalid compHead!!!\n");
        size_t count = compHead->payloadLen;
        src += sizeof(dl_comp_head);
        src += count;
        sum += count;
        blockCount--;
    } while (blockCount > 0);

    return sum;
}
#endif

#if 0
dl_comp_return_t DLCompressDFP::compress_sum(const int8_t *invec, int8_t *inoutvec, size_t blockCount)
{
    dl_comp_return_t ret      = DL_COMP_OK;
    const size_t blockSize  = sizeof(dl_comp_head) + DL_COMP_BLOCK_NUM;
    size_t inCount          = get_dataCount_in_compressed_buffer((const int8_t*)invec, blockCount);
    size_t outCount         = get_dataCount_in_compressed_buffer((const int8_t*)inoutvec, blockCount);

    DLCP_ASSERT(inCount == outCount, "inCount is not equal to outCount");

    float deqBuf1[DL_COMP_BLOCK_NUM];
    float deqBuf2[DL_COMP_BLOCK_NUM];

    for (size_t i = 0; i < inCount; i += DL_COMP_BLOCK_NUM, invec += blockSize, inoutvec += blockSize) {
        size_t compBlock = (i + DL_COMP_BLOCK_NUM) < inCount ? DL_COMP_BLOCK_NUM : (inCount - i);
        decompress_buffer(invec, deqBuf1, 1);
        decompress_buffer(inoutvec, deqBuf2, 1);
        if (!avx512_enabled_) {
            dl_comp_float_vector_add(deqBuf2, deqBuf1, compBlock);
        } else {
            dl_comp_avx512_float_vector_add(deqBuf2, deqBuf1, compBlock);
        }
        ret = compress_buffer(deqBuf1, inoutvec, compBlock, false);
        if (ret != DL_COMP_OK) {
            return ret;
        }
    }

    return ret; 
}
#endif

dl_comp_return_t DLCompressDFP::compress_sum2(const int8_t *invec, int8_t *inoutvec, size_t blockCount)
{
    const size_t blockSize  = sizeof(dl_comp_head) + DL_COMP_BLOCK_NUM;
    dl_comp_return_t ret = DL_COMP_OK;
    // size_t count          = get_dataCount_in_compressed_buffer((const int8_t*)invec, blockCount);
           
    if (!avx512_enabled_) { 
        for (size_t i = 0; i < blockCount; i++, invec += blockSize, inoutvec += blockSize) {
            ret = compress_block_sum(invec, inoutvec);
            if (ret != DL_COMP_OK) {
                return ret;
            }         
        }
    } else {
        for (size_t i = 0; i < blockCount; i++, invec += blockSize, inoutvec += blockSize) {
            ret = compress_block_sum2(invec, inoutvec);
            if (ret != DL_COMP_OK) {
                return ret;
            }
        }
    }
 
    return ret; 
}

dl_comp_return_t DLCompressDFP::compress_block_sum(const int8_t *invec, int8_t *inoutvec)
{
    dl_comp_head *inHead   = (dl_comp_head *)invec;
    dl_comp_head *outHead  = (dl_comp_head *)inoutvec;

    size_t count    = inHead->payloadLen;
    int inScale     = inHead->exponent;
    int outScale    = outHead->exponent;

    if ((inHead->magic != DL_COMP_HEAD_MAGIC) || (outHead->magic != DL_COMP_HEAD_MAGIC)) {
        return DL_COMP_FAIL_INVALID_COMPRESSED_FORMAT;
    }

    if (inScale == 0) {
        // Means invec contain all 0.
        return DL_COMP_OK;
    }

    if (outScale == 0) {
        // Means outvec contain all 0.
        memcpy(inoutvec, invec, sizeof(dl_comp_head) + count);
        return DL_COMP_OK;
    }

    // Since scale is 2 exponent, if their gap is bigger than 128 (we don't need to sum up)
    if (std::abs(inScale - outScale) > 8) {
        if (outScale < inScale) {
            return DL_COMP_OK;
        } else {
            memcpy(inoutvec, invec, sizeof(dl_comp_head) + count);
            return DL_COMP_OK;
        }
    }
   
    int resvec[DL_COMP_BLOCK_NUM] = {0}; 
    int minScale        = std::min(inScale, outScale);
    int inScaleGap      = inScale - minScale;
    int outScaleGap     = outScale - minScale;
    int8_t left;
    int max_abs = 0;


    invec       += sizeof(dl_comp_head);
    inoutvec    += sizeof(dl_comp_head);
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < count; i++) {
        left = invec[i] >> inScaleGap;
        int8_t right = inoutvec[i] >> outScaleGap;
        resvec[i] = left + right;
        // This is for compensation of final right shift
        // To make it an unbiased estimator, we only 
        // compensate when left number is 
        resvec[i] += resvec[i] & left & 1;
        max_abs |= (resvec[i] > 0 ? resvec[i] : (-resvec[i]));
    }

    if (max_abs >= 128) {
        minScale -= 1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < count; i++) {
            inoutvec[i] = resvec[i] >> 1;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < count; i++) {
            inoutvec[i] = resvec[i];
        }
    }

    outHead->exponent = minScale; 
    return DL_COMP_OK;
}

dl_comp_return_t DLCompressDFP::compress_block_sum2(const int8_t *invec, int8_t *inoutvec)
{
    dl_comp_head *inHead   = (dl_comp_head *)invec;
    dl_comp_head *outHead  = (dl_comp_head *)inoutvec;

    size_t count    = inHead->payloadLen;
    int inScale     = inHead->exponent;
    int outScale    = outHead->exponent;

    if ((inHead->magic != DL_COMP_HEAD_MAGIC) || (outHead->magic != DL_COMP_HEAD_MAGIC)) {
        return DL_COMP_FAIL_INVALID_COMPRESSED_FORMAT;
    }

    if (count % 16 != 0) {
        return compress_block_sum(invec, inoutvec);
    }

    if (inScale == 0) {
        // Means invec contain all 0.
        return DL_COMP_OK;
    }

    if (outScale == 0) {
        // Means outvec contain all 0.
        memcpy(inoutvec, invec, sizeof(dl_comp_head) + count);
        return DL_COMP_OK;
    }

    // Since scale is 2 exponent, if their gap is bigger than 128 (we don't need to sum up)
    if (std::abs(inScale - outScale) > 7) {
        if (outScale < inScale) {
            return DL_COMP_OK;
        } else {
            memcpy(inoutvec, invec, sizeof(dl_comp_head) + count);
            return DL_COMP_OK;
        }
    }

    int32_t resvec[DL_COMP_BLOCK_NUM] = {0};
    int minScale        = std::min(inScale, outScale);
    int inScaleGap      = inScale - minScale;
    int outScaleGap     = outScale - minScale;
    int max_abs = 0;
    size_t group_size = 16;
    __mmask16 mask = _mm512_int2mask(0xFFFF);
    __m512i i32_one_v = _mm512_set1_epi32(1);
    __m512i i32_or_v = _mm512_set1_epi32(0);

    invec       += sizeof(dl_comp_head);
    inoutvec    += sizeof(dl_comp_head);        

    for (size_t i = 0; i < count; i += group_size) {
        const int8_t *i8_left     = invec + i;
        int8_t *i8_right    = inoutvec + i;
        int32_t *i32_result  = resvec + i;
        __m128i i8_left_v   = _mm_maskz_loadu_epi8(mask, i8_left);
        __m128i i8_right_v  = _mm_maskz_loadu_epi8(mask, i8_right);
        __m512i i32_left_v  = _mm512_cvtepi8_epi32(i8_left_v);
        __m512i i32_right_v = _mm512_cvtepi8_epi32(i8_right_v);
        i32_left_v          = _mm512_srai_epi32(i32_left_v, inScaleGap);
        i32_right_v         = _mm512_srai_epi32(i32_right_v, outScaleGap);
        __m512i i32_result_v= _mm512_add_epi32(i32_left_v, i32_right_v);
        //compensation
        __m512i i32_comp_v  = _mm512_and_epi32(i32_result_v, i32_left_v);
        i32_comp_v          = _mm512_and_epi32(i32_comp_v, i32_one_v);
        i32_result_v        = _mm512_add_epi32(i32_result_v, i32_comp_v);
        _mm512_mask_storeu_epi32(i32_result, mask, i32_result_v);
        // To get or of while result
        i32_result_v        = _mm512_abs_epi32(i32_result_v);
        i32_or_v            = _mm512_or_epi32(i32_result_v, i32_or_v);
    } 

    max_abs = _mm512_reduce_or_epi32(i32_or_v);

    if (max_abs >= 128) {
        minScale -= 1;
        for (size_t i = 0; i < count; i += group_size) {
            int32_t *i32_res    = resvec + i;
            int8_t *i8_inout    = inoutvec + i;
            __m512i i32resvec_v = _mm512_loadu_si512(i32_res);
            i32resvec_v         = _mm512_srai_epi32(i32resvec_v, 1);
            _mm512_mask_cvtsepi32_storeu_epi8(i8_inout, mask, i32resvec_v);
        }
    } else {
        for (size_t i = 0; i < count; i += group_size) {
            int32_t *i32_res    = resvec + i;
            int8_t *i8_inout    = inoutvec + i;
            __m512i i32resvec_v = _mm512_loadu_si512(i32_res);
             _mm512_mask_cvtsepi32_storeu_epi8(i8_inout, mask, i32resvec_v);
        }
    }

    outHead->exponent = minScale;
    return DL_COMP_OK;
}

#if 0
void DLCompressDFP::dump_compressed_buffer(const int8_t *src, size_t blockCount) 
{
    dl_comp_head *compHead = NULL;

    if (blockCount == 0) return;

    DLCP_LOG(INFO, "Enter function dump_compressed_buffer...\n");
    do {
        compHead = (dl_comp_head *)src;
        if (compHead->magic != DL_COMP_HEAD_MAGIC) {
            DLCP_LOG(INFO, "Invalid compHead!!!\n");
            return;
        }
        size_t count = compHead->payloadLen;
        int scale = compHead->exponent;
        DLCP_LOG(INFO, "count = %lu Scale = %d\n", (unsigned long)count, scale);
        float pow2_scale = std::pow(2, scale);
        src += sizeof(dl_comp_head);
        for (size_t i = 0; i < count; i++) {
            float d_value = ((float)src[i])/pow2_scale;
            DLCP_LOG(INFO, "compressed value %d decompressed value %f\n", src[i], d_value);
        }
        src += count;
        blockCount--;
    } while (blockCount > 0);
    DLCP_LOG(INFO, "End of function dump_compressed_buffer...\n");
}

bool DLCompressDFP::check_compressed_buffer(const float *comp1, const int8_t *comp2, const float *diff, size_t blockCount)
{
    float epislon = 1e-9;
    dl_comp_head *compHead = NULL;

    do {
        compHead = (dl_comp_head *)comp2;
        if (compHead->magic != DL_COMP_HEAD_MAGIC) {
            DLCP_LOG(ERROR, "Invalid compHead!!!\n");
            return false;
        }
        size_t count = compHead->payloadLen;
        int scale = compHead->exponent;
        comp2 += sizeof(dl_comp_head);
        float pow2_scale = std::pow(2, scale);
        for (size_t i = 0; i < count; i++) {
            float d_value = ((float)comp2[i])/pow2_scale;
            if (d_value * comp1[i] < 0.0f) {
                DLCP_LOG(ERROR, "detected big gap src = %f d_value = %f diff = %f\n", comp1[i], d_value, diff[i]);
                DLCP_LOG(ERROR, "scale = %d, pow2_scale = %f, compressed_value = %d\n", scale, std::pow(2, scale), comp2[i]);
                return false;
            }
        }
        comp1 += count;
        comp2 += count;
        diff  += count;
        blockCount--;
    } while (blockCount > 0);

    return true;
}

dl_comp_return_t compress_helper(float *src, int8_t *dst, float *diff, dl_comp_method_t method, size_t count)
{
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(method);
    dl_comp_return_t ret = compInst->compress_buffer(src, dst, diff, count);
    return ret;
}
#endif

#if 0
dl_comp_return_t decompress_helper(const int8_t *src, float *dst, dl_comp_method_t method)
{
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(method);
    return compInst->decompress_buffer(src, dst, 0);
}
#endif

void dl_comp_float_vector_add(const float* invec, float *inoutvec, size_t count)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < count; ++i) {
        inoutvec[i] += invec[i];
    }
}

void dl_comp_avx512_float_vector_add(const float* invec, float *inoutvec, size_t count)
{
    // If count is smaller than 16 we use non-avx512 implementation
    // 16 is the element number which one avx512 register can hold
    if (count < 16) {
        return dl_comp_float_vector_add(invec, inoutvec, count);
    }

    // If count can't be divided by 16, we handle tailing remainder
    // with non-avx512 imeplementation
    if (count % 16 != 0) {
        size_t remainder = count % 16;
        count -= remainder;
        dl_comp_float_vector_add(invec+count, inoutvec+count, remainder);
    }

    size_t group_size = 16;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < count; idx += group_size) {
        const float *fvec1  = invec + idx;
        float *fvec2        = inoutvec + idx;
        __m512 operand1     = _mm512_loadu_ps(fvec1);
        __m512 operand2     = _mm512_loadu_ps(fvec2);
        __m512 result       = _mm512_add_ps(operand1, operand2);
        _mm512_storeu_ps(fvec2, result);
    }
}

