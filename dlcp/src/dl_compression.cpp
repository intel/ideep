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


#include <stdio.h>

#include "dl_compression.h"
#include "dl_compression_impl.hpp"


dl_comp_return_t dl_comp_compress_buffer( const void *src,
                                          void *dst,
                                          size_t dataCount,
                                          void *diff,
                                          dl_comp_data_type_t src_data_type,
                                          size_t comp_ratio,
                                          dl_comp_method_t method )
{
    // Parameter checking
    if (src_data_type != DL_COMP_FLOAT32) {
        return DL_COMP_FAIL_SRC_DATA_TYPE_NOT_SUPPORTED;
    }

    if (comp_ratio != 4) {
        return DL_COMP_FAIL_RATIO_NOT_SUPPORTED;
    }

    if (method != DL_COMP_DFP) {
        return DL_COMP_FAIL_COMP_METHOD_NOT_SUPPORTED;
    }

    // Do compession
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(DL_COMP_DFP);
    
    return compInst->compress_buffer((float *)src,
                                     (int8_t *)dst,
                                     (float *)diff,
                                     dataCount,
                                     src == dst);
}

dl_comp_return_t dl_comp_decompress_buffer( const void *src, 
                                            void *dst, 
                                            size_t dataCount )
{
    dl_comp_head *compHead = (dl_comp_head *)src;

    if (compHead->magic != DL_COMP_HEAD_MAGIC) {
        // This is a work-around for MLSL. Because in MPI_Test
        // sometimes an already de-compressed buffer may be sent 
        // to compress lib to do de-compressed buffer. So we
        // simply ignore it in this case.
        return DL_COMP_OK;
    }

    size_t blockCount = dataCount % DL_COMP_BLOCK_NUM == 0 ? (dataCount / DL_COMP_BLOCK_NUM) : (dataCount / DL_COMP_BLOCK_NUM + 1);
    // do de-compression
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(DL_COMP_DFP);

    return compInst->decompress_buffer((const int8_t *)src, (float *)dst, blockCount);
}

dl_comp_return_t dl_comp_compressed_buffer_sum( const void *inBuffer1, 
                                                const void *inBuffer2,
                                                size_t dataCount,
                                                void *outBuffer )
{
    return DL_COMP_FAIL_NOT_SUPPORTED;
}

size_t dl_comp_get_sizeof_block( dl_comp_data_type_t src_data_type, 
                                 size_t comp_ratio, 
                                 dl_comp_method_t method )
{
    size_t blockSize = 0;
    if (src_data_type == DL_COMP_FLOAT32 &&
        comp_ratio == 4 &&
        method == DL_COMP_DFP) {
        blockSize = sizeof(int8_t) * DL_COMP_BLOCK_NUM + sizeof(dl_comp_head);
    }

    return blockSize;
}

size_t dl_comp_get_elem_num_in_block()
{
    return DL_COMP_BLOCK_NUM;
}

dl_comp_return_t  dl_comp_compressed_buffer_reduce_sum( const void *inBuffer, 
                                                        void *inoutBuffer,
                                                        size_t blockCount )
{
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(DL_COMP_DFP);
    
    return compInst->compress_sum2((const int8_t *)inBuffer, (int8_t *)inoutBuffer, blockCount);
}

size_t dl_comp_convert_block_count(size_t dataCount)
{
    size_t blockCount = dataCount % DL_COMP_BLOCK_NUM == 0 ? 
                        (dataCount / DL_COMP_BLOCK_NUM) : (dataCount / DL_COMP_BLOCK_NUM + 1);
    return blockCount;
}

bool dl_comp_check_running_environ()
{
  // Currently, we only check whether avx512 instruction supported.
  return dl_comp_check_avx512_supported(); 
}

int dl_comp_compress_buffer_FLOAT32ToINT8( const void *srcBuffer,
                                            void *dstBuffer,
                                            void *diff,
                                            size_t dataCount)
{
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(DL_COMP_DFP);

    dl_comp_return_t ret =  compInst->compress_buffer((float *)srcBuffer,
                                                      (int8_t *)dstBuffer,
                                                      (float *)diff,
                                                      dataCount,
                                                      srcBuffer == dstBuffer);
    return ret;
}

int dl_comp_decompress_buffer_INT8ToFLOAT32(const void *srcBuffer,
                                            void *dstBuffer,
                                            size_t dataCount)
{
    dl_comp_head *compHead = (dl_comp_head *)srcBuffer;

    if (compHead->magic != DL_COMP_HEAD_MAGIC) {
        // This is a work-around for MLSL. Because in MPI_Test
        // sometimes an already de-compressed buffer may be sent 
        // to compress lib to do de-compressed buffer. So we
        // simply ignore it in this case.
        return DL_COMP_OK;
    }

    // do de-compression
    size_t blockCount = dataCount % DL_COMP_BLOCK_NUM == 0 ? 
                        (dataCount / DL_COMP_BLOCK_NUM) : (dataCount / DL_COMP_BLOCK_NUM + 1); 
    DLCompressBase *compInst = DLCompressBase::get_compression_instance(DL_COMP_DFP);
    dl_comp_return_t ret = compInst->decompress_buffer((const int8_t *)srcBuffer, (float *)dstBuffer, blockCount);

    return ret;
}
