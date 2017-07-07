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


#ifndef DL_COMPRESSION_H
#define DL_COMPRESSION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DL_COMP_NONE = 0,
    DL_COMP_DFP = 1,
} dl_comp_method_t;

typedef enum {
    DL_COMP_OK = 0,
    DL_COMP_FAIL = 1,
    DL_COMP_FAIL_SRC_DATA_TYPE_NOT_SUPPORTED = 2,
    DL_COMP_FAIL_RATIO_NOT_SUPPORTED = 3,
    DL_COMP_FAIL_COMP_METHOD_NOT_SUPPORTED = 4,
    DL_COMP_FAIL_INVALID_COMPRESSED_FORMAT = 5,
    DL_COMP_FAIL_NOT_SUPPORTED = 6
} dl_comp_return_t;

typedef enum {
    DL_COMP_INT8    = 0,
    DL_COMP_FLOAT16 = 1,
    DL_COMP_FLOAT32 = 2,
    DL_COMP_FLOAT64 = 3,
} dl_comp_data_type_t;

// Compress src buffer into dst buffer.
//
// Parameters:
// src [in] pointer to src buffer
// dst [out] pointer to dst buffer
// dataCount [in] num of element needs to be compressed
// diff [in/out] place the precision lost from the last compress
//               return the precision lost from this compress. 
//               If you don't care about lost precision, you can
//               set it NULL pointer.
// src_data_type [in/out] data type in src buffer              
// comp_ratio [in] compression ratio, it should only be 2,4,8,16,32.
//                 e.g. If we compress FLOAT32 to INT8, the comp_ratio
//                 is 4.
// method [in] compression algorithm
// Returns:
// compress successful or not. DL_COMP_OK means successful, otherwise not.
dl_comp_return_t dl_comp_compress_buffer( const void *src, 
                                          void *dst, 
                                          size_t dataCount, 
                                          void *diff, 
                                          dl_comp_data_type_t src_data_type,
                                          size_t comp_ratio,
                                          dl_comp_method_t method );

// de-Compress src buffer into dst buffer.
// 
// Parameters:
// src [in] pointer to src buffer
// dst [out] pointer to dst buffer
// dataCount [in] num of element needs to be de-Compressed
// Returns:
// de-compress successful or not.
dl_comp_return_t dl_comp_decompress_buffer( const void *src, 
                                            void *dst, 
                                            size_t dataCount );

// Sum up compressed data from two input buffer and put the result
// in the outBuffer.
// 
// Parameters:
// inBuffer1 [in] pointer to quantized data vector
// inBuffer2 [in] pointer to quantized data vector
// dataCount [in] num of element in inBuffer1 and inBuffer2 
//                needs to be sum up.
// outBuffer [out] pointer to quantized data vector and the result
//                      will be placed in this inoutBuffer.
// Returns:
// sum up successful or not.
dl_comp_return_t dl_comp_compressed_buffer_sum( const void *inBuffer1, 
                                                const void *inBuffer2,
                                                size_t dataCount,
                                                void *outBuffer ); 

// Get compress meta data info(block). Some operation like multi-node all-reduce
// will divide payload into parts to enhance communication efficiency.This api 
// is to notify of the compressed meta data info: The minimum slicing granularity.
// Its size is related with src DataType, comp_ratio, compression algorithm
//
// Parameters:
// srcDataType [in] data type of src data before compression.
// comp_ratio [in] compression ratio
// method [in] compression algorithm
// Returns:
// N/A.
size_t dl_comp_get_sizeof_block( dl_comp_data_type_t src_data_type, 
                                 size_t comp_ratio, 
                                 dl_comp_method_t method );

// Sum up two buffer's compressed data and put the result in
// second buffer.Please attention here we use blockCount 
// as input parameter. 1 block can contain multiple data.
// 
// Parameters:
// inBuffer [in] pointer to quantized data
// inoutBuffer [in/out] pointer to quantized data. Result will 
//                      be placed in this buffer.
// Returns:
// 
dl_comp_return_t  dl_comp_compressed_buffer_reduce_sum( const void *inBuffer, 
                                                        void *inoutBuffer,
                                                        size_t blockCount );

// Util function for converting data count into block count.
//
// Prameters:
// dataCount [in] num of digit
// Returns:
// return corresponding num of block
size_t dl_comp_convert_block_count(size_t dataCount);

// Util function to get how many elements in one block.
// Parameters:
// N/A
// Returns:
// return how many elements in one block
size_t dl_comp_get_elem_num_in_block();

// Check Running Environment.
// Parameters:
// N/A
// Returns:
// If check successful, return true. otherwise false.
// If false, pls disable quantization functionality.
// E.g. We sugguest not to use quantization on machine
// not support avx512 instructions, because there's no 
// performance gain.
bool dl_comp_check_running_environ();

// Uitl function for compress float32 data to int8
// Parameters:
// srcBuffer [in] src float32 data
// dstBuffer [out] dst int8 data
// diff: [in/out] precision lost in compression
// dataCount [in] data count
// Return:
// If successful, return 0, otherwise error code.
int dl_comp_compress_buffer_FLOAT32ToINT8( const void *srcBuffer,
                                           void *dstBuffer,
                                           void *diff,
                                           size_t dataCount);

// Util function for de-compress int8 to float32
// Parameters:
// srcBuffer [in] contain int8 compressed data
// dstBuffer [out] de-comressed float32 data
// dataCount [in] data count
// Return:
// If successful, return 0, othersise error code.
int dl_comp_decompress_buffer_INT8ToFLOAT32(const void *srcBuffer,
                                            void *dstBuffer,
                                            size_t dataCount);

#ifdef __cplusplus
}
#endif

#endif
