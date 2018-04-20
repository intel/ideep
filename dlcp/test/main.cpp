
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>

#include "dl_compression.h"

#define DATA_LEN    100000000

float data1[DATA_LEN];

float data2[DATA_LEN];

void dataSetUp(void);

bool test_compress_buffer();

bool test_decompress_buffer();

bool test_compressed_buffer_reduce_sum();

bool test_compressed_buffer_sum();

#if 0
void addVec(const float *vec1, const float *vec2, float *vec3, int count) {
    for (int i = 0; i < count; i++) {
        vec3[i] = vec1[i] + vec2[i];
    }
}

void cmpVec(const float *vec1, const float *vec2, int count) {
    for (int i = 0; i < count; i++) {
        if (std::abs(vec1[i] - vec2[i]) > 1e-3) {
            printf("Detect big gap index: %d\n", i);
        }
    }
}
#endif
float getSum(const float *src, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += src[i];
    }
    return sum;
}

#if 0
void dumpVec(const float *vec, int count) {
    for (int i = 0; i < count; i++) {
        printf("vec[%d] = %lf\n", i, vec[i]);
    }
}
#endif

float sumVec(const float *vec1, const float *vec2, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum = sum + vec1[i] + vec2[i];
    //    printf("data1[%d] = %lf data2[%d] = %lf sum = %lf\n", i, vec1[i], i, vec2[i], vec1[i] + vec2[i]);
    }
    return sum;
}

#if 0
float sumVec2(const float *vec1, const float *vec2, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum = sum + vec1[i] + vec2[i];
    //    printf("tempData1[%d] = %lf tempData2[%d] = %lf sum = %lf\n", i, vec1[i], i, vec2[i], vec1[i] + vec2[i]);
    }
    return sum;
}
#endif

int main(int argc, char *argv[])
{
    dataSetUp();

    if (!test_compress_buffer()) {
        printf("test_compress_buffer failure!\n");
    } else {
        printf("test_compress_buffer successful!\n");
    }

    if (!test_decompress_buffer()) {
        printf("test_decompress_buffer failure!\n");
    } else {
        printf("test_decompress_buffer successful!\n");
    }

    if (!test_compressed_buffer_reduce_sum()) {
        printf("test_compressed_buffer_reduce_sum failure!\n");
    } else {
        printf("test_compressed_buffer_reduce_sum successful!\n");
    }

    if (!test_compressed_buffer_sum()) {
        printf("test_compressed_buffer_sum failure!\n");
    } else {
        printf("test_compressed_buffer_sum successful!\n");
    }

    return 0;
}

void dataSetUp()
{
    srand((int)time(0));

    for (int i = 0; i < DATA_LEN; i++) {
        data1[i] = (rand() % 10000) / (-100000.f) ;
    }
    
    for (int i = 0; i < DATA_LEN; i++) {
        data2[i] = (rand() % 10000) / (-100000.f);
    }
}

bool test_compress_buffer()
{
    float *tempData = (float *)malloc(sizeof(float) * DATA_LEN);
    memcpy(tempData, data1, sizeof(float) * DATA_LEN);

    dl_comp_return_t ret = dl_comp_compress_buffer((const void *)tempData,
                                                   tempData,
                                                   DATA_LEN,
                                                   NULL,
                                                   DL_COMP_FLOAT32,
                                                   4,
                                                   DL_COMP_DFP);
    free(tempData);
    if (ret != DL_COMP_OK) {
        printf("compress failed error = %d!\n", ret);
        return false;
    }

    return true;
}

bool test_decompress_buffer()
{
    float *tempData = (float *)malloc(sizeof(float) * DATA_LEN);
    float *diff = (float *)malloc(sizeof(float) * DATA_LEN);
    memcpy(tempData, data1, sizeof(float) * DATA_LEN);
    memset(diff, 0, sizeof(float) * DATA_LEN); 

    printf("before compress Total Sum: %f\n", getSum(data1, DATA_LEN));
    dl_comp_return_t ret = dl_comp_compress_buffer((const void *)tempData,
                                                   tempData,
                                                   DATA_LEN,
                                                   diff,
                                                   DL_COMP_FLOAT32,
                                                   4,
                                                   DL_COMP_DFP);
    if (ret != DL_COMP_OK) {
        printf("compress failed error = %d!\n", ret);
        free(tempData);
        free(diff);
        return false;
    }

    ret = dl_comp_decompress_buffer((const void *)tempData,
                                    tempData,
                                    DATA_LEN);
    if (ret != DL_COMP_OK) {
        printf("de-compress failed error = %d!\n", ret);
        free(tempData);
        free(diff);
        return false;
    }

    printf("after compress Total Sum: %f diff: %f\n", getSum(tempData, DATA_LEN), getSum(diff, DATA_LEN));
    printf("after diff compensation Total Sum: %f\n", sumVec(tempData, diff, DATA_LEN));
    free(tempData);
    free(diff);
    return true;
}

bool test_compressed_buffer_reduce_sum()
{
    float *tempData1 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *tempData2 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *tempData3 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *tempData4 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *sum1 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *sum2 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *sum3 = (float *)malloc(sizeof(float) * DATA_LEN);
    memcpy(tempData1, data1, sizeof(float) * DATA_LEN); 
    memcpy(tempData2, data2, sizeof(float) * DATA_LEN); 

    dl_comp_return_t ret = dl_comp_compress_buffer((const void *)tempData1,
                                                   tempData1,
                                                   DATA_LEN,
                                                   NULL,
                                                   DL_COMP_FLOAT32,
                                                   4,
                                                   DL_COMP_DFP);

    if (ret != DL_COMP_OK) {
        printf("compress failed error = %d!\n", ret);
        free(tempData1);
        free(tempData2);
        free(tempData3);
        free(tempData4);
        free(sum1);
        free(sum2);
        free(sum3);
        return false;
    }
    
    ret = dl_comp_compress_buffer((const void *)tempData2,
                                  tempData2,
                                  DATA_LEN,
                                  NULL,
                                  DL_COMP_FLOAT32,
                                  4,
                                  DL_COMP_DFP);

    if (ret != DL_COMP_OK) {
        printf("compress failed error = %d!\n", ret);
        free(tempData1);
        free(tempData2);
        free(tempData3);
        free(tempData4);
        free(sum1);
        free(sum2);
        free(sum3);
        return false;
    }

#if 0
    ret = dl_comp_decompress_buffer((const void *)tempData1,
                                    (void *)tempData3,
                                    DATA_LEN);
    ret = dl_comp_decompress_buffer((const void *)tempData2,
                                    (void *)tempData4,
                                    DATA_LEN);
      
    printf("orig data sum = %lf\n", sumVec(data1, data2, DATA_LEN));
    printf("new data sum = %lf\n", sumVec2(tempData3, tempData4, DATA_LEN));
#endif

#if 1
    size_t blockCount = dl_comp_convert_block_count(DATA_LEN);

    ret = dl_comp_compressed_buffer_reduce_sum((const void *)tempData1,
                                               (void *)tempData2,
                                               blockCount);

    if (ret != DL_COMP_OK) {
        printf("reduce sum failed error = %d!\n", ret);
        free(tempData1);
        free(tempData2);
        free(tempData3);
        free(tempData4);
        free(sum1);
        free(sum2);
        free(sum3);
        return false;
    }
   
    ret = dl_comp_decompress_buffer((const void *)tempData2,
                                    (void *)tempData2,
                                    DATA_LEN);

    if (ret != DL_COMP_OK) {
        printf("de compress failed error = %d!\n", ret);
        free(tempData1);
        free(tempData2);
        free(tempData3);
        free(tempData4);
        free(sum1);
        free(sum2);
        free(sum3);
        return false;
    }

    printf("orig data sum = %lf\n", sumVec(data1, data2, DATA_LEN));
    printf("new reduce sum = %lf\n", getSum(tempData2, DATA_LEN));
#endif

//    addVec(data1, data2, sum1, DATA_LEN);
//    addVec(tempData3, tempData4, sum2, DATA_LEN);

//    printf("start to cmp sum1 and tempData2!\n");
//    cmpVec(sum1, tempData2, DATA_LEN);

//   printf("start to cmp sum2 and sum1!\n");
//    cmpVec(sum2, sum1, DATA_LEN);


    free(tempData1);
    free(tempData2);
    free(tempData3);
    free(tempData4);
    free(sum1);
    free(sum2);
    free(sum3);
    return true;
}

bool test_compressed_buffer_sum()
{
    float *tempData1 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *tempData2 = (float *)malloc(sizeof(float) * DATA_LEN);
    float *sum1 = (float *)malloc(sizeof(float) * DATA_LEN);

    memcpy(tempData1, data1, sizeof(float) * DATA_LEN);
    memcpy(tempData2, data2, sizeof(float) * DATA_LEN);

    dl_comp_get_sizeof_block(DL_COMP_FLOAT32,
                             4,
                             DL_COMP_DFP);
    
    dl_comp_return_t ret = dl_comp_compress_buffer((const void *)tempData1,
                                                   tempData1,
                                                   DATA_LEN,
                                                   NULL,
                                                   DL_COMP_FLOAT32,
                                                   4,
                                                   DL_COMP_DFP);

    if (ret != DL_COMP_OK) {
        printf("compress failed error = %d!\n", ret);
        free(tempData1);
        free(tempData2);
        free(sum1);
        return false;
    }

    dl_comp_compress_buffer_FLOAT32ToINT8((const void *)tempData2,
                                          tempData2,
                                          NULL,
                                          DATA_LEN);

    dl_comp_compressed_buffer_sum(tempData1,
                                  tempData2,
                                  DATA_LEN,
                                  sum1);

    dl_comp_get_elem_num_in_block();

    dl_comp_decompress_buffer_INT8ToFLOAT32(sum1, sum1, DATA_LEN);

    return true;
            
}
