// device specific code

#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "pal.h"
#include "knobs.h"

static void calculate2_fp32(
    #if PRINT_CALC_TRACE
    int id, int state,
    #endif
    void *write_buf, const void *buf_src1, const void *buf_src2, int count);

static void calculate2_int32(
    #if PRINT_CALC_TRACE
    int id, int state,
    #endif
    void *write_buf, const void *buf_src1, const void *buf_src2, int count);

struct type_handler type_handlers[] = {
    [TR_FP32]  = {TR_FP32,  4, calculate2_fp32},
    //[TR_FP16]  = {TR_FP16,  2, calculate2_fp16},
    [TR_FP16]  = {TR_FP16,  2, NULL},
    [TR_INT32] = {TR_INT32, 4, calculate2_int32},
};

#if PRINT_CALC_TRACE
static void fp32_vector_add(int id, int state, float *outvec, const float *invec1, const float *invec2, size_t count);
static void int32_vector_add(int id, int state, int *outvec, const int *invec1, const int *invec2, size_t count);
#endif
static void avx256_fp32_vector_add(float *outvec, const float *invec1, const float *invec2, size_t count);
static void avx256_int32_vector_add(int *outvec, const int *invec1, const int *invec2, size_t count);

static void calculate2_fp32(
    #if PRINT_CALC_TRACE
    int id, int state,
    #endif
    void *write_buf, const void *buf_src1, const void *buf_src2, int count)
{
    #if PRINT_CALC_TRACE
    fp32_vector_add(id, state, (float*)write_buf, (float*)buf_src1, (float*)buf_src2, count);
    #elseif BYPASS_CALC
    #else
    avx256_fp32_vector_add((float*)write_buf, (float*)buf_src1, (float*)buf_src2, count);
    #endif
}

static void calculate2_int32(
    #if PRINT_CALC_TRACE
    int id, int state,
    #endif
    void *write_buf, const void *buf_src1, const void *buf_src2, int count)
{
    #if PRINT_CALC_TRACE
    int32_vector_add(id, state, (int*)write_buf, (int*)buf_src1, (int*)buf_src2, count);
    #elseif BYPASS_CALC
    #else
    avx256_int32_vector_add((int*)write_buf, (int*)buf_src1, (int*)buf_src2, count);
    #endif
}

#if PRINT_CALC_TRACE
static void fp32_vector_add(int id, int state, float *outvec, const float *invec1, const float *invec2, size_t count)
{
    #if PRINT_CALC_TRACE
    if (count == 1) {
        printf ("%d-%d \t%f=%f+%f *%p=*%p+*%p\n", id, state,
                invec1[0]+invec2[0], invec1[0], invec2[0],
                outvec, invec1, invec2);
    }
    #endif
    for (size_t i = 0; i < count; ++i) {
        outvec[i] = invec1[i] + invec2[i];
    }
}
#endif

#if PRINT_CALC_TRACE
static void int32_vector_add(int id, int state, int *outvec, const int *invec1, const int *invec2, size_t count)
{
    #if PRINT_CALC_TRACE
    if (count == 1) {
        printf ("%d-%d \t%d=%d+%d *%p=*%p+*%p\n", id, state,
                invec1[0]+invec2[0], invec1[0], invec2[0],
                outvec, invec1, invec2);
    }
    #endif
    for (size_t i = 0; i < count; ++i) {
        outvec[i] = invec1[i] + invec2[i];
    }
}
#endif

static void avx256_fp32_vector_add(float *outvec, const float *invec1, const float *invec2, size_t count)
{
    size_t group_size = 8;
    size_t idx = 0;
    size_t count_major = count - (count%group_size);
    for (; idx < count_major; idx += group_size) {
        const float *vec0  = invec1 + idx;
        const float *vec1  = invec2 + idx;
        float *vec3  = outvec + idx;
        __m256 operand0     = _mm256_loadu_ps(vec0);
        __m256 operand1     = _mm256_loadu_ps(vec1);
        operand0            = _mm256_add_ps(operand1, operand0);
        _mm256_storeu_ps(vec3, operand0);
    }
    for (; idx < count; idx++) {
        outvec[idx] = invec1[idx]+invec2[idx];
    }
}

static void avx256_int32_vector_add(int *outvec, const int *invec1, const int *invec2, size_t count)
{
    //size_t group_size = 8;
    size_t idx = 0;
    #if 0
    size_t count_major = count - (count%group_size);
    for (; idx < count_major; idx += group_size) {
        const __m256i *vec0  = (__m256i*)(invec1 + idx);
        const __m256i *vec1  = (__m256i*)(invec2 + idx);
        __m256i *vec3  = (__m256i*)(outvec + idx);
        __m256i operand0     = _mm256_loadu_si256(vec0);
        __m256i operand1     = _mm256_loadu_si256(vec1);
        operand0            = _mm256_add_epi32(operand1, operand0);
        _mm256_storeu_si256(vec3, operand0);
    }
    #endif
    for (; idx < count; idx++) {
        outvec[idx] = invec1[idx]+invec2[idx];
    }
}

void copy_device_mem(void *dst, void *src, size_t size)
{
    memcpy (dst, src, size);
}

void *alloc_device_mem(size_t size)
{
    void *ptr = malloc(size);
    if (ptr==NULL) {
        printf ("OOM\n");
        exit(0);
    }
    return ptr;
}

void *alloc_host_mem(size_t size)
{
    void *ptr = malloc(size);
    if (ptr==NULL) {
        printf ("OOM\n");
        exit(0);
    }
    return ptr;
}

void free_device_mem(void *ptr)
{
    free(ptr);
}

void free_host_mem(void *ptr)
{
    free(ptr);
}

void comm_init(int *rank, int *world_size)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, world_size);
}

void comm_finalize(void)
{
    MPI_Finalize();
}

void comm_send(void *buf, size_t size, int to_rank, struct comm_req *request)
{
    MPI_Isend(buf, size, MPI_BYTE, to_rank, 0, MPI_COMM_WORLD, &(request->req));
}

// return value: -1: no thing to receieve
//               >=0: receieve size in byte = return value
int comm_probe(int from_rank)
{
    int flag;
    MPI_Status status;
    int count;

    MPI_Iprobe(from_rank, 0, MPI_COMM_WORLD, &flag, &status);

    if (!flag) {
        return -1;
    }

    MPI_Get_count(&status, MPI_BYTE, &count);
    return count;
}

void comm_recv(void *buf, size_t size, int from_rank, struct comm_req *request)
{
    MPI_Irecv(buf, size, MPI_BYTE, from_rank, 0, MPI_COMM_WORLD, &(request->req));
}

bool comm_test(struct comm_req *request)
{
    int flag;
    MPI_Test(&(request->req), &flag, MPI_STATUS_IGNORE);
    return flag;
}

