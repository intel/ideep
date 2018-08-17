// Author: Ma, Guokai (guokai.ma@intel.com)
// Copyright 2018 www.intel.com
// This code is modified to demostrate total reduce, an allreduce implementation for repetitive occuring pattern

// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
//
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <TR_interface.h>
#include "../time.h"

#define ITER 100
#define PAYLOAD_COUNT 10
#define SMALL_SCALE 0.1
#define SMALL_SIZE 3
#define REVERSE_ISSUE
//#define ARTIFICAL_NUMBER
#define RUN_INPLACE 1
#define RUN_NON_INPLACE 0

static inline int get_layer_size(int id, int num_elements)
{
    if (id==0) return num_elements;
    return num_elements*SMALL_SCALE+SMALL_SIZE;
}

static inline int gen_rand()
{
    int ret_val = rand()%1000;
    return ret_val;
}

// Creates an array of random numbers. Each number has a value from 0 - 1
int *create_rand_sum(int num_elements, int id, int world_size) {
    srand(id);
    int *buf = (int *)malloc(sizeof(int) * num_elements);
    for (int i = 0; i < num_elements; i++) {
        buf[i] = 0.0;
    }
    for (int node=0; node <world_size; node++) {
        for (int i = 0; i < num_elements; i++) {
            buf[i] += gen_rand();
        }
    }
    return buf;
}

int *create_rand_nums(int num_elements, int id, int rank) {
    srand(id);
    int *buf = (int *)malloc(sizeof(int) * num_elements);
    for (int node=0; node<rank; node++) {
        for (int i = 0; i<num_elements; i++) {
            buf[i] = gen_rand();
        }
    }
    for (int i = 0; i<num_elements; i++) {
        buf[i] = gen_rand();
    }
    return buf;
}

static inline int calc_val(int id, int rank, int index)
{
    return (id+1)+(rank+1)/100.0+(index+1)/10000.0;
}

int *create_artifical_sum(int num_elements, int id, int world_size)
{
    int *buf = (int *)malloc(sizeof(int) * num_elements);
    for (int rank=0; rank<world_size; rank++) {
        for (int i = 0; i < num_elements; i++) {
            if (rank == 0) {
                buf[i] = calc_val(id, rank, i);
            } else {
                buf[i] += calc_val(id, rank, i);
            }
        }
    }
    return buf;
}

int *create_artifical_nums(int num_elements, int id, int rank)
{
    int *buf = (int *)malloc(sizeof(int) * num_elements);
    for (int i = 0; i < num_elements; i++) {
        buf[i] = calc_val(id, rank, i);
    }
    return buf;
}

void calc_delta(int id, int* buf1, int* buf2, size_t num_elements)
{
    // Clean up
    int total = 0, max_delta = 0, first_delta = 0;
    int total_diff = 0;
    for (size_t i=0; i<num_elements; i++) {
        int delta = buf1[i] - buf2[i];
        if (delta < 0) delta = -delta;
        total = total + delta;
        if (delta > max_delta) {
            max_delta = delta;
        }
        if (delta > 0 && !(first_delta > 0)) {
            first_delta = delta;
        }
        if (delta > 0) {
            total_diff ++;
            if (total_diff <= 3)
                printf("\nindex=%ld buf1=%d buf2=%d", i, buf1[i], buf2[i]);
        }
    }
    if (total_diff > 0) {
        printf ("\nid=%d, max_delta %d num_elems %ld, total_diff=%d\n", id, max_delta, num_elements, total_diff);
        exit (1);
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: avg num_elements\n");
        exit(1);
    }

    system("/bin/hostname");
    int num_elements = atoi(argv[1]);

    // init with no affinity
    TR_init(-1);

    int world_size = TR_get_world_size();
    int world_rank = TR_get_rank();

    if(world_rank==0) printf ("%d: number of bytes per node is %d, world_size=%d, rank=%d\n\n", getpid(), num_elements*4, world_size, world_rank);

    int *send_buf[PAYLOAD_COUNT];
    int *send_buf_ref[PAYLOAD_COUNT];
    int *recv_buf_ref[PAYLOAD_COUNT];

    for (int layer=0; layer<PAYLOAD_COUNT; layer++) {
        if (world_rank == 0) {
            printf ("initialize layer %d\n", layer);
        }
        send_buf[layer] =
            #ifdef ARTIFICAL_NUMBER
            create_artifical_nums(get_layer_size(layer, num_elements), layer, world_rank);
            #else
            create_rand_nums(get_layer_size(layer, num_elements), layer, world_rank);
            #endif
        send_buf_ref[layer] = (int *)malloc(sizeof(int) * get_layer_size(layer, num_elements));
        memcpy(send_buf_ref[layer], send_buf[layer], sizeof(int) * get_layer_size(layer, num_elements));
        recv_buf_ref[layer] =
            #ifdef ARTIFICAL_NUMBER
            create_artifical_sum(get_layer_size(layer, num_elements), layer, world_size);
            #else
            create_rand_sum(get_layer_size(layer, num_elements), layer, world_size);
            #endif
    }


    int *recv_buf[PAYLOAD_COUNT];
    for (int i=0; i<PAYLOAD_COUNT; i++) {
        recv_buf[i] = (int *)malloc(sizeof(int) * get_layer_size(i, num_elements));
    }

    size_t total_elements;

    #if RUN_INPLACE
    for (int index=0; index<ITER; index++) {
        if(world_rank==0) printf ("**************total reduce iallreduce, inplace iTER=%d**************************\r", index);

        total_elements = 0;
        for (int i=0; i<PAYLOAD_COUNT; i++) {
            memcpy(recv_buf[i], send_buf[i], get_layer_size(i, num_elements)*sizeof(int));
        }

        #ifndef REVERSE_ISSUE
        for (int i=0; i<PAYLOAD_COUNT; i++) {
            size_t num = get_layer_size(i, num_elements);
            total_elements += num;
            TR_iallreduce(i, i, TR_IN_PLACE, recv_buf[i], num, TR_INT32, NULL);
        }
        #else
        if (world_rank % 2) {
            for (int i=0; i<PAYLOAD_COUNT; i++) {
                size_t num = get_layer_size(i, num_elements);
                total_elements += num;
                TR_iallreduce(i, i, TR_IN_PLACE, recv_buf[i], num, TR_INT32, NULL);
            }
        } else {
            for (int i=PAYLOAD_COUNT-1; i>=0; i--) {
                size_t num = get_layer_size(i, num_elements);
                total_elements += num;
                TR_iallreduce(i, 0, TR_IN_PLACE, recv_buf[i], num, TR_INT32, NULL);
            }
        }
        #endif

        for (int i=PAYLOAD_COUNT-1; i>=0; i--) {
            TR_wait(i);
        }

        TR_barrier();

        for (int i=0; i<PAYLOAD_COUNT; i++) {
            calc_delta(i, recv_buf_ref[i], recv_buf[i], get_layer_size(i, num_elements));
        }
    }
    #endif

    #if RUN_NON_INPLACE
    for (int index=0; index<ITER; index++) {
        total_elements = 0;
        if(world_rank==0) printf ("**************total reduce iallreduce, iTER=%d**************************\r", index);

        for (int i=0; i<PAYLOAD_COUNT; i++) {
            for (int j=0; j<get_layer_size(i, num_elements); j++) {
                recv_buf[i][j] = 0.0001*world_rank+0.1*j;
            }
        }

        for (int i=0; i<PAYLOAD_COUNT; i++) {
            size_t num = get_layer_size(i, num_elements);
            total_elements += num;
            TR_iallreduce(i+PAYLOAD_COUNT, i, send_buf[i], recv_buf[i], num, TR_INT32, NULL);
        }

        for (int i=PAYLOAD_COUNT-1; i>=0; i--) {
            TR_wait(i+PAYLOAD_COUNT);
        }
        TR_barrier();
        for (int i=0; i<PAYLOAD_COUNT; i++) {
            calc_delta(i, recv_buf_ref[i], recv_buf[i], get_layer_size(i, num_elements));
        }
    }
    #endif

    TR_finalize();
    exit(0);
}
