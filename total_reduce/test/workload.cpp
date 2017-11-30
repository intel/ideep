// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Program that computes the average of an array of elements in parallel using
// MPI_Reduce.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <TR_interface.h>
#include "../time.h"

#ifdef USE_MLSL
#include "mlsl.hpp"
using namespace MLSL;
using namespace std;

Distribution *dist;
#endif

#define EVENTS(x) x, sizeof(x)/sizeof(x[0])

#define SCALE 0.7

typedef struct event {
    float start_time, wait_time;
    int size;
    float* payload;
    float* recv_buf;
    float overdue;
    #ifdef USE_MLSL
    CommReq *req;
    #endif
} event_t;

event_t inception_v3[] = {
    #include "inception_v3.h"
};

event_t vgg16[] = {
    #include "vgg16.h"
};

event_t res50[] = {
    #include "resnet50.h"
};

event_t test_events_1[] = {
    {0.2, 0.21, 10000, NULL},
    {0, 1.2, 100000000, NULL},
};

void init_events(int seed, float scale, event_t *events, size_t size)
{
    if (seed != 0)
        srand(seed);

    for (size_t i=0; i<size; i++) {
        events[i].start_time *= scale;
        events[i].wait_time *= scale;
        if (events[i].payload == NULL) {
            #ifdef USE_MLSL
            events[i].payload = Environment::GetEnv().Alloc(events[i].size*sizeof(float), sizeof(float));
            events[i].recv_buf = Environment::GetEnv().Alloc(events[i].size*sizeof(float), sizeof(float));
            #else
            events[i].payload  = (float*)malloc(events[i].size*sizeof(float));
            events[i].recv_buf = (float*)malloc(events[i].size*sizeof(float));
            #endif
        }
        for (int j=0; j<events[i].size; j++) {
            events[i].payload[j] = (float)(rand())/RAND_MAX-0.5;
        }
    }
}

void execute_events(event_t *events, size_t size)
{
    float time0 = 0.0;
    for (int i=size-1; i>=0; i--) {
        sleepf(events[i].start_time - time0);
        #ifdef USE_MLSL
        events[i].req = dist->AllReduce(events[i].payload, events[i].recv_buf, events[i].size, DT_FLOAT, RT_SUM, GT_DATA);
        #else
        TR_iallreduce(i, 0, events[i].payload, events[i].recv_buf, events[i].size, TR_FP32, NULL);
        #endif
        time0 = events[i].start_time;
    }
;
    for (size_t i=0; i<size; i++) {
        sleepf(events[i].wait_time - time0);
        float start = get_time();
        #ifdef USE_MLSL
        Environment::GetEnv().Wait(events[i].req);
        #else
        TR_wait(i);
        #endif
        float end = get_time();
        time0 = events[i].wait_time;
        events[i].overdue = end-start;
    }
}

void print_events(event_t *events, size_t size)
{
    float total_wait_time = 0.0;
    size_t total_size = 0;
    for (int i=size-1; i>=0; i--) {
        if (events[i].overdue>0.001)
            printf ("layer %3d: size=%10d, start=%f, end=%f, overdue=%.*f\n", i, events[i].size,
                events[i].start_time, events[i].wait_time,
                events[i].overdue==0.0?0:6, events[i].overdue);
        total_wait_time += events[i].overdue;
        total_size += events[i].size;
    }
    printf ("total_wait_time = %f\n", total_wait_time);
    printf ("total_size = %ld\n", total_size);
}

#define str2(x) str(x)
#define str(x) #x

int main(int argc, char** argv) {

    #define ITER 15

    #ifdef USE_MLSL
    size_t process_count;
    Environment::GetEnv().Init(&argc, &argv);
    process_count =  Environment::GetEnv().GetProcessCount();
    dist = Environment::GetEnv().CreateDistribution(process_count, 1);
    init_time();
    #else
    TR_init();
    #endif

    #ifdef USE_MLSL
    if (Environment::GetEnv().GetProcessIdx() == 0)
    #else
    if (TR_get_rank()==0)
    #endif
        printf ("simulating %s with %s\n", str2(EVENT_NAME), 
            #ifdef USE_MLSL
            "mlsl"
            #else
            "total reduce"
            #endif
            );

    float begin, end, total=0.0;
    init_events(12345, SCALE, EVENTS(EVENT_NAME));
    for (int i=0; i<ITER; i++) {
        begin = get_time();
        execute_events(EVENTS(EVENT_NAME));
        #ifndef USE_MLSL
        TR_barrier();
        #endif
        end = get_time();
        total+=end-begin;
        //printf ("iteration %d: span = %f\t\t\t\n", i, end - begin);
    }
    #ifdef USE_MLSL
    if (Environment::GetEnv().GetProcessIdx() == 0) {
    #else
    if (TR_get_rank()==0) {
    #endif
        print_events(EVENTS(EVENT_NAME));
        printf ("avg=%f\n", total/ITER);
    }

    #ifdef USE_MLSL
    Environment::GetEnv().Finalize();
    #else
    TR_finalize();
    #endif

    return 0;
}
