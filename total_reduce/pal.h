#ifndef UTIL_H
#define UTIL_H

#include <stdbool.h>
#include <mpi.h>
#include <TR_interface.h>
#include "knobs.h"

struct type_handler {
    enum TR_datatype type;
    int element_size;
    void (*calculate2)(
                #if PRINT_CALC_TRACE
                int id, int state,
                #endif
                void *write_buf, const void *buf_src1, const void *buf_src2, int count);
};

extern struct type_handler type_handlers[];

void copy_device_mem(void *dst, void *src, size_t size);

void *alloc_host_mem(size_t size);
void *alloc_device_mem(size_t size);
void free_host_mem(void *ptr);
void free_device_mem(void *ptr);

struct comm_req {
    MPI_Request req;
};

void comm_init(int *, int *);
void comm_finalize(void);
void comm_send(void *buf, size_t size, int to_rank, struct comm_req *request);
int comm_probe(int from_rank);
void comm_recv(void *buf, size_t size, int from_rank, struct comm_req *request);
bool comm_test(struct comm_req *request);

#endif



