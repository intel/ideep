#ifndef __RING__H__
#define __RING__H__
#include <stdbool.h>

bool ring_send_step_header(int id, int state, int iter, void *send_buf, void *recv_buf,
                           size_t num_elements, int element_size, int world_size, int world_rank, int send_rank);
void ring_send_step_body(int id, int state, void *send_buf, void *recv_buf,
                         size_t num_elements, int element_size, int world_size, int world_rank, int send_rank);
void *ring_get_recv_buf(int state, void *send_buf, void *recv_buf,
                         size_t num_elements, int element_size, int world_size, int world_rank);
void ring_get_compute_buffers(int state, void *send_buf, void *pending_buf, void *recv_buf,
                              size_t num_elements, int element_size, int world_size, int world_rank,
                              void **out_buf, void **in_buf1, void **in_buf2, size_t *size);
#endif
