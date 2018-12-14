#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ring.h"
#include "pal.h"
#include "total_reduce.h"
#if PROFILE>=2
    #include "time.h"
#endif

/*
    Implementation of ring algorithm for total reduce
*/

static void ring_get_chunk_size (size_t *chunk_size, size_t *remainder_size, size_t num_elements, int world_size)
{
    *chunk_size = num_elements/world_size;
    if (*chunk_size >= 8) {
        *chunk_size = *chunk_size - (*chunk_size % 8); // make things slightly faster
    }
    *remainder_size = num_elements - *chunk_size * (world_size-1);
}

// return true if a 'micro body' is sent along with header
// return false if body should be sent out seperately
bool ring_send_step_header(int id, int state, int iter,
                           void *src_buf, void *dst_buf,
                           size_t num_elements, int element_size,
                           int world_size, int world_rank, int send_rank)
{
    size_t chunk_size, remainder_size, send_size, send_byte_size;
    ring_get_chunk_size (&chunk_size, &remainder_size, num_elements, world_size);

    int lmt_stage_1 = world_size-1;
    int lmt_stage_2 = 2*world_size-2;

    assert (state < lmt_stage_2);

    struct message_header *send_header = total_reduce_get_send_header();
    struct comm_req *send_header_request = total_reduce_get_send_header_request();

    // DO NOT try to merge these two path for code reuse, they don't make it faster
    // and will make code harder to read
    if (state >= 0 && state < lmt_stage_1) {
        int i = world_rank - state;
        int send_chunk_idx = (i+world_size) % world_size;

        send_size = (send_chunk_idx==world_size-1)?remainder_size:chunk_size;
        send_byte_size = send_size *element_size;

        send_header->id = id;
        send_header->iter = iter;
        send_header->state = state;
        send_header->size = send_byte_size;

        if (send_byte_size >0 && send_byte_size <= MICRO_MESSAGE_SIZE) {
            void *send_buf_p = (char*)(state==0 ? src_buf : dst_buf) + element_size*send_chunk_idx*chunk_size;
            memcpy (send_header->micro_body, send_buf_p, send_byte_size);
        }

        comm_send(send_header,
             (char*)(send_header->micro_body) - (char*)send_header +
                (send_byte_size<=MICRO_MESSAGE_SIZE?send_byte_size:0),
             send_rank, send_header_request);
    } else {
        assert (state >=lmt_stage_1 && state <lmt_stage_2);
        int i = world_rank+1+(world_size-1)-state;
        int send_chunk_idx = (i+world_size) % world_size;

        send_size = (send_chunk_idx==world_size-1)?remainder_size:chunk_size;
        send_byte_size = send_size *element_size;

        send_header->id = id;
        send_header->iter = iter;
        send_header->state = state;
        send_header->size = send_byte_size;

        if (send_byte_size >0 && send_byte_size <= MICRO_MESSAGE_SIZE) {
            void *send_buf_p = (char*)dst_buf + element_size*send_chunk_idx*chunk_size;
            memcpy (send_header->micro_body, send_buf_p, send_byte_size);
        }
        comm_send(send_header,
             (char*)(send_header->micro_body) - (char*)send_header +
                (send_byte_size<=MICRO_MESSAGE_SIZE?send_byte_size:0),
             send_rank, send_header_request);
    }
    return send_byte_size <= MICRO_MESSAGE_SIZE;
}

void ring_send_step_body(int id, int state, void *src_buf, void *dst_buf, size_t num_elements, int element_size,
                         int world_size, int world_rank, int send_rank)
{
    size_t chunk_size, remainder_size;
    ring_get_chunk_size (&chunk_size, &remainder_size, num_elements, world_size);

    int lmt_stage_1 = world_size-1;
    int lmt_stage_2 = 2*world_size-2;

    assert (state < lmt_stage_2);

    // DO NOT try to merge these two path for code reuse, they don't make it faster and will make code harder to read
    if (state >= 0 && state < lmt_stage_1) {
        int i = world_rank - state;
        int send_chunk_idx = (i+world_size) % world_size;

        size_t send_size = (send_chunk_idx==world_size-1)?remainder_size:chunk_size;

        struct comm_req *send_body_request = total_reduce_get_send_body_request(id, send_size);

        void *send_buf_p = (char*)(state==0 ? src_buf : dst_buf) + element_size*send_chunk_idx*chunk_size;
        comm_send(send_buf_p, send_size*element_size, send_rank, send_body_request);
    } else {
        assert (state >=lmt_stage_1 && state <lmt_stage_2);
        int i = world_rank+1+(world_size-1)-state;
        int send_chunk_idx = (i+world_size) % world_size;

        size_t send_size = (send_chunk_idx==world_size-1)?remainder_size:chunk_size;

        struct comm_req *send_body_request = total_reduce_get_send_body_request(id, send_size);

        void *send_buf_p = (char*)dst_buf + element_size*send_chunk_idx*chunk_size;
        comm_send(send_buf_p, send_size*element_size, send_rank, send_body_request);
    }
}

void *ring_get_recv_buf(int state, void *dst_buf, void *recv_buf, size_t num_elements, int element_size,
                         int world_size, int world_rank)
{
    size_t chunk_size, remainder_size;
    void *real_recv_buf;

    ring_get_chunk_size (&chunk_size, &remainder_size, num_elements, world_size);

    int lmt_stage_1 = world_size-1;
    int lmt_stage_2 = 2*world_size-2;

    if (state>=lmt_stage_2) printf ("state=%d\n", state);
    assert (state < lmt_stage_2);

    // DO NOT try to merge these two path for code reuse, they don't make it faster and will make code harder to read
    int i;
    if (state >= 0 && state < lmt_stage_1) {
        i = world_rank - state;
        real_recv_buf = recv_buf;
    } else {
        assert (state >=lmt_stage_1 && state <lmt_stage_2);
        i = world_rank+1+(world_size-1)-state;
        real_recv_buf = dst_buf;
    }

    int    recv_chunk_idx = (i-1+world_size) % world_size;

    void *recv_buf_p     = (char*)real_recv_buf+element_size*recv_chunk_idx*chunk_size;
    return recv_buf_p;
}

void ring_get_compute_buffers(int state, void *send_buf, void *pending_buf, void *recv_buf,
                              size_t num_elements, int element_size,
                              int world_size, int world_rank,
                              // output
                              void **out_buf, void **in_buf1, void **in_buf2, size_t *size)
{
    size_t chunk_size, remainder_size;
    ring_get_chunk_size (&chunk_size, &remainder_size, num_elements, world_size);

    int lmt_stage_1 = world_size-1;
    int lmt_stage_2 = 2*world_size-2;

    assert (state < lmt_stage_2);

    if (state >= 0 && state < lmt_stage_1) {
        int i = world_rank - state;
        int recv_chunk_idx = (i-1+world_size) % world_size;

        *out_buf = recv_buf;
        *in_buf1 = pending_buf;
        *in_buf2 = (char*)send_buf+element_size*recv_chunk_idx*chunk_size,
        *size = (recv_chunk_idx==world_size-1)?remainder_size:chunk_size;
    } else {
        *out_buf = *in_buf1 = *in_buf2 = NULL;
        *size = 0;
    }
}
