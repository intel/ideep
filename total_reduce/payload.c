#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

#include "payload.h"
#include "time.h"
#include "total_reduce.h"
#include "ring.h"
#include "pending_message.h"
#include "compute_request.h"
#include "pal.h"
#include "knobs.h"

struct buf_pool {
    size_t size;
    struct buf_pool *next;
};

static struct buf_pool inner_buf_pool;

static pthread_mutex_t inner_buf_pool_mutex;

// needs MT protection
static struct payload *payload_list = NULL;

static pthread_mutex_t payload_list_mutex;

void payload_list_init(void)
{
    assert (payload_list == NULL);
    pthread_mutex_init(&payload_list_mutex, NULL);
    pthread_mutex_init(&inner_buf_pool_mutex, NULL);
    payload_list = (struct payload*)alloc_host_mem(sizeof(struct payload));
    payload_list->next = NULL;

    inner_buf_pool.size = sizeof(inner_buf_pool);
    inner_buf_pool.next = NULL;
}

static void payload_add(struct payload *payload)
{
    assert (payload);

    pthread_mutex_lock(&payload_list_mutex);

    if (payload->next == NULL) {
        struct payload *cur = payload_list;
        while (cur->next) {
            if (payload->priority >= cur->next->priority) {
                break;
            }
            cur = cur->next;
        }
        payload->next = cur->next;
        cur->next = payload;
    }

    pthread_mutex_unlock(&payload_list_mutex);
}

static void* payload_alloc_inner_buf(size_t size);
struct payload *payload_new_or_reuse(int id, int priority, enum total_reduce_op op, size_t count,
                            void *in_buf, void *out_buf, TR_datatype data_type, void (*callback)(int))
{
    assert (out_buf);
    assert (in_buf != out_buf);   // currently only support none inplace only
    assert (op == ALLREDUCE);

    struct payload *payload = payload_get_from_id_nolock(id);
    if (payload == NULL) {
        payload = (struct payload*)alloc_host_mem(sizeof(struct payload));

        payload->next = NULL;
        payload->id = id;
        payload->iter = 0;
        payload->count = count;
        payload->data_type = data_type;
        payload->calculate2 = type_handlers[data_type].calculate2;
        payload->element_size = type_handlers[data_type].element_size;
        payload->op = op;
        payload->in_buf = in_buf;
        if (payload->in_buf == TR_IN_PLACE) {
            payload->inner_buf = payload_alloc_inner_buf(count*payload->element_size);
        } else {
            payload->inner_buf = NULL;
        }
        payload->algorithm = RING;
        payload->priority = priority;
        payload->out_buf = out_buf;

        payload->send_state = 0;
        payload->recv_state = 0;
        payload->comp_state = 0;

        payload->time_start = get_time();
        payload->time_end = -1.0;
        payload->time_due = -1.0;

        payload->callback = callback;

        payload_add(payload);
    } else {
        pthread_mutex_lock(&payload_list_mutex);
        assert (payload->count == count);
        assert (payload->data_type == data_type);
        assert (payload->op == op);
        assert (payload->priority == priority);
        if (payload->in_buf == TR_IN_PLACE) {
            assert (in_buf == TR_IN_PLACE);
            payload->inner_buf = payload_alloc_inner_buf(count*payload->element_size);
        } else {
            assert (in_buf != TR_IN_PLACE);
            payload->in_buf = in_buf;
        }
        payload->out_buf = out_buf;


        payload->iter++;

        payload->time_start = get_time();
        payload->time_end = -1.0;
        payload->time_due = -1.0;

        payload->callback = callback;
        payload->recv_state = 0;
        payload->comp_state = 0;
        payload->send_state = 0;
        pthread_mutex_unlock(&payload_list_mutex);
    }

    return payload;
}

struct payload *payload_get_from_id(int id)
{
    pthread_mutex_lock(&payload_list_mutex);
    struct payload *cur = payload_list;
    while (cur->next) {
        cur = cur->next;
        if (cur->id == id) {
            pthread_mutex_unlock(&payload_list_mutex);
            return cur;
        }
    }
    pthread_mutex_unlock(&payload_list_mutex);
    return NULL;
}

struct payload *payload_get_from_id_nolock(int id)
{
    struct payload *cur = payload_list;
    while (cur->next) {
        cur = cur->next;
        if (cur->id == id) {
            return cur;
        }
    }
    return NULL;
}

bool payload_overdue_p (struct payload *payload)
{
    return (payload->time_due >= 0.0);
}

static void payload_recycle_inner_buf(struct payload *payload);
// if called from user thread, external = true
// if called from total reduce thread, external = false
bool payload_check_done_p (struct payload *payload, bool external)
{
    int world_size = total_reduce_get_world_size();

    if (payload->send_state == 2*world_size - 2 &&
        payload->recv_state == 2*world_size - 2 &&
        payload->comp_state == 2*world_size - 2 ) {
        if (payload->time_end < 0) {
            payload->time_end = get_time();
        }
        if (payload->callback != NULL) {
            if (!external) {
                payload->callback(payload->id);
                payload->callback = NULL;
                payload_recycle_inner_buf(payload);
                return true;
            } else {
                return false;
            }
        }
        if (!external) {
            payload_recycle_inner_buf(payload);
        }
        return true;
    }
    assert (payload->send_state <= 2*world_size -2);
    assert (payload->recv_state <= 2*world_size -2);
    assert (payload->comp_state <= 2*world_size -2);
    return false;
}

static bool payload_check_ready_p (struct payload *payload)
{
    if (total_reduce_has_active_send_request_p(payload->id)) {
        // cannot make a payload ready if there is an active send request for it
        return false;
    }

    if (payload->send_state > payload->recv_state)
        return false;
    if (payload->send_state > payload->comp_state)
        return false;
    return !payload_check_done_p(payload, false);
}

// pick the payload with highest priority
struct payload *payload_pick_ready(
    #if PROFILE>=1
    int *profile_flag
    #else
    void
    #endif
)
{
    #if PROFILE>=1
    *profile_flag = 0;
    #endif
    struct payload *last_ready_small = NULL;
    struct payload *last_ready_large = NULL;
    bool has_overdue = false;

    pthread_mutex_lock(&payload_list_mutex);
    struct payload *cur = payload_list;
    while (cur->next) {
        cur = cur->next;

        // there might be overdue payload that is not ready
        // we should be aware of that
        if (!has_overdue && !payload_check_done_p(cur, false)) {
            if (payload_overdue_p(cur)) {
                has_overdue = true;
            }
        }

        if (payload_check_ready_p(cur)) {
            if (payload_overdue_p(cur)) {
                last_ready_small = last_ready_large = cur;
                break;
            }
            if (last_ready_small == NULL &&
                    cur->count/total_reduce_get_world_size()<LARGE_CHUNK_SIZE) {
                last_ready_small = cur;
            }
            if (last_ready_large == NULL &&
                    cur->count/total_reduce_get_world_size()>=LARGE_CHUNK_SIZE) {
                last_ready_large = cur;
            }
        }
    }
    pthread_mutex_unlock(&payload_list_mutex);

    bool sending_large_body_p = total_reduce_sending_large_body_p();

    if (last_ready_small != NULL && last_ready_large !=NULL) {
        if(last_ready_small == last_ready_large) {
            return last_ready_small;
        }

        bool small_overdue = payload_overdue_p(last_ready_small);
        bool large_overdue = payload_overdue_p(last_ready_large);

        if(small_overdue) {
            return last_ready_small;
        } else if (large_overdue) {
            return last_ready_large;
        }

        if (sending_large_body_p || has_overdue) {
            return last_ready_small;
        } else {
            return last_ready_large;
        }
    }

    if (last_ready_small != NULL) {
        return last_ready_small;
    }

    if (last_ready_large != NULL) {
        if (!sending_large_body_p && !has_overdue) {
            return last_ready_large;
        }
        #if PROFILE>=1
        *profile_flag = 1; // mark as large payload blocked
        #endif
        return NULL;
    }

    return NULL;
}

void print_payload_list(void)
{
    int world_size = total_reduce_get_world_size();

    pthread_mutex_lock(&payload_list_mutex);
    struct payload *cur = payload_list;
    while (cur->next) {
        cur = cur->next;

        size_t count = cur->count;
        float time_start = cur->time_start;
        float time_end = cur->time_end;
        float time_due = cur->time_due;
        float time_span = time_end-time_start;
        float time_overdue = time_due > time_end ? 0 : time_end-time_due;

        float time_bound = 2.0*(world_size-1)/world_size*count*32/10/1000000000;

        printf ("id=%3d, "
        #ifdef DEBUG
        "send_state=%d, recv_state=%d, comp_state=%d, "
        #endif
        "count=%10ld, time_bound=%.3f, span=%.3f, overdue=%.3f, start=%.3f, end=%.3f, due=%.3f\n",
        cur->id,
        #ifdef DEBUG
        cur->send_state, cur->recv_state, cur->comp_state,
        #endif
        cur->count, time_bound, time_span, time_overdue, time_start, time_end, time_due);
    }
    pthread_mutex_unlock(&payload_list_mutex);
}

void free_payload_list(void)
{
    pthread_mutex_lock(&payload_list_mutex);
    struct payload *cur = payload_list;
    cur = cur->next;
    while (cur) {
        struct payload *to_be_freed = cur;
        cur = cur->next;
        if (to_be_freed->inner_buf != NULL) {
            free(to_be_freed->inner_buf);
        }
        free(to_be_freed);
    }
    pthread_mutex_unlock(&payload_list_mutex);
}

bool payload_expecting(struct payload *payload, struct message_header *header)
{
    assert (payload);
    assert (header);

    return (payload->iter == header->iter &&
            payload->recv_state == header->state &&
            payload->comp_state == header->state);

}

// if called from user thread, external = true
// if called from total reduce thread, external = false
bool payload_all_done_p(bool external)
{
    pthread_mutex_lock(&payload_list_mutex);

    struct payload *cur = payload_list;
    while (cur->next) {
        cur = cur->next;

        // implementation depends on policy, currently using the naive implementation
        if (!payload_check_done_p(cur, external)) {
            pthread_mutex_unlock(&payload_list_mutex);
            return false;
        }
    }

    pthread_mutex_unlock(&payload_list_mutex);
    return true;
}

static inline void *get_src_ptr(struct payload *payload)
{
    if (payload->in_buf == TR_IN_PLACE) {
        assert (payload->inner_buf!=NULL);
        return payload->out_buf;
    } else {
        assert (payload->inner_buf==NULL);
        return payload->in_buf;
    }
}

static inline void *get_dst_ptr(struct payload *payload)
{
    return payload->out_buf;
}

// return true if a 'micro body' had been sent along with header
// return false if the body needs to be sent seperately
bool payload_send_step_header(struct payload *payload)
{
    int world_size = total_reduce_get_world_size();
    int world_rank = total_reduce_get_rank();
    int send_rank = total_reduce_get_succ_rank();

    void *src_buf, *dst_buf;

    src_buf = get_src_ptr(payload);
    dst_buf = get_dst_ptr(payload);

    return ring_send_step_header(payload->id, payload->send_state, payload->iter,
                          src_buf, dst_buf, payload->count, payload->element_size,
                          world_size, world_rank, send_rank);
}

static inline void *get_recv_ptr(struct payload *payload)
{
    if (payload->in_buf == TR_IN_PLACE) {
        assert (payload->inner_buf!=NULL);
        return payload->inner_buf;
    } else {
        assert (payload->inner_buf==NULL);
        return payload->out_buf;
    }
}

void payload_send_step_body(struct payload *payload)
{
    int world_size = total_reduce_get_world_size();
    int world_rank = total_reduce_get_rank();
    int send_rank = total_reduce_get_succ_rank();

    void *src_buf, *dst_buf;

    src_buf = get_src_ptr(payload);
    dst_buf = get_dst_ptr(payload);

    ring_send_step_body(payload->id, payload->send_state,
                        src_buf, dst_buf, payload->count, payload->element_size,
                        world_size, world_rank, send_rank);
}

void *payload_get_recv_buf  (struct payload *payload)
{
    int world_size = total_reduce_get_world_size();
    int world_rank = total_reduce_get_rank();

    void *dst_buf, *recv_buf;

    dst_buf = get_dst_ptr(payload);
    recv_buf = get_recv_ptr(payload);

    return ring_get_recv_buf(payload->recv_state,
                             dst_buf, recv_buf, payload->count, payload->element_size,
                             world_size, world_rank);
}

static void *payload_get_dst_buf  (struct payload *payload)
{
    int world_size = total_reduce_get_world_size();
    int world_rank = total_reduce_get_rank();

    void *dst_buf = get_dst_ptr(payload);

    return ring_get_recv_buf(payload->recv_state,
                             dst_buf, dst_buf, payload->count, payload->element_size,
                             world_size, world_rank);
}

bool payload_do_compute(struct payload *payload, struct pending_message *message)
{
    int world_size = total_reduce_get_world_size();
    int world_rank = total_reduce_get_rank();

    void *out_buf, *in_buf1, *in_buf2, *src_buf;
    size_t size;
    void *recv_buf = payload_get_recv_buf(payload);
    void *dst_buf = payload_get_dst_buf(payload);
    bool ret_val = false;

    src_buf = get_src_ptr(payload);

    ring_get_compute_buffers(payload->recv_state,
                             src_buf, message?message->buf:recv_buf, dst_buf,
                             payload->count, payload->element_size, world_size, world_rank,
                             &out_buf, &in_buf1, &in_buf2, &size);
    if (out_buf != NULL) {
        if (!FORCE_CONCURRENT_COMPUTING &&
            (FORCE_SERIAL_COMPUTING || payload->time_due >= 0.0 || payload->count <SMALL_MESSAGE_SIZE)) {
            // overdue payload are urgent, just calculate it
            // small payload has low cost to compute, just compute it

            payload->calculate2(
                    #if PRINT_CALC_TRACE
                    payload->id, payload->comp_state,
                    #endif
                    out_buf, in_buf1, in_buf2, size);
            payload->comp_state++;
            if (message) {
                ret_val = true;
            }
        } else {
            compute_request_list_add(payload, out_buf, in_buf1, in_buf2, size, message);
        }
    } else {
        if (message) {
            // this is the only place where a device memory copy will happen
            // memory copy should be avoided, but here concurrency is more
            // important
            copy_device_mem(recv_buf, message->buf, message->header.size);
        }
        payload->comp_state++;
    }
    payload->recv_state++;
    payload_check_done_p(payload, false);
    return ret_val;
}

static void* payload_alloc_inner_buf(size_t size)
{
    struct buf_pool *ptr = &inner_buf_pool;
    struct buf_pool *ret = NULL;

    pthread_mutex_lock(&inner_buf_pool_mutex);
    while(ptr->next) {
        if (ptr->next->size == size) {
            ret = ptr->next;
            ptr->next = ret->next;
            pthread_mutex_unlock(&inner_buf_pool_mutex);
            return ret;
        }
        ptr = ptr->next;
    }

    pthread_mutex_unlock(&inner_buf_pool_mutex);
    return alloc_device_mem(size);
}

static void payload_recycle_inner_buf(struct payload *payload)
{
    if (payload->inner_buf != NULL) {
        pthread_mutex_lock(&inner_buf_pool_mutex);
        size_t size = payload->count*payload->element_size;
        if (size < sizeof(inner_buf_pool)) {
            free_device_mem(payload->inner_buf);
        } else {
            struct buf_pool *ptr = (struct buf_pool*)payload->inner_buf;
            ptr->size = size;
            ptr->next = inner_buf_pool.next;
            inner_buf_pool.next = ptr;
        }
        pthread_mutex_unlock(&inner_buf_pool_mutex);
        payload->inner_buf = NULL;
    }
}
