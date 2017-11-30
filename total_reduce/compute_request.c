#include <assert.h>
#include <stdlib.h>
#include "pal.h"
#include "time.h"
#include "total_reduce.h"
#include "pending_message.h"
#include "compute_request.h"
#include "payload.h"
#include "knobs.h"

#if PROFILE>=1
float make_compute_progress;
#endif

static struct compute_request *compute_request_list = NULL;

void compute_request_list_init(void)
{
    assert (compute_request_list == NULL);
    compute_request_list = (struct compute_request*) alloc_host_mem(sizeof(struct compute_request));
    compute_request_list -> next = NULL;
}

bool compute_request_list_empty_p(void)
{
    assert (compute_request_list);
    return compute_request_list->next == NULL;
}

void compute_request_list_add(struct payload *payload, void *out_buf, void *in_buf1, void *in_buf2,
                              size_t size, struct pending_message *message)
{
    struct compute_request *ptr = (struct compute_request*) alloc_host_mem(sizeof(struct compute_request));
    ptr->next    = NULL;
    ptr->payload = payload;
    ptr->out_buf = out_buf;
    ptr->in_buf1 = in_buf1;
    ptr->in_buf2 = in_buf2;
    ptr->size    = size;
    ptr->message = message;
    ptr->progress = 0;

    struct compute_request *cur = compute_request_list;
    while (cur->next) {
        cur = cur->next;
    }

    cur ->next = ptr;
}

void compute_request_progress(void)
{
    #if PROFILE>=1
    update_time();
    #endif

    struct compute_request *cur = compute_request_list;
    struct compute_request *next = compute_request_list->next;
    size_t quota = COMPUTE_CHUNK_SIZE;

    while (next && quota>0) {
        size_t remain = next->size - next->progress;

        char *out_buf = (char*)(next->out_buf);
        char *in_buf1 = (char*)(next->in_buf1);
        char *in_buf2 = (char*)(next->in_buf2);
        int element_size = next->payload->element_size;

        if (remain > quota) {
            next->payload->calculate2(
                       #if PRINT_CALC_TRACE
                       next->payload->id, next->payload->comp_state,
                       #endif
                       out_buf+next->progress*element_size,
                       in_buf1+next->progress*element_size,
                       in_buf2+next->progress*element_size,
                       quota);
            next->progress += quota;
            break;
        } else {
            next->payload->calculate2(
                       #if PRINT_CALC_TRACE
                       next->payload->id, next->payload->comp_state,
                       #endif
                       out_buf+next->progress*element_size,
                       in_buf1+next->progress*element_size,
                       in_buf2+next->progress*element_size,
                       remain);
            next->payload->comp_state++;
            if (next->message) {
                free_device_mem(next->message->buf);
                free_host_mem(next->message);
            }
            cur->next = next->next;
            free_host_mem(next);
            next = cur->next;
            quota = quota - remain;
        }
    }

    #if PROFILE>=1
    accu_time(&make_compute_progress);
    #endif
}
