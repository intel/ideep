#ifndef __PAYLOAD__H__
#define __PAYLOAD__H__

#include "total_reduce.h"
#include "pending_message.h"
#include "pal.h"

enum total_reduce_op {ALLREDUCE};
enum total_reduce_algorithm {RING};

struct payload {
    struct payload *next;

    int id;
    int iter;
    int priority;
    enum total_reduce_op op;
    size_t count;
    void *in_buf;
    void *out_buf;
    void *inner_buf;
    TR_datatype data_type;
    int element_size;
    enum total_reduce_algorithm algorithm;
    int send_state;
    int recv_state;
    int comp_state;

    float time_start;
    float time_end;
    float time_due;

    void (*callback)(int);

    void (*calculate2)(
                #if PRINT_CALC_TRACE
                int id, int state,
                #endif
                void *write_buf, const void *buf_src1, const void *buf_src2, int count);
};

void payload_list_init(void);
struct payload *payload_new_or_reuse(int id, int priority, enum total_reduce_op op, size_t size,
                                     void *in_buf, void *out_buf, TR_datatype data_type, void (*callback)(int));
struct payload *payload_get_from_id(int id);
struct payload *payload_get_from_id_nolock(int id);
bool payload_check_done_p (struct payload *payload, bool external);
struct payload *payload_pick_ready(
#if PROFILE>=1
int *profile_flag
#else
void
#endif
);
void print_payload_list(void);
void free_payload_list(void);
bool payload_expecting(struct payload *payload, struct message_header *header);
bool payload_all_done_p(bool external);
bool payload_send_step_header(struct payload *payload);
void payload_send_step_body(struct payload *payload);
void *payload_get_recv_buf  (struct payload *payload);
bool payload_do_compute(struct payload *payload, struct pending_message *message);

#endif
