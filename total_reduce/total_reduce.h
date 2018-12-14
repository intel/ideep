#ifndef __TOTAL_REDUCE_INTERNAL__H__
#define __TOTAL_REDUCE_INTERNAL__H__
#include "pal.h"
#include "knobs.h"

struct message_header {
    int id;
    int iter;
    int state;
    int size;
    char micro_body[MICRO_MESSAGE_SIZE];
};

struct message_header *total_reduce_get_send_header(void);
struct comm_req *total_reduce_get_send_header_request(void);
struct comm_req *total_reduce_get_send_body_request(int id, size_t size);
bool total_reduce_has_active_send_request_p(int id);
bool total_reduce_sending_large_body_p(void);
int total_reduce_get_world_size(void);
int total_reduce_get_rank(void);
int total_reduce_get_pred_rank(void);
int total_reduce_get_succ_rank(void);
void total_reduce_allreduce(int id, int priority,
                            void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype);
void total_reduce_iallreduce(int id, int priority,
                             void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype,
                             void (*callback)(int));
void total_reduce_bcast(int id, int priority, void *buffer, size_t num_elements, TR_datatype datatype, int root);
void total_reduce_barrier(void);

void total_reduce_init();
void total_reduce_finalize();

#if PROFILE>=2
extern float t_send_body_start;
extern size_t send_body_size;
#endif

#if PROFILE>=1
extern float check_pending_list;
#endif

#endif
