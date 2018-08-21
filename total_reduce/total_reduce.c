#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <math.h>

#include <pthread.h>

#include "pal.h"
#include "time.h"
#include "ring.h"
#include "total_reduce.h"
#include "pending_message.h"
#include "compute_request.h"
#include "payload.h"
#include "knobs.h"

// does not need MT protection
struct message_header send_header;
struct comm_req send_header_request;

static struct message_header recv_header;
static struct comm_req recv_header_request;

struct comm_body_info {
    struct comm_req request;
    int id;
    bool active_p;
    bool pending_p;
    struct pending_message *pending_message;
    size_t size;  // the size is a rough estimation, do not use it as exact size
    #if PROFILE >= 1
    float t_start;
    #endif
}
send_body_info[SEND_CONCURRENCY] = {{{0}}},
recv_body_info[RECV_CONCURRENCY] = {{{0}}};

#if PROFILE>=1
static float check_ready_payload;
static float check_ready_payload_fail;
static float check_ready_payload_block;
static float check_sending_header;
static float check_sending_body;
static float probe_recving_header;
static float check_recving_header;
static float check_recving_body;
float check_pending_list;
float start_recving_header_time;
float total_recving_header_span = 0.0;
float max_recving_header_span = 0.0;
int total_recving_header_count = 0;

static int send_counts[SEND_CONCURRENCY+1] = {0};
static int recv_counts[RECV_CONCURRENCY+1] = {0};
static int check_ready_block_count = 0;
static int check_ready_fail_count = 0;
#endif

static int world_size;
static int my_rank;
static int pred_rank;
static int succ_rank;

static void* total_reduce_thread_func(void *ptr);


static struct thread_args {
    int pred_rank;
    int my_rank;
    int succ_rank;
    int world_size;
} thread_args;

// varable that controls total_reduce loop, set to false if want to stop total_reduce loop
static bool volatile total_reduce_on = true;

static int get_inactive_send_request(void)
{
    for (int i=0; i<SEND_CONCURRENCY; i++) {
        if (!send_body_info[i].active_p) {
            return i;
        }
    }
    assert(!"should not get here\n");
}

static int get_inactive_recv_request(void)
{
    for (int i=0; i<RECV_CONCURRENCY; i++) {
        if (!recv_body_info[i].active_p) {
            return i;
        }
    }
    assert(!"should not get here\n");
}

bool total_reduce_has_active_send_request_p(int id)
{
    for (int i=0; i<SEND_CONCURRENCY; i++) {
        if (send_body_info[i].active_p && send_body_info[i].id == id) {
            return true;
        }
    }
    return false;
}

bool total_reduce_sending_large_body_p(void)
{
    for (int i=0; i<SEND_CONCURRENCY; i++) {
        if (send_body_info[i].active_p && send_body_info[i].size >= LARGE_CHUNK_SIZE) {
            return true;
        }
    }
    return false;
}

struct message_header *total_reduce_get_send_header(void) {return &send_header; }
struct comm_req *total_reduce_get_send_header_request(void) { return &send_header_request; }
struct comm_req *total_reduce_get_send_body_request(int id, size_t size)
{
    int index = get_inactive_send_request();
    send_body_info[index].active_p = true;
    send_body_info[index].id = id;
    send_body_info[index].size = size;
    #if PROFILE >= 2
    send_body_info[index].t_start = get_time();
    #endif
    return &(send_body_info[index].request);
}

int total_reduce_get_world_size(void) { return world_size; }
int total_reduce_get_rank(void) { return my_rank; }
int total_reduce_get_pred_rank(void) { return pred_rank; }
int total_reduce_get_succ_rank(void) { return succ_rank; }

static pthread_t total_reduce_thread;

// total reduce implementation
void total_reduce_init(int affinity)
{
    // initialize MPI
    comm_init(&my_rank, &world_size);

    pred_rank = (my_rank+world_size-1) % world_size;
    succ_rank = (my_rank+1) % world_size;

    // initialize total reduce
    static bool total_reduce_inited = false;

    assert (!total_reduce_inited);

    total_reduce_on = true;

    init_time();

    payload_list_init();
    pending_message_list_init();
    compute_request_list_init();

    //init thread for total reduce
    thread_args.pred_rank  = pred_rank;
    thread_args.my_rank    = my_rank;
    thread_args.succ_rank  = succ_rank;
    thread_args.world_size = world_size;
    pthread_create (&total_reduce_thread, NULL, total_reduce_thread_func, (void*)&thread_args);
    if (affinity >= 0) {
        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);
        CPU_SET(affinity, &cpuset);
        pthread_setaffinity_np(total_reduce_thread, 1, &cpuset);
    }
}

void total_reduce_finalize(void)
{
    total_reduce_on = false;
    pthread_join(total_reduce_thread, NULL);
    comm_finalize();
}

static bool total_reduce_check_grace_exit_p()
{
    if (!total_reduce_on) {
        return payload_all_done_p(false);
    }
    return false;
}

void total_reduce_allreduce(int id, int priority,
                            void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype)
{
    struct payload * payload = payload_new_or_reuse(id, priority, ALLREDUCE, num_elements,
                                                    send_buf, recv_buf, datatype, NULL);
    payload->time_due = get_time();
    while(1) {
        if (payload_check_done_p(payload, true))
            break;
    }
}

void total_reduce_iallreduce(int id, int priority,
                   void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype,
                   void (*callback)(int))
{
    payload_new_or_reuse(id, priority, ALLREDUCE, num_elements, send_buf, recv_buf, datatype, callback);
}

void total_reduce_bcast(int id, int priority, void *buffer, size_t num_elements, TR_datatype datatype, int root)
{
    if (total_reduce_get_rank() !=root) {
        bzero (buffer, num_elements*sizeof(float));
    }
    total_reduce_allreduce(id, priority, TR_IN_PLACE, buffer, num_elements, datatype);
}

void total_reduce_barrier(void)
{
    int dummy[1] = {0};
    total_reduce_allreduce(-1, 0, TR_IN_PLACE, dummy, 1, TR_INT32);
}

static bool message_sending_header_p   = false;
static int message_sending_body_count  = 0;
static bool message_recving_header_p   = false;
static int message_recving_body_count = 0;
static struct payload* sending_payload;
static bool sending_micro_body_p;

static void do_start_sending_header(void);
static void do_start_sending_body(void);
static void do_probe_recving_header(void);
static void do_start_recving_body(void);
static void do_check_sending_body(void);
static void do_check_recving_body(void);

static void* total_reduce_thread_func(void *ptr)
{
    pred_rank  = ((struct thread_args*)ptr)->pred_rank;
    my_rank    = ((struct thread_args*)ptr)->my_rank;
    succ_rank  = ((struct thread_args*)ptr)->succ_rank;
    world_size = ((struct thread_args*)ptr)->world_size;

    #if PROFILE>=1
    check_ready_payload = 0.0;
    check_ready_payload_fail = 0.0;
    check_sending_header = 0.0;
    check_sending_body = 0.0;
    probe_recving_header = 0.0;
    check_recving_header = 0.0;
    check_recving_body = 0.0;
    check_pending_list = 0.0;
    make_compute_progress = 0.0;
    #endif

    #ifdef DEBUG
    bool debug_printed = false;
    #endif

    float max_iter_time=0.0;
    float total_iter_time=0.0;
    float total_iter_count=0;
    float prev_iter_time = get_time();
    for (;;) {
        float cur_iter_time = get_time();
        total_iter_time += cur_iter_time - prev_iter_time;
        total_iter_count ++;
        if (cur_iter_time - prev_iter_time >max_iter_time) {
            max_iter_time = cur_iter_time - prev_iter_time;
        }
        prev_iter_time = cur_iter_time;

        if (total_reduce_check_grace_exit_p()) { break; }

        do_start_sending_header();

        do_start_sending_body();

        do_probe_recving_header();

        do_start_recving_body();

        do_check_sending_body();

        do_check_recving_body();

        pending_message_process();

        compute_request_progress();

        #ifdef DEBUG
        if (get_time() > 10 && !debug_printed) {
            printf ("message_sending_header_p = %d, "
                    "message_sending_body_count = %d, "
                    "message_recving_header_p = %d, "
                    "message_recving_body_count = %d, ",
                message_sending_header_p,
                message_sending_body_count,
                message_recving_header_p,
                message_recving_body_count
            );
            print_payload_list();
            debug_printed = true;
        }
        #endif
    }

    // after loop

    #if PROFILE >= 2
    if (my_rank == 0)
        print_payload_list();
    #endif
    #if PROFILE >= 1
    update_time();
    if (my_rank == 0) {
        float dark_matter = time_now - (
                            check_ready_payload
                          + check_ready_payload_fail
                          + check_ready_payload_block
                          + check_sending_header
                          + check_sending_body
                          + probe_recving_header
                          + check_recving_header
                          + check_recving_body
                          + check_pending_list
                          + make_compute_progress
                        );
        float overhead = check_ready_payload
                   + check_ready_payload_fail
                   + check_ready_payload_block
                   + check_sending_header
                   + probe_recving_header
                   + check_recving_header
                   + check_pending_list;
        printf ("times:"
            " total=%.3f\toverhead=%.1f%%\tdark_matter=%.1f%%\n"
            "       check_ready_payload_time=%.1f%%(hit=%.1f%% idle=%.1f%% block=%.1f%%)\n"
            "       sending_header_time=%.1f%%\tsending_body_time=%.1f%%\n"
            "       probe_recving_header=%.1f%%\trecving_header_time=%.1f%%\trecving_body_time=%.1f%%\n"
            "       check_pending_list=%.1f%%\tmake_compute_progress=%.1f%%\n",
            time_now,
            overhead/time_now*100,
            dark_matter/time_now*100,
            (check_ready_payload
            +check_ready_payload_fail
            +check_ready_payload_block)/time_now*100,
            check_ready_payload/time_now*100,
            check_ready_payload_fail/time_now*100,
            check_ready_payload_block/time_now*100,
            check_sending_header/time_now*100,
            check_sending_body/time_now*100,
            probe_recving_header/time_now*100,
            check_recving_header/time_now*100,
            check_recving_body/time_now*100,
            check_pending_list/time_now*100,
            make_compute_progress/time_now*100);

        int total_send_count=0, total_recv_count=0;

        for (int i=0; i<SEND_CONCURRENCY; i++) {
            total_send_count += send_counts[i];
        }
        for (int i=0; i<RECV_CONCURRENCY; i++) {
            total_recv_count += recv_counts[i];
        }
        printf ("total_loop_count=%d\n", total_send_count);
        printf ("check_ready_block_counts: %.1f%%\n", check_ready_block_count*100.0/total_send_count);
        printf ("check_ready_fail_counts: %.1f%%\n", check_ready_fail_count*100.0/total_send_count);
        printf ("send_counts:\n");
        for (int i=0; i<SEND_CONCURRENCY; i++) {
            printf ("%d: %.1f%%(%d)\n", i, send_counts[i]*100.0/total_send_count, send_counts[i]);
        }
        printf ("recv_counts:\n");
        for (int i=0; i<RECV_CONCURRENCY; i++) {
            printf ("%d: %.1f%%(%d)\n", i, recv_counts[i]*100.0/total_recv_count, recv_counts[i]);
        }
        printf ("average recv header span %f\n", total_recving_header_span/total_recving_header_count);
        printf ("maximum recv header span %f\n", max_recving_header_span);
        printf ("average iteration time %f\n", total_iter_time/total_iter_count);
        printf ("max iteration time %f\n", max_iter_time);
    }
    #endif

    assert(compute_request_list_empty_p());
    assert(pending_message_list_empty_p());
    free_payload_list();

    return NULL;
}

static void do_start_sending_header(void)
{
    #if PROFILE>=1
    send_counts[message_sending_body_count]++;
    #endif

    if (!message_sending_header_p && message_sending_body_count<SEND_CONCURRENCY) {
        #if PROFILE>=1
        update_time();
        #endif

        #if PROFILE>=1
        int profile_flag;
        #endif

        // progress send part
        struct payload *payload = payload_pick_ready(
                                                #if PROFILE>=1
                                                    &profile_flag
                                                #endif
                                                    );
        if (payload!= NULL) {
            sending_micro_body_p = payload_send_step_header(payload);
            message_sending_header_p = true;
            sending_payload = payload;
            #if PROFILE>=1
            accu_time(&check_ready_payload);
            #endif
        } else {
            #if PROFILE>=1
            if (profile_flag == 1) {
                accu_time(&check_ready_payload_block);
                check_ready_block_count++;
            } else {
                accu_time(&check_ready_payload_fail);
                if (message_sending_body_count == 0) {
                    check_ready_fail_count++;
                }
            }
            #endif
        }
    }
}

static void do_start_sending_body(void)
{
    if (message_sending_header_p) {
        #if PROFILE>=1
        update_time();
        #endif

        if (sending_micro_body_p) {
            int flag = 0;
            flag = comm_test(&send_header_request);
            if (flag) {
                message_sending_header_p = false;
                sending_payload->send_state++;
                payload_check_done_p(sending_payload, false);
            }
        } else {
            int flag = 0;
            flag = comm_test(&send_header_request);
            if (flag) {
                payload_send_step_body(sending_payload);
                message_sending_header_p = false;
                message_sending_body_count++;
                assert (message_sending_body_count <= SEND_CONCURRENCY);
            }
        }

        #if PROFILE>=1
        accu_time(&check_sending_header);
        #endif
    }
}

static void do_probe_recving_header(void)
{
    static float last_check_recv_header_time = 0.0;
    if (!message_recving_header_p && message_recving_body_count<RECV_CONCURRENCY) {
        #if PROFILE>=1
        float cur_time = update_time();
        #else
        float cur_time = get_time();
        #endif

        if (cur_time - last_check_recv_header_time > -0.00001) {
            int count=0;

            count = comm_probe(pred_rank);
            if (count >= 0) {
                comm_recv(&recv_header, count, pred_rank, &recv_header_request);
                #if PROFILE>=1
                start_recving_header_time = get_time();
                #endif
                message_recving_header_p = true;
            }
            last_check_recv_header_time = cur_time;
        }

        #if PROFILE>=1
        accu_time(&probe_recving_header);
        #endif
    }
}

static void do_start_recving_body(void)
{
    if (message_recving_header_p) {
        #if PROFILE>=1
        update_time();
        #endif

        int flag = 0;
        flag = comm_test(&recv_header_request);
        if (flag) {
            #if PROFILE>=1
            float recv_header_span = get_time() - start_recving_header_time;
            total_recving_header_span = recv_header_span;
            total_recving_header_count ++;
            if (recv_header_span > max_recving_header_span) {
                max_recving_header_span = recv_header_span;
            }
            #endif
            // only allow one active recvier for each id, avoid out-of-order in same id
            struct payload *payload = payload_get_from_id (recv_header.id);

            if (payload && payload_expecting(payload, &recv_header)) {
                float *recv_buf = (float*)payload_get_recv_buf (payload);
                if (recv_header.size <= MICRO_MESSAGE_SIZE) {
                    memcpy(recv_buf, recv_header.micro_body, recv_header.size);
                    payload_do_compute(payload, NULL);
                } else {
                    int index = get_inactive_recv_request();

                    comm_recv(recv_buf, recv_header.size, pred_rank, &(recv_body_info[index].request));
                    recv_body_info[index].pending_p = false;
                    recv_body_info[index].active_p = true;
                    recv_body_info[index].id = recv_header.id;
                    #if PROFILE>=1
                    recv_body_info[index].t_start = get_time();
                    #endif
                    message_recving_body_count++;
                }
            } else { // pending message
                if (recv_header.size <= MICRO_MESSAGE_SIZE) {
                    struct pending_message *pending_message = pending_message_new (recv_header);
                    memcpy(pending_message->buf, recv_header.micro_body, recv_header.size);
                    pending_message_add (pending_message);
                } else {
                    int index = get_inactive_recv_request();

                    recv_body_info[index].pending_message = pending_message_new (recv_header);
                    comm_recv(recv_body_info[index].pending_message->buf,
                         recv_header.size, pred_rank,
                         &(recv_body_info[index].request));
                    recv_body_info[index].pending_p = true;
                    recv_body_info[index].active_p = true;
                    recv_body_info[index].id = recv_header.id;
                    #if PROFILE>=1
                    recv_body_info[index].t_start = get_time();
                    #endif
                    message_recving_body_count++;
                }
            }

            message_recving_header_p = false;
            assert (message_recving_body_count <= RECV_CONCURRENCY);
        }

        #if PROFILE>=1
        accu_time(&check_recving_header);
        #endif
    }
}

static void do_check_sending_body(void)
{
    if (message_sending_body_count > 0) {
        #if PROFILE>=1
        update_time();
        #endif

        int flag[SEND_CONCURRENCY] = {0};
        for (int i=0; i<SEND_CONCURRENCY; i++) {
            if (send_body_info[i].active_p) {
                flag[i] = comm_test(&(send_body_info[i].request));
            }
        }
        for (int i=0; i<SEND_CONCURRENCY; i++) {
            if (flag[i]) {
                #if PROFILE >= 3
                float time =  get_time()-send_body_info[i].t_start;
                printf ("send_body_size = %d, send_time = %f, bandwidth=%f,\n",
                        send_body_info[i].send_body_size, time,
                        send_body_info[i].send_body_size*32/time/1000000000);
                #endif

                struct payload *payload = payload_get_from_id(send_body_info[i].id);
                assert (payload);
                payload->send_state++;
                payload_check_done_p(payload, false);
                send_body_info[i].active_p = false;
                message_sending_body_count--;
            }
        }

        #if PROFILE>=1
        accu_time(&check_sending_body);
        #endif
    }
}

static void do_check_recving_body(void)
{
    #if PROFILE>=1
    recv_counts[message_recving_body_count]++;
    #endif
    if (message_recving_body_count > 0) {
        #if PROFILE>=1
        update_time();
        #endif

        int flag[RECV_CONCURRENCY] = {0};
        for (int i=0; i<RECV_CONCURRENCY; i++) {
            if (recv_body_info[i].active_p) {
                flag[i] = comm_test(&(recv_body_info[i].request));
            }
        }

        for (int i=0; i<RECV_CONCURRENCY; i++) {
            if (flag[i]) {
                if (!recv_body_info[i].pending_p) {
                    struct payload *payload = payload_get_from_id (recv_body_info[i].id);
                    payload_do_compute(payload, NULL);
                } else {
                    pending_message_add (recv_body_info[i].pending_message);
                }

                recv_body_info[i].active_p = false;
                message_recving_body_count--;
            }
        }

        #if PROFILE>=1
        accu_time(&check_recving_body);
        #endif
    }
}
