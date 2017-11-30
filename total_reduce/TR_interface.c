#include <stdlib.h>
#include <assert.h>

#include "pal.h"
#include "ring.h"
#include "payload.h"
#include "time.h"
#include "total_reduce.h"
#include "TR_interface.h"

void TR_init(void)
{
    total_reduce_init();
}

int TR_get_world_size(void)
{
    return total_reduce_get_world_size();
}

int TR_get_rank(void)
{
    return total_reduce_get_rank();
}

void TR_allreduce(int id, int priority,
                  void *send_buf, void *recv_buf, size_t num_elements, TR_Datatype datatype)
{
    assert (id >= 0);
    total_reduce_allreduce(id, priority, send_buf, recv_buf, num_elements, datatype);
}

void TR_iallreduce(int id, int priority,
                   void *send_buf, void *recv_buf, size_t num_elements, TR_Datatype datatype,
                   void (*callback)(int))
{
    assert (id >= 0);
    total_reduce_iallreduce(id, priority, send_buf, recv_buf, num_elements, datatype, callback);
}

void TR_bcast(int id, int priority,
              void *buffer, size_t num_elements, TR_Datatype datatype, int root)
{
    total_reduce_bcast(id, priority, buffer, num_elements, datatype, root);
}

void TR_wait(int id)
{
    struct payload * payload = payload_get_from_id(id);
    if (payload == NULL) {
        return;
    }
    payload->time_due = get_time();
    while(1) {
        if (payload_check_done_p(payload)) {
            break;
        }
    }
}

bool TR_test(int id, enum urgency urgency)
{
    struct payload * payload = payload_get_from_id(id);
    if (payload == NULL) {
        return false;
    }

    if (urgency == NEED) {
        payload->time_due = get_time();
    }

    bool ret_val = payload_check_done_p(payload);
    return ret_val;
}

void TR_set_urgent(int id)
{
    TR_test(id, NEED);
}

void TR_barrier(void)
{
    total_reduce_barrier();
}

void TR_finalize(void)
{
    total_reduce_finalize();
}
