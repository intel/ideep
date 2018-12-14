#include <stdlib.h>
#include <assert.h>

#include "TR_interface.h"

bool TR_available(void)
{
    return false;
}

void TR_init(int affinity)
{
    assert (!"Should not get here");
}

int TR_get_world_size(void)
{
    assert (!"Should not get here");
    return 0;
}

int TR_get_rank(void)
{
    assert (!"Should not get here");
    return 0;
}

void TR_allreduce(int id, int priority,
                  void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype)
{
    assert (!"Should not get here");
}

void TR_iallreduce(int id, int priority,
                   void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype,
                   void (*callback)(int))
{
    assert (!"Should not get here");
}

void TR_bcast(int id, int priority,
              void *buffer, size_t num_elements, TR_datatype datatype, int root)
{
    assert (!"Should not get here");
}

void TR_wait(int id)
{
    assert (!"Should not get here");
}

bool TR_test(int id, TR_urgency urgency)
{
    assert (!"Should not get here");
    return false;
}

void TR_set_urgent(int id)
{
    assert (!"Should not get here");
}

void TR_barrier(void)
{
    assert (!"Should not get here");
}

void TR_finalize(void)
{
    assert (!"Should not get here");
}
