#include <stdlib.h>
#include <assert.h>
#include "total_reduce.h"
#include "pending_message.h"
#include "payload.h"
#include "time.h"
#include "pal.h"

static struct pending_message *pending_message_list = NULL;

void pending_message_list_init(void)
{
    assert (pending_message_list == NULL);
    pending_message_list = (struct pending_message*)alloc_host_mem(sizeof(struct pending_message));
    pending_message_list -> next = NULL;
}

bool pending_message_list_empty_p(void)
{
    assert (pending_message_list);
    return pending_message_list->next == NULL;
}

static struct pending_message *pending_message_first_ready(void)
{
    struct pending_message *cur = pending_message_list;
    while (cur->next) {
        cur = cur->next;
        int id = cur->header.id;
        struct payload *payload = payload_get_from_id(id);
        if (payload!=NULL &&
            payload_expecting(payload, &cur->header)) {
            return cur;
        }
    }
    return NULL;
}

struct pending_message *pending_message_new (struct message_header header)
{
    struct pending_message *pending_message = (struct pending_message*)alloc_host_mem(sizeof(struct pending_message));
    pending_message->header = header;
    pending_message->next = NULL;
    pending_message->buf = alloc_device_mem(header.size);
    return pending_message;
}

static void pending_message_detach(struct pending_message *message)
{
    assert (message != NULL);

    struct pending_message *cur = pending_message_list;
    while (cur->next) {
        if (cur->next == message) {
            cur->next = message->next;
            break;
        }
        cur = cur->next;
    }
}

static void pending_message_delete(struct pending_message *message)
{
    pending_message_detach(message);
    free_device_mem(message->buf);
    free_host_mem(message);
}

void pending_message_add (struct pending_message *message)
{
    assert (message != NULL);

    struct pending_message *cur = pending_message_list;
    while (cur->next) {
        cur = cur->next;
    }
    cur->next = message;
}

void pending_message_process(void)
{
    #if PROFILE>=1
    update_time();
    #endif
    do {
        struct pending_message *message = pending_message_first_ready();
        if (message == NULL) {
            break;
        }

        struct payload *payload = payload_get_from_id(message->header.id);

        bool need_delete_message = payload_do_compute(payload, message);
        if (need_delete_message) {
            pending_message_delete(message);
        } else {
            pending_message_detach(message);
        }
    } while(1);
    #if PROFILE>=1
    accu_time(&check_pending_list);
    #endif
}
