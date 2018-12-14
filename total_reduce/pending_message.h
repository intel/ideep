#ifndef __PENDING_MESSAGE__H__
#define __PENDING_MESSAGE__H__

#include <stdlib.h>
struct pending_message {
    struct pending_message *next;
    struct message_header header;
    void *buf;
};

void pending_message_list_init(void);
bool pending_message_list_empty_p(void);
struct pending_message *pending_message_new (struct message_header header);
void pending_message_add (struct pending_message *message);
void pending_message_process(void);

#endif
