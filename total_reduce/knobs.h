#ifndef __KNOBS__H__
#define __KNOBS__H__

// how many concurrent isend body and irecv body can coexist
#define SEND_CONCURRENCY 1
#define RECV_CONCURRENCY 1

// the size of CHUNK that is considered 'large'.  There will be only one large chunk in flight
#define LARGE_CHUNK_SIZE 10000

// the size of message that is considered 'small'.  Small message will have
// bigger priority in computation and communication
#define SMALL_MESSAGE_SIZE 1024

// the size of message that is considered 'micro', micro message will be pack in header directly
#define MICRO_MESSAGE_SIZE 4096

// the size of CHUNK that each compute will take.  This number decide the concurrency
// between computing and data transfer
#define COMPUTE_CHUNK_SIZE 16384

// for debugging purpose, force computing to be serial, instead of concurrent to communication
#define FORCE_SERIAL_COMPUTING false
// for debugging purpose, force computing to be concurrent, no computing is blocking
#define FORCE_CONCURRENT_COMPUTING false

// for debugging purpose, print calculation trace if only one element pair are added up in each step
#define PRINT_CALC_TRACE false

#define BYPASS_CALC false

#endif
