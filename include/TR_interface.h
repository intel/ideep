#ifndef __TOTAL_REDUCE__H__
#define __TOTAL_REDUCE__H__

#define EXPORT __attribute__((visibility("default")))

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef enum TR_urgency {TR_NEED, TR_GREEDY} TR_urgency;

typedef enum TR_datatype {TR_FP32, TR_FP16, TR_INT32} TR_datatype;

#define TR_IN_PLACE NULL

EXPORT bool TR_available(void);

EXPORT void TR_init(int affinity);

EXPORT int TR_get_world_size(void);
EXPORT int TR_get_rank(void);

EXPORT void TR_allreduce(int id, int priority,
                  void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype);
EXPORT void TR_iallreduce(int id, int priority,
                   void *send_buf, void *recv_buf, size_t num_elements, TR_datatype datatype,
                   void (*callback)(int));
EXPORT void TR_bcast(int id, int priority,
              void *buffer, size_t num_elements, TR_datatype datatype, int root);
EXPORT void TR_wait(int id);
EXPORT bool TR_test(int id, TR_urgency urgency);
EXPORT void TR_set_urgent(int id);
EXPORT void TR_barrier(void);

EXPORT void TR_finalize(void);

#ifdef __cplusplus
}
#endif

#endif
