#ifndef __COMPUTE_REQUEST__H__
#define __COMPUTE_REQUEST__H__
struct compute_request {
    struct compute_request *next;
    struct payload *payload;
    void *out_buf;
    void *in_buf1;
    void *in_buf2;
    size_t size;
    struct pending_message *message;
    int    progress;
};

#if PROFILE>=1
extern float make_compute_progress;
#endif

void compute_request_list_init(void);
bool compute_request_list_empty_p(void);
void compute_request_list_add(struct payload *payload, void *out_buf, void *in_buf1, void *in_buf2, size_t size,
                              struct pending_message *message);
void compute_request_progress(void);
#endif
