#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <errno.h>
#include "time.h"

#if PROFILE>=1
float time_now;
#endif

struct timeval t_begin;

void init_time(void)
{
    gettimeofday(&t_begin, NULL);
}

float get_time(void)
{
    struct timeval t_now;
    gettimeofday(&t_now, NULL);
    return t_now.tv_sec-t_begin.tv_sec+(t_now.tv_usec-t_begin.tv_usec)/1000000.0;
}

void sleepf(float second)
{
    assert (second>=0);
    struct timespec span, remain, *val, *rem, *swap_temp;
    span.tv_sec = (int)second;
    span.tv_nsec = (second - (int)second) * 1000000000;

    val = &span;
    rem = &remain;

    while (nanosleep(val, rem) && errno==EINTR) {
        swap_temp = val;
        val = rem;
        rem = swap_temp;
    }
}

#if PROFILE>=1
float update_time(void)
{
    time_now = get_time();
    return time_now;
}

void accu_time(float *time_var)
{
    *time_var += get_time() - time_now;
}
#endif
