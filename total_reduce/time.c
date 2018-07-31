#include <time.h>
#include <sys/time.h>
#include <assert.h>
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
    float start = get_time();
    while (get_time() - start <= second);
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
