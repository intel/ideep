#ifndef __TIME__H__
#define __TIME__H__

#ifdef __cplusplus
extern "C" {
#endif

#if PROFILE>=1
extern float time_now;
#endif

void init_time(void);
float get_time(void);
void sleepf(float second);

#if PROFILE>=1
float update_time(void);
void accu_time(float *time_var);
#endif

#ifdef __cplusplus
}
#endif

#endif

