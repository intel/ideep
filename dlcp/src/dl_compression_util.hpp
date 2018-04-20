/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


#ifndef DL_COMPRESSION_UTIL_HPP
#define DL_COMPRESSION_UTIL_HPP

#include <assert.h>
#include <cstring>
#include <ctype.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

extern int g_log_th; // log threash hold

#define GET_TID() syscall(SYS_gettid)
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define DLCP_LOG(log_level, fmt, ...)   \
do {                                    \
    if (log_level <= g_log_th)          \
    {                                   \
        char time_buf[20];              \
        dl_comp_get_time(time_buf, 20); \
        switch (log_level)              \
        {                               \
            case ERROR:                 \
            {                           \
                printf("%s: ERROR: (%ld): %s:%d " fmt "\n", time_buf, GET_TID(),   \
                       __FUNCTION__, __LINE__, ##__VA_ARGS__);                     \ 
                break;                                                             \
            }                                                                      \
            case INFO:                                                             \
            {                                                                      \
                printf("(%ld):" fmt "\n", GET_TID(), ##__VA_ARGS__);               \
                break;                                                             \
            }                                                                      \
            case DEBUG:                                                            \
            case TRACE:                                                            \
            {                                                                      \
                printf("%s: (%ld): %s:%d " fmt "\n", time_buf, GET_TID(),          \
                       __FUNCTION__, __LINE__, ##__VA_ARGS__);                     \
                break;                                                             \
            }                                                                      \
            default:                                                               \
            {                                                                      \
                printf("(%ld):" fmt "\n", GET_TID(), ##__VA_ARGS__);               \
            }                                                                      \
        }                                                                          \
        fflush(stdout);                                                            \
    }                                                                              \
} while (0)

#define DLCP_ASSERT(cond, fmt, ...)                                                     \
do                                                                                      \
{                                                                                       \
    if (!(cond))                                                                        \
    {                                                                                   \
        fprintf(stderr, "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n",               \
                GET_TID(), __FILENAME__, __FUNCTION__, __LINE__, #cond, ##__VA_ARGS__); \
        fflush(stderr);                                                                 \
        _exit(1);                                                                       \
    }                                                                                   \
} while(0)

enum LogLevel
{
    ERROR = 0,
    INFO,
    DEBUG,
    TRACE
};

void dl_comp_get_time(char *buf, size_t buf_size);

#endif
