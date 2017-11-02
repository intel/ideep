/*******************************************************************************
 * * Copyright 2016-2017 Intel Corporation
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *     http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * *******************************************************************************/


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
                printf("%s: ERROR: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),   \
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
                printf("%s: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),          \
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
