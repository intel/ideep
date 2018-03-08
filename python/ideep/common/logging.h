#pragma once
#include <iostream>

class Log {
};

static Log s_log;

template<typename T>
Log& operator<<(Log& log, const T& obj)
{
    return log;
}


#define LOG(level) s_log
