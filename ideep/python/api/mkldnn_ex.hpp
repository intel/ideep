#pragma once

#include <mkldnn.hpp>

using namespace mkldnn;
static memory::format get_fmt(memory::primitive_desc &mpd) {
    memory::format mkl_fmt = static_cast<memory::format>(
            mpd.desc().data.format);
    return mkl_fmt; 
}
