/*
 *COPYRIGHT
 *All modification made by Intel Corporation: © 2017 Intel Corporation.
 *Copyright (c) 2015 Preferred Infrastructure, Inc.
 *Copyright (c) 2015 Preferred Networks, Inc.
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


#ifndef _DLCP_PY_H_
#define _DLCP_PY_H_

#include "dl_compression.h"
#include "mdarray.h"
#include "tensor.h"

class dlCompression {
public:
    enum {
        dl_comp_none = DL_COMP_NONE,
        dl_comp_dfp = DL_COMP_DFP,
    };

    enum {
        dl_comp_ok = DL_COMP_OK,
        dl_comp_fail = DL_COMP_FAIL,
        dl_comp_fail_src_data_type_not_supported =
            DL_COMP_FAIL_SRC_DATA_TYPE_NOT_SUPPORTED,
        dl_comp_fail_ratio_not_supported =
            DL_COMP_FAIL_RATIO_NOT_SUPPORTED,
        dl_comp_fail_comp_method_not_supported =
            DL_COMP_FAIL_COMP_METHOD_NOT_SUPPORTED,
        dl_comp_fail_invalid_compressed_format =
            DL_COMP_FAIL_INVALID_COMPRESSED_FORMAT,
        dl_comp_fail_not_supported =
            DL_COMP_FAIL_NOT_SUPPORTED,
    };

    static bool available;

    static void init() {
        available = dl_comp_check_running_environ();
    }

    static bool is_available() {
        return available;
    }

    static int Compress(mdarray *src, mdarray *dst,
        mdarray *diff, size_t ratio, int method) {
        if (!is_available())
            return DL_COMP_FAIL_NOT_SUPPORTED;

        if (src->get()->tensor()->size() !=
            dst->get()->tensor()->size())
            return DL_COMP_FAIL;

        if (src->get()->tensor()->type() !=
            dst->get()->tensor()->type())
            return DL_COMP_FAIL;

        int dtype = -1;
        switch (src->get()->tensor()->type()) {
        case SINT8:
            dtype = DL_COMP_INT8;
            break;

        case FLOAT32:
            dtype = DL_COMP_FLOAT32;
            break;

        default:
            break;
        }

        if (-1 == dtype)
            return DL_COMP_FAIL_SRC_DATA_TYPE_NOT_SUPPORTED;

        return dl_comp_compress_buffer(src->get()->tensor()->data(),
            dst->get()->tensor()->data(), src->get()->tensor()->size(),
            diff ? diff->get()->tensor()->data() : nullptr,
            (dl_comp_data_type_t)dtype, ratio, (dl_comp_method_t)method);
    }

    static int Decompress(mdarray *src, mdarray *dst) {
        if (!is_available())
            return DL_COMP_FAIL_NOT_SUPPORTED;

        if (src->get()->tensor()->size() !=
            dst->get()->tensor()->size())
            return DL_COMP_FAIL;

        if (src->get()->tensor()->type() !=
            dst->get()->tensor()->type())
            return DL_COMP_FAIL;

        return dl_comp_decompress_buffer(src->get()->tensor()->data(),
                                         dst->get()->tensor()->data(),
                                         src->get()->tensor()->size());
    }
};

#endif
