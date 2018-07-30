/*
 *COPYRIGHT
 *All modification made by Intel Corporation: © 2018 Intel Corporation.
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

#pragma once
#include <Python.h>
#include "mdarray.h"
#include "ideep.hpp"
#include "TR_interface.h"
#include "tensor.hpp"

//using tensor = ideep::tensor;

class distribute {
public:
    //using tensor = ideep::tensor;
    using data_type_t = mkldnn::memory::data_type;

    enum tr_urgency {
        tr_need = TR_NEED,
        tr_greedy = TR_GREEDY
    };

    enum tr_error_code {
        tr_success,
        tr_fail,
        tr_type_not_supported
    };

    static bool available() {
        return TR_available();
    }

    static void init() {
        TR_init();
    }

    static int get_world_size() {
        return TR_get_world_size();
    }

    static int get_rank() {
        return TR_get_rank();
    }

    static tr_error_code allreduce(int id, int priority, mdarray *send_recv_buf) {
        return allreduce(id, priority, send_recv_buf, send_recv_buf);
    }

    static tr_error_code allreduce(int id, int priority, mdarray *send_buf, mdarray *recv_buf) {
        if (send_buf->get()->get_nelems() != recv_buf->get()->get_nelems()) {
            return tr_fail;
        }
        if (send_buf->get()->get_data_type() != recv_buf->get()->get_data_type()) {
            return tr_fail;
        }

        TR_datatype datatype;

        switch (send_buf->get()->get_data_type()) {
        case data_type_t::f32:
            datatype = TR_FP32;
            break;

        case data_type_t::s32:
            datatype = TR_INT32;
            break;

        default:
            return tr_type_not_supported;
        }

        size_t num_elements = send_buf->get()->get_nelems();

        TR_allreduce(id, priority, send_buf==recv_buf?TR_IN_PLACE:send_buf,
                     recv_buf->get()->get_data_handle(), num_elements, datatype);

        return tr_success;
    }

    static tr_error_code iallreduce(int id, int priority, mdarray *send_recv_buf) {
        return iallreduce(id, priority, send_recv_buf, send_recv_buf);
    }

    static tr_error_code iallreduce(int id, int priority, mdarray *send_buf, mdarray *recv_buf) {
        if (send_buf->get()->get_nelems() != recv_buf->get()->get_nelems()) {
            return tr_fail;
        }
        if (send_buf->get()->get_data_type() != recv_buf->get()->get_data_type()) {
            return tr_fail;
        }

        TR_datatype datatype;

        switch (send_buf->get()->get_data_type()) {
        case data_type_t::f32:
            datatype = TR_FP32;
            break;

        case data_type_t::s32:
            datatype = TR_INT32;
            break;

        default:
            return tr_type_not_supported;
        }

        size_t num_elements = send_buf->get()->get_nelems();

        TR_iallreduce(id, priority, send_buf==recv_buf?TR_IN_PLACE:send_buf,
                      recv_buf->get()->get_data_handle(), num_elements, datatype, NULL);

        return tr_success;
    }

    static tr_error_code bcast(int id, int priority, mdarray *buf, int root) {
        TR_datatype datatype;

        switch (buf->get()->get_data_type()) {
        case data_type_t::f32:
            datatype = TR_FP32;
            break;

        case data_type_t::s32:
            datatype = TR_INT32;
            break;

        default:
            return tr_type_not_supported;
        }

        size_t num_elements = buf->get()->get_nelems();

        TR_bcast(id, priority, buf->get()->get_data_handle(), num_elements, datatype, root);

        return tr_success;
    }

    static void wait(int id) {
        TR_wait(id);
    }

    static bool test(int id, TR_urgency urgency) {
        return TR_test(id, urgency);
    }

    static void set_urgent(int id) {
        TR_set_urgent(id);
    }

    static void barrier() {
        TR_barrier();
    }

    static void finalize() {
        TR_finalize();
    }
};
