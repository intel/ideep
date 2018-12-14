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
#include "ideep.hpp"
#include "TR_interface.h"
#include "assert.h"
#include "tensor.hpp"

#define null_callback (void(*)(int))NULL

namespace ideep {
class distribute {
public:
    using tensor = ideep::tensor;
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

    // return value:
    //      true  - multinode support is enabled in ideep
    //      false - no multinode support in ideep
    static bool available() {
        return TR_available();
    }

    // call once before using distribute collectives
    // any distribute collective must be called AFTER this call returns
    static void init() {
        TR_init(-1);
    }

    // same as above, but set affinity of work thread to affinity
    static void init(int affinity) {
        assert (affinity >= 0);
        TR_init(affinity);
    }

    // get number of nodes
    static int get_world_size() {
        return TR_get_world_size();
    }

    // get a unique rank of current node, rank is from 0 to number-of-nodes - 1
    static int get_rank() {
        return TR_get_rank();
    }

    /*
        allreduce interface families
        variants are:
            with/without id
            inplace/non-inplace
    */

    // blocking allreduce add send_recv_buf together elementwisely across
    // different nodes with same id
    // id must >= 0, ids smaller than 0 are reserved
    static tr_error_code allreduce(int id, tensor &send_recv_buf) {
        assert (id >= 0);
        return _allreduce(id, send_recv_buf, send_recv_buf);
    }

    // same as above, result is written in recv_buf
    static tr_error_code allreduce(int id, tensor &send_buf, tensor &recv_buf) {
        assert (id >= 0);
        return _allreduce(id, send_buf, recv_buf);
    }

    /* without id */

    // when not supplying id, caller should gurantee call sequence have same order between all nodes
    // note: you can still use out-of-order call sequence for all call with id
    static tr_error_code allreduce(tensor &send_recv_buf) {
        int id = _get_new_implicit_id();
        return _allreduce(id, send_recv_buf, send_recv_buf);
    }

    static tr_error_code allreduce(tensor &send_buf, tensor &recv_buf) {
        int id = _get_new_implicit_id();
        return _allreduce(id, send_buf, recv_buf);
    }

    /*
        iallreduce interface families
        variants are:
            with/without id
            inpace/non-inpace
            with/without callback
    */

    // non-blocking iallreduce add send_recv_buf together elementwisely across
    // different nodes with same id
    static tr_error_code iallreduce(int id, tensor &send_recv_buf) {
        assert (id >= 0);
        return _iallreduce(id, send_recv_buf, send_recv_buf, null_callback);
    }

    // same as above, a python callback function is supplied to be called
    // when iallreduce is done.   The callback is always initiated from
    // a thread managed by distributed module.  callback implementation is
    // responsible for thread safety
    static tr_error_code iallreduce(int id, tensor &send_recv_buf, void (*callback)(int)) {
        assert (id >= 0);
        return _iallreduce(id, send_recv_buf, send_recv_buf, callback);
    }

    // same as above, the result is written in recv_buf
    static tr_error_code iallreduce(int id, tensor &send_buf, tensor &recv_buf) {
        assert (id >= 0);
        return _iallreduce(id, send_buf, recv_buf, null_callback);
    }

    // same as above with callback
    static tr_error_code iallreduce(int id, tensor &send_buf, tensor &recv_buf, void (*callback)(int)) {
        assert (id >= 0);
        return _iallreduce(id, send_buf, recv_buf, callback);
    }

    /* without id */

    static tr_error_code iallreduce(tensor &send_recv_buf, int &id) {
        id = _get_new_implicit_id();
        return _iallreduce(id, send_recv_buf, send_recv_buf, null_callback);
    }

    static tr_error_code iallreduce(tensor &send_recv_buf, void (*callback)(int), int &id) {
        id = _get_new_implicit_id();
        return _iallreduce(id, send_recv_buf, send_recv_buf, callback);
    }

    static tr_error_code iallreduce(tensor &send_buf, tensor &recv_buf, int &id) {
        id = _get_new_implicit_id();
        return _iallreduce(id, send_buf, recv_buf, null_callback);
    }

    static tr_error_code iallreduce(tensor &send_buf, tensor &recv_buf, void (*callback)(int), int &id) {
        id = _get_new_implicit_id();
        return _iallreduce(id, send_buf, recv_buf, callback);
    }

    /*
        bcast
    */

    // blocking broadcast buf from root node to other nodes
    static tr_error_code bcast(int id, tensor &buf, int root) {
        assert (id >= 0);
        return _bcast(id, buf, root);
    }

    /* without id */

    static tr_error_code bcast(tensor &buf, int root) {
        int id = _get_new_implicit_id();
        return _bcast(id, buf, root);
    }

    // wait for ID to finish
    static void wait(int id) {
        TR_wait(id);
    }

    // check whether collective with id is finished or not
    // if urgency is tr_greedy, the check is opportunistic
    // if urgency is tr_need,   caller is indicating the collective is blocking
    //                          other on going activities
    static bool test(int id, tr_urgency urgency) {
        TR_urgency u = TR_GREEDY;
        switch (urgency) {
        case tr_need:   u = TR_NEED;   break;
        case tr_greedy: u = TR_GREEDY; break;
        default: assert (!"should not get here");
        }
        return TR_test(id, u);
    }

    static bool test(int id) {
        return TR_test(id, TR_GREEDY);
    }

    // indicating collective is blocking on going activities
    static void set_urgent(int id) {
        TR_set_urgent(id);
    }

    // wait for all collective to finish before return
    static void barrier() {
        TR_barrier();
    }

    // release all the resources for communication
    static void finalize() {
        TR_finalize();
    }

private:

    static tr_error_code _allreduce(int id, tensor &send_buf, tensor &recv_buf) {
        if (send_buf.get_nelems() != recv_buf.get_nelems()) {
            return tr_fail;
        }
        if (send_buf.get_data_type() != recv_buf.get_data_type()) {
            return tr_fail;
        }

        TR_datatype datatype;

        switch (send_buf.get_data_type()) {
        case data_type_t::f32:
            datatype = TR_FP32;
            break;

        case data_type_t::s32:
            datatype = TR_INT32;
            break;

        default:
            return tr_type_not_supported;
        }

        size_t num_elements = send_buf.get_nelems();

        TR_allreduce(id, 0, send_buf==recv_buf?TR_IN_PLACE:send_buf.get_data_handle(),
                     recv_buf.get_data_handle(), num_elements, datatype);

        return tr_success;
    }

    static tr_error_code _iallreduce(int id, tensor &send_buf, tensor &recv_buf, void (*callback)(int)) {
        if (send_buf.get_nelems() != recv_buf.get_nelems()) {
            return tr_fail;
        }
        if (send_buf.get_data_type() != recv_buf.get_data_type()) {
            return tr_fail;
        }

        TR_datatype datatype;

        switch (send_buf.get_data_type()) {
        case data_type_t::f32:
            datatype = TR_FP32;
            break;

        case data_type_t::s32:
            datatype = TR_INT32;
            break;

        default:
            return tr_type_not_supported;
        }

        size_t num_elements = send_buf.get_nelems();

        TR_iallreduce(id, 0, send_buf==recv_buf?TR_IN_PLACE:send_buf.get_data_handle(),
                      recv_buf.get_data_handle(), num_elements, datatype, callback);

        return tr_success;
    }

    static tr_error_code _bcast(int id, tensor &buf, int root) {
        TR_datatype datatype;

        switch (buf.get_data_type()) {
        case data_type_t::f32:
            datatype = TR_FP32;
            break;

        case data_type_t::s32:
            datatype = TR_INT32;
            break;

        default:
            return tr_type_not_supported;
        }

        size_t num_elements = buf.get_nelems();

        TR_bcast(id, 0, buf.get_data_handle(), num_elements, datatype, root);

        return tr_success;
    }

    static int _get_new_implicit_id(void) {
        // implicity id is negative and start from -2, then goes like -3, -4, -5, etc.
        // -1 is reserved for intenal purpose
        static int implicit_id = -2;
        assert (implicit_id < 0);
        int ret_val = implicit_id;
        implicit_id--;
        return ret_val;
    }
};

}
