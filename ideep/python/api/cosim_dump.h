/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _COSIOM_DUMP_H_
#define _COSIOM_DUMP_H_

#include <mkldnn.hpp>
#include <fstream>
#include <cstdarg>
#include "mdarray.h"
#include "dump_desc.h"

namespace mkldnn {

class cosim_dump {
public:
    cosim_dump(operation_kind aop_kind);

    void dump_memory(parm_kind aparm_kind, const memory &mem);

    void dump_int_parms(parm_kind aparm_kind, int nargs, ...);

    void dump_double_parms(parm_kind aparm_kind, int nargs, ...);

    virtual ~cosim_dump();

private:
    cosim_dump() {}

    std::fstream dfile;

    DumpHeader header;
};

class cosim_check {
public:
    cosim_check();

    void set_act_view(Py_buffer *view);

    void set_ref_view(Py_buffer *view);

    bool expect_allclose(double atol, double rtol);

    virtual ~cosim_check() {};

private:
    struct buf_view {
        Py_ssize_t len;
        Py_ssize_t itemsize;
        memory::data_type dtype;
        int ndim;
        Py_ssize_t strides[TENSOR_MAX_DIMS];
        Py_ssize_t shape[TENSOR_MAX_DIMS];
        void *buf;
    };

    void set_view(Py_buffer *view, buf_view *buf);

    buf_view act;
    buf_view ref;
};

};

#endif
