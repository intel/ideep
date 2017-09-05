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

%module (package="mkldnn") cosim_dump
%{
  #define SWIG_FILE_WITH_INIT
  #include <mkldnn.hpp>
  #include <fstream>
  #include "mdarray.h"
  #include "dump_desc.h"
  #include "cosim_dump.h"
%}

%init %{
  import_array();
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import memory.i
%import mdarray.i

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

enum operation_kind {
    cdump_op_invalid = 0,
    cdump_op_conv_forward,
    cdump_op_conv_backward,
    cdump_op_lrn_forward,
    cdump_op_lrn_backward,
    cdump_op_max_pooling_forward,
    cdump_op_max_pooling_backward,
    cdump_op_avg_pooling_forward,
    cdump_op_avg_pooling_backward,
    cdump_op_max
};

enum parm_kind {
    cdump_memory_invalid = 0,
    cdump_src_memory,
    cdump_weight_memory,
    cdump_bias_memory,
    cdump_diff_dst_memory,
    cdump_conv_int_parms,
    cdump_lrn_local_size,
    cdump_lrn_doulbe_parms,
    cdump_max_pooling_int_parms,
    cdump_avg_pooling_int_parms,
    cdump_memory_max
};

%varargs(TENSOR_MAX_DIMS, int arg = 0) cosim_dump::dump_int_parms;
%varargs(TENSOR_MAX_DIMS, double arg = 0.0f) cosim_dump::dump_double_parms;

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
        mkldnn::memory::data_type dtype;
        int ndim;
        Py_ssize_t strides[TENSOR_MAX_DIMS];
        Py_ssize_t shape[TENSOR_MAX_DIMS];
        void *buf;
    };

    void set_view(Py_buffer *view, buf_view *buf);

    buf_view act;
    buf_view ref;
};

} // namespace mkldnn
