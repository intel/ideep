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

#include "cosim_dump.h"
#include <string>
#include <cmath>
#include <cstdlib>
#include <assert.h>

namespace mkldnn {

cosim_dump::cosim_dump(operation_kind aop_kind) {
    header.id_num = CDUMP_ID_NUM;
    header.mkldnn_ver = 0;
    header.op_kind = aop_kind;

    const char* dname = NULL;
    switch(header.op_kind) {
        case cdump_op_conv_forward:
            dname = "Conv_forward.cdump";
            break;
        case cdump_op_conv_backward:
            dname = "Conv_backward.cdump";
            break;
        case cdump_op_lrn_forward:
            dname = "Lrn_forward.cdump";
            break;
        case cdump_op_lrn_backward:
            dname = "Lrn_backward.cdump";
            break;
        case cdump_op_max_pooling_forward:
            dname = "MaxPooling_forward.cdump";
            break;
        case cdump_op_max_pooling_backward:
            dname = "MaxPooling_backward.cdump";
            break;
        case cdump_op_avg_pooling_forward:
            dname = "AvgPooling_forward.cdump";
            break;
        case cdump_op_avg_pooling_backward:
            dname = "AvgPooling_backward.cdump";
            break;
        default:
            dname =  "Cosim_dump.cdump";
            break;
    }

    dfile.open(dname, std::ios::binary | std::ios::trunc | std::ios::out);
    if (!dfile.is_open() || !dfile.good()) {
        printf("Failed to open dump file %s\n", dname);
        return;
    }

    dfile.write((const char*)&header, sizeof(DumpHeader));
    if (!dfile.good()) {
        printf("Failed to write file header to dump file %s\n", dname);
        dfile.close();
        return;
    }

    dfile.write((const char*)&dummy_mdesc, sizeof(mkldnn_memory_desc_t));
    if (!dfile.good()) {
        printf("Failed to write dummy_mdesc to dump file %s\n", dname);
        dfile.close();
        return;
    }
}

cosim_dump::~cosim_dump() {
    if (dfile.is_open()) {
        dfile.close();
    }
}

void cosim_dump::dump_memory(parm_kind aparm_kind, const memory &mem) {
    if (!dfile.is_open()) {
        printf("FATAL: the dump file is unavailable!\n");
        return;
    }

    auto mp = mem.get_primitive_desc();

    DumpDesc dd;
    dd.pa_kind = aparm_kind;
    dd.desc_size = sizeof(mkldnn_memory_desc_t);
    dd.data_size = mp.get_size();
    dfile.write((const char*)&dd, sizeof(DumpDesc));
    if (!dfile.good()) {
        printf("Failed to write memory DumpDesc to dump file!\n");
        return;
    }

    auto md = mp.desc();
    dfile.write(reinterpret_cast<const char*>(&md.data), dd.desc_size);
    if (!dfile.good()) {
        printf("Failed to write memory desc to dump file!\n");
        return;
    }

    void* data = mem.get_data_handle();
    dfile.write(reinterpret_cast<const char*>(data), dd.data_size);
    if (!dfile.good()) {
        printf("Failed to write memory data to dump file!\n");
        return;
    }
}

void cosim_dump::dump_int_parms(parm_kind aparm_kind, int nargs, ...) {
    assert(nargs <= TENSOR_MAX_DIMS);

    if (!dfile.is_open()) {
        printf("FATAL: the dump file is unavailable!\n");
        return;
    }

    DumpDesc dd;
    dd.pa_kind = aparm_kind;
    dd.desc_size = 0;
    dd.data_size = 0;

    int i = 0;
    va_list vl;
    va_start(vl, nargs);
    for (i = 0; i < nargs; i++) {
        dd.iparms[i] = va_arg(vl, int);
    }
    va_end(vl);

    dfile.write(reinterpret_cast<const char*>(&dd), sizeof(DumpDesc));
    if (!dfile.good()) {
        printf("Failed to write int DumpDesc to dump file!\n");
        return;
    }
}

void cosim_dump::dump_double_parms(parm_kind aparm_kind, int nargs, ...) {
    assert(nargs <= TENSOR_MAX_DIMS);

    if (!dfile.is_open()) {
        printf("FATAL: the dump file is unavailable!\n");
        return;
    }

    DumpDesc dd;
    dd.pa_kind = aparm_kind;
    dd.desc_size = 0;
    dd.data_size = 0;

    int i = 0;
    va_list vl;
    va_start(vl, nargs);
    for (i = 0; i < nargs; i++) {
        dd.dparms[i] = va_arg(vl, double);
    }
    va_end(vl);

    dfile.write(reinterpret_cast<const char*>(&dd), sizeof(DumpDesc));
    if (!dfile.good()) {
        printf("Failed to write double DumpDesc to dump file!\n");
        return;
    }
}

cosim_check::cosim_check() {
    act.len = 0;
    act.itemsize = 0;
    act.dtype = memory::data_undef;
    act.ndim = 0; // scalar is not supported
    act.buf = nullptr;

    ref.len = 0;
    ref.itemsize = 0;
    ref.dtype = memory::data_undef;
    ref.ndim = 0; // scalar is not supported
    ref.buf = nullptr;
}

void cosim_check::set_act_view(Py_buffer *view) {
    set_view(view, &act);
}

void cosim_check::set_ref_view(Py_buffer *view) {
    set_view(view, &ref);
}

void cosim_check::set_view(Py_buffer *view, cosim_check::buf_view *buf) {
    assert(view != nullptr && buf != nullptr);

    buf->len = view->len;
    buf->itemsize = view->itemsize;

    buf->dtype = memory::data_undef;
    std::string format(view->format);
    if (view->itemsize == 4) {
      if (std::string::npos != format.find_last_of('f')) {
        buf->dtype = memory::f32;
      } else if (std::string::npos != format.find_last_of('i')) {
        buf->dtype = memory::s32;
      }
    }

    buf->ndim = view->ndim;
    for (int i = 0; i < view->ndim; i++) {
        buf->strides[i] = view->strides[i];
        buf->shape[i] = view->shape[i];
    }

    buf->buf = view->buf;
}

bool cosim_check::expect_allclose(double atol, double rtol) {
    if ((act.len != ref.len) ||
        (act.dtype != ref.dtype) ||
        (act.itemsize != ref.itemsize) ||
        (act.ndim != ref.ndim)) {
        printf("WARNING: act & ref are not matched!\n");
        return false;
    }

    if (act.len <= 0 ||
        act.itemsize <= 0 ||
        act.dtype == memory::data_undef ||
        act.ndim > TENSOR_MAX_DIMS ||
        act.ndim <= 0 ||
        act.buf == nullptr ||
        ref.buf == nullptr) {
        printf("WARNING: act & ref are not initialized!\n");
        return false;
    }

    int total = 1;
    int ndim = act.ndim;
    for (int i = 0; i < ndim; i++) {
        if ((act.shape[i] != ref.shape[i]) || (act.shape[i] <= 0)) {
            printf("WARNING: act & ref have different/wrong shape!\n");
            return false;
        }
        total *= act.shape[i];
    }

    if (total != act.len / act.itemsize) {
        printf("WARNING: something wrong in the shape of act & ref!\n");
        return false;
    }

    if (act.buf == ref.buf) {
        printf("NOTE: act & ref have same data buffer\n");
        return true;
    }

    int mismatched = 0;
    if (act.dtype == memory::s32) {
        int *abuf = static_cast<int*>(act.buf);
        int *rbuf = static_cast<int*>(ref.buf);
#   pragma omp parallel for schedule(static)
        for (int j = 0; j < total; j++) {
            int aval = abuf[j];
            int rval = rbuf[j];
            if (aval != rval) {
                if (mismatched == 0) {
                    printf("[ __act__ , __ref__ , __diff__ , #index]\n");
                }
                printf("[%d, %d, %d, #%d]\n", aval, rval, abs(aval - rval), j);
                mismatched ++;
            }
        }

    } else if (act.dtype == memory::f32) {
        float *abuf = static_cast<float*>(act.buf);
        float *rbuf = static_cast<float*>(ref.buf);
#   pragma omp parallel for schedule(static)
        for (int j = 0; j < total; j++) {
            float aval = abuf[j];
            float rval = rbuf[j];
            float diff = fabs(aval - rval);
            if (diff > (atol + rtol * fabs(rval))) {
                if (mismatched == 0) {
                    printf("[ __act__ , __ref__ , __diff__ , #index ]\n");
                }
                printf("[ %.10lf, %.10lf, %.10lf, #%d ]\n", aval, rval, diff, j);
                mismatched ++;
            }
        }

    } else {
        printf("WARNING: wrong dtype!\n");
        return false;
    }

    if (mismatched != 0) {
        printf("size: %d ndim: %d shape: [ ", total, ndim);
        for (int k = 0; k < act.ndim; k++) {
            printf("%d ", static_cast<int>(act.shape[k]));
        }
        printf("]\nmismatch rate: %.10lf%%\n",
                ((double)mismatched / (double)total) * (double)100.00f);

        return false;
    }

    return true;
}

};
