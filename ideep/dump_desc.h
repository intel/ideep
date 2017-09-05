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

#ifndef _DUMP_DESC_H_
#define _DUMP_DESC_H_

#include <mkldnn.hpp>

namespace mkldnn {

/***
XXX: Dump file layout
-------------------------
| DumpHeader            |
-------------------------
| dummy_mdesc           |
-------------------------
| DumpDesc1             |
-------------------------
| mkldnn_memory_desc_t1 |
-------------------------
| ...data...            |
-------------------------
| DumpDesc2             |
-------------------------
| mkldnn_memory_desc_t2 |
-------------------------
| ...data...            |
-------------------------
| ...........           |
-------------------------
***/

static const mkldnn_memory_desc_t dummy_mdesc = {
    .primitive_kind = mkldnn_undefined_primitive,
    .ndims = TENSOR_MAX_DIMS,
    .dims = {1},
    .data_type = mkldnn_data_type_undef,
    .format = mkldnn_format_undef,
    .layout_desc = {
        .blocking = {
            .block_dims = {2},
            .strides = {3},
            .padding_dims = {4},
            .offset_padding_to_data = {5},
            .offset_padding = 0xf0f05050,
        },
    },
};

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

#define CDUMP_ID_NUM  0xA0A05050

struct DumpHeader {
    int             id_num;
    int             mkldnn_ver;
    operation_kind  op_kind;
}__attribute__ ((packed));

struct DumpDesc {
    parm_kind   pa_kind;
    int         desc_size;
    int         data_size;
    union {
        int     iparms[TENSOR_MAX_DIMS];
        double  dparms[TENSOR_MAX_DIMS];
    };
}__attribute__ ((packed));

};

#endif
