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

%module api
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include <mkldnn.hpp>
  using mkldnn::handle_traits;
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%define %sequence_like(integer_sequence_compitable_type)
%typemap(typecheck) (integer_sequence_compitable_type) {
  $1 = PySequence_Check($input);
}

%typemap(in) (integer_sequence_compitable_type) (int count) {
  count = PySequence_Size($input);

  for (int i =0; i < count; i ++) {
    PyObject *o = PySequence_GetItem($input, i);
    $1.push_back(PyLong_AsLong(o));
  }
}
%enddef

%sequence_like(mkldnn::memory::dims);

namespace mkldnn {

template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) {}
    handle &operator=(const handle &&other) = delete;

protected:
    handle(T t = 0, bool weak = false): _data(0);
    bool operator==(const T other) const;
    bool operator!=(const T other) const;

public:
    handle(const handle &other);

    handle &operator=(const handle &other);

    void reset(T t, bool weak = false);

    T get() const;

    bool operator==(const handle &other) const;
    bool operator!=(const handle &other) const;
};

namespace c_api {
  %include c_api.i
}

%template (mkldnn_primitive_t_handle) handle< mkldnn_primitive_t >;
%template (mkldnn_engine_t_handle) handle< mkldnn_engine_t >;
%template (mkldnn_primitive_desc_t_handle) handle < mkldnn_primitive_desc_t >;
%template (mkldnn_stream_t_handle) handle< mkldnn_stream_t >;

%rename (primitive_at) primitive::at;
class primitive: public handle<mkldnn_primitive_t> {
    friend struct error;
    friend struct stream;
    friend class primitive_at;
    using handle::handle;
public:
    struct at {
        mkldnn_primitive_at_t data;

        at(const primitive &aprimitive, size_t at = 0);
        inline operator primitive() const;
    };

    inline const_mkldnn_primitive_desc_t get_primitive_desc() const;
};

struct error: public std::exception {
    mkldnn_status_t status;
    std::string message;
    primitive error_primitive;

    error(mkldnn_status_t astatus, std::string amessage,
            mkldnn_primitive_t aerror_primitive = 0);

    static void wrap_c_api(mkldnn_status_t status,
            std::string message,
            mkldnn_primitive_t *error_primitive = 0);
};

struct engine: public handle<mkldnn_engine_t> {

    enum kind {
        any = mkldnn_any_engine,
        cpu = mkldnn_cpu,
    };

    static size_t get_count(kind akind) {
        return mkldnn_engine_get_count(convert_to_c(akind));
    }

    engine(kind akind, size_t index);

    // XXX: Solve it! explicit engine(const mkldnn_engine_t& aengine);
};

%rename (memory_desc) memory::desc;
%rename (memory_primitive_desc) memory::primitive_desc;

%exception memory::primitive_desc::primitive_desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

%exception memory::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct memory: public primitive  {
    private:
    std::shared_ptr<char> _handle;

    public:
    typedef std::vector<int> dims; /*manual scrip*/

    template <typename T> static void validate_dims(std::vector<T> v);

    enum data_type {
        data_undef = mkldnn_data_type_undef,
        f32 = mkldnn_f32,
        s32 = mkldnn_s32,
    };

    enum format {
        format_undef = mkldnn_format_undef,
        any = mkldnn_any,
        blocked = mkldnn_blocked,
        x = mkldnn_x,
        nc = mkldnn_nc,
        nchw = mkldnn_nchw,
        nhwc = mkldnn_nhwc,
        chwn = mkldnn_chwn,
        nChw8c = mkldnn_nChw8c,
        nChw16c = mkldnn_nChw16c,
        oi = mkldnn_oi,
        io = mkldnn_io,
        oihw = mkldnn_oihw,
        ihwo = mkldnn_ihwo,
        oIhw8i = mkldnn_oIhw8i,
        oIhw16i = mkldnn_oIhw16i,
        OIhw8i8o = mkldnn_OIhw8i8o,
        OIhw16i16o = mkldnn_OIhw16i16o,
        OIhw8o8i = mkldnn_OIhw8o8i,
        OIhw16o16i = mkldnn_OIhw16o16i,
        Ohwi8o = mkldnn_Ohwi8o,
        Ohwi16o = mkldnn_Ohwi16o,
        goihw = mkldnn_goihw,
        gOIhw8i8o = mkldnn_gOIhw8i8o,
        gOIhw16i16o = mkldnn_gOIhw16i16o,
        gOIhw8o8i = mkldnn_gOIhw8o8i,
        gOIhw16o16i = mkldnn_gOIhw16o16i,
    };

    struct desc {
        friend struct memory;
        mkldnn_memory_desc_t data;

        desc(dims adims, data_type adata_type,
                format aformat);
        desc(const mkldnn_memory_desc_t &adata);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        friend struct memory;
        primitive_desc() {}

        primitive_desc(const desc &adesc, const engine &aengine);

        memory::desc desc();

        size_t get_size() const;

        bool operator==(const primitive_desc &other) const;

        bool operator!=(const primitive_desc &other) const;
    };

    memory(const primitive &aprimitive);

    memory(const primitive_desc &adesc);

    memory(const primitive_desc &adesc, void *ahandle);

    primitive_desc get_primitive_desc() const;

    inline void *get_data_handle() const;

    inline void set_data_handle(void *handle) const;

    // XXX: Trivial, can delete them?
    static mkldnn_data_type_t convert_to_c(data_type adata_type);

    static mkldnn_memory_format_t convert_to_c(format aformat);

};

enum query {
    undef = mkldnn_query_undef,

    eengine = mkldnn_query_engine,
    primitive_kind = mkldnn_query_primitive_kind,

    num_of_inputs_s32 = mkldnn_query_num_of_inputs_s32,
    num_of_outputs_s32 = mkldnn_query_num_of_outputs_s32,

    time_estimate_f64 = mkldnn_query_time_estimate_f64,
    memory_consumption_s64 = mkldnn_query_memory_consumption_s64,

    memory_d = mkldnn_query_memory_d,
    convolution_d = mkldnn_query_convolution_d,
    relu_d = mkldnn_query_relu_d,
    softmax_d = mkldnn_query_softmax_d,
    pooling_d = mkldnn_query_pooling_d,
    lrn_d = mkldnn_query_lrn_d,
    batch_normalization_d = mkldnn_query_batch_normalization_d,
    inner_product_d = mkldnn_query_inner_product_d,
    convolution_relu_d = mkldnn_query_convolution_relu_d,

    input_pd = mkldnn_query_input_pd,
    output_pd = mkldnn_query_output_pd,
    src_pd = mkldnn_query_src_pd,
    diff_src_pd = mkldnn_query_diff_src_pd,
    weights_pd = mkldnn_query_weights_pd,
    diff_weights_pd = mkldnn_query_diff_weights_pd,
    dst_pd = mkldnn_query_dst_pd,
    diff_dst_pd = mkldnn_query_diff_dst_pd,
    workspace_pd = mkldnn_query_workspace_pd,
};
inline mkldnn_query_t convert_to_c(query aquery) {
    return static_cast<mkldnn_query_t>(aquery);
}

enum padding_kind {
    zero = mkldnn_padding_zero
};
inline mkldnn_padding_kind_t convert_to_c(padding_kind kind) {
    return static_cast<mkldnn_padding_kind_t>(kind);
}

enum prop_kind {
    forward_training = mkldnn_forward_training,
    forward_scoring = mkldnn_forward_scoring,
    forward_inference = mkldnn_forward_inference,
    forward = mkldnn_forward,
    backward = mkldnn_backward,
    backward_data = mkldnn_backward_data,
    backward_weights = mkldnn_backward_weights,
    backward_bias = mkldnn_backward_bias
};
inline mkldnn_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<mkldnn_prop_kind_t>(kind);
}

enum algorithm {
    convolution_direct = mkldnn_convolution_direct,
    lrn_across_channels = mkldnn_lrn_across_channels,
    lrn_within_channel  = mkldnn_lrn_within_channel,
    pooling_max = mkldnn_pooling_max,
    pooling_avg = mkldnn_pooling_avg
};

enum batch_normalization_flag {
    use_global_stats = mkldnn_use_global_stats,
    use_scale_shift = mkldnn_use_scaleshift,
    omit_stats = mkldnn_omit_stats
};

static mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<mkldnn_alg_kind_t>(aalgorithm);
}

%rename (reorder_primitive_desc) reorder::primitive_desc;
struct reorder : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const memory::primitive_desc &input,
                       const memory::primitive_desc &output);
    };

    reorder(const primitive_desc &aprimitive_desc,
            const primitive::at &input, const memory &output);

    reorder(const primitive::at &input, const memory &output);
};

%rename (view_primitive_desc) view::primitive_desc;
struct view : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const memory::primitive_desc &input, memory::dims dims,
                memory::dims offsets);

        memory::primitive_desc dst_primitive_desc() const;
    };

    view(const primitive_desc &view_pd, primitive::at input);

    view(memory input, memory::dims dims, memory::dims offsets);
};

%rename (concat_primitive_desc) concat::primitive_desc;
struct concat : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
                std::vector<memory::primitive_desc> inputs);

        primitive_desc(const memory::desc &output, int concat_dimension,
                std::vector<memory::primitive_desc> inputs);

        primitive_desc(int concat_dimension,
                std::vector<memory::primitive_desc> inputs);

        memory::primitive_desc dst_primitive_desc() const;

    };

    concat(const primitive_desc &concat_pd,
            std::vector<primitive::at> &inputs, const memory &output);
};

%rename (sum_primitive_desc) sum::primitive_desc;
struct sum : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
                std::vector<memory::primitive_desc> inputs);

        primitive_desc(const memory::desc &output, std::vector<double> scale,
                std::vector<memory::primitive_desc> inputs);

        primitive_desc(std::vector<double> scale,
                std::vector<memory::primitive_desc> inputs);

        memory::primitive_desc dst_primitive_desc() const;

    };

    sum(const primitive_desc &sum_pd,
            std::vector<primitive::at> &inputs, const memory &output);
};

%exception stream::submit {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct stream: public handle<mkldnn_stream_t> {
    using handle::handle;

    enum kind { any = mkldnn_stream_kind_t::mkldnn_any_stream,
        eager = mkldnn_stream_kind_t::mkldnn_eager,
        lazy = mkldnn_stream_kind_t::mkldnn_lazy };

    static mkldnn_stream_kind_t convert_to_c(kind akind);

    stream(kind akind);

    stream &submit(std::vector<primitive> primitives);

    bool wait(bool block = true);

    stream &rerun();
};

%rename (convolution_forward_desc) convolution_forward::desc;
%rename (convolution_forward_primitive_desc) convolution_forward::primitive_desc;
struct convolution_forward: public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc src_primitive_desc() const;

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc bias_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    convolution_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst);

    convolution_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst);
};

%rename (convolution_backward_desc) convolution_backward::desc;
%rename (convolution_backward_primitive_desc) convolution_backward::primitive_desc;
struct convolution_backward_data : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc diff_src_primitive_desc() const;

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc diff_dst_primitive_desc() const;
    };

    convolution_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at &weights,
            const memory &diff_src);
};

%rename (convolution_backward_weights_desc) convolution_backward_weights::desc;
%rename (convolution_backward_weights_primitive_desc) convolution_backward_weights::primitive_desc;
struct convolution_backward_weights : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);

        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc src_primitive_desc() const;

        memory::primitive_desc diff_weights_primitive_desc() const;

        memory::primitive_desc diff_bias_primitive_desc() const;

        memory::primitive_desc diff_dst_primitive_desc() const;
    };

    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights, const memory &diff_bias);

    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights);
};

%rename (convolution_relu_forward_desc) convolution_relu_forward::desc;
%rename (convolution_relu_forward_primitive_desc) convolution_relu_forward::primitive_desc;
struct convolution_relu_forward : public primitive {
    struct desc {
        mkldnn_convolution_relu_desc_t data;
        desc(const convolution_forward::desc conv_desc,
                const double negative_slope);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);
    };

    convolution_relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst);

    convolution_relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst);
};

%rename (lrn_forward_desc) lrn_forward::desc;
%rename (lrn_forward_primitive_desc) lrn_forward::primitive_desc;
struct lrn_forward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, double alpha, double beta);
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, double alpha, double beta, double k);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc src_primitive_desc() const;

        memory::primitive_desc workspace_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &workspace,
            const memory &dst);

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst);
};

%rename (lrn_backward_desc) lrn_backward::desc;
%rename (lrn_backward_primitive_desc) lrn_backward::primitive_desc;
struct lrn_backward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, double alpha, double beta);
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, double alpha, double beta, double k);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const lrn_forward::primitive_desc &hint_fwd_primitive_desc);

        memory::primitive_desc diff_src_primitive_desc() const;

        memory::primitive_desc workspace_primitive_desc() const;

        memory::primitive_desc diff_dst_primitive_desc() const;
    };

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src);

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src);
};

%rename (pooling_forward_desc) pooling_forward::desc;
%rename (pooling_forward_primitive_desc) pooling_forward::primitive_desc;
struct pooling_forward : public primitive {
    struct desc {
        mkldnn_pooling_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims kernel,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc workspace_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    pooling_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst);

    pooling_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst, const memory &workspace);
};

%rename (pooling_backward_desc) pooling_backward::desc;
%rename (pooling_backward_primitive_desc) pooling_backward::primitive_desc;
struct pooling_backward : public primitive {
    struct desc {
        mkldnn_pooling_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims &strides,
                const memory::dims &kernel,
                const memory::dims &padding_l,
                const memory::dims &padding_r,
                const padding_kind apadding_kind);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const pooling_forward::primitive_desc &hint_fwd_primitive_desc);
        
        memory::primitive_desc diff_src_primitive_desc() const;
    };

    pooling_backward(const primitive_desc &aprimitive_desc, const primitive::at &diff_dst,
            const memory &diff_src);

    pooling_backward(const primitive_desc &aprimitive_desc, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src);
};

%rename (relu_forward_desc) relu_forward::desc;
%rename (relu_forward_primitive_desc) relu_forward::primitive_desc;
struct relu_forward : public primitive {
    struct desc {
        mkldnn_relu_desc_t data;
        // template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                float negative_slope);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc dst_primitive_desc() const;
    };

    relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst);
};

%rename (relu_backward_desc) relu_backward::desc;
%rename (relu_backward_primitive_desc) relu_backward::primitive_desc;
struct relu_backward : public primitive {
    struct desc {
        mkldnn_relu_desc_t data;
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
            float negative_slope);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const relu_forward::primitive_desc &hint_fwd_primitive_desc);
        
        memory::primitive_desc diff_src_primitive_desc() const;
    };

    relu_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src);
};

%rename (softmax_forward_desc) softmax_forward::desc;
%rename (softmax_forward_primitive_desc) softmax_forward::primitive_desc;
struct softmax_forward : public primitive {
    struct desc {
        mkldnn_softmax_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int softmax_axis);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);
    };

    softmax_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst);
};

%rename (batch_normalization_forward_desc) batch_normalization_forward::desc;
%rename (batch_normalization_forward_primitive) batch_normalization_forward::primitive_desc;
struct batch_normalization_forward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &src_desc, T epsilon,
                unsigned flags);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc mean_primitive_desc() const;

        memory::primitive_desc variance_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &weights,
            const memory &dst);

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const memory &dst);

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst, const memory &mean, const memory &variance);

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst, const memory &mean,
            const memory &variance);

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst);

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst);
};

%rename (batch_normalization_backward_desc) batch_normalization_backward::desc;
%rename (batch_normalization_backward_primitive_desc) batch_normalization_backward::primitive_desc;
struct batch_normalization_backward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T epsilon, unsigned flags);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const batch_normalization_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc diff_weights_primitive_desc() const;

        memory::primitive_desc mean_primitive_desc() const;

        memory::primitive_desc variance_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    // Prop_kind == backward
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const primitive::at &weights, const memory &diff_src,
            const memory &diff_weights);

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance,const primitive::at &diff_dst,
            const primitive::at &weights,  const memory &diff_src);

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const memory &diff_src);
};

%rename (inner_product_forward_desc) inner_product_forward::desc;
%rename (inner_product_forward_primitive_desc) inner_product_forward::primitive_desc;

%exception inner_product_forward::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct inner_product_forward: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc);

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc src_primitive_desc() const;

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc bias_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const primitive::at &bias, const memory &dst);

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const memory &dst);
};

%rename (inner_product_backward_data_desc) inner_product_backward_data::desc;
%rename (inner_product_backward_data_primitive_desc) inner_product_backward_data::primitive_desc;
struct inner_product_backward_data: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc diff_dst_primitive_desc() const;

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc diff_src_primitive_desc() const;
    };

    inner_product_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at weights,
            const memory &diff_src);
};

%rename (inner_product_backward_weights_desc) inner_product_backward_weights::desc;
%rename (inner_product_backward_weights_primitive_desc) inner_product_backward_weights::primitive_desc;
struct inner_product_backward_weights: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc);
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc diff_dst_primitive_desc() const;

        memory::primitive_desc diff_weights_primitive_desc() const;

        memory::primitive_desc diff_bias_primitive_desc() const;

        memory::primitive_desc src_primitive_desc() const;
    };

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights);

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights, const memory &diff_bias);
};

} // namespace mkldnn

%template (memory_dims) std::vector<int>;
%template (primitive_vector) std::vector<mkldnn::primitive>;

%inline %{
  void *get_dummy() {
    return reinterpret_cast<void *>(0x4);
  }
%}
