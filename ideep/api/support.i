%module (package="mkldnn.api") support
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

namespace mkldnn {

namespace c_api {
  %import c_api.i
}

template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) {}

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

%template (engine_handle) handle< mkldnn_engine_t >;

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

%rename (at) primitive::at;
%template (primitive_handle) handle<mkldnn_primitive_t>;

class primitive: public handle<mkldnn_primitive_t> {
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
    eltwise_d = mkldnn_query_eltwise_d,
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
    convolution_winograd = mkldnn_convolution_winograd,
    eltwise_relu = mkldnn_eltwise_relu,
    lrn_across_channels = mkldnn_lrn_across_channels,
    lrn_within_channel  = mkldnn_lrn_within_channel,
    pooling_max = mkldnn_pooling_max,
    pooling_avg = mkldnn_pooling_avg,
    pooling_avg_include_padding = mkldnn_pooling_avg_include_padding,
    pooling_avg_exclude_padding = mkldnn_pooling_avg_exclude_padding
};

enum batch_normalization_flag {
    use_global_stats = mkldnn_use_global_stats,
    use_scale_shift = mkldnn_use_scaleshift,
    omit_stats = mkldnn_omit_stats
};

static mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<mkldnn_alg_kind_t>(aalgorithm);
}

%exception stream::submit {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

%template (stream_handle_t) handle< mkldnn_stream_t >;

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

}

%template (primitive_list) std::vector<mkldnn::primitive>;
