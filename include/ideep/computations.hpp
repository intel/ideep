#ifndef IDEEP_HPP
#define IDEEP_HPP

#include "abstract_types.hpp"
#include "tensor.hpp"
#include "lru_cache.hpp"
#include "utils.hpp"

namespace ideep {

using tdims_t = tensor::dims;
using tview_t = tensor::view;
using tdesc_t = tensor::descriptor;
using tdtype_t = tensor::data_type;

/// A group of primitive descriptors, pack related reorder descriptors
/// with computational descriptor.
class descriptor_group: public c_wrapper<mkldnn_primitive_desc_t> {
  friend class primitive_group;
public:
  /// Post ops for fusion operations
  class post_ops : public c_wrapper<mkldnn_post_ops_t> {
  public:
    post_ops() : c_wrapper([]() {
      mkldnn_post_ops_t result;
      error::wrap_c_api(mkldnn_post_ops_create(&result), "could not create post operation sequence");
      return result;
    }()) {}

    int num_ops() const {
      return mkldnn_post_ops_len(get());
    }

    kind op_kind(int index) const {
      IDEEP_ENFORCE(index < num_ops(), "post_ops index is out of range");
      return static_cast<kind>(mkldnn_post_ops_get_kind(get(), index));
    }

    bool has_op_kind(kind op_kind) const {
      for (int i = 0; i < num_ops(); i++) {
        if (op_kind == this->op_kind(i)) {
          return true;
        }
      }
      return false;
    }

    bool non_negitive_output() const {
      auto last = num_ops() - 1;
      if (last < 0) {
        return false;
      }

      auto params = get_params(last);
      if (std::get<0>(params) != kind::eltwise
          || std::get<1>(params) <= 0.f || std::get<2>(params) != 0.f
          || std::get<3>(params) != 0.f || std::get<4>(params) != algorithm::eltwise_relu)
        return false;

      return true;
    }

    void append(kind op_kind,
        float scale, float alpha, float beta, algorithm alg) {
      switch(op_kind) {
        case kind::sum:
          error::wrap_c_api(mkldnn_post_ops_append_sum(get(), scale), "could not append sum");
          break;
        case kind::eltwise:
          error::wrap_c_api(mkldnn_post_ops_append_eltwise(
                get(), scale, convert_to_c(alg), alpha, beta), "could not append eltwise");
          break;
        default:
          error::wrap_c_api(mkldnn_invalid_arguments, "Unsupport op kind");
      }
    }

    std::tuple<kind, float, float, float, algorithm> get_params(int index) const {
      mkldnn_alg_kind_t c_alg = mkldnn_eltwise_relu;
      float scale = 1.0, alpha = 1.0, beta = 0.0;

      auto akind = op_kind(index);
      switch(akind) {
        case kind::sum:
          error::wrap_c_api(mkldnn_post_ops_get_params_sum(get(), index, &scale),
              "could not get sum params");
          break;
        case kind::eltwise:
          error::wrap_c_api(mkldnn_post_ops_get_params_eltwise(
                get(), index, &scale, &c_alg, &alpha, &beta),
              "could not get eltwise params");
          break;
        default:
          error::wrap_c_api(mkldnn_invalid_arguments, "could not get params");
          break;
      }

      return std::make_tuple(akind, scale, alpha, beta, static_cast<algorithm>(c_alg));
    }

    void to_bytes(utils::bytestring& bytes) const {

      for (int i = 0; i < num_ops(); i ++) {
        kind akind;
        algorithm alg;
        float scale = 1.0, alpha = 1.0, beta = 0.0;
        std::tie(akind, scale, alpha, beta, alg) = get_params(i);

        switch(akind) {
          case kind::sum:
            utils::to_bytes(bytes, akind);
            bytes.append(1, '.');
            utils::to_bytes(bytes, scale);
            break;
          case kind::eltwise:
            utils::to_bytes(bytes, akind);
            bytes.append(1, '.');
            utils::to_bytes(bytes, scale);
            bytes.append(1, '.');
            utils::to_bytes(bytes, alpha);
            bytes.append(1, '.');
            utils::to_bytes(bytes, beta);
            bytes.append(1, '.');
            utils::to_bytes(bytes, alg);
          default:
            break;
        }
      }
    }

  public:
    // Helper factory
    static post_ops sum(float scale = 1.0) {
      post_ops ret;
      ret.append(kind::sum, scale, 1.0, 0.0, algorithm::eltwise_relu);
      return ret;
    }

    static post_ops relu(float scale = 1.f, float alpha = 0.f, float beta = 0.f) {
      post_ops ret;
      ret.append(kind::eltwise, scale, alpha, beta, algorithm::eltwise_relu);
      return ret;
    }

    static post_ops residual(
        float sum_scale = 1.0, float relu_scale = 1.0, float alpha = 0.f, float beta = 0.f) {
      post_ops ret;
      ret.append(kind::sum, sum_scale, 1.0, 0.0, algorithm::eltwise_relu);
      ret.append(kind::eltwise, relu_scale, alpha, beta, algorithm::eltwise_relu);
      return ret;
    }
  };

  /// Attribute class for extra information into computations, including
  /// post operations, rounding mode, etc.
  class attr_t : public c_wrapper<mkldnn_primitive_attr_t> {
  public:
    attr_t() : c_wrapper([]() {
      mkldnn_primitive_attr_t result;
      error::wrap_c_api(mkldnn_primitive_attr_create(&result), "could not create a primitive attr");
      return result;
    }()) {}

    attr_t(int mask, scale_t &scales, round_mode mode = round_mode::round_nearest)
      : c_wrapper([]() {
      mkldnn_primitive_attr_t result;
      error::wrap_c_api(mkldnn_primitive_attr_create(&result), "could not create a primitive attr");
      return result; }()) {
      set_output_scales(mask, scales);
      set_int_output_round_mode(round_mode::round_nearest);
    }

    round_mode get_int_output_round_mode() const {
      mkldnn_round_mode_t result;
      error::wrap_c_api(mkldnn_primitive_attr_get_int_output_round_mode(get(), &result),
          "could not get int output round mode");
      return round_mode(result);
    }

    void set_int_output_round_mode(round_mode mode) {
      error::wrap_c_api(mkldnn_primitive_attr_set_int_output_round_mode(
            get(), mkldnn::convert_to_c(mode)),
          "could not set int output round mode");
    }

    std::pair<scale_t, int> get_output_scales() const {
      int count, c_mask;
      const float *c_scales;
      error::wrap_c_api(mkldnn_primitive_attr_get_output_scales(
            get(), &count, &c_mask, &c_scales),
          "could not get int output scales");
      return std::make_pair(scale_t(c_scales, c_scales + count), c_mask);
    }

    void set_output_scales(int mask, const scale_t &scales) {
      error::wrap_c_api(mkldnn_primitive_attr_set_output_scales(
            get(), (int)scales.size(), mask, &scales[0]),
          "could not set int output scales");
    }

    const post_ops get_post_ops() const {
      const_mkldnn_post_ops_t c_result;
      error::wrap_c_api(mkldnn_primitive_attr_get_post_ops(get(), &c_result),
          "could not get post operatoion sequence");
      post_ops result;
      result.reset(const_cast<mkldnn_post_ops_t>(c_result), true);
      return result;
    }

    void set_post_ops(post_ops ops) {
      error::wrap_c_api(mkldnn_primitive_attr_set_post_ops(get(), ops.get()),
          "could not set post operation sequence");
    }

    void to_bytes(utils::bytestring& bytes) const {
      get_post_ops().to_bytes(bytes);
      auto scales = get_output_scales();
      utils::to_bytes(bytes, scales.first);
      utils::to_bytes(bytes, scales.second);
    }

  public:
    // Helper factory
    static inline attr_t fuse_sum(float scale = 1.0) {
      attr_t attr;
      attr.set_post_ops(post_ops::sum(scale));
      return attr;
    }

    static inline attr_t fuse_relu(float scale = 1.0, float alpha = 0.f, float beta = 0.f) {
      attr_t attr;
      attr.set_post_ops(post_ops::relu(scale, alpha, beta));
      return attr;
    }

    static inline attr_t residual( float sum_scale = 1.0, float relu_scale = 1.0,
        float alpha = 0.f, float beta = 0.f) {
      attr_t attr;
      attr.set_post_ops(post_ops::residual(sum_scale, relu_scale, alpha, beta));
      return attr;
    }

    static inline attr_t attr_post_ops(post_ops post) {
      attr_t attr;
      attr.set_post_ops(post);
      return attr;
    }
  };

protected:
  std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
      const std::vector<tdesc_t> &inputs) {
    std::vector<const_mkldnn_primitive_desc_t> c_api_inputs;
    c_api_inputs.reserve(inputs.size());

    auto convert_to_c = [](const tdesc_t &d) { return d.get(); };
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(c_api_inputs), convert_to_c);
    return c_api_inputs;
  }

public:
  descriptor_group() = default;

  /// Query interface
  tdesc_t expected_descriptor_of(mkldnn::query q, int index = 0) const {
    const_mkldnn_primitive_desc_t const_cdesc =
        mkldnn_primitive_desc_query_pd(get(), mkldnn::convert_to_c(q), index);
    return param::descriptor(const_cdesc);
  }

  /// Query expected input descriptor
  tdesc_t expected_input_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::input_pd, index);
  }

  /// Query expected output descriptor
  tdesc_t expected_output_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::output_pd, index);
  }

  /// Query expected src descriptor
  tdesc_t expected_src_descriptor() const {
    return expected_descriptor_of(mkldnn::src_pd);
  }

  /// Query expected weights descriptor
  tdesc_t expected_weights_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd);
  }

  /// Query expected bias descriptor
  tdesc_t expected_bias_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd, 1);
  }

  /// Query expected dst descriptor
  tdesc_t expected_dst_descriptor() const {
    return expected_descriptor_of(mkldnn::dst_pd, 0);
  }

  /// Query expected workspace descriptor
  tdesc_t expected_workspace_descriptor() const {
    return expected_descriptor_of(mkldnn::workspace_pd, 0);
  }

  /// Query expected gradient X descriptor
  tdesc_t expected_gradx_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_src_pd, 0);
  }

  /// Query expected gradient Y descriptor
  tdesc_t expected_grady_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_dst_pd, 0);
  }

  /// Qeury expected weights gradient descriptor
  tdesc_t expected_gradw_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 0);
  }

  /// Qeury expected bias gradient descriptor
  tdesc_t expected_gradb_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 1);
  }

  /// Query interface
  tdesc_t dup_descriptor_of(mkldnn::query q, int index = 0) const {
    mkldnn_primitive_desc_t cdesc;
    const_mkldnn_primitive_desc_t const_cdesc =
        mkldnn_primitive_desc_query_pd(get(), mkldnn::convert_to_c(q), index);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
        "could not clone a src primititve descriptor");
    return param::descriptor(cdesc);
  }

  /// Query expected input descriptor
  tdesc_t dup_input_descriptor(int index) const {
    return dup_descriptor_of(mkldnn::input_pd, index);
  }

  /// Query expected output descriptor
  tdesc_t dup_output_descriptor(int index) const {
    return dup_descriptor_of(mkldnn::output_pd, index);
  }

  /// Query expected src descriptor
  tdesc_t dup_src_descriptor() const {
    return dup_descriptor_of(mkldnn::src_pd);
  }

  /// Query expected weights descriptor
  tdesc_t dup_weights_descriptor() const {
    return dup_descriptor_of(mkldnn::weights_pd);
  }

  /// Query expected bias descriptor
  tdesc_t dup_bias_descriptor() const {
    return dup_descriptor_of(mkldnn::weights_pd, 1);
  }

  /// Query expected dst descriptor
  tdesc_t dup_dst_descriptor() const {
    return dup_descriptor_of(mkldnn::dst_pd, 0);
  }

  /// Query expected workspace descriptor
  tdesc_t dup_workspace_descriptor() const {
    return dup_descriptor_of(mkldnn::workspace_pd, 0);
  }

  /// Query expected gradient X descriptor
  tdesc_t dup_gradx_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_src_pd, 0);
  }

  /// Query expected gradient Y descriptor
  tdesc_t dup_grady_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_dst_pd, 0);
  }

  /// Qeury expected weights gradient descriptor
  tdesc_t dup_gradw_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_weights_pd, 0);
  }

  /// Qeury expected bias gradient descriptor
  tdesc_t dup_gradb_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_weights_pd, 1);
  }

  /// Query number of inputs
  int num_of_inputs() const {
      return mkldnn_primitive_desc_query_s32(get(),
         mkldnn::convert_to_c(mkldnn::num_of_inputs_s32), 0);
  }

  /// Query number of outputs
  int num_of_outputs() const {
      return mkldnn_primitive_desc_query_s32(get(),
         mkldnn::convert_to_c(mkldnn::num_of_outputs_s32), 0);
  }
};

/// A group of primitives, pack related reorder with computation.
/// It serves as a base class of computation
class primitive_group: public c_wrapper<mkldnn_primitive_t> {
public:
  primitive_group() = default;

  /// Returns the internal structure of primitive descriptor.
  const_mkldnn_primitive_desc_t get_mkldnn_primitive_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &cdesc),
        "could not get primitive descriptor from a memory primitive");
    return cdesc;
  }

  /// Query interface
  tdesc_t expected_descriptor_of(mkldnn::query q, int index = 0) const {
    const_mkldnn_primitive_desc_t const_cdesc = mkldnn_primitive_desc_query_pd(
        get_mkldnn_primitive_desc_t(), mkldnn::convert_to_c(q), index);
    return tdesc_t(const_cdesc);
  }

  /// Query interface
  tdesc_t dup_descriptor_of(mkldnn::query q, int index = 0) const {
    mkldnn_primitive_desc_t cdesc;
    const_mkldnn_primitive_desc_t const_cdesc = mkldnn_primitive_desc_query_pd(
        get_mkldnn_primitive_desc_t(), mkldnn::convert_to_c(q), index);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
        "could not clone a src primititve descriptor");
    return tdesc_t(cdesc);
  }

protected:
  /// Specific query interface, not valid for all computations.
  tdesc_t expected_input_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::input_pd, index);
  }

  tdesc_t expected_output_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::output_pd, index);
  }

  tdesc_t expected_src_descriptor() const {
    return expected_descriptor_of(mkldnn::src_pd);
  }

  tdesc_t expected_weights_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd);
  }

  tdesc_t expected_bias_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd, 1);
  }

  tdesc_t expected_dst_descriptor() const {
    return expected_descriptor_of(mkldnn::dst_pd, 0);
  }

  tdesc_t expected_workspace_descriptor() const {
    return expected_descriptor_of(mkldnn::workspace_pd, 0);
  }

  tdesc_t expected_gradx_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_src_pd, 0);
  }

  tdesc_t expected_grady_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_dst_pd, 0);
  }

  tdesc_t expected_gradw_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 0);
  }

  tdesc_t expected_gradb_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 1);
  }

  /// Specific query interface, not valid for all computations.
  tdesc_t dup_input_descriptor(int index) const {
    return dup_descriptor_of(mkldnn::input_pd, index);
  }

  tdesc_t dup_output_descriptor(int index) const {
    return dup_descriptor_of(mkldnn::output_pd, index);
  }

  tdesc_t dup_src_descriptor() const {
    return dup_descriptor_of(mkldnn::src_pd);
  }

  tdesc_t dup_weights_descriptor() const {
    return dup_descriptor_of(mkldnn::weights_pd);
  }

  tdesc_t dup_bias_descriptor() const {
    return dup_descriptor_of(mkldnn::weights_pd, 1);
  }

  tdesc_t dup_dst_descriptor() const {
    return dup_descriptor_of(mkldnn::dst_pd, 0);
  }

  tdesc_t dup_workspace_descriptor() const {
    return dup_descriptor_of(mkldnn::workspace_pd, 0);
  }

  tdesc_t dup_gradx_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_src_pd, 0);
  }

  tdesc_t dup_grady_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_dst_pd, 0);
  }

  tdesc_t dup_gradw_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_weights_pd, 0);
  }

  tdesc_t dup_gradb_descriptor() const {
    return dup_descriptor_of(mkldnn::diff_weights_pd, 1);
  }

  void execute(stream &parallel_control) {
    std::vector<mkldnn_primitive_t> execution_sequence;
    mkldnn_primitive_t c_api_error_primitive;

    execution_sequence.push_back(get());
    error::wrap_c_api(mkldnn_stream_submit(
          parallel_control.get(), execution_sequence.size(),
          &execution_sequence[0], &c_api_error_primitive),
        "could not execute the computation");
  }
};

struct reorder: public c_wrapper<mkldnn_primitive_t>,
  public utils::computation_cache<reorder> {
  struct descriptor : public c_wrapper<mkldnn_primitive_desc_t> {
    using attr_t = descriptor_group::attr_t;
    using post_ops = descriptor_group::post_ops;

    descriptor(const c_wrapper<mkldnn_primitive_desc_t> &input,
        const tdesc_t &output,
        const attr_t& attr = attr_t()) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
            &result, input.get(), output.get(), attr.get()),
          "could not create a reorder primitive descriptor");
      reset(result);
    }
  };

public:
  using attr_t = descriptor::attr_t;

  reorder() = default;

  void init(const tdesc_t& src_desc, const tdesc_t& dst_desc, const attr_t& attr = attr_t()) {
    mkldnn_primitive_desc_t desc;
    error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
          &desc, src_desc.get(), dst_desc.get(), attr.get()),
        "could not create a reorder primitive descriptor");
    c_wrapper<mkldnn_primitive_desc_t> sg(desc);

    in_.init(src_desc, nullptr);
    out_.init(dst_desc, nullptr);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in_.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out_.get() };
    error::wrap_c_api(mkldnn_primitive_create(&result, desc, inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }

  void init(const tview_t& view, const tdesc_t& src_desc,
      const tdesc_t& dst_desc, const attr_t& attr = attr_t()) {
    mkldnn_primitive_desc_t desc;
    error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
          &desc, view.get(), dst_desc.get(), attr.get()),
        "could not create a reorder primitive descriptor");
    c_wrapper<mkldnn_primitive_desc_t> sg(desc);

    in_.init(src_desc, nullptr);
    out_.init(dst_desc, nullptr);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in_.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out_.get() };
    error::wrap_c_api(mkldnn_primitive_create(&result, desc, inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }

  void init(const tdesc_t& src_desc, const tview_t& view,
      const tdesc_t& dst_desc, const attr_t& attr = attr_t()) {
    mkldnn_primitive_desc_t desc;
    error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
          &desc, src_desc.get(), view.get(), attr.get()),
        "could not create a reorder primitive descriptor");
    c_wrapper<mkldnn_primitive_desc_t> sg(desc);

    in_.init(src_desc, nullptr);
    out_.init(dst_desc, nullptr);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in_.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out_.get() };
    error::wrap_c_api(mkldnn_primitive_create(&result, desc, inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }

  template<typename T, typename... Ts>
  reorder(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void operator() (const tensor &input, const tensor &output) {
    IDEEP_ENFORCE(!(input.get_data_type() == tdtype_t::s8
          && output.get_data_type() == tdtype_t::u8),
        "Not support the reorder of s8 to u8 to avoid overflow.");
    IDEEP_ENFORCE(input.get_descriptor() == in_.get_descriptor()
        && output.get_descriptor() == out_.get_descriptor(),
        "Unmatch tensor descriptor in reorder");

    in_.set_data_handle(input.get_data_handle());
    out_.set_data_handle(output.get_data_handle());

    std::vector<mkldnn_primitive_t> execution_sequence = {get()};
    mkldnn_primitive_t c_api_error_primitive;

    error::wrap_c_api(mkldnn_stream_submit(
          stream::default_stream().get(), execution_sequence.size(),
          &execution_sequence[0], &c_api_error_primitive),
        "could not execute reorder");
  }

  static void compute( const tensor& input, tensor& output, const attr_t& attr = attr_t()) {
    if (input.is_empty() || output.is_empty())
      return;

    // TODO:it will be remove when deconvolution in mkl-dnn support iohw format.
    auto input_in = input;
    if (input_in.is_iohw_public_layout()) {
      tensor::iohw_definedby_blocked(input_in);
    }

    key_t key;
    if (output.get_internal_format() == static_cast<format>(mkldnn_blocked) &&
        input_in.get_internal_format() == static_cast<format>(mkldnn_blocked)) {
      utils::create_key(key, input_in, output, attr);
    } else if (output.get_internal_format() == static_cast<format>(mkldnn_blocked)) {
      utils::create_key(key, input_in.get_dims(), input_in.get_data_type(),
          input_in.get_internal_format(), output, attr);
    } else if (input_in.get_internal_format() == static_cast<format>(mkldnn_blocked)) {
      utils::create_key(key, input_in, output.get_dims(), output.get_data_type(),
          output.get_internal_format(), attr);
    } else {
      utils::create_key(key, input_in.get_dims(), input_in.get_data_type(),
          input_in.get_internal_format(), output.get_dims(), output.get_data_type(),
          output.get_internal_format(), attr);
    }

    fetch_or_create_m(op, key, input_in.get_descriptor(), output.get_descriptor(), attr);
    op(input_in, output);
  }

  // TODO: make this right
  static tensor compute( const tensor &input, const tdims_t &volume, const tdims_t &start) {
    key_t key;
    utils::create_key(key, input.get_dims(), input.get_data_type(),
        input.get_internal_format(), volume, start);

    auto view = input.create_view(volume, start);
    tensor gx;
    gx.init<reorder>(view.expected_dst_descriptor());

    fetch_or_create_m(op, key, view, input.get_descriptor(), gx.get_descriptor());
    op(input, gx);
    return gx;
  }

protected:
  tensor in_, out_;
};

struct direct_copy : public reorder {
public:
  using reorder::reorder;

  static void compute(const tensor& input, tensor& output) {
    if (input.is_empty() || input == output) {
      return;
    }

    output.reinit<direct_copy>(input.get_descriptor());
    reorder::compute(input, output);
    if (input.has_scale()) {
      output.set_scale(input.get_scale());
    }
  }
};

struct spliter : public reorder {
public:
  using reorder::reorder;

  static std::vector<tensor> compute(const tensor& input,
      std::vector<int32_t>& axis_info, int axis, bool add_axis) {
    reorder reorder_;
    std::vector<tensor> outputs;
    tdims_t output_dims(input.get_dims());
    tdims_t offset_dims(output_dims.size(), 0);
    IDEEP_ENFORCE(axis < input.ndims(), "invalid axis in split");

    for (unsigned i = 0; i < axis_info.size(); ++i) {
      output_dims[axis] = axis_info[i];
      auto view = input.create_view(output_dims, offset_dims);
      tensor output(view.expected_dst_descriptor());
      reorder_.init(view, input.get_descriptor(), output.get_descriptor());
      reorder_(input, output);
      if (input.has_scale()) output.set_scale(input.get_scale());

      if (add_axis) {
        tdims_t out_dims(output_dims);
        out_dims.erase(out_dims.begin() + axis);
        output.reshape(out_dims);
      }

      outputs.emplace_back(output);
      offset_dims[axis] += axis_info[i];
    }

    return outputs;
  }
};

/// Computation class, abstruct of computation
struct computation : public primitive_group {
public:
  template<class ...T>
  using s_vector = utils::s_vector<T...>;

  computation() = default;

  inline void init_internal(const descriptor_group &adesc) {
    inouts_ = s_vector<tensor>((unsigned)(inputs_num_ + outputs_num_));

    std::unique_ptr<mkldnn_primitive_at_t []> inputs(new mkldnn_primitive_at_t [inputs_num_]);
    for (int i =0; i < inputs_num_; i ++) {
      inouts_[i] = {adesc.expected_input_descriptor(i), nullptr };
      inputs[i] = { inouts_[i].get(), 0 };
    }

    std::unique_ptr<const_mkldnn_primitive_t []> outputs(new const_mkldnn_primitive_t [outputs_num_]);
    for (int i = 0; i < outputs_num_; i ++) {
      inouts_[i + inputs_num_] = {adesc.expected_output_descriptor(i), nullptr };
      outputs[i] = inouts_[i + inputs_num_].get();
    }

    mkldnn_primitive_t result;
    error::wrap_c_api(mkldnn_primitive_create(&result, adesc.get(), inputs.get(), outputs.get()),
        "could not create a computation primitive");
    reset(result);
  }

  void init(const descriptor_group& adesc, const std::vector<tdesc_t> &args) {
    IDEEP_ENFORCE(adesc.num_of_inputs() == (int)args.size(), "Unmatch the number of inputs");
    inputs_num_ = (int)args.size();
    outputs_num_ = adesc.num_of_outputs();
    init_internal(adesc);
  }

  template<typename... Ts>
  void init(const descriptor_group &adesc, const Ts&... args) {
    inputs_num_ = adesc.num_of_inputs();
    outputs_num_ = adesc.num_of_outputs();
    init_internal(adesc);
  }

  void connect_handle_for(int index, const tensor& atensor) {
    IDEEP_ENFORCE(inouts_[(unsigned)index].get_descriptor() == atensor.get_descriptor(),
        "Incorrect tensor descriptor");
    inouts_[(unsigned)index].set_data_handle(atensor.get_data_handle<false>());
  }

  void connect_handle_for(const std::vector<tensor>& inputs, const tensor& output) {
    int i = 0;
    for(; (unsigned)i < inputs.size(); i++) {
      connect_handle_for(i, inputs[(unsigned)i]);
    }
    connect_handle_for(i, output);
  }

  template<typename ...Params>
  void connect_handle_for(int index, const tensor& first, const Params&... rest) {
    connect_handle_for(index, first);
    connect_handle_for(index + 1, rest...);
  }

  void execute(const std::vector<tensor>& inputs, const tensor& outputs) {
    connect_handle_for(inputs, outputs);
    stream parallel_control = stream::default_stream();
    primitive_group::execute(parallel_control);
  }

  template<typename ...Params>
  void execute(const tensor& arg0, const Params&... args) {
    connect_handle_for(0, arg0, args...);
    stream parallel_control = stream::default_stream();
    primitive_group::execute(parallel_control);
  }

  int num_of_inputs() const {
    IDEEP_ENFORCE(inouts_.size() == (inputs_num_ + outputs_num_),
        "Incorrect number of inputs and outputs");
    return inputs_num_;
  }

  int num_of_outputs() const {
    IDEEP_ENFORCE(inouts_.size() == (inputs_num_ + outputs_num_),
        "Incorrect number of inputs and outputs");
    return outputs_num_;
  }

private:
  int inputs_num_;
  int outputs_num_;
  s_vector<tensor> inouts_;
};

struct sum : public computation,
  public utils::computation_cache<sum> {
  struct descriptor : public descriptor_group {
    descriptor(const scale_t &scales, const std::vector<tdesc_t> &inputs) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_sum_primitive_desc_create(
            &result, nullptr, (int)c_api_inputs.size(), &scales[0], &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    descriptor(const scale_t &scales, const std::vector<tdesc_t> &inputs, const tdesc_t& output_desc) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_sum_primitive_desc_create(
              &result, output_desc.get_mkldnn_memory_desc_t(),
              (int)c_api_inputs.size(), &scales[0], &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::execute;
  using computation::expected_dst_descriptor;

  sum() = default;

  void init(const scale_t &scales, const std::vector<tdesc_t> &inputs) {
    descriptor forward_descriptor(scales, inputs);
    computation::init(forward_descriptor, inputs);
  }

  void init(const scale_t &scales, const std::vector<tdesc_t> &inputs,
      const tdesc_t& output) {
    descriptor forward_descriptor(scales, inputs, output);
    computation::init(forward_descriptor, inputs);
  }

  sum(const scale_t &scales, const std::vector<tdesc_t> &inputs_desc,
      const tdesc_t& output_desc) {
    init(scales, inputs_desc, output_desc);
  }

  sum(const scale_t& scales, const std::vector<tdesc_t>& inputs_desc) {
    init(scales, inputs_desc);
  }

  void execute(const std::vector<tensor>& inputs, const tensor& output) {
    computation::execute(inputs, output);
  }

  static void compute(const scale_t &scales, const std::vector<tensor>& inputs, tensor& output) {
    std::vector<tensor> inputs_in;
    std::vector<tdesc_t> inputs_desc;
    for (auto in : inputs) {
      auto _in = in;
      if (in.get_data_type() != tdtype_t::f32) {
        _in.init<sum>({in.get_dims(), tdtype_t::f32});
        IDEEP_ENFORCE(in.has_scale(), "Can not find scales");
        IDEEP_ENFORCE(in.get_scale().size() == 1, "Incorrect scale size");
        auto scale = IDEEP_DEF_SCALE;
        scale[0] /= in.get_scale()[0];
        reorder::compute(in, _in, {0, scale});
      }
      inputs_in.push_back(_in);
      inputs_desc.push_back(_in.get_descriptor());
    }

    if (output != inputs_in[0]) {
      sum comp(scales, inputs_desc);
      output.reinit<sum>(comp.expected_dst_descriptor());
      comp.execute(inputs_in, output);
    } else {
      sum comp(scales, inputs_desc, output.get_descriptor());
      comp.execute(inputs_in, output);
    }
  }
};

/// Convolution forward computation, this class represent a MKL-DNN
/// convolution forward process, also manage old computation instances.
struct convolution_forward: public computation,
  public utils::computation_cache<convolution_forward> {
  /// Descriptor class for describing convolution forward process
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &src_desc, const tdesc_t &weights_desc, const tdesc_t &bias_desc,
        const tdesc_t &dst_desc, const tdims_t& strides, const tdims_t& dilates,
        const tdims_t& padding_l, const tdims_t& padding_r, const attr_t& attr = attr_t(),
        algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
        padding_kind apadding_kind = padding_kind::zero) {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto bias_data = bias_desc.format_any();
      auto dst_data = attr.get_post_ops().has_op_kind(kind::sum) ?
        *dst_desc.get_mkldnn_memory_desc_t() : dst_desc.format_any();
      tdims_t dilates_in {0, 0};
      if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
        dilates_in = dilates;
        IDEEP_STD_EACH_SUB(dilates_in, 1);
      }
      error::wrap_c_api(mkldnn_dilated_convolution_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
            &src_data, &weights_data, &bias_data, &dst_data, &strides[0], &dilates_in[0],
            &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a dilated convolution forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
        &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
          "could not create a convolution forward primitive descriptor");
      reset(result);
    }

    descriptor(const tdesc_t &src_desc, const tdesc_t &weights_desc, const tdesc_t &dst_desc,
        const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l, const tdims_t& padding_r,
        const attr_t& attr = attr_t(), algorithm aalgorithm = algorithm::convolution_direct,
        prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto dst_data = attr.get_post_ops().has_op_kind(kind::sum) ?
        *dst_desc.get_mkldnn_memory_desc_t() : dst_desc.format_any();
      tdims_t dilates_in {0, 0};
      if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
        dilates_in = dilates;
        IDEEP_STD_EACH_SUB(dilates_in, 1);
      }
      error::wrap_c_api(mkldnn_dilated_convolution_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
            &src_data, &weights_data, nullptr, &dst_data, &strides[0], &dilates_in[0],
            &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a dilated convolution forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
            &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
          "could not create a convolution forward primitive descriptor");
      reset(result);
    }
  };

 public:
  using attr_t = descriptor::attr_t;
  using computation::expected_input_descriptor;
  using computation::expected_dst_descriptor;
  using computation::expected_weights_descriptor;

  template<typename T, typename ...Ts>
  convolution_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  template<typename T, typename ...Ts,
    typename = typename std::enable_if<std::is_same<T, tdesc_t>::value>::type>
  void init(const tdesc_t &src_desc, const tdesc_t &weights_desc,
      const tdesc_t &bias, const T &dst, Ts&&... args) {
    descriptor forward_descriptor(src_desc, weights_desc, bias, dst, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, src_desc, weights_desc, bias);
  }

  template<typename T, typename ...Ts,
    typename = typename std::enable_if<std::is_same<T, tdims_t>::value>::type>
  void init(const tdesc_t &src_desc, const tdesc_t &weights_desc,
      const tdesc_t &dst, const T something, Ts&&... args) {
    descriptor forward_descriptor(src_desc, weights_desc, dst, something, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, src_desc, weights_desc);
  }

  void execute(const tensor& src, const tensor& weights, const tensor& dst) {
    computation::execute(src, weights, dst);
  }

  void execute(const tensor& src, const tensor& weights, const tensor& bias, const tensor& dst) {
    computation::execute(src, weights, bias, dst);
  }

  template <bool with_bias>
  static void compute_impl(convolution_forward &comp, const tensor& src,
      const tensor& weights, const tensor& bias, tensor& dst) {
    auto src_in = src;
    if (comp.src_reorder_) {
      src_in = *comp.src_in_;
      comp.src_reorder_->operator()(src, src_in);
    }

    // solve the problem that slow reorder from nchw
    auto _weights = weights.as_weights();
    auto weights_in = _weights;
    if (comp.weights_reorder_) {
      weights_in = *comp.weights_in_;
      comp.weights_reorder_->operator()(_weights, weights_in);
    }

    if (comp.dst_exp_desc_) {
      dst.reinit<convolution_forward>(*comp.dst_exp_desc_);
    }
    if (comp.dst_scales_) {
      dst.set_scale(*comp.dst_scales_);
    }

    if (with_bias) {
      auto bias_in = bias;
      if (comp.bias_reorder_) {
        bias_in = *comp.bias_in_;
        comp.bias_reorder_->operator()(bias, bias_in);
      }
      comp.execute(src_in, weights_in, bias_in, dst);

    } else {
      comp.execute(src_in, weights_in, dst);
    }

    if (comp.dst_u8_desc_) {
      dst.set_descriptor(*comp.dst_u8_desc_);
    }
  }

  template <bool with_bias, typename ...Ts>
  static void compute_impl(key_t &key, const tensor& src, const tensor& weights, const tensor& bias,
      const tdims_t& dst_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, const scale_t& src_scales,
      const scale_t& weights_scales, const scale_t& dst_scales, const attr_t& attr,
      const lowp_kind alowp_kind, Ts&&... args) {
    scale_t dst_scales_in;
    auto dst_data_type = tdtype_t::f32;
    auto& post_ops = attr.get_post_ops();
    tdesc_t src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;

    auto weights_scales_in = weights.has_scale() ? weights.get_scale() : weights_scales;
    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(alowp_kind == LOWP_U8S8 || alowp_kind == LOWP_S8S8, "Unsupported lowp kind");
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
      auto src_scales_in = src.has_scale() ? src.get_scale()
        : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);

      // determine dst data type
      if (post_ops.has_op_kind(kind::sum)) {
        dst_data_type = dst.get_data_type();
      } else if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = tdtype_t::f32;
      } else if (post_ops.non_negitive_output()){
        dst_data_type = tdtype_t::u8;
      } else {
        dst_data_type = tdtype_t::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = (dst_scales.empty() || dst_data_type == tdtype_t::f32)
        ? IDEEP_DEF_SCALE : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
      }
      op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), op_scales);
      op_attr.set_int_output_round_mode(round_mode::round_nearest);

      if (post_ops.has_op_kind(kind::sum)) {
        float sum_scale = dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
        if (post_ops.has_op_kind(kind::eltwise)) {
          op_attr.set_post_ops(descriptor::post_ops::residual(sum_scale));
        } else {
          op_attr.set_post_ops(descriptor::post_ops::sum(sum_scale));
        }
      } else if (post_ops.has_op_kind(kind::eltwise)) {
        op_attr.set_post_ops(descriptor::post_ops::relu());
      }

      src_desc = {src.get_dims(), alowp_kind == LOWP_U8S8 ? tdtype_t::u8 : tdtype_t::s8};
      if (src.get_data_type() == tdtype_t::f32) {
        src_attr = {0 , src_scales_in};
      }

      weights_desc = {weights.get_dims(), tdtype_t::s8};
      if (weights.get_data_type() == tdtype_t::f32) {
        weights_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, weights.is_grouped()), weights_scales_in};
      }

      if (with_bias) {
        bias_desc = {bias.get_dims(), tdtype_t::s32};
        if (bias.get_data_type() == tdtype_t::f32) {
          bias_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), bias_scales};
        }
      }
    } else {
      op_attr = attr;

      src_desc = {src.get_dims(), tdtype_t::f32};
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }

      weights_desc = weights.get_descriptor();
      IDEEP_ENFORCE(weights.get_data_type() == tdtype_t::f32, "Incorrect data type in weights");

      if (with_bias) {
        IDEEP_ENFORCE(bias.get_data_type() == tdtype_t::f32, "Incorrect data type in bias");
        bias_desc = bias.get_descriptor();
      }
    }

    if (key.empty()) {
      if (with_bias)
        utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(),
            weights.get_data_type(), weights.get_dims(), weights.get_internal_format(), bias.get_dims(),
            strides, dilates, padding_l, padding_r, op_attr, src_scales, dst_scales, args...);
      else
        utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(),
            weights.get_data_type(), weights.get_dims(), weights.get_internal_format(),
            strides, dilates, padding_l, padding_r, op_attr, src_scales, dst_scales, args...);
    }

    auto dst_format = post_ops.has_op_kind(kind::sum) ?
      dst.get_internal_format() : engine::default_format(dst_dims.size());
    tdesc_t dst_desc_in(dst_dims, dst_data_type, dst_format);

    auto it = find(key);
    if (it == end()) {
      it = with_bias
        ? create(key, src_desc, weights_desc, bias_desc, dst_desc_in, strides, dilates,
            padding_l, padding_r, op_attr, std::forward<Ts>(args)...)
        : create(key, src_desc, weights_desc, dst_desc_in, strides, dilates, padding_l, padding_r,
            op_attr, std::forward<Ts>(args)...);
    }
    auto comp = fetch(it);

    auto src_in = src;
    if (src.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_forward>(comp.expected_src_descriptor());
      comp.src_reorder_.reset(new reorder);
      comp.src_reorder_->init(src.get_descriptor(), src_in.get_descriptor(), src_attr);
      comp.src_reorder_->operator()(src, src_in);
    }

    // solve the problem that slow reorder from nchw
    auto _weights = weights.as_weights();
    auto weights_in = _weights;
    if (_weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<convolution_forward>(comp.expected_weights_descriptor());
      comp.weights_reorder_.reset(new reorder);
      comp.weights_reorder_->init(_weights.get_descriptor(), weights_in.get_descriptor(), weights_attr);
      comp.weights_reorder_->operator()(_weights, weights_in);
    }

    auto dst_desc = comp.expected_dst_descriptor();
    if (dst.get_descriptor() != dst_desc) {
      comp.dst_exp_desc_.reset(new tdesc_t(dst_desc));
      IDEEP_ENFORCE(!post_ops.has_op_kind(kind::sum), "Unmatch format or data type in Conv Sum fusion");
      dst.reinit<convolution_forward>(dst_desc);
    }

    if (!dst_scales.empty() && dst_data_type != tdtype_t::f32) {
      dst.set_scale(dst_scales_in);
      comp.dst_scales_.reset(new scale_t(dst_scales_in));
    }

    if (with_bias) {
      auto bias_in = bias;
      if (bias.get_descriptor() != bias_desc) {
        bias_in.init<convolution_forward>(bias_desc);
        comp.bias_reorder_.reset(new reorder);
        comp.bias_reorder_->init(bias.get_descriptor(), bias_in.get_descriptor(), bias_attr);
        comp.bias_reorder_->operator()(bias, bias_in);
      }

      comp.execute(src_in, weights_in, bias_in, dst);
      comp.bias_in_ = std::make_shared<tensor>(bias_in);
    } else {
      comp.execute(src_in, weights_in, dst);
    }

    if (post_ops.non_negitive_output() && dst.get_data_type() == tdtype_t::s8) {
      tdesc_t dst_u8_desc { dst.get_dims(), tdtype_t::u8, dst.get_internal_format()};
      dst.set_descriptor(dst_u8_desc);
      comp.dst_u8_desc_ = std::make_shared<tdesc_t>(dst_u8_desc);
    }

    comp.src_in_ = std::make_shared<tensor>(src_in);
    comp.weights_in_ = std::make_shared<tensor>(weights_in);
    update(comp, it);
  }

  template<bool with_bias = true>
  static void compute(key_t &key, const tensor &src, const tensor& weights, const tensor& bias,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero, const lowp_kind alowp_kind = LOWP_U8S8) {
    auto weights_in = weights;
    weights_in.make_group(group);

    // FIXME: workaroud winograd format issue in inference
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd && aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    auto it = key.empty() ? end() : find(key);
    if (it != end()) {
      compute_impl<with_bias>(fetch(it), src, weights_in, bias, dst);
    } else {
      compute_impl<with_bias>(key, src, weights_in, bias, result_dims, dst, strides, dilates,
          padding_l, padding_r, src_scales, weights_scales, dst_scales, attr, alowp_kind,
          aalgorithm, apkind, appading_kind);
    }
  }

  static void compute(key_t &key, const tensor &src, const tensor& weights,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero, const lowp_kind alowp_kind = LOWP_U8S8) {
    static tensor dummy_bias;
    compute<false>(key, src, weights, dummy_bias, result_dims, dst, strides, dilates,
        padding_l, padding_r, group, src_scales, weights_scales, dst_scales, attr,
        aalgorithm, aprop_kind, appading_kind, alowp_kind);
  }

  static tdesc_t expected_weights_descriptor( const tdims_t& weights_dims,
      tdtype_t dtype = tdtype_t::f32, const tdims_t& strides = {1, 1},
      const tdims_t& padding_l = {0, 0}, const tdims_t& padding_r = {0, 0},
      const tdims_t& dilates = {0, 0}, int group = 1, algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward, tdtype_t x_dtype = tdtype_t::f32,
      const tdims_t& src_dims = tdims_t()) {
    auto dims_in = weights_dims;
    if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(dims_in)) {
      tensor::group_dims(dims_in, group);
    }
    auto ndims = dims_in.size();
    auto grouped = IDEEP_IS_GROUPED_4DIMS(dims_in);
    auto g = grouped ? dims_in[0] : 1;

    tdims_t dilates_in {0, 0};
    if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
      dilates_in = dilates;
      IDEEP_STD_EACH_SUB(dilates_in, 1);
    }

    IDEEP_ENFORCE(!(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    auto ic = g * dims_in[1 + grouped];
    auto oc = g * dims_in[0 + grouped];
    auto kh = dims_in[ndims - 2];
    auto kw = dims_in[ndims - 1];
    int mb, h, w;
    if (src_dims.empty()) {
      // Construct a dummy case
      mb = 1;
      h = 2 * kh;
      w = 4 * kw;
    } else {
      // Use the real data
      mb = src_dims[0];
      h = src_dims[2];
      w = src_dims[3];
    }
    auto oh = (h - ((kh - 1) * (dilates_in[0] + 1) + 1) + (padding_l[0] + padding_r[0])) / strides[0] + 1;
    auto ow = (w - ((kw - 1) * (dilates_in[1] + 1) + 1) + (padding_l[1] + padding_r[1])) / strides[1] + 1;

    tdims_t x_dims = { mb, ic, h, w};
    tdims_t y_dims = { mb, oc, oh, ow};
    auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;
    tdesc_t x_desc(x_dims, x_dtype, format::nchw);
    tdesc_t y_desc(y_dims, y_dtype, format::nchw);
    tdesc_t weights_desc(dims_in, dtype, grouped ? format::goihw : format::oihw);

    // FIXME: workaroud winograd format issue in inference
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd && aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    convolution_forward comp(x_desc, weights_desc, y_desc, strides, dilates, padding_l, padding_r,
        attr_t(), aalgorithm, apkind);
    return comp.dup_weights_descriptor();
  }

private:
  std::shared_ptr<reorder> src_reorder_, weights_reorder_, bias_reorder_;
  std::shared_ptr<tensor> src_in_, weights_in_, bias_in_;
  std::shared_ptr<tdesc_t> dst_exp_desc_;
  std::shared_ptr<tdesc_t> dst_u8_desc_;
  std::shared_ptr<scale_t> dst_scales_;
};

struct convolution_backward_data : public computation,
  public utils::computation_cache<convolution_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &grady_desc, const tdesc_t &weights_desc, const tdesc_t &gradx_desc,
        const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::convolution_direct, padding_kind apadding_kind = padding_kind::zero)
      : hint_(gradx_desc, weights_desc, grady_desc, strides, dilates, padding_l, padding_r)  {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto diff_src_any = gradx_desc.format_any();
      auto weights_any = weights_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      tdims_t dilates_in {0, 0};
      if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
        dilates_in = dilates;
        IDEEP_STD_EACH_SUB(dilates_in, 1);
      }
      error::wrap_c_api(mkldnn_dilated_convolution_backward_data_desc_init(
            &data, convert_to_c(aalgorithm), &diff_src_any, &weights_any, &diff_dst_any,
            &strides[0], &dilates_in[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward data descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward data primitive descriptor");
      reset(result);
    }
  private:
    convolution_forward::descriptor hint_;
  };
public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &grady_desc, const tdesc_t &weights_desc,
      const tdesc_t &gradx_desc, Ts&&... args) {
    descriptor backward_data_descriptor(grady_desc, weights_desc, gradx_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor, grady_desc, weights_desc);
  }

  template<typename T, typename ...Ts>
  convolution_backward_data (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& grady, const tensor& weights, const tensor& gradx) {
    computation::execute(grady, weights, gradx);
  }

  template<typename ...Ts>
  static void compute_impl(const tensor& grady, const tensor& weights,
      const tdims_t& gradx_dims, tensor& gradx, Ts&&... args) {
    tdesc_t result_desc(gradx_dims, grady.get_data_type());
    key_t key;
    utils::create_key(key, grady.get_data_type(), grady.get_dims(),
        weights.get_dims(), gradx_dims, args...);

    fetch_or_create_m(comp, key, grady.get_descriptor(),
        weights.get_descriptor(), result_desc, std::forward<Ts>(args)...);

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<convolution_backward_data>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    // solve the problem that slow reorder from nchw
    auto _weights = weights.as_weights();
    auto weights_in = _weights;
    if (_weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<convolution_backward_data>(comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    gradx.reinit<convolution_backward_data>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }

  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims,
      tensor& gradx, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    compute_impl(grady, weights, gradx_dims, gradx, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }

  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims,
      tensor& gradx, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, const int group, algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    auto weights_in = weights;
    weights_in.make_group(group);
    compute_impl(grady, weights_in, gradx_dims, gradx, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }
};

struct convolution_backward_weights : public computation,
  public utils::computation_cache<convolution_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &x_desc, const tdesc_t &grady_desc, const tdesc_t &gradw_desc,
        const tdesc_t &gradb_desc, const tdims_t& strides, const tdims_t& dilates,
        const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        padding_kind apadding_kind = padding_kind::zero)
      : hint_(x_desc, gradw_desc, gradb_desc, grady_desc,
         strides, dilates, padding_l, padding_r) {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto src_any = x_desc.format_any();
      auto diff_weights_any = gradw_desc.format_any();
      auto diff_bias_any = gradb_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      tdims_t dilates_in {0, 0};
      if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
        dilates_in = dilates;
        IDEEP_STD_EACH_SUB(dilates_in, 1);
      }
      error::wrap_c_api( mkldnn_dilated_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any, &diff_weights_any, &diff_bias_any,
            &diff_dst_any, &strides[0], &dilates_in[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward weights primitive descriptor");
      reset(result);
    }

    descriptor(const tdesc_t &x_desc, const tdesc_t &grady_desc, const tdesc_t &gradw_desc,
        const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::convolution_direct, padding_kind apadding_kind = padding_kind::zero)
    : hint_(x_desc, gradw_desc, grady_desc,
        strides, dilates, padding_l, padding_r) {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto src_any = x_desc.format_any();
      auto diff_weights_any = gradw_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      tdims_t dilates_in {0, 0};
      if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
        dilates_in = dilates;
        IDEEP_STD_EACH_SUB(dilates_in, 1);
      }
      error::wrap_c_api( mkldnn_dilated_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any, &diff_weights_any, nullptr, &diff_dst_any,
            &strides[0], &dilates_in[0],  &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward weights primitive descriptor");
      reset(result);
    }
  private:
    convolution_forward::descriptor hint_;
  };
public:
  using computation::expected_gradw_descriptor;
  using computation::expected_gradb_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &x_desc, const tdesc_t &grady_desc,
      const tdesc_t &gradw_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, grady_desc, gradw_desc, std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor, x_desc, grady_desc);
  }

  template<typename T, typename ...Ts>
  convolution_backward_weights (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& grady, const tensor& gradw, const tensor& grad_bias) {
    computation::execute(src, grady, gradw, grad_bias);
  }

  void execute(const tensor& src, const tensor& grady, const tensor& gradw) {
    computation::execute(src, grady, gradw);
  }

  template<typename ...Ts>
  static void compute_impl(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, Ts&&... args) {
    tdesc_t gradw_desc(gradw_dims, src.get_data_type());
    tdesc_t gradb_desc(tdims_t {grady.get_dim(1)}, src.get_data_type());

    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(),
        grady.get_dims(), gradw_dims, grady.get_dim(1), args...);

    fetch_or_create_m(comp, key, src.get_descriptor(), grady.get_descriptor(), gradw_desc,
        gradb_desc, std::forward<Ts>(args)...);

    auto src_in = src;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<convolution_backward_weights>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<convolution_backward_weights>(comp.expected_gradw_descriptor());
    gradb.reinit<convolution_backward_weights>(comp.expected_gradb_descriptor());
    comp.execute(src_in, grady_in, gradw, gradb);
  }

  template<typename ...Ts>
  static void compute_impl(const tensor& src, const tensor& grady,
      const tdims_t& gradw_dims, tensor& gradw, Ts&&... args) {
    tdesc_t gradw_desc(gradw_dims, src.get_data_type());

    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), grady.get_dims(), gradw_dims, args...);

    fetch_or_create_m(comp, key, src.get_descriptor(),
        grady.get_descriptor(), gradw_desc, std::forward<Ts>(args)...);

    auto src_in = src;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<convolution_backward_weights>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<convolution_backward_weights>(comp.expected_gradw_descriptor());
    comp.execute(src_in, grady_in, gradw);
  }

  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r,
      algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    compute_impl(src, grady, gradw_dims, gradw, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }

  static void compute(const tensor& src,
      const tensor& grady, const tdims_t& gradw_dims, tensor& gradw, tensor& gradb,
      const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    compute_impl(src, grady, gradw_dims, gradw, gradb, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }

  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, const int group,
      algorithm aalgorithm = algorithm::convolution_direct, padding_kind apadding_kind = padding_kind::zero) {
    auto gw_dims_in = gradw_dims;
    if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(gradw_dims)) {
      tensor::group_dims(gw_dims_in, group);
    }
    compute_impl(src, grady, gw_dims_in, gradw, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);

    if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(gradw_dims)) {
      IDEEP_ENFORCE(group == gradw.get_dim(0), "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims[0] == group * gradw.get_dim(1), "invalid dim 1 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims.size() == gradw.ndims() - 1, "invalid ndim in grouped gradw");
      gradw.reshape(gradw_dims);
    }
  }

  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, const int group, algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    auto gw_dims_in = gradw_dims;
    if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(gradw_dims)) {
      tensor::group_dims(gw_dims_in, group);
    }
    compute_impl(src, grady, gw_dims_in, gradw, gradb,
        strides, dilates, padding_l, padding_r, aalgorithm, apadding_kind);

    if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(gradw_dims)) {
      IDEEP_ENFORCE(group == gradw.get_dim(0), "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims[0] == group * gradw.get_dim(1), "invalid dim 1 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims.size() == gradw.ndims() - 1, "invalid ndim in grouped gradw");
      gradw.reshape(gradw_dims);
    }
  }
};

struct convolution_transpose_forward : public computation,
      public utils::computation_cache<convolution_transpose_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_desc, const tdesc_t& weights_desc, const tdesc_t& bias_desc,
        const tdesc_t& dst_desc, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
        const attr_t& attr = attr_t(), algorithm aalgorithm = algorithm::deconvolution_direct,
        prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
      utils::validate_dims(strides, padding_l, padding_r);
      mkldnn_deconvolution_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto bias_data = bias_desc.format_any();
      auto dst_data = dst_desc.format_any();
      error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(
              &data, mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
              &src_data, &weights_data, &bias_data, &dst_data, &strides[0], &padding_l[0],
              &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution forward descriptor(bias)");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
              &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
          "could not create a deconvolution forward primitive descriptor(bias)");
      reset(result);
    }

    descriptor(const tdesc_t& src_desc, const tdesc_t& weights_desc, const tdesc_t& dst_desc,
        const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r, const attr_t& attr = attr_t(),
        algorithm aalgorithm = algorithm::deconvolution_direct, prop_kind aprop_kind = prop_kind::forward,
        padding_kind apadding_kind = padding_kind::zero) {
      utils::validate_dims(strides, padding_l, padding_r);
      mkldnn_deconvolution_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto dst_data = dst_desc.format_any();
      error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(
              &data, mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm), &src_data,
              &weights_data, nullptr, &dst_data, &strides[0], &padding_l[0], &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution forward descriptor(no bias)");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
              &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
          "could not create a deconvolution forward primitive descriptor(no bias)");
      reset(result);
    }
  };

 public:
  using attr_t = descriptor::attr_t;
  using computation::expected_dst_descriptor;
  using computation::expected_input_descriptor;
  using computation::expected_weights_descriptor;

  template <typename T, typename... Ts, typename = typename std::enable_if<
          std::is_same<T, tdesc_t>::value>::type>
  void init( const tdesc_t& src_desc, const tdesc_t& weights_desc,
      const tdesc_t& bias, const T& dst, Ts&&... args) {
    descriptor forward_descriptor(src_desc, weights_desc, bias, dst, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, src_desc, weights_desc, bias);
  }

  template <typename T, typename... Ts,
           typename = typename std::enable_if<std::is_same<T, tdims_t>::value>::type>
  void init( const tdesc_t& src_desc, const tdesc_t& weights_desc,
      const tdesc_t& dst, const T something, Ts&&... args) {
    descriptor forward_descriptor(src_desc, weights_desc, dst, something, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, src_desc, weights_desc);
  }

  template <typename T, typename... Ts>
  convolution_transpose_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& weights, const tensor& dst) {
    computation::execute(src, weights, dst);
  }

  void execute(const tensor& src, const tensor& weights, const tensor& bias, const tensor& dst) {
    computation::execute(src, weights, bias, dst);
  }

  template <typename... Ts>
  static void compute_impl(const tensor& src, const tensor& weights, const tensor& bias,
      const tdims_t& dst_dims, tensor& dst, Ts&&... args) {
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), weights.get_dims(), bias.get_dims(),
        dst_dims, args...);

    fetch_or_create_m(comp, key, src.get_descriptor(), weights.get_descriptor(),
        bias.get_descriptor(), tdesc_t{dst_dims, src.get_data_type()}, std::forward<Ts>(args)...);

    auto src_in = src;
    if (src.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_transpose_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    // solve the problem that slow reorder from nchw
    auto _weights = weights.as_weights();
    auto weights_in = _weights;
    if (_weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<convolution_transpose_forward>(comp.expected_weights_descriptor());
      reorder::compute(_weights, weights_in);
    }

    auto dst_desc = comp.expected_dst_descriptor();
    dst.reinit<convolution_transpose_forward>(std::move(dst_desc));
    comp.execute(src_in, weights_in, bias, dst);
  }

  template <typename... Ts>
  static void compute_impl(const tensor& src, const tensor& weights, const tdims_t& dst_dims,
      tensor& dst, Ts&&... args) {
    tdesc_t result_desc(dst_dims, src.get_data_type());
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), weights.get_dims(), dst_dims, args...);

    fetch_or_create_m(comp, key, src.get_descriptor(), weights.get_descriptor(),
        tdesc_t{dst_dims, src.get_data_type()}, std::forward<Ts>(args)...);

    auto src_in = src;
    if (src.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_transpose_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    // solve the problem that slow reorder from nchw
    auto _weights = weights.as_weights();
    auto weights_in = _weights;
    if (_weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<convolution_transpose_forward>(comp.expected_weights_descriptor());
      reorder::compute(_weights.as_weights(), weights_in);
    }

    auto dst_desc = comp.expected_dst_descriptor();
    dst.reinit<convolution_transpose_forward>(std::move(dst_desc));
    comp.execute(src_in, weights_in, dst);
  }

  static void compute(const tensor& src, const tensor& weights, const tdims_t& result_dims,
      tensor& dst, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
      const attr_t& attr = attr_t(), algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward, padding_kind appading_kind = padding_kind::zero) {
    compute_impl(src, weights, result_dims, dst, strides, padding_l, padding_r, attr,
        aalgorithm, aprop_kind, appading_kind);
  }

  static void compute( const tensor& src, const tensor& weights, const tensor& bias, const tdims_t& result_dims,
      tensor& dst, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
      const attr_t& attr = attr_t(), algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward, padding_kind appading_kind = padding_kind::zero) {
    compute_impl(src, weights, bias, result_dims, dst, strides, padding_l, padding_r,
        attr, aalgorithm, aprop_kind, appading_kind);
  }

  static tdesc_t expected_weights_descriptor( const tdims_t& weights_dims,
      tdtype_t dtype = tdtype_t::f32, const tdims_t& strides = {1, 1},
      const tdims_t& padding_l = {0, 0}, const tdims_t& padding_r = {0, 0}, int group = 1) {
    auto dims_in = weights_dims;
    if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(dims_in)) {
      tensor::group_dims(dims_in, group);
    }

    auto ndims = dims_in.size();
    auto grouped = IDEEP_IS_GROUPED_4DIMS(dims_in);
    auto g = grouped ? dims_in[0] : 1;
    auto ic = g * dims_in[1 + grouped];
    auto oc = g * dims_in[0 + grouped];
    auto kh = dims_in[ndims - 2];
    auto kw = dims_in[ndims - 1];
    auto h = 8 * kh;
    auto w = 8 * kw;
    auto oh = (h - 1) * strides[0] + kh - padding_l[0] - padding_r[0];
    auto ow = (w - 1) * strides[1] + kw - padding_l[1] - padding_r[1];

    tdims_t x_dims = {1, ic, h, w};
    tdims_t y_dims = {1, oc, oh, ow};
    auto x_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::u8;
    auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;

    tdesc_t x_desc(x_dims, x_dtype, format::nchw);
    tdesc_t y_desc(y_dims, y_dtype, format::nchw);
    tdesc_t weights_desc(dims_in, dtype, grouped ? format::goihw : format::oihw);

    convolution_transpose_forward comp( x_desc, weights_desc, y_desc, strides, padding_l, padding_r);
    return comp.dup_weights_descriptor();
  }
};

struct convolution_transpose_backward_data : public computation,
      public utils::computation_cache<convolution_transpose_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& grady_desc, const tdesc_t& weights_desc, const tdesc_t& gradx_desc,
        const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::deconvolution_direct, padding_kind apadding_kind = padding_kind::zero)
        : hint_(gradx_desc, weights_desc, grady_desc, strides, padding_l, padding_r) {
      utils::validate_dims(strides, padding_l, padding_r);
      auto diff_src_any = gradx_desc.format_any();
      auto weights_any = weights_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();

      mkldnn_deconvolution_desc_t data;
      error::wrap_c_api(mkldnn_deconvolution_backward_data_desc_init(
              &data, convert_to_c(aalgorithm), &diff_src_any, &weights_any, &diff_dst_any,
              &strides[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward data descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a deconvolution backward data primitive descriptor");
      reset(result);
    }

   private:
    convolution_transpose_forward::descriptor hint_;
  };

 public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template <typename... Ts>
  void init(const tdesc_t& grady_desc, const tdesc_t& weights_desc,
      const tdesc_t& gradx_desc, Ts&&... args) {
    descriptor backward_data_descriptor(
        grady_desc, weights_desc, gradx_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor, grady_desc, weights_desc);
  }

  template <typename T, typename... Ts>
  convolution_transpose_backward_data(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& grady, const tensor& weights, const tensor& gradx) {
    computation::execute(grady, weights, gradx);
  }

  template <typename... Ts>
  static void compute_impl(const tensor& grady, const tensor& weights,
      const tdims_t& gradx_dims, tensor& gradx, Ts&&... args) {
    tdesc_t result_desc(gradx_dims, grady.get_data_type());
    tdesc_t weight_desc;
    tdims_t oihw_dims;
    bool is_iohw = weights.is_iohw_public_layout();
    if (is_iohw) {
      oihw_dims = {weights.get_dim(1), weights.get_dim(0), weights.get_dim(2), weights.get_dim(3)};
      tdesc_t desc(oihw_dims, weights.get_data_type(), format::oihw);
      weight_desc = desc;
    }

    key_t key;
    utils::create_key(key, grady.get_data_type(), grady.get_dims(),
      is_iohw ? oihw_dims : weights.get_dims(), gradx_dims, args...);
    fetch_or_create_m(comp, key, grady.get_descriptor(),
        is_iohw ? weight_desc : weights.get_descriptor(), result_desc, std::forward<Ts>(args)...);

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<convolution_transpose_backward_data>( comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    // solve the problem that slow reorder from nchw
    auto _weights = weights;
    auto weights_in = _weights;
    if (_weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<convolution_transpose_backward_data>(comp.expected_weights_descriptor());
      reorder::compute(_weights, weights_in);
    }

    gradx.reinit<convolution_transpose_backward_data>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }

  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims,
      tensor& gradx, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
      algorithm aalgorithm = algorithm::deconvolution_direct, padding_kind apadding_kind = padding_kind::zero) {
    compute_impl(grady, weights, gradx_dims, gradx, strides, padding_l, padding_r,
        aalgorithm, apadding_kind);
  }
};

struct convolution_transpose_backward_weights
    : public computation,
      public utils::computation_cache<convolution_transpose_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, const tdesc_t& grady_desc, const tdesc_t& gradw_desc,
        const tdesc_t& gradb_desc, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::deconvolution_direct, padding_kind apadding_kind = padding_kind::zero)
        : hint_(x_desc, gradw_desc, gradb_desc, grady_desc, strides, padding_l, padding_r) {
      utils::validate_dims(strides, padding_l, padding_r);
      mkldnn_deconvolution_desc_t data;
      auto src_any = x_desc.format_any();
      auto diff_weights_any = gradw_desc.format_any();
      auto diff_bias_any = gradb_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();

      error::wrap_c_api(mkldnn_deconvolution_backward_weights_desc_init(
              &data, convert_to_c(aalgorithm), &src_any, &diff_weights_any, &diff_bias_any,
              &diff_dst_any, &strides[0], &padding_l[0], &padding_r[0],
              mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api( mkldnn_primitive_desc_create(
              &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a deconvolution backward weights primitive descriptor");
      reset(result);
    }

    descriptor(const tdesc_t& x_desc, const tdesc_t& grady_desc, const tdesc_t& gradw_desc,
        const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::deconvolution_direct, padding_kind apadding_kind = padding_kind::zero)
        : hint_(x_desc, gradw_desc, grady_desc, strides, padding_l, padding_r) {
      utils::validate_dims(strides, padding_l, padding_r);
      mkldnn_deconvolution_desc_t data;
      auto src_any = x_desc.format_any();
      auto diff_weights_any = gradw_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      error::wrap_c_api(mkldnn_deconvolution_backward_weights_desc_init(
              &data, convert_to_c(aalgorithm), &src_any, &diff_weights_any, nullptr, &diff_dst_any,
              &strides[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
              &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a deconvolution backward weights primitive descriptor");
      reset(result);
    }

   private:
    convolution_transpose_forward::descriptor hint_;
  };

 public:
  using computation::expected_gradb_descriptor;
  using computation::expected_gradw_descriptor;

  template <typename... Ts>
  void init(const tdesc_t& x_desc, const tdesc_t& grady_desc,
      const tdesc_t& gradw_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, grady_desc, gradw_desc, std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor, x_desc, grady_desc);
  }

  template <typename T, typename... Ts>
  convolution_transpose_backward_weights(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& grady, const tensor& gradw, const tensor& grad_bias) {
    computation::execute(src, grady, gradw, grad_bias);
  }

  void execute(const tensor& src, const tensor& grady, const tensor& gradw) {
    computation::execute(src, grady, gradw);
  }

  /*
   * This interface require MKL-DNN fixed
   * https://github.com/intel/mkl-dnn/commit/86f152b614c947b87633062a182c57775856a348
   */
  template <typename... Ts>
  static void compute_impl(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gbias, Ts&&... args) {
    tdesc_t gradw_desc(gradw_dims, src.get_data_type());
    tdesc_t gradb_desc(tdims_t{grady.get_dim(1)}, src.get_data_type());

    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), grady.get_dims(), gradw_dims,
        grady.get_dim(1), args...);
    fetch_or_create_m(comp, key, src.get_descriptor(), grady.get_descriptor(), gradw_desc,
        gradb_desc, std::forward<Ts>(args)...);

    auto src_in = src;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_transpose_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<convolution_transpose_backward_weights>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<convolution_transpose_backward_weights>(comp.expected_gradw_descriptor());
    gbias.reinit<convolution_transpose_backward_weights>(comp.expected_gradb_descriptor());
    comp.execute(src_in, grady_in, gradw, gbias);
  }

  template <typename... Ts>
  static void compute_impl(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, Ts&&... args) {
    tdesc_t gradw_desc(gradw_dims, src.get_data_type());

    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), grady.get_dims(), gradw_dims, args...);
    fetch_or_create_m(comp, key, src.get_descriptor(), grady.get_descriptor(), gradw_desc,
        std::forward<Ts>(args)...);

    auto src_in = src;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<convolution_transpose_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<convolution_transpose_backward_weights>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<convolution_transpose_backward_weights>(comp.expected_gradw_descriptor());
    comp.execute(src_in, grady_in, gradw);
  }

  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, const tdims_t& strides, const tdims_t& padding_l,
      const tdims_t& padding_r, algorithm aalgorithm = algorithm::deconvolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    compute_impl(src, grady, gradw_dims, gradw, strides, padding_l, padding_r,
        aalgorithm, apadding_kind);
  }

  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, const tdims_t& strides, const tdims_t& padding_l,
      const tdims_t& padding_r, algorithm aalgorithm = algorithm::deconvolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    compute_impl(src, grady, gradw_dims, gradw, gradb, strides, padding_l, padding_r,
        aalgorithm, apadding_kind);
  }
};

struct lrn_forward : public computation,
  public utils::computation_cache<lrn_forward> {
  struct descriptor : public descriptor_group {
    descriptor (const tdesc_t &x_desc, int local_size, float alpha, float beta, float k = 1.0,
        algorithm aalgorithm = algorithm::lrn_across_channels, prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_lrn_desc_t data;
      auto src_data = x_desc.get_mkldnn_memory_desc_t();
      error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data, mkldnn::convert_to_c(aprop_kind),
            convert_to_c(aalgorithm), src_data, local_size, alpha, beta, k),
          "could not create a lrn forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a lrn forward primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::expected_dst_descriptor;
  using computation::expected_workspace_descriptor;
  using computation::expected_src_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &x_desc, Ts&&... args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  template<typename T, typename ...Ts>
  lrn_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor &src, const tensor& dst, const tensor& workspace) {
    computation::execute(src, dst, workspace);
  }

  void execute(const tensor &src, tensor& dst) {
    if (dst.has_extra())
      computation::execute(src, dst, *dst.get_extra());
    else
      computation::execute(src, dst);
  }

  static void compute(key_t &key, const tensor& src, tensor& dst, int local_size, float alpha,
      float beta, float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels,
      prop_kind aprop_kind = prop_kind::forward_training) {

    auto src_in = src;
    tdesc_t src_desc;
    scale_t src_scales(IDEEP_DEF_SCALE);
    if (src_in.has_scale()) {
      IDEEP_ENFORCE(src_in.get_data_type() != tdtype_t::f32, "Incorrect data type");
      IDEEP_ENFORCE(src_in.get_scale().size() == 1, "Invalid scale size");
      src_desc = {src_in.get_dims(), tdtype_t::f32};
      src_scales[0] /= src_in.get_scale()[0];
    } else {
      src_desc = src_in.get_descriptor();
      IDEEP_ENFORCE(src_in.get_data_type() == tdtype_t::f32, "Incorrect src data type");
    }

    if (key.empty())
      utils::create_key(key, src_desc.get_data_type(), src_desc.get_dims(),
          src_desc.get_internal_format(), local_size, alpha, beta, k, aalgorithm, aprop_kind);

    fetch_or_create_m(comp, key, src_desc, local_size, alpha, beta, k, aalgorithm, aprop_kind);

    bool with_workspace = aprop_kind == prop_kind::forward_training;

    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<lrn_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in, {0, src_scales});
    }

    if (dst != src) {
      dst.reinit<lrn_forward>(comp.expected_dst_descriptor());
      if (with_workspace)
        dst.init_extra<lrn_forward>(comp.expected_workspace_descriptor());
    }

    comp.execute(src_in, dst);
  }

  static void compute(const tensor& src, tensor& dst, int local_size, float alpha, float beta,
      float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels,
      prop_kind aprop_kind = prop_kind::forward_training) {
    key_t key;
    compute(key, src, dst, local_size, alpha, beta, k, aalgorithm, aprop_kind);
  }
};

struct lrn_backward : public computation,
  public utils::computation_cache<lrn_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &x_desc, const tdesc_t &gx_desc, int local_size, float alpha,
        float beta, float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels)
      : hint_(x_desc, local_size, alpha, beta, k, aalgorithm) {
      mkldnn_lrn_desc_t data;
      error::wrap_c_api(mkldnn_lrn_backward_desc_init(
            &data, convert_to_c(aalgorithm), gx_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), local_size, alpha, beta, k),
          "could not create a lrn backward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a backward lrn primitive descriptor");
      reset(result);
    }

  private:
    lrn_forward::descriptor hint_;
  };
public:
  using computation::expected_gradx_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &x_desc,
      const tdesc_t &grady_desc, Ts&&... args) {
    descriptor backward_data_descriptor(x_desc, grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor, x_desc, grady_desc);
  }

  template<typename T, typename ...Ts>
  lrn_backward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& y, const tensor& gradx) {
    if (num_of_inputs() == 2)
      computation::execute(x, grady, gradx);
    else
      computation::execute(x, grady, *y.get_extra(), gradx);
  }

  static void compute(const tensor& x, const tensor& grady, const tensor& y, tensor& gradx,
      int local_size, float alpha, float beta, float k = 1.0,
      algorithm aalgorithm = algorithm::lrn_across_channels) {
    key_t key;
    utils::create_key(key, x.get_data_type(), x.get_dims(),
        x.get_internal_format(), local_size, alpha, beta, k, aalgorithm);

    fetch_or_create_m(comp, key, x.get_descriptor(),
        grady.get_descriptor(), local_size, alpha, beta, k, aalgorithm);

    gradx.reinit<lrn_backward>(comp.expected_gradx_descriptor());
    comp.execute(x, grady, y, gradx);
  }
};

struct pooling_forward : public computation,
  public utils::computation_cache<pooling_forward> {
  struct descriptor : descriptor_group {
    descriptor() = default;
    descriptor(const tdesc_t &x_desc, const tdesc_t &y_desc, const tdims_t& strides,
        const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r, algorithm aalgorithm,
        prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
      utils::validate_dims(strides, kernel, padding_l, padding_r);
      auto src_data = x_desc.get_mkldnn_memory_desc_t();
      auto dst_data = y_desc.format_any();
      mkldnn_pooling_desc_t data;
      error::wrap_c_api(mkldnn_pooling_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm), src_data, &dst_data,
            &strides[0], &kernel[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not init a forward pooling descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a forward pooling primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::expected_dst_descriptor;
  using computation::expected_workspace_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  template<typename T, typename ...Ts>
  pooling_forward(T arg, Ts &&...args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor &src, const tensor &dst, const tensor &workspace) {
    computation::execute(src, dst, workspace);
  }

  void execute(const tensor &src, tensor &dst) {
    if (dst.has_extra())
      computation::execute(src, dst, *dst.get_extra());
    else
      computation::execute(src, dst);
  }

  static void compute(key_t &key, const tensor& src, const tdims_t& dst_dims, tensor& dst,
      const tdims_t& strides, const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r,
      algorithm aalgorithm, prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
    if (key.empty())
      utils::create_key(key, src.get_data_type(), src.get_dims(),
          src.get_internal_format(), dst_dims, strides, kernel, padding_l,
          padding_r, aalgorithm, aprop_kind, apadding_kind);

    tdesc_t dst_desc(dst_dims, src.get_data_type());
    fetch_or_create_m(comp, key, src.get_descriptor(), dst_desc, strides, kernel, padding_l,
        padding_r, aalgorithm, aprop_kind, apadding_kind);

    bool with_workspace = true && aprop_kind == prop_kind::forward_training
        && aalgorithm == mkldnn::pooling_max;

    if (dst != src) {
      dst.reinit<pooling_forward>(comp.expected_dst_descriptor());
      if (with_workspace)
        dst.init_extra<pooling_forward>(comp.expected_workspace_descriptor());
      if (src.has_scale()) {
        dst.set_scale(src.get_scale());
      }
    }

    comp.execute(src, dst);
  }

  static void compute(const tensor& src, const tdims_t& dst_dims, tensor& dst, const tdims_t& strides,
      const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r, algorithm aalgorithm,
      prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
    key_t key;
    compute(key, src, dst_dims, dst, strides, kernel,
        padding_l, padding_r, aalgorithm, aprop_kind, apadding_kind);
  }
};

struct pooling_backward : public computation,
  public utils::computation_cache<pooling_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &gradx_desc, const tdesc_t &grady_desc, const tdims_t& strides,
        const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm, padding_kind apadding_kind = padding_kind::zero)
      : hint_([&]() {
              utils::validate_dims(strides, kernel, padding_l, padding_r);
              auto gradx_data = gradx_desc.get_mkldnn_memory_desc_t();
              auto grady_data = grady_desc.format_any();
              mkldnn_pooling_desc_t data;
              error::wrap_c_api(mkldnn_pooling_forward_desc_init(
                    &data, mkldnn::convert_to_c(prop_kind::forward), convert_to_c(aalgorithm),
                    gradx_data, &grady_data, &strides[0], &kernel[0], &padding_l[0], &padding_r[0],
                    mkldnn::convert_to_c(apadding_kind)),
                  "could not init a forward pooling descriptor");
              mkldnn_primitive_desc_t result;
              error::wrap_c_api(mkldnn_primitive_desc_create(
                    &result, &data, engine::cpu_engine().get(), nullptr),
                  "could not create a forward pooling primitive descriptor");

              pooling_forward::descriptor hint;
              hint.reset(result);
              return hint;
            } ()) {
      utils::validate_dims(strides, kernel, padding_l, padding_r);
      auto gradx_data = gradx_desc.format_any();
      mkldnn_pooling_desc_t data;
      error::wrap_c_api(mkldnn_pooling_backward_desc_init(
            &data, convert_to_c(aalgorithm), &gradx_data, grady_desc.get_mkldnn_memory_desc_t(),
            &strides[0], &kernel[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not init a backward pooling descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a backward pooling primitive descriptor");
      reset(result);
    }
  private:
    pooling_forward::descriptor hint_;
  };
public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &gradx_desc,
      const tdesc_t &grady_desc, Ts &&...args) {
    descriptor backward_descriptor(gradx_desc, grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor, grady_desc, gradx_desc);
  }

  template<typename T, typename ...Ts>
  pooling_backward(T arg, Ts &&...args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& grady, const tensor& y, const tensor& gradx) {
    if (num_of_inputs() == 1)
      computation::execute(grady, gradx);
    else
      computation::execute(grady, *y.get_extra(), gradx);
  }

  static void compute(const tensor& grady, const tensor& y, const tensor& x, tensor& gradx,
      const tdims_t& strides, const tdims_t& kernel, const tdims_t& padding_l,
      const tdims_t& padding_r, algorithm aalgorithm, padding_kind apadding_kind = padding_kind::zero) {
    auto grady_in = grady;
    if (grady.get_internal_format() != x.get_internal_format()) {
      grady_in.init<pooling_backward>({grady.get_dims(),grady.get_data_type(), x.get_internal_format()});
      reorder::compute(grady, grady_in);
    }

    key_t key;
    utils::create_key(key, grady_in.get_data_type(), grady_in.get_dims(),
        grady_in.get_internal_format(), x.get_dims(), strides, kernel, padding_l,
        padding_r, aalgorithm, apadding_kind);

    fetch_or_create_m(comp, key, x.get_descriptor(), grady_in.get_descriptor(),
        strides, kernel, padding_l, padding_r, aalgorithm, apadding_kind);

    gradx.reinit<pooling_backward>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, y, gradx);
  }
};

struct eltwise_forward : public computation,
  public utils::computation_cache<eltwise_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &x_desc, float alpha = 0.0, float beta = 0.0,
        algorithm alg_kind = algorithm::eltwise_relu, prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_eltwise_desc_t data;
      error::wrap_c_api(mkldnn_eltwise_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), mkldnn::convert_to_c(alg_kind),
            x_desc.get_mkldnn_memory_desc_t(), alpha, beta),
          "could not create a eltwise forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a eltwise forward primitive descriptor");
      reset(result);
    }
  };

public:
  using computation::computation;
  using computation::expected_dst_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  template<typename T, typename ...Ts>
  eltwise_forward(T arg, Ts &&...args) {
    init(std::forward<T>(arg), std::forward<Ts>(args)...);
  }

  void execute(const tensor &x, const tensor &y) {
    computation::execute(x, y);
  }

  static void compute(key_t &key, const tensor& src, tensor& dst,
      algorithm aalgorithm = algorithm::eltwise_relu, prop_kind aprop_kind = prop_kind::forward,
      float alpha = 0.0, float beta = 0.0) {
    auto src_in = src;
    if (aalgorithm != algorithm::eltwise_relu
        && src.get_data_type() != tdtype_t::f32) {
      src_in.init<eltwise_forward>({src.get_dims(), tdtype_t::f32});
      IDEEP_ENFORCE(src.has_scale(), "Can not find scales");
      IDEEP_ENFORCE(src.get_scale().size() == 1, "Incorrect scale size");
      auto scale = IDEEP_DEF_SCALE;
      scale[0] /= src.get_scale()[0];
      reorder::compute(src, src_in, {0, scale});
    }

    if (key.empty())
      utils::create_key(key, src_in.get_data_type(), src_in.get_dims(),
          src_in.get_internal_format(), alpha, beta, aalgorithm, aprop_kind);

    fetch_or_create_m(comp, key, src_in.get_descriptor(),
        alpha, beta, aalgorithm, aprop_kind);

    if (dst != src) {
      dst.reinit<eltwise_forward>(src_in.get_descriptor());
      if (src_in.has_scale()) dst.set_scale(src_in.get_scale());
    }

    comp.execute(src_in, dst);
    if (dst.has_scale() && aalgorithm == algorithm::eltwise_relu
        && dst.get_data_type() == tdtype_t::s8)
      dst.set_descriptor({dst.get_dims(), tdtype_t::u8, dst.get_internal_format()});
  }

  static void compute(const tensor& src, tensor& dst, algorithm aalgorithm = algorithm::eltwise_relu,
      prop_kind aprop_kind = prop_kind::forward, float alpha = 0.0, float beta = 0.0) {
    key_t key;
    compute(key, src, dst, aalgorithm, aprop_kind, alpha, beta);
  }
};

struct eltwise_backward : public computation,
  public utils::computation_cache<eltwise_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &grady_desc, const tdesc_t &x_desc,
        float alpha = 0.0, float beta = 0.0, algorithm alg_kind = algorithm::eltwise_relu)
      : hint_(x_desc, alg_kind) {
      mkldnn_eltwise_desc_t data;
      error::wrap_c_api(mkldnn_eltwise_backward_desc_init(
            &data, mkldnn::convert_to_c(alg_kind), grady_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), static_cast<float>(alpha), static_cast<float>(beta)),
          "could not create a eltwise backward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a eltwise backward primitive descriptor");
      reset(result);
    }
  private:
    eltwise_forward::descriptor hint_;
  };
public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &grady_desc, const tdesc_t &x_desc, Ts &&...args) {
    descriptor backward_descriptor(grady_desc, x_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor, grady_desc, x_desc);
  }

  template<typename T, typename ...Ts>
  eltwise_backward(T grady_desc, T src_desc, Ts &&...args) {
    init(std::forward<T>(grady_desc), std::forward<T>(src_desc), std::forward<Ts>(args)...);
  }

  void execute(const tensor &x, const tensor &grady, const tensor &gradx) {
    computation::execute(x, grady, gradx);
  }

  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  static void compute(const tensor& src, const tensor& grady, tensor& gradx,
      algorithm aalgorithm = algorithm::eltwise_relu, float alpha = 0.0, float beta = 0.0) {
    // if grady is from outside, make it ours
    tensor grady_in = grady;
    if (grady.get_internal_format() != src.get_internal_format()) {
      grady_in.init<eltwise_backward>(src.get_descriptor());
      reorder::compute(grady, grady_in);
      if (grady == gradx) {
        gradx.set_descriptor(grady_in.get_descriptor());
      }
    }

    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(),
        alpha, beta, aalgorithm);

    fetch_or_create_m(comp, key, grady_in.get_descriptor(),
        src.get_descriptor(), alpha, beta, aalgorithm);

    if (grady != gradx)
      gradx.reinit<eltwise_backward>(comp.expected_gradx_descriptor());

    comp.execute(src, grady_in, gradx);
  }
};

struct channel_shuffle_forward: public computation,
  public utils::computation_cache<channel_shuffle_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &src_desc, const int group_size, const int axis = 1,
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_shuffle_desc_t data;
      error::wrap_c_api(mkldnn_shuffle_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind),
            src_desc.get_mkldnn_memory_desc_t(), axis, group_size),
          "could not create a shuffle forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a shuffle forward primitive descriptor");
      reset(result);
    }
  };
public:
  template<typename ...Ts>
  void init(const tdesc_t &x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  template<typename T, typename ...Ts>
  channel_shuffle_forward(T arg, Ts &&...args) {
    init(std::forward<T>(arg), std::forward<Ts>(args)...);
  }

  void execute(const tensor &x, const tensor &y) {
    computation::execute(x, y);
  }

  static void compute(const tensor& src, tensor& dst, const int group,
      const int axis = 1, prop_kind aprop_kind = prop_kind::forward) {
    IDEEP_ENFORCE(src.get_dim(axis) % group == 0, "Invalid channel and group");
    IDEEP_ENFORCE(src.get_data_type() == tdtype_t::f32, "invalid data type");

    auto group_size = src.get_dim(axis) / group;
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(),
        src.get_internal_format(), group_size, axis, aprop_kind);
    fetch_or_create_m(comp, key, src.get_descriptor(), group_size, axis, aprop_kind);

    auto src_in = src;
    if (src.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<channel_shuffle_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    if (dst != src) {
      dst.reinit<channel_shuffle_forward>(comp.expected_dst_descriptor());
    }

    comp.execute(src_in, dst);
  }
};

struct channel_shuffle_backward : public computation,
  public utils::computation_cache<channel_shuffle_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &grady_desc, const int group_size, const int axis = 1) {
      mkldnn_shuffle_desc_t data;
      error::wrap_c_api(mkldnn_shuffle_backward_desc_init(
            &data, grady_desc.get_mkldnn_memory_desc_t(), static_cast<int>(axis),
            static_cast<int>(group_size)),
          "could not create a shuffle backward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a shuffle backward primitive descriptor");
      reset(result);
    }
  };
public:
  template<typename ...Ts>
  void init(const tdesc_t &grady_desc, Ts &&...args) {
    descriptor backward_descriptor( grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor, grady_desc);
  }

  template<typename T, typename ...Ts>
  channel_shuffle_backward(T grady_desc, Ts &&...args) {
    init(std::forward<T>(grady_desc), std::forward<Ts>(args)...);
  }

  void execute(const tensor &grady, const tensor &gradx) {
    computation::execute(grady, gradx);
  }

  static void compute(const tensor& grady, tensor& gradx, const int group, const int axis = 1) {
    auto group_size = grady.get_dim(axis) / group;
    key_t key;
    utils::create_key(key, grady.get_data_type(), grady.get_dims(),
        grady.get_internal_format(), group_size, axis);
    fetch_or_create_m(comp, key, grady.get_descriptor(), group_size, axis);

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<channel_shuffle_backward>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    if (gradx != grady)
      gradx.reinit<channel_shuffle_backward>(comp.expected_gradx_descriptor());

    comp.execute(grady_in, gradx);
  }
};

struct concat : public computation,
  public utils::computation_cache<concat> {
  struct descriptor : public descriptor_group {
    descriptor(int concat_dimension, const std::vector<tdesc_t> &inputs) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_concat_primitive_desc_create(
            &result, nullptr, (int)c_api_inputs.size(), concat_dimension, &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }

    descriptor(int concat_dimension, const std::vector<tdesc_t> &inputs, const tdesc_t out_desc) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_concat_primitive_desc_create(
            &result, out_desc.get_mkldnn_memory_desc_t(), (int)c_api_inputs.size(),
            concat_dimension, &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::execute;
  using computation::expected_dst_descriptor;

  void init(int concat_dimension, const std::vector<tdesc_t> &inputs) {
    descriptor forward_descriptor (concat_dimension, inputs);
    computation::init(forward_descriptor, inputs);
  }

  concat(int concat_dimension, const std::vector<tdesc_t> &inputs) {
    init(concat_dimension, inputs);
  }

  void execute(const std::vector<tensor> &inputs, const tensor &output) {
    computation::execute(inputs, output);
  }

  static void compute(key_t &key, std::vector<tensor>& inputs, int axis, tensor& output) {
    std::vector<tdesc_t> tdesc;
    std::vector<tdtype_t> inputs_dt;
    std::vector<tdims_t> inputs_dims;
    std::vector<format> inputs_format;
    for (tensor elems : inputs) {
      tdesc.push_back(elems.get_descriptor());
      inputs_dt.push_back(elems.get_data_type());
      inputs_dims.push_back(elems.get_dims());
      inputs_format.push_back(elems.get_internal_format());
    }

    if (key.empty())
      utils::create_key(key, inputs_dt, inputs_dims, inputs_format, axis);

    // FIXME
    // currently align all inputs format with first one
    std::vector<tensor> inputs_in;
    inputs_in.push_back(inputs[0]);
    for (int i = 1; i < tdesc.size(); i++) {
      auto src_in = inputs[i];
      if (inputs_format[i] != inputs_format[0]) {
        src_in.init<concat>({inputs_dims[i], inputs_dt[i], inputs_format[0]});
        reorder::compute(inputs[i], src_in);
      }
      inputs_in.push_back(src_in);
      tdesc[i] = src_in.get_descriptor();
    }

    fetch_or_create_m(comp, key, axis, tdesc);
    output.reinit<concat>(comp.expected_dst_descriptor());

    comp.execute(inputs_in, output);
  }

  static void compute(std::vector<tensor>& inputs, int axis, tensor& output) {
    key_t key;
    compute(key, inputs, axis, output);
  }

  static std::vector<int32_t> compute( std::vector<tensor>& inputs, int axis, bool add_axis, tensor& dst) {
    IDEEP_ENFORCE(axis < (inputs[0].ndims() + (add_axis ? 1 : 0)), "invalid axis in concat");
    for (int i = 0; i < inputs[0].ndims(); i++) {
      if (i == axis && !add_axis) continue;
      for (unsigned j = 1; j <inputs.size(); j++) {
        IDEEP_ENFORCE(inputs[j].get_dim(i) == inputs[0].get_dim(i), "invalid input dims in concat");
      }
    }

    int32_t dst_channels = 0;
    std::vector<int32_t> axis_info(inputs.size(), 0);
    for (unsigned k = 0; k <inputs.size(); k++) {
      axis_info[k] = add_axis ? 1 : inputs[k].get_dim(axis);
      dst_channels += axis_info[k];
    }

    tdims_t dst_dims(inputs[0].get_dims());
    if (add_axis)
      dst_dims.insert(dst_dims.begin() + axis, dst_channels);
    else
      dst_dims[axis] = dst_channels;

    auto dst_data_type = inputs[0].get_data_type();
    auto dst_format = inputs[0].get_internal_format();
    scale_t min_scale(IDEEP_DEF_SCALE);
    if (dst_data_type != tdtype_t::f32) {
      min_scale[0] = std::numeric_limits<float>::max();
      for (auto i : inputs) {
        if (i.get_data_type() != dst_data_type) {
          min_scale = IDEEP_DEF_SCALE;
          dst_data_type = tdtype_t::f32;
          break;
        }
        if (i.has_scale() && (min_scale[0] > i.get_scale()[0])) {
          IDEEP_ENFORCE(i.get_scale().size() == 1, "incorrect scale size");
          min_scale[0] = i.get_scale()[0];
        }
      }
    }

    tdims_t offset_dims(dst_dims.size(), 0);
    if (add_axis)
      dst.reinit({dst_dims, dst_data_type});
    else
      dst.reinit({dst_dims, dst_data_type, dst_format});
    if (dst_data_type != tdtype_t::f32)
      dst.set_scale(min_scale);

    reorder reorder_;
    scale_t scales(1);
    // FIXME: To avoid view issue in mkldnn
    // NOTE: In mkldnn concat, dim 3 and 6+ are not supported.
    // Morewhile, the tensor shape must be blockable to create a view.
    if (!add_axis && dst_dims.size() != 3 && dst_dims.size() < 6) {
      for (unsigned k = 0; k < inputs.size(); k++) {
        if (!inputs[k].is_limited_blockable()) {
          for (int i = 0; i < inputs.size(); ++i) {
            float input_scale = inputs[i].has_scale() ? inputs[i].get_scale()[0] : 1.0f;
            if (inputs[i].get_data_type() != dst_data_type || input_scale - min_scale[0] != 0) {
              scales[0] = min_scale[0] / input_scale;
              tensor input_fp = inputs[i];
              input_fp.reinit({inputs[i].get_dims(), dst_data_type, inputs[i].get_internal_format()});
              reorder_.init(inputs[i].get_descriptor(), input_fp.get_descriptor(), {0, scales});
              reorder_(inputs[i], input_fp);
              inputs[i] = input_fp;
            }
          }
          compute(inputs, axis, dst);
          return axis_info;
        }
      }
    }

    for (unsigned i = 0; i < inputs.size(); ++i) {
      scales[0] = min_scale[0] / (inputs[i].has_scale() ? inputs[i].get_scale()[0] : 1.0f);
      if (add_axis) {
        tdims_t in_dims(inputs[i].get_dims());
        in_dims.insert(in_dims.begin() + axis, 1);
        tdesc_t in_desc(inputs[i].get_descriptor().reshape(in_dims));
        auto view = dst.create_view(in_dims, offset_dims);
        reorder_.init(in_desc, view, dst.get_descriptor(), {0, scales});
        reorder_({in_desc, inputs[i].get_data_handle()}, dst);
      } else {
        auto view = dst.create_view(inputs[i].get_dims(), offset_dims);
        reorder_.init(inputs[i].get_descriptor(), view, dst.get_descriptor(), {0, scales});
        reorder_(inputs[i], dst);
      }
      offset_dims[axis] += axis_info[i];
    }

    return axis_info;
  }
};

struct softmax_forward : public computation {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &x_desc, int softmax_axis, prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_softmax_desc_t data;
      error::wrap_c_api(mkldnn_softmax_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), x_desc.get_mkldnn_memory_desc_t(), softmax_axis),
          "could not create a softmax forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
              &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a softmax forward primitive descriptor");
      reset(result);
    }
  };

public:
  using computation::expected_dst_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t& src_desc, const tdesc_t& dst_desc, Ts&&... args) {
    descriptor softmax_descriptor(src_desc, std::forward<Ts>(args)...);
    computation::init(softmax_descriptor, src_desc, dst_desc);
  }

  void execute(const tensor& src, const tensor& dst) {
    computation::execute(src, dst);
  }
};

struct batch_norm_forward_base : public computation {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &src_desc, float epsilon, unsigned flags, prop_kind aprop_kind) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(mkldnn_batch_normalization_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(), epsilon, flags),
          "could not create a batch normalization forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a batch normalization forward primitive descriptor");
      reset(result);
    }

    descriptor(const tdesc_t &src_desc, float epsilon, attr_t attr, unsigned flags, prop_kind aprop_kind) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(mkldnn_batch_normalization_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(), epsilon, flags),
          "could not create a batch normalization forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
          &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
      "could not create a batch normalization forward primitive descriptor");
      reset(result);
    }
  };

public:
  using computation::expected_dst_descriptor;

  template<typename... Ts>
  void init(float epsilon, unsigned flags, prop_kind aprop_kind, const tdesc_t &src_desc, Ts&... rest) {
    descriptor batch_norm_forward(src_desc, epsilon, flags, aprop_kind);
    init(batch_norm_forward, src_desc, rest...);
  }

  /// Execute interface for (1, 0) (stats_is_src, use_scaleshift)
  void execute(const tensor& src, const tensor& mean, const tensor& variance, const tensor& dst) {
    computation::execute(src, mean, variance, dst);
  }

  /// Execute interface for (1, 1)
  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& weights, const tensor& dst) {
    computation::execute(src, mean, variance, weights, dst);
  }
};

struct batch_normalization_forward_inference : public batch_norm_forward_base,
  public utils::computation_cache<batch_normalization_forward_inference> {
public:
  using batch_norm_forward_base::execute;

  /// Execute interface for  (0, 0)
  void execute(const tensor& src, const tensor& dst) {
    computation::execute(src, dst);
  }

  /// Execute interface for  (0, 1)
  void execute(const tensor& src, const tensor& weights, const tensor& dst) {
    computation::execute(src, weights, dst);
  }

public:

  void init(const tdesc_t& src_desc, float epsilon,
      unsigned flag = batch_normalization_flag::use_global_stats | batch_normalization_flag::use_scale_shift) {
    descriptor batch_norm_forward(src_desc, epsilon, flag, prop_kind::forward_scoring);
    weights_.init(batch_norm_forward.expected_weights_descriptor());
    computation::init(batch_norm_forward);
  }

  template<typename T, typename ...Ts>
  batch_normalization_forward_inference(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  /// More functionality in this interface
  void execute(const tensor& src, const tensor& scale, const tensor& shift, const tensor& dst) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    std::memcpy((char *)weights_.get_data_handle() + scale.get_size(), shift.get_data_handle(), shift.get_size());
    computation::execute(src, weights_, dst);
  }

  /// More functionality in this interface
  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, const tensor& dst) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    std::memcpy((char *)weights_.get_data_handle() + scale.get_size(), shift.get_data_handle(), shift.get_size());
    computation::execute(src, mean, variance, weights_, dst);
  }

  using computation::expected_dst_descriptor;

  // Inplace support?
  static void compute(key_t &key, const tensor& src, const tensor& scale,
      const tensor& shift, tensor& dst, float epsilon) {
    auto src_in = src;
    if (src.get_data_type() != tdtype_t::f32) {
      src_in.init<batch_normalization_forward_inference>( {src.get_dims(), tdtype_t::f32});
      IDEEP_ENFORCE(src.has_scale(), "Can not find scales");
      auto src_scales = IDEEP_DEF_SCALE;
      src_scales[0] /= src.get_scale()[0];
      reorder::compute(src, src_in, {0, src_scales});
    }

    if (key.empty())
      utils::create_key(key, src_in.get_data_type(), src_in.get_dims(),
          src_in.get_internal_format(), 3, epsilon);

    fetch_or_create_m(comp, key, src_in.get_descriptor(),
        batch_normalization_flag::use_scale_shift, epsilon);

    if (dst != src)
      dst.reinit<batch_normalization_forward_inference>(comp.expected_dst_descriptor());
    comp.execute(src_in, scale, shift, dst);
  }

  static void compute(key_t &key, const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, tensor& dst, float epsilon) {
    auto src_in = src;
    if (src.get_data_type() != tdtype_t::f32) {
      src_in.init<batch_normalization_forward_inference>( {src.get_dims(), tdtype_t::f32});
      IDEEP_ENFORCE(src.has_scale(), "Can not find scales");
      auto src_scales = IDEEP_DEF_SCALE;
      src_scales[0] /= src.get_scale()[0];
      reorder::compute(src, src_in, {0, src_scales});
    }

    if (key.empty())
      utils::create_key(key, src_in.get_data_type(), src_in.get_dims(),
          src_in.get_internal_format(), 5, epsilon);

    fetch_or_create_m(comp, key, src_in.get_descriptor(), epsilon);

    if (dst != src) {
      dst.reinit<batch_normalization_forward_inference>(comp.expected_dst_descriptor());
    }
    comp.execute(src_in, mean, variance, scale, shift, dst);
  }

  static void compute(const tensor& src, const tensor& scale,
      const tensor& shift, tensor& dst, float epsilon) {
    key_t key;
    compute(key, src, scale, shift, dst, epsilon);
  }

  static void compute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, tensor& dst, float epsilon) {
    key_t key;
    compute(key, src, mean, variance, scale, shift, dst, epsilon);
  }

private:
  tensor weights_;
};

struct batch_normalization_forward_training : public batch_norm_forward_base,
  public utils::computation_cache<batch_normalization_forward_training> {
  float get_epsilon() const {
    const mkldnn_batch_normalization_desc_t *p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d), 0, (void *)&p_desc),
      "could not query batch normalization descriptor");
    return p_desc->batch_norm_epsilon;
  }
public:
  using computation::expected_dst_descriptor;
  using batch_norm_forward_base::execute;

  void init(const tdesc_t& src_desc, const tdesc_t& scale, const tdesc_t& shift, float momentum,
      float epsilon, unsigned flags = batch_normalization_flag::use_scale_shift) {
    // IDEEP_ENFORCE(scale.ndims() == 1 && shift.ndims() == 1, "Incorrect dims");
    descriptor batch_norm_forward(src_desc, epsilon, flags, prop_kind::forward_training);
    computation::init(batch_norm_forward, src_desc);

    // We borrown scale and bias for the shape of mean and variance
    weights_.init(batch_norm_forward.expected_weights_descriptor());
    sum_.init({momentum, 1.f - momentum}, {scale, shift});
  }

  template<typename T, typename... Ts>
  batch_normalization_forward_training (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  /// Execute interface for (0, 0)
  void execute(const tensor& src, const tensor& dst, const tensor& mean, const tensor& variance) {
    computation::execute(src, dst, mean, variance);
  }

  /// Execute interface for (0, 1)
  void execute(const tensor& src, const tensor& weights, const tensor& dst,
      const tensor& mean, const tensor& variance) {
    computation::execute(src, weights, dst, mean, variance);
  }

  void execute(const tensor& src, const tensor& scale, const tensor& shift,
      const tensor& dst, const tensor& mean, const tensor& variance) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    std::memcpy((char *)weights_.get_data_handle() + scale.get_size(),
        shift.get_data_handle(), shift.get_size());
    computation::execute(src, weights_, dst, mean, variance);
  }

  void running_statistic(const tensor& mean, const tensor& variance,
      const tensor& running_mean, const tensor& running_var) {
    // TODO: provide accelerated version
    std::vector<tensor> inputs_for_mean {running_mean, mean};
    std::vector<tensor> inputs_for_var {running_var, variance};
    sum_.execute(inputs_for_mean, running_mean);
    sum_.execute(inputs_for_var, running_var);
  }

  // TODO: deprecates these two
  tdesc_t expected_mean_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 1);
  }

  tdesc_t expected_variance_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 2);
  }

  // TODO: this is good one
  tdesc_t expected_statistic_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 1);
  }

  static void compute(const tensor& src, const tensor& scale, const tensor& shift,
      tensor& dst, tensor& mean, tensor& variance, float momentum, float epsilon) {
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(), epsilon);

    fetch_or_create_m(comp, key, src.get_descriptor(),
        scale.get_descriptor(), shift.get_descriptor(), momentum, epsilon);
    comp.eps = epsilon;

    dst.reinit<batch_normalization_forward_training>(comp.expected_dst_descriptor());
    mean.reinit(comp.expected_statistic_descriptor());
    variance.reinit(comp.expected_statistic_descriptor());

    comp.execute(src, scale, shift, dst, mean, variance);
  }

  static void compute(const tensor& src, const tensor& scale, const tensor& shift, tensor& dst,
      tensor& mean, tensor& variance, tensor& running_mean, tensor& running_var, float momentum, float epsilon) {
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(), epsilon);

    fetch_or_create_m(comp, key, src.get_descriptor(),
        scale.get_descriptor(), shift.get_descriptor(), momentum, epsilon);

    // TODO: Substitue running statistics calculation with lighter version
    dst.reinit<batch_normalization_forward_training>(comp.expected_dst_descriptor());
    mean.reinit(comp.expected_statistic_descriptor());
    variance.reinit(comp.expected_statistic_descriptor());
    if (running_mean.get_descriptor() != comp.expected_statistic_descriptor()){
      running_mean.reinit(comp.expected_statistic_descriptor());
      std::memset(running_mean.get_data_handle(), 0, running_mean.get_size());
    }
    if (running_var.get_descriptor() != comp.expected_statistic_descriptor()){
      running_var.reinit(comp.expected_statistic_descriptor());
      auto p = static_cast<float *>(running_var.get_data_handle());
      std::fill_n(p, running_var.get_nelems(), 1);
    }

    comp.execute(src, scale, shift, dst, mean, variance);
    comp.running_statistic(mean, variance, running_mean, running_var);
  }

private:
  tensor weights_;
  sum sum_;
  float eps;
};

struct batch_normalization_backward : public computation,
  public utils::computation_cache<batch_normalization_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &gradx_desc, const tdesc_t &x_desc,
        float epsilon, unsigned flags, prop_kind aprop_kind)
      : hint_(x_desc, epsilon, flags, prop_kind::forward_training) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(mkldnn_batch_normalization_backward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), gradx_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), static_cast<float>(epsilon), flags),
          "could not create a batch normalization backward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
          &result, &data, engine::cpu_engine().get(), hint_.get()),
        "could not create a batch normalization backward primitive descriptor");
      reset(result);
    }
  private:
    batch_normalization_forward_training::descriptor hint_;
  };

  float get_epsilon() const {
    const mkldnn_batch_normalization_desc_t *p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d), 0, (void *)&p_desc),
      "could not query batch normalization descriptor");
    return p_desc->batch_norm_epsilon;
  }

public:
  using computation::expected_input_descriptor;
  using computation::expected_gradx_descriptor;

  tdesc_t expected_grad_scale_descriptor() const {
    return expected_descriptor_of(query::src_pd, 2);
  }
  tdesc_t expected_grad_shift_descriptor() const {
    return expected_descriptor_of(query::src_pd, 1);
  }
  tdesc_t expected_statistic_descriptor() const {
    return expected_descriptor_of(query::src_pd, 1);
  }

  prop_kind get_prop_kind() const {
    const mkldnn_batch_normalization_desc_t *p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d), 0, (void *)&p_desc),
      "could not query batch normalization descriptor");
    return static_cast<prop_kind>(p_desc->prop_kind);
  }

  void init(const tdesc_t& gradx_desc, const tdesc_t& src_desc, float epsilon,
      unsigned flags = batch_normalization_flag::use_scale_shift, prop_kind aprop_kind=prop_kind::backward) {
    descriptor batch_norm_backward(gradx_desc, src_desc, epsilon, flags, aprop_kind);
    computation::init(batch_norm_backward);
    weights_.init(batch_norm_backward.expected_weights_descriptor());
    grad_scale_shift_.init(batch_norm_backward.expected_weights_descriptor());
  }

  template<typename T, typename ...Ts>
  batch_normalization_backward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx, const tensor& gradw) {
    // We can sure that only scale is matter at this place
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx, gradw);
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance, const tensor& grady,
      const tensor& scale, const tensor& gradx, const tensor& gradw, const tensor& grad_shift) {
    // protect API integraty, should we use solid check instead of assert?
    IDEEP_ENFORCE(get_prop_kind() == prop_kind::backward, "Incorrect prop_kind");
    // We can sure that only scale is matter at this place
    // And single thread of memcpy should be fast enough
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());

    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(), get_epsilon());
    fetch_or_create_m(comp, key, src.get_descriptor(), src.get_descriptor(), get_epsilon());
    grad_scale_shift_.reinit(comp.expected_gradw_descriptor());

    computation::execute(src, mean, variance, grady, weights_, gradx, grad_scale_shift_);
    std::memcpy(gradw.get_data_handle(), (char *)grad_scale_shift_.get_data_handle(), gradw.get_size());
    std::memcpy(grad_shift.get_data_handle(),
        (char *)grad_scale_shift_.get_data_handle() + gradw.get_size(), grad_shift.get_size());
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx) {
    IDEEP_ENFORCE(get_prop_kind() == prop_kind::backward_data, "Incorrect prop_kind");
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx);
  }

  static void compute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, tensor& gradx, tensor& gradw, float epsilon) {
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(), epsilon);

    fetch_or_create_m(comp, key, src.get_descriptor(), src.get_descriptor(), epsilon);

    auto grady_in = grady;
    if (grady_in.get_descriptor() != comp.expected_input_descriptor(3)) {
      grady_in.reinit<batch_normalization_backward>(comp.expected_input_descriptor(3));
      reorder::compute(grady, grady_in);
    }

    gradx.reinit<batch_normalization_backward>(comp.expected_gradx_descriptor());
    gradw.reinit(comp.expected_gradw_descriptor());

    comp.execute(src, mean, variance, grady_in, scale, gradx, gradw);
  }

  static void compute(const tensor& src, const tensor& mean,
      const tensor& variance, const tensor& grady, const tensor& scale,
      tensor& gradx, tensor& grad_scale, tensor& grad_shift, float epsilon) {
    key_t key;
    utils::create_key(key, src.get_data_type(), src.get_dims(), src.get_internal_format(), epsilon);
    fetch_or_create_m(comp, key, src.get_descriptor(), src.get_descriptor(), epsilon);

    auto grady_in = grady;
    if (grady_in.get_descriptor() != comp.expected_input_descriptor(3)) {
      grady_in.reinit<batch_normalization_backward>(comp.expected_input_descriptor(3));
      reorder::compute(grady, grady_in);
    }

    gradx.reinit<batch_normalization_backward>(comp.expected_gradx_descriptor());
    grad_scale.reinit(mean.get_descriptor());
    grad_shift.reinit(mean.get_descriptor());

    comp.execute(src, mean, variance, grady_in, scale, gradx, grad_scale, grad_shift);
  }

private:
  tensor weights_;
  tensor grad_scale_shift_;
};

struct inner_product_forward: public computation,
  public utils::computation_cache<inner_product_forward> {
  struct descriptor: public descriptor_group {
    descriptor(const tdesc_t &src_desc, const tdesc_t &weights_desc, const tdesc_t &bias_desc,
        const tdesc_t &dst_desc, prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_inner_product_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto bias_data = bias_desc.format_any();
      auto dst_data = dst_desc.format_any();

      error::wrap_c_api( mkldnn_inner_product_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), &src_data, &weights_data, &bias_data, &dst_data),
          "could not create a inner product forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a inner product forward primitive descriptor");
      reset(result);
    }

    descriptor(const tdesc_t &src_desc, const tdesc_t &weights_desc, const tdesc_t &dst_desc,
            prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_inner_product_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto dst_data = dst_desc.format_any();

      error::wrap_c_api( mkldnn_inner_product_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), &src_data, &weights_data, nullptr, &dst_data),
          "could not create a inner product forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a inner product forward primitive descriptor");
      reset(result);
    }
  };
 public:
  using computation::execute;
  using computation::expected_dst_descriptor;
  using computation::expected_weights_descriptor;
  using computation::expected_src_descriptor;

  void init(const tdesc_t &src_desc, const tdesc_t &weights_desc, const tdesc_t &dst_desc) {
    descriptor forward_descriptor(src_desc, weights_desc, dst_desc);
    computation::init(forward_descriptor, src_desc, weights_desc);
  }

  void init(const tdesc_t &src_desc, const tdesc_t &weights_desc,
      const tdesc_t &bias_desc, const tdesc_t &dst_desc) {
    descriptor forward_descriptor(src_desc, weights_desc, bias_desc, dst_desc);
    computation::init(forward_descriptor, src_desc, weights_desc, bias_desc);
  }

  template<typename T, typename ...Ts>
  inner_product_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  static void compute(key_t &key, const tensor& src, const tensor& weights,
      const tensor& bias, tensor& dst) {
    auto weights_in = weights;
    auto src_in = src.has_scale() ? src.to_public() : src;
    auto bias_in = bias.has_scale() ? bias.to_public() : bias;

    if (src_in.ndims() != weights_in.ndims()) {
      auto ndims = src_in.is_public_format() ? weights_in.ndims() : src_in.ndims();
      if (ndims != src_in.ndims()) {
        auto new_dims = weights_in.get_dims();
        new_dims[0] = src_in.get_dim(0);
        src_in.reshape(new_dims);
      } else if (ndims != weights_in.ndims()) {
        auto new_dims = src_in.get_dims();
        new_dims[0] = weights_in.get_dim(0);
        weights_in.reshape(new_dims);
      }
    }
    IDEEP_ENFORCE(src_in.ndims() == weights_in.ndims(), "Invalid dims in src or weights");
    IDEEP_ENFORCE(!weights_in.has_scale() && weights_in.get_data_type() == tdtype_t::f32,
          "INT8 mode is not supported");

    auto src_desc = src_in.get_descriptor();
    IDEEP_ENFORCE(src_in.get_data_type() == tdtype_t::f32, "Incorrect src data type");

    tdims_t dst_dims = {src_desc.get_dim(0), weights_in.get_dim(0)};
    tdesc_t dst_desc(dst_dims, src_desc.get_data_type());

    if (key.empty())
      utils::create_key(key, src_desc.get_data_type(), src_desc.get_dims(),
          weights_in.get_dims(), bias_in.get_dims(), dst_dims);

    fetch_or_create_m(comp, key, src_desc,
        weights_in.get_descriptor(), bias_in.get_descriptor(), dst_desc);

    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<inner_product_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    if (weights_in.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<inner_product_forward>(comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    dst.reinit<inner_product_forward>(comp.expected_dst_descriptor());
    comp.execute(src_in, weights_in, bias_in, dst);
  }

  static void compute(const tensor& src, const tensor& weights,
      const tensor& bias, tensor& dst) {
    key_t key;
    compute(key, src, weights, bias, dst);
  }

  static void compute(key_t & key, const tensor& src, const tensor& weights, tensor& dst) {
    auto weights_in = weights;
    auto _src = src.has_scale() ? src.to_public() : src;
    auto src_in = _src;

    if (src_in.ndims() != weights_in.ndims()) {
      auto ndims = src_in.is_public_format() ? weights_in.ndims() : src_in.ndims();
      if (ndims != src_in.ndims()) {
        auto new_dims = weights_in.get_dims();
        new_dims[0] = src_in.get_dim(0);
        src_in.reshape(new_dims);
      } else if (ndims != weights_in.ndims()) {
        auto new_dims = src_in.get_dims();
        new_dims[0] = weights_in.get_dim(0);
        weights_in.reshape(new_dims);
      }
    }
    IDEEP_ENFORCE(src_in.ndims() == weights_in.ndims(), "Invalid dims in src or weights");
    IDEEP_ENFORCE(!weights_in.has_scale() && weights_in.get_data_type() == tdtype_t::f32,
          "INT8 mode is not supported");

    auto src_desc = src_in.get_descriptor();
    IDEEP_ENFORCE(src_in.get_data_type() == tdtype_t::f32, "Incorrect src data type");

    tdims_t dst_dims = {src_desc.get_dim(0), weights_in.get_dim(0)};
    tdesc_t dst_desc(dst_dims, src_desc.get_data_type());

    if (key.empty())
      utils::create_key(key, src_desc.get_data_type(), src_desc.get_dims(), weights_in.get_dims(), dst_dims);

    fetch_or_create_m(comp, key, src_desc, weights_in.get_descriptor(), dst_desc);

    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<inner_product_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }

    if (weights_in.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<inner_product_forward>(comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    dst.reinit<inner_product_forward>(comp.expected_dst_descriptor());
    comp.execute(src_in, weights_in, dst);
  }

  static void compute(const tensor& src, const tensor& weights, tensor& dst) {
    key_t key;
    compute(key, src, weights, dst);
  }

  static tdesc_t expected_weights_descriptor( const tdims_t& weights_dims, tdtype_t dtype = tdtype_t::f32) {
    auto x_dims = weights_dims;
    x_dims[0] = 1;
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(), "Invalid dims for data and weights");
    tdesc_t x_desc(x_dims, dtype, ndims == 2 ? format::nc : format::nchw);
    tdesc_t y_desc(y_dims, dtype, format::nc);
    tdesc_t weights_desc(weights_dims, dtype, ndims == 2 ? format::oi : format::oihw);

    inner_product_forward comp(x_desc, weights_desc, y_desc);
    return comp.dup_weights_descriptor();
  }
};

// TODO: parameter sequence adjust?
struct inner_product_backward_data: public computation,
  public utils::computation_cache<inner_product_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &gradx_desc, const tdesc_t &weights_desc, const tdesc_t &grady_desc)
      : hint_(gradx_desc, weights_desc, grady_desc) {
      auto diff_src_data = gradx_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      mkldnn_inner_product_desc_t data;
      error::wrap_c_api(mkldnn_inner_product_backward_data_desc_init(
            &data, &diff_src_data, &weights_data, &diff_dst_data),
          "could not create a inner product backward data descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a inner product backward data primitive descriptor");
      reset(result);
    }
  private:
    inner_product_forward::descriptor hint_;
  };

public:
  using computation::expected_gradx_descriptor;
  using computation::expected_grady_descriptor;
  using computation::expected_weights_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &gradx_desc, const tdesc_t &weights_desc, const tdesc_t &grady_desc) {
    descriptor backward_data_descriptor(gradx_desc, weights_desc, grady_desc);
    computation::init(backward_data_descriptor, grady_desc, weights_desc);
  }

  template<typename T, typename ...Ts>
  inner_product_backward_data(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& grady, const tensor& weights, const tensor& gradx) {
    computation::execute(grady, weights, gradx);
  }

  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims, tensor& gradx) {
    auto weights_in = weights;
    if (gradx_dims.size() != weights_in.ndims()) {
      auto new_dims = gradx_dims;
      new_dims[0] = weights_in.get_dim(0);
      weights_in.reshape(new_dims);
    }
    IDEEP_ENFORCE(gradx_dims.size() == weights_in.ndims(), "Invalid dims in src or weights");

    tdesc_t gradx_desc(gradx_dims, grady.get_data_type());

    key_t key;
    utils::create_key(key, grady.get_data_type(), grady.get_dims(), weights_in.get_dims(), gradx_dims);

    fetch_or_create_m(comp, key, gradx_desc, weights_in.get_descriptor(), grady.get_descriptor());

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<inner_product_backward_data>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    if (weights_in.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<inner_product_backward_data>(comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    gradx.reinit<inner_product_backward_data>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }
};

struct inner_product_backward_weights : public computation,
  public utils::computation_cache<inner_product_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t &x_desc, const tdesc_t &gradw_desc,
        const tdesc_t &gradb_desc, const tdesc_t &grady_desc)
      : hint_(x_desc, gradw_desc, gradb_desc, grady_desc) {
      mkldnn_inner_product_desc_t data;
      auto src_data = x_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      auto diff_weights_data = gradw_desc.format_any();
      auto diff_bias_data = gradb_desc.format_any();
      error::wrap_c_api( mkldnn_inner_product_backward_weights_desc_init(
            &data, &src_data, &diff_weights_data, &diff_bias_data, &diff_dst_data),
          "could not create a inner product backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a inner product backward weights primitive descriptor");
      reset(result);
    }

    descriptor(const tdesc_t &x_desc, const tdesc_t &gradw_desc, const tdesc_t &grady_desc)
    : hint_(x_desc, gradw_desc, grady_desc) {
      mkldnn_inner_product_desc_t data;
      auto src_data = x_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      auto diff_weights_data = gradw_desc.format_any();
      error::wrap_c_api( mkldnn_inner_product_backward_weights_desc_init(
          &data, &src_data, &diff_weights_data, nullptr, &diff_dst_data),
          "could not create a inner product backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a inner product backward weights primitive descriptor");
      reset(result);
    }
  private:
    inner_product_forward::descriptor hint_;
  };
public:
  using computation::expected_gradw_descriptor;
  using computation::expected_gradb_descriptor;

  template<typename ...Ts>
  void init(const tdesc_t &x_desc, const tdesc_t &grady_desc, const tdesc_t &gradw_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, grady_desc, gradw_desc, std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor, x_desc, grady_desc);
  }

  template<typename T, typename ...Ts>
  inner_product_backward_weights(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& gradw) {
    computation::execute(x, grady, gradw);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& gradw, const tensor& gradb) {
    computation::execute(x, grady, gradw, gradb);
  }

  static void compute(const tensor& x, const tensor& grady, tensor& gradw) {
    auto gradw_dims = x.get_dims();
    gradw_dims[0] = grady.get_dim(1);
    tdesc_t gradw_desc(gradw_dims, grady.get_data_type());

    key_t key;
    utils::create_key(key, x.get_data_type(), x.get_dims(), gradw_dims, grady.get_dims());

    fetch_or_create_m(comp, key, x.get_descriptor(), gradw_desc, grady.get_descriptor());

    auto x_in = x;
    if (x.get_descriptor() != comp.expected_src_descriptor()) {
      x_in.init<inner_product_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(x, x_in);
    }

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<inner_product_backward_weights>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<inner_product_backward_weights>(comp.expected_gradw_descriptor());
    comp.execute(x_in, grady_in, gradw);
  }

  static void compute(const tensor& x, const tensor& grady, tensor& gradw, tensor& gradb) {
    auto gradw_dims = x.get_dims();
    gradw_dims[0] = grady.get_dim(1);

    tdims_t gradb_dims = {grady.get_dim(1)};
    tdesc_t gradw_desc(gradw_dims, x.get_data_type());
    tdesc_t gradb_desc(gradb_dims, x.get_data_type());

    key_t key;
    utils::create_key(key, x.get_data_type(), x.get_dims(), gradw_dims, gradb_dims, grady.get_dims());
    fetch_or_create_m(comp, key, x.get_descriptor(), gradw_desc, gradb_desc, grady.get_descriptor());

    auto x_in = x;
    if (x.get_descriptor() != comp.expected_src_descriptor()) {
      x_in.init<inner_product_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(x, x_in);
    }

    auto grady_in = grady;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<inner_product_backward_weights>(comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<inner_product_backward_weights>(comp.expected_gradw_descriptor());
    gradb.reinit(comp.expected_gradb_descriptor());

    comp.execute(x_in, grady_in, gradw, gradb);
  }
};

struct dropout_forward {
public:
  template<class T>
  static void compute_impl(const tensor& src, float ratio, tensor& dst, tensor& mask) {
    dropout_forward comp;
    mask.reinit<dropout_forward>(src.get_descriptor());
    dst.reinit<dropout_forward>(src.get_descriptor());
    if (src.has_scale()) dst.set_scale(src.get_scale());

    const auto scale = 1.0 / (1.0 - ratio);
    const auto size = src.get_nelems();
    std::unique_ptr<int[]> bernouli_nums(new int[size]);
    utils::bernoulli_generate(size, 1.0 - ratio, bernouli_nums.get());

    const auto src_data = static_cast<T *>(src.get_data_handle());
    const auto mask_data = static_cast<T *>(mask.get_data_handle());
    const auto dst_data = static_cast<T *>(dst.get_data_handle());

    # pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      mask_data[i] = bernouli_nums[i] * scale;
      dst_data[i] = mask_data[i] * src_data[i];
    }
  }

  static void compute(const tensor &src, float ratio,
      tensor& dst, tensor& mask) {
    switch(src.get_data_type()) {
    case tdtype_t::f32:
      compute_impl<float>(src, ratio, dst, mask);
      break;
    case tdtype_t::s32:
      compute_impl<int32_t>(src, ratio, dst, mask);
      break;
    case tdtype_t::s16:
      compute_impl<int16_t>(src, ratio, dst, mask);
      break;
    case tdtype_t::s8:
      compute_impl<int8_t>(src, ratio, dst, mask);
      break;
    case tdtype_t::u8:
      compute_impl<uint8_t>(src, ratio, dst, mask);
      break;
    default:
      throw error(mkldnn_invalid_arguments, "Unsupported mkldnn data type!");
    }
  }
};

struct dropout_backward {
public:
  template<class T>
  static void compute_impl(const tensor& mask, const tensor& gy, tensor& gx) {
    dropout_backward comp;
    gx.reinit<dropout_backward>(gy.get_descriptor());

    const auto size = mask.get_nelems();
    const auto mask_data = static_cast<T *>(mask.get_data_handle());
    const auto gy_data = static_cast<T *>(gy.get_data_handle());
    const auto gx_data = static_cast<T *>(gx.get_data_handle());

    # pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      gx_data[i] = mask_data[i] * gy_data[i];
    }
  }

  static void compute(const tensor &mask, const tensor &gy, tensor& gx) {
    switch(gy.get_data_type()) {
    case tdtype_t::f32:
      compute_impl<float>(mask, gy, gx);
      break;
    case tdtype_t::s32:
      compute_impl<int32_t>(mask, gy, gx);
      break;
    case tdtype_t::s16:
      compute_impl<int16_t>(mask, gy, gx);
      break;
    case tdtype_t::s8:
      compute_impl<int8_t>(mask, gy, gx);
      break;
    case tdtype_t::u8:
      compute_impl<uint8_t>(mask, gy, gx);
      break;
    default:
      throw error(mkldnn_invalid_arguments, "Unsupported mkldnn data type!");
    }
  }
};

} // namespace ideep

#endif
