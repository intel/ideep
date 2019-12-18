#ifndef IDEEP_COMPUTATIONS_HPP
#define IDEEP_COMPUTATIONS_HPP

#include "abstract_types.hpp"
#include "tensor.hpp"
#include "lru_cache.hpp"
#include "utils.hpp"
#include "primitives.hpp"

namespace ideep {

using tdims_t = tensor::dims;
using tdesc_t = tensor::descriptor;
using tdtype_t = tensor::data_type;
using treorder_t = tensor::reorder;
using attr_t = descriptor_group::attr_t;

struct direct_copy : public utils::computation_cache<direct_copy> {
public:
  direct_copy(tdesc_t desc) {
    treorder_t::reorder_desc rdesc(desc, desc);
    reorder_.init(rdesc, desc, desc);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& input, tensor& output) {
    if (input.is_empty() || input == output) {
      return;
    }

    tdesc_t input_desc = input.get_descriptor();
    output.reinit<alloc>(input.get_descriptor());

    key_t key;
    check_or_create_k(key, input);
    fetch_or_create_m(comp, key, input_desc);

    comp.reorder_(input, output);
    if (input.has_scale()) {
      output.set_scale(input.get_scale());
    }
  }
private:
    treorder_t reorder_;
};

struct spliter : public utils::computation_cache<spliter> {
public:
  template<class ...T>
  using s_vector = utils::s_vector<T...>;

  spliter(int32_t inputs_num) { reorders_.assign(inputs_num, nullptr); }

  template<class alloc = utils::allocator>
  static std::vector<tensor> compute(const tensor& input, std::vector<int32_t>& axis_info, int axis, bool add_axis) {
    std::vector<tensor> outputs;
    tdims_t output_dims(input.get_dims());
    tdims_t offset_dims(output_dims.size(), 0);
    IDEEP_ENFORCE(axis < input.ndims(), "invalid axis in split");

    key_t key;
    check_or_create_k(key, input, axis_info, axis, add_axis);
    fetch_or_create_m(comp, key, axis_info.size());

    for (unsigned i = 0; i < axis_info.size(); ++i) {
      output_dims[axis] = axis_info[i];
      auto view = input.create_view(output_dims, offset_dims);
      tensor output;
      output.init<alloc>(view.expected_dst_descriptor());
      if (comp.reorders_[i]) {
        comp.reorders_[i]->operator()(input, output);
      } else {
        treorder_t reorder_(view, input.get_descriptor(), output.get_descriptor());
        reorder_(input, output);
        comp.reorders_[i] = std::make_shared<treorder_t>(reorder_);
      }
      if (input.has_scale()) output.set_scale(input.get_scale());

      if (add_axis) {
        tdims_t out_dims(output_dims);
        out_dims.erase(out_dims.begin() + axis);
        output.reshape<alloc>(out_dims);
      }

      outputs.emplace_back(output);
      offset_dims[axis] += axis_info[i];
    }

    return outputs;
  }
private:
  s_vector<std::shared_ptr<treorder_t>> reorders_;
};

/// Computation class, abstruct of computation
struct computation : public primitive_group {
public:
  template<class ...T>
  using s_vector = utils::s_vector<T...>;

  computation() = default;

  inline void init_internal(const descriptor_group& adesc) {
    inouts_ = s_vector<tensor>((unsigned)(inputs_num_ + outputs_num_));

    std::unique_ptr<mkldnn_primitive_at_t []> inputs(new mkldnn_primitive_at_t [inputs_num_]);
    for (int i =0; i < inputs_num_; i ++) {
      inouts_[i] = {adesc.expected_descriptor_of(query::input_pd, i), nullptr };
      inputs[i] = { inouts_[i].get(), 0 };
    }

    std::unique_ptr<const_mkldnn_primitive_t []> outputs(new const_mkldnn_primitive_t [outputs_num_]);
    for (int i = 0; i < outputs_num_; i ++) {
      inouts_[i + inputs_num_] = {adesc.expected_descriptor_of(query::output_pd, i), nullptr };
      outputs[i] = inouts_[i + inputs_num_].get();
    }

    create_primitive(adesc, inputs.get(), outputs.get());
  }

  void init(const descriptor_group& adesc, const std::vector<tdesc_t>& args) {
    IDEEP_ENFORCE(adesc.num_of_inputs() == (int)args.size(), "Unmatch the number of inputs");
    inputs_num_ = (int)args.size();
    outputs_num_ = adesc.num_of_outputs();
    init_internal(adesc);
    reorders_.assign(inputs_num_, nullptr);
    ins_buf_.assign(inputs_num_, std::make_shared<tensor>(tensor()));
  }

  template<typename... Ts>
  void init(const descriptor_group& adesc) {
    inputs_num_ = adesc.num_of_inputs();
    outputs_num_ = adesc.num_of_outputs();
    init_internal(adesc);
    reorders_.assign(inputs_num_, nullptr);
    ins_buf_.assign(inputs_num_, std::make_shared<tensor>(tensor()));
  }

  template<class alloc = utils::allocator>
  const tensor& transform_input_cache(int index, const tensor& input, attr_t attr = attr_t()) {
    // NOTE: To cache the reorder for specific computation, the input tensor and attr must
    // be recorded in the computation key with all details.
    IDEEP_ENFORCE(index < inputs_num_, "Invalid input index");
    if (ins_buf_[index] == nullptr) {
      return input;
    }
    if (reorders_[index]) {
      reorders_[index]->operator()(input, *ins_buf_[index]);
      return *ins_buf_[index];
    }

    auto src_desc = input.get_descriptor();
    auto dst_desc = inouts_[index].get_descriptor();
    if (src_desc == dst_desc) {
      ins_buf_[index] = nullptr;
      return input;
    }

    tensor in_ten;
    in_ten.init<alloc>(dst_desc);
    treorder_t reorder_ (src_desc, dst_desc, attr);
    reorder_(input, in_ten);
    ins_buf_[index] = std::make_shared<tensor>(in_ten);
    reorders_[index] = std::make_shared<treorder_t>(reorder_);
    return *ins_buf_[index];
  }

  template<class alloc = utils::allocator>
  const tensor& transform_input_uncache(int index, const tensor& input, attr_t attr = attr_t()) {
    IDEEP_ENFORCE(index < inputs_num_, "Invalid input index");
    auto src_desc = input.get_descriptor();
    auto dst_desc = inouts_[index].get_descriptor();
    if (src_desc == dst_desc) {
      return input;
    }

    tensor in_ten;
    in_ten.init<alloc>(dst_desc);
    treorder_t reorder_ (src_desc, dst_desc, attr);
    reorder_(input, in_ten);
    return in_ten;
  }

  void connect_handle_for(int index, const tensor& atensor) {
    inouts_[(unsigned)index].set_data_handle(atensor.get_data_handle());
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

  void execute(const std::vector<tensor>& inputs, const tensor& output) {
    connect_handle_for(inputs, output);
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
  s_vector<shared_ptr<treorder_t>> reorders_;
  s_vector<shared_ptr<tensor>> ins_buf_;
};

struct sum : public computation,
  public utils::computation_cache<sum> {
  struct descriptor : public descriptor_group {
    descriptor(const scale_t& scales, const std::vector<tdesc_t>& inputs) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = tdesc_t::convert_to_c(inputs);
      error::wrap_c_api(mkldnn_sum_primitive_desc_create(
            &result, nullptr, (int)c_api_inputs.size(), &scales[0], &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    descriptor(const scale_t& scales, const std::vector<tdesc_t>& inputs, const tdesc_t& output_desc) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = tdesc_t::convert_to_c(inputs);
      error::wrap_c_api(mkldnn_sum_primitive_desc_create(
              &result, output_desc.get_mkldnn_memory_desc_t(), (int)c_api_inputs.size(), &scales[0], &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }
  };

public:
  sum() = default;

  void init(const scale_t& scales, const std::vector<tdesc_t>& inputs) {
    descriptor forward_descriptor(scales, inputs);
    computation::init(forward_descriptor, inputs);
  }

  void init(const scale_t& scales, const std::vector<tdesc_t>& inputs, const tdesc_t& output) {
    descriptor forward_descriptor(scales, inputs, output);
    computation::init(forward_descriptor, inputs);
  }

  sum(const scale_t& scales, const std::vector<tdesc_t>& inputs_desc, const tdesc_t& output_desc, bool inplace = false) {
    if (inplace) {
      init(scales, inputs_desc, output_desc);
    } else {
      init(scales, inputs_desc);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const scale_t& scales, const std::vector<tensor>& inputs, tensor& output) {
    std::vector<tensor> inputs_in;
    std::vector<tdesc_t> inputs_desc;
    for (auto in : inputs) {
      auto _in = in;
      if (in.get_data_type() != tdtype_t::f32) {
        _in.init<alloc>({in.get_dims(), tdtype_t::f32});
        IDEEP_ENFORCE(in.has_scale(), "Can not find scales");
        IDEEP_ENFORCE(in.get_scale().size() == 1, "Incorrect scale size");
        auto scale = IDEEP_DEF_SCALE;
        scale[0] /= in.get_scale()[0];
        treorder_t::compute(in, _in, {0, scale});
      }
      inputs_in.push_back(_in);
      inputs_desc.push_back(_in.get_descriptor());
    }

    std::vector<tdtype_t> inputs_dt;
    std::vector<tdims_t> inputs_dims;
    std::vector<format> inputs_format;
    for (tensor elems : inputs_in) {
      inputs_dt.push_back(elems.get_data_type());
      inputs_dims.push_back(elems.get_dims());
      inputs_format.push_back(elems.get_internal_format());
    }

    key_t key;
    bool inplace = (output == inputs_in[0]);
    check_or_create_k(key, inputs_dt, inputs_dims, inputs_format, scales, inplace);
    fetch_or_create_m(comp, key, scales, inputs_desc, output.get_descriptor(), inplace);

    if (!inplace)
      output.reinit<alloc>(comp.expected_dst_descriptor());
    comp.execute(inputs_in, output);
  }
};

/// Convolution forward computation, this class represent a MKL-DNN
/// convolution forward process, also manage old computation instances.
struct convolution_forward: public computation,
  public utils::computation_cache<convolution_forward> {
  /// Descriptor class for describing convolution forward process
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_desc, const tdesc_t& weights_desc, const tdesc_t& bias_desc,
        const tdesc_t& dst_desc, const tdims_t& strides, const tdims_t& dilates,
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
      auto dilates_in = utils::get_compatible_dilates(dilates, src_desc.ndims());
      error::wrap_c_api(mkldnn_dilated_convolution_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
            &src_data, &weights_data, &bias_data, &dst_data, &strides[0], &dilates_in[0],
            &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a dilated convolution forward descriptor");
      create_primitive_desc_v2(data, attr);
    }
  };

 public:

  template<typename ...Ts>
  convolution_forward(const tdesc_t& src_desc, Ts&&... args) {
    descriptor forward_descriptor(src_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template <class alloc, bool with_bias>
  static void compute_impl(convolution_forward& comp, const tensor& src,
      const tensor& weights, const tensor& bias, tensor& dst) {
    auto src_in = comp.transform_input_cache<alloc>(0, src);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights.as_weights());

    if (comp.dst_exp_desc_) {
      dst.reinit<alloc>(*comp.dst_exp_desc_);
    }
    if (comp.dst_scales_) {
      dst.set_scale(*comp.dst_scales_);
    }

    if (with_bias) {
      auto bias_in = comp.transform_input_cache<alloc>(2, bias);
      comp.execute(src_in, weights_in, bias_in, dst);
    } else {
      comp.execute(src_in, weights_in, dst);
    }

    if (comp.dst_u8_desc_) {
      dst.set_descriptor(*comp.dst_u8_desc_);
    }
  }

  template <class alloc, bool with_bias, typename ...Ts>
  static void compute_impl(key_t& key, const tensor& src, const tensor& weights, const tensor& bias,
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

    auto dst_format = post_ops.has_op_kind(kind::sum) ?
      dst.get_internal_format() : engine::default_format(dst_dims.size());
    tdesc_t dst_desc_in(dst_dims, dst_data_type, dst_format);

    check_or_create_k(key, src, weights, with_bias, strides, dilates, padding_l, padding_r,
        op_attr, src_scales, dst_scales, args...);
    fetch_or_create_m(comp, key, src_desc, weights_desc, bias_desc, dst_desc_in, strides, dilates,
            padding_l, padding_r, op_attr, std::forward<Ts>(args)...);

    auto src_in = comp.transform_input_cache<alloc>(0, src, src_attr);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights.as_weights(), weights_attr);

    auto dst_desc = comp.expected_dst_descriptor();
    if (dst.get_descriptor() != dst_desc) {
      comp.dst_exp_desc_.reset(new tdesc_t(dst_desc));
      IDEEP_ENFORCE(!post_ops.has_op_kind(kind::sum), "Unmatch format or data type in Conv Sum fusion");
      dst.reinit<alloc>(dst_desc);
    }

    if (!dst_scales.empty() && dst_data_type != tdtype_t::f32) {
      dst.set_scale(dst_scales_in);
      comp.dst_scales_.reset(new scale_t(dst_scales_in));
    }

    if (with_bias) {
      auto bias_in = comp.transform_input_cache<alloc>(2, bias, bias_attr);
      comp.execute(src_in, weights_in, bias_in, dst);
    } else {
      comp.execute(src_in, weights_in, dst);
    }

    if (post_ops.non_negitive_output() && dst.get_data_type() == tdtype_t::s8) {
      tdesc_t dst_u8_desc { dst.get_dims(), tdtype_t::u8, dst.get_internal_format()};
      dst.set_descriptor(dst_u8_desc);
      comp.dst_u8_desc_ = std::make_shared<tdesc_t>(dst_u8_desc);
    }

    update(comp, it);
  }

  template<class alloc = utils::allocator, bool with_bias = true>
  static void compute(key_t& key, const tensor& src, const tensor& weights, const tensor& bias,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero, const lowp_kind alowp_kind = LOWP_U8S8) {
    auto weights_in = weights;
    weights_in.make_group(group);

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the mkldnn_wino_fmt for weights is required by winograd primitive.
    // Then, in the cases of variable input shape, the detials of mkldnn_wino_fmt will be changed.
    // And, extra weihgts reorder is inevitable each time, leading to bad performance.
    // Here, we set the prop_kind to forward, in order to reorder and cache weights as blocked format,
    // instead of mkldnn_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd && aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    auto it = key.empty() ? end() : find(key);
    if (it != end()) {
      compute_impl<alloc, with_bias>(fetch(it), src, weights_in, bias, dst);
    } else {
      compute_impl<alloc, with_bias>(key, src, weights_in, bias, result_dims, dst, strides, dilates,
          padding_l, padding_r, src_scales, weights_scales, dst_scales, attr, alowp_kind,
          aalgorithm, apkind, appading_kind);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, const tensor& weights,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero, const lowp_kind alowp_kind = LOWP_U8S8) {
    static tensor dummy_bias;
    compute<alloc, false>(key, src, weights, dummy_bias, result_dims, dst, strides, dilates,
        padding_l, padding_r, group, src_scales, weights_scales, dst_scales, attr,
        aalgorithm, aprop_kind, appading_kind, alowp_kind);
  }

  template<class alloc = utils::allocator, bool with_bias = true>
  static void compute(const tensor& src, const tensor& weights, const tensor& bias,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero, const lowp_kind alowp_kind = LOWP_U8S8) {
    key_t key;
    compute<alloc, with_bias>(key, src, weights, bias, result_dims, dst, strides, dilates,
        padding_l, padding_r, group, src_scales, weights_scales, dst_scales, attr,
        aalgorithm, aprop_kind, appading_kind, alowp_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& weights,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero, const lowp_kind alowp_kind = LOWP_U8S8) {
    static tensor dummy_bias;
    compute<alloc, false>(src, weights, dummy_bias, result_dims, dst, strides, dilates,
        padding_l, padding_r, group, src_scales, weights_scales, dst_scales, attr,
        aalgorithm, aprop_kind, appading_kind, alowp_kind);
  }

  template<class alloc = utils::allocator, bool with_bias = true>
  static void compute(const tensor& src, const tensor& weights, const tensor& bias,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group, const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero) {
    key_t key;
    static scale_t dummy_scales;
    compute<alloc, with_bias>(key, src, weights, bias, result_dims, dst, strides, dilates,
        padding_l, padding_r, group, dummy_scales, dummy_scales, dummy_scales, attr,
        aalgorithm, aprop_kind, appading_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& weights,
      const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, int group, const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
      padding_kind appading_kind = padding_kind::zero) {
    static tensor dummy_bias;
    compute<alloc, false>(src, weights, dummy_bias, result_dims, dst, strides, dilates,
        padding_l, padding_r, group, attr, aalgorithm, aprop_kind, appading_kind);
  }

  static tdesc_t expected_weights_descriptor(const tdims_t& weights_dims,
      tdtype_t dtype = tdtype_t::f32, const tdims_t& strides = {1, 1},
      const tdims_t& padding_l = {0, 0}, const tdims_t& padding_r = {0, 0},
      const tdims_t& dilates = {0, 0}, int group = 1, algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward, tdtype_t x_dtype = tdtype_t::f32,
      const tdims_t& src_dims = tdims_t()) {
    auto dims_in = weights_dims;
    auto src_size =  strides.size() + 2; // we should give the stride to make check 2d or 3d conv
    auto grouped = IDEEP_IS_GROUPED(src_size, dims_in);
    if (group > 1 && !grouped) {
      tensor::group_dims(dims_in, group);
    }
    auto ndims = dims_in.size();
    auto g = grouped ? dims_in[0] : 1;
    auto dilates_in = utils::get_compatible_dilates(dilates, src_size);

    IDEEP_ENFORCE(!(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    tdims_t x_dims;
    tdims_t y_dims;
    auto ic = g * dims_in[1 + grouped];
    auto oc = g * dims_in[0 + grouped];
    tdims_t kernel_size;
    if (src_size == 5){
      kernel_size.push_back(dims_in[ndims - 3]);
    }
    kernel_size.push_back(dims_in[ndims - 2]);
    kernel_size.push_back(dims_in[ndims - 1]);
    auto src_format = src_size == 4 ? format::nchw : format::ncdhw;
    if (src_dims.empty()) {
      // Construct a dummy case
      x_dims.push_back(1);
      x_dims.push_back(ic);
      y_dims.push_back(1);
      y_dims.push_back(oc);
      if (src_size == 4) {
        x_dims.push_back(2 * kernel_size[0]);
        x_dims.push_back(4 * kernel_size[1]);
      } else {
        x_dims.push_back(2 * kernel_size[0]);
        x_dims.push_back(4 * kernel_size[1]);
        x_dims.push_back(8 * kernel_size[2]);
      }
    } else {
      // Use the real data
      for (auto i=0; i < src_size; ++i) {
         x_dims.push_back(src_dims[i]);
      }
      y_dims.push_back(src_dims[0]);
      y_dims.push_back(oc);
    }
    for (auto d = 2; d < src_size; ++d) {
      auto out_size = (x_dims[d] - ((kernel_size[d-2] - 1) * (dilates_in[d-2] + 1) + 1)
          + (padding_l[d-2] + padding_r[d-2])) / strides[d-2] + 1;
      y_dims.push_back(out_size);
    }

    auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;
    tdesc_t x_desc(x_dims, x_dtype, src_format);
    tdesc_t y_desc(y_dims, y_dtype, src_format);
    auto weight_format = grouped ? format::goihw : format::oihw;
    if (src_size == 5) {
      weight_format = grouped ? format::goidhw : format::oidhw;
    }
    tdesc_t weights_desc(dims_in, dtype, weight_format);

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the mkldnn_wino_fmt for weights is required by winograd primitive.
    // Then, in the cases of variable input shape, the detials of mkldnn_wino_fmt will be changed.
    // And, extra weihgts reorder is inevitable each time, leading to bad performance.
    // Here, we set the prop_kind to forward, in order to reorder and cache weights as blocked format,
    // instead of mkldnn_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd && aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    convolution_forward comp(x_desc, weights_desc, tdesc_t(), y_desc, strides, dilates, padding_l, padding_r,
        attr_t(), aalgorithm, apkind);
    return comp.dup_descriptor_of(query::weights_pd);
  }

private:
  std::shared_ptr<tdesc_t> dst_exp_desc_;
  std::shared_ptr<tdesc_t> dst_u8_desc_;
  std::shared_ptr<scale_t> dst_scales_;
};

struct convolution_backward_data : public computation,
  public utils::computation_cache<convolution_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& grady_desc, const tdesc_t& weights_desc, const tdesc_t& gradx_desc,
        const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::convolution_direct, padding_kind apadding_kind = padding_kind::zero)
      : hint_(gradx_desc, weights_desc, tdesc_t(), grady_desc, strides, dilates, padding_l, padding_r)  {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto diff_src_any = gradx_desc.format_any();
      auto weights_any = weights_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      auto dilates_in = utils::get_compatible_dilates(dilates, grady_desc.ndims());
      error::wrap_c_api(mkldnn_dilated_convolution_backward_data_desc_init(
            &data, convert_to_c(aalgorithm), &diff_src_any, &weights_any, &diff_dst_any,
            &strides[0], &dilates_in[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward data descriptor");
      create_primitive_desc(data, hint_.get());
    }
  private:
    convolution_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  convolution_backward_data(const tdesc_t& grady_desc, Ts&&... args) {
    descriptor backward_data_descriptor(grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor);
  }

  template<class alloc, typename ...Ts>
  static void compute_impl(const tensor& grady, const tensor& weights,
      const tdims_t& gradx_dims, tensor& gradx, Ts&&... args) {
    tdesc_t result_desc(gradx_dims, grady.get_data_type());
    key_t key;
    check_or_create_k(key, grady, weights, gradx_dims, args...);
    fetch_or_create_m(comp, key, grady.get_descriptor(),
        weights.get_descriptor(), result_desc, std::forward<Ts>(args)...);

    auto grady_in = comp.transform_input_cache<alloc>(0, grady);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights.as_weights());
    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims,
      tensor& gradx, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, const int group, algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    auto weights_in = weights;
    weights_in.make_group(group);
    compute_impl<alloc>(grady, weights_in, gradx_dims, gradx, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }
};

struct convolution_backward_weights : public computation,
  public utils::computation_cache<convolution_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, const tdesc_t& grady_desc, const tdesc_t& gradw_desc,
        const tdesc_t& gradb_desc, const tdims_t& strides, const tdims_t& dilates,
        const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        padding_kind apadding_kind = padding_kind::zero)
      : hint_(x_desc, gradw_desc, gradb_desc, grady_desc, strides, dilates, padding_l, padding_r) {
      utils::validate_dims(strides, dilates, padding_l, padding_r);
      mkldnn_convolution_desc_t data;
      auto src_any = x_desc.format_any();
      auto diff_weights_any = gradw_desc.format_any();
      auto diff_bias_any = gradb_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      auto dilates_in = utils::get_compatible_dilates(dilates, grady_desc.ndims());
      error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any, &diff_weights_any, &diff_bias_any,
            &diff_dst_any, &strides[0], &dilates_in[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      create_primitive_desc(data, hint_.get());
    }

  private:
    convolution_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  convolution_backward_weights (const tdesc_t& x_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor);
  }

  template<class alloc, bool with_gradb, typename ...Ts>
  static void compute_impl(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, Ts&&... args) {
    tdesc_t gradw_desc(gradw_dims, src.get_data_type());
    tdesc_t gradb_desc;
    if (with_gradb) {
      gradb_desc = {{grady.get_dim(1)}, src.get_data_type()};
    }

    key_t key;
    check_or_create_k(key, src, with_gradb, grady, gradw_dims, args...);
    fetch_or_create_m(comp, key, src.get_descriptor(), grady.get_descriptor(), gradw_desc,
        gradb_desc, std::forward<Ts>(args)...);

    auto src_in = comp.transform_input_cache<alloc>(0, src);
    auto grady_in = comp.transform_input_cache<alloc>(1, grady);
    gradw.reinit<alloc>(comp.expected_gradw_descriptor());

    if (with_gradb) {
      gradb.reinit<alloc>(comp.expected_gradb_descriptor());
      comp.execute(src_in, grady_in, gradw, gradb);
    } else {
      comp.execute(src_in, grady_in, gradw);
    }
  }

  template<class alloc = utils::allocator, bool with_gradb = true>
  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, const int group, algorithm aalgorithm = algorithm::convolution_direct,
      padding_kind apadding_kind = padding_kind::zero) {
    auto gw_dims_in = gradw_dims;
    auto src_size = grady.ndims();
    auto grouped = IDEEP_IS_GROUPED(src_size, gradw_dims);
    if (group > 1 && !grouped) {
      tensor::group_dims(gw_dims_in, group);
    }
    compute_impl<alloc, with_gradb>(src, grady, gw_dims_in, gradw, gradb,
        strides, dilates, padding_l, padding_r, aalgorithm, apadding_kind);
    if (group > 1 && !grouped) {
      IDEEP_ENFORCE(group == gradw.get_dim(0), "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims[0] == group * gradw.get_dim(1), "invalid dim 1 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims.size() == gradw.ndims() - 1, "invalid ndim in grouped gradw");
      gradw.reshape<alloc>(gradw_dims);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, const int group,
      algorithm aalgorithm = algorithm::convolution_direct, padding_kind apadding_kind = padding_kind::zero) {
    static tensor dummy_gradb;
    compute<alloc, false>(src, grady, gradw_dims, gradw, dummy_gradb, strides, dilates, padding_l,
        padding_r, group, aalgorithm, apadding_kind);
  }
};

struct convolution_transpose_forward : public computation,
      public utils::computation_cache<convolution_transpose_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_desc, const tdesc_t& weights_desc, const tdesc_t& bias_desc,
        const tdesc_t& dst_desc, const tdims_t& strides, const tdims_t& dilates,
        const tdims_t& padding_l, const tdims_t& padding_r, const attr_t& attr = attr_t(),
        algorithm aalgorithm = algorithm::deconvolution_direct,
        prop_kind aprop_kind = prop_kind::forward,
        padding_kind apadding_kind = padding_kind::zero) {
      utils::validate_dims(strides, padding_l, padding_r);
      mkldnn_deconvolution_desc_t data;
      auto src_data = src_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto bias_data = bias_desc.format_any();
      auto dst_data = dst_desc.format_any();
      auto dilates_in = utils::get_compatible_dilates(dilates, src_desc.ndims());
      error::wrap_c_api(
          mkldnn_dilated_deconvolution_forward_desc_init(&data, mkldnn::convert_to_c(aprop_kind),
              convert_to_c(aalgorithm), &src_data, &weights_data, &bias_data, &dst_data, &strides[0],
              &dilates_in[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution forward descriptor");
      create_primitive_desc_v2(data, attr);
    }
  };

 public:

  template <typename... Ts>
  convolution_transpose_forward(const tdesc_t& src_desc, Ts&&... args) {
    descriptor forward_descriptor(src_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template <class alloc, bool with_bias, typename... Ts>
  static void compute_impl(const tensor& src, const tensor& weights, const tensor& bias,
      const tdims_t& dst_dims, tensor& dst, Ts&&... args) {
    IDEEP_ENFORCE(!weights.has_scale(), "Unsupported lowp input");
    key_t key;
    check_or_create_k(key, src, weights, with_bias, dst_dims, args...);
    fetch_or_create_m(comp, key, src.get_descriptor(), weights.get_descriptor(),
        bias.get_descriptor(), tdesc_t{dst_dims, src.get_data_type()}, std::forward<Ts>(args)...);

    auto src_in = comp.transform_input_cache<alloc>(0, src);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights.as_weights());

    dst.reinit<alloc>(comp.expected_dst_descriptor());
    if (with_bias) {
      comp.execute(src_in, weights_in, bias, dst);
    } else {
      comp.execute(src_in, weights_in, dst);
    }
  }

  template<class alloc = utils::allocator, bool with_bias = true>
  static void compute(const tensor& src, const tensor& weights, const tensor& bias, const tdims_t& result_dims,
      tensor& dst, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
      const tdims_t& dilates = {1, 1}, const attr_t& attr = attr_t(), algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward, padding_kind appading_kind = padding_kind::zero, const int group = 1) {
    auto weights_in = weights;
    weights_in.make_group(group, true);
    compute_impl<alloc, with_bias>(src, weights_in, bias, result_dims, dst, strides, dilates, padding_l, padding_r,
        attr, aalgorithm, aprop_kind, appading_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& weights, const tdims_t& result_dims,
      tensor& dst, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
      const tdims_t& dilates = {1, 1}, const attr_t& attr = attr_t(), algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward, padding_kind appading_kind = padding_kind::zero, const int group = 1) {
    static tensor dummy_bias;
    compute<alloc, false>(src, weights, dummy_bias,  result_dims, dst, strides, padding_l, padding_r,
        dilates, attr, aalgorithm, aprop_kind, appading_kind, group);
  }

  static tdesc_t expected_weights_descriptor(const tdims_t& weights_dims,
      tdtype_t dtype = tdtype_t::f32, const tdims_t& strides = {1, 1}, const tdims_t& padding_l = {0, 0},
      const tdims_t& padding_r = {0, 0}, const int group = 1, const tdims_t& dilates = {1, 1}) {
    auto dims_in = weights_dims;
    auto src_size =  strides.size() + 2; // we should give the stride to make check 2d or 3d conv
    auto grouped = IDEEP_IS_GROUPED(src_size, dims_in);
    if (group > 1 && !grouped) {
      tensor::group_dims(dims_in, group, true);
    }
    auto ndims = dims_in.size();
     auto g = grouped ? dims_in[0] : 1;
    tdims_t x_dims, y_dims, kernel_size;
    if (src_size == 5){
      kernel_size.push_back(dims_in[ndims - 3]);
    }
    kernel_size.push_back(dims_in[ndims - 2]);
    kernel_size.push_back(dims_in[ndims - 1]);

    auto ic = g * dims_in[1 + grouped];
    auto oc = g * dims_in[0 + grouped];
    x_dims.push_back(1);
    x_dims.push_back(ic);
    y_dims.push_back(1);
    y_dims.push_back(oc);
    if (src_size == 4) {
      x_dims.push_back(8 * kernel_size[0]);
      x_dims.push_back(8 * kernel_size[1]);
    } else {
      x_dims.push_back(8 * kernel_size[0]);
      x_dims.push_back(8 * kernel_size[1]);
      x_dims.push_back(8 * kernel_size[2]);
    }
    for (auto d = 2; d < src_size; ++d) {
      auto out_size = (x_dims[d] -1) * strides[d-2] + (1 + (kernel_size[d-2] - 1) * dilates[d-2])
          - padding_l[d-2] - padding_r[d-2];
      y_dims.push_back(out_size);
    }
    auto x_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::u8;
    auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;
    auto src_format = src_size == 4 ? format::nchw : format::ncdhw;
    tdesc_t x_desc(x_dims, x_dtype, src_format);
    tdesc_t y_desc(y_dims, y_dtype, src_format);
    auto weight_format = grouped ? format::goihw : format::oihw;
    if (src_size == 5) {
      weight_format = grouped ? format::goidhw : format::oidhw;
    }
    tdesc_t weights_desc(dims_in, dtype, weight_format);

    convolution_transpose_forward comp(x_desc, weights_desc, tdesc_t(), y_desc,
        strides, dilates, padding_l, padding_r);
    return comp.dup_descriptor_of(query::weights_pd);
  }
};

struct convolution_transpose_backward_data : public computation,
      public utils::computation_cache<convolution_transpose_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& grady_desc, const tdesc_t& weights_desc, const tdesc_t& gradx_desc,
        const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm = algorithm::deconvolution_direct, padding_kind apadding_kind = padding_kind::zero)
        : hint_(gradx_desc, weights_desc, tdesc_t(), grady_desc, strides, dilates, padding_l, padding_r) {
      utils::validate_dims(strides, padding_l, padding_r);
      auto diff_src_any = gradx_desc.format_any();
      auto weights_any = weights_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      auto dilates_in = utils::get_compatible_dilates(dilates, grady_desc.ndims());
      mkldnn_deconvolution_desc_t data;
      error::wrap_c_api(mkldnn_dilated_deconvolution_backward_data_desc_init(
              &data, convert_to_c(aalgorithm), &diff_src_any, &weights_any, &diff_dst_any, &strides[0],
              &dilates_in[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward data descriptor");
      create_primitive_desc(data, hint_.get());
    }

   private:
    convolution_transpose_forward::descriptor hint_;
  };

 public:
  template <typename... Ts>
  convolution_transpose_backward_data(const tdesc_t& grady_desc, Ts&&... args) {
    descriptor backward_data_descriptor(grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor);
  }

  template <class alloc, typename... Ts>
  static void compute_impl(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims,
      tensor& gradx, Ts&&... args) {
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
    check_or_create_k(key, grady, weights, oihw_dims, gradx_dims, args...);
    fetch_or_create_m(comp, key, grady.get_descriptor(),
        is_iohw ? weight_desc : weights.get_descriptor(), result_desc, std::forward<Ts>(args)...);

    auto grady_in = comp.transform_input_cache<alloc>(0, grady);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights);

    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }

  template <class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims,
      tensor& gradx, const tdims_t& strides, const tdims_t& padding_l, const tdims_t& padding_r,
      const tdims_t& dilates = {1, 1}, algorithm aalgorithm = algorithm::deconvolution_direct,
      padding_kind apadding_kind = padding_kind::zero, const int group = 1) {
    auto weights_in = weights;
    weights_in.make_group(group, true);
    compute_impl<alloc>(grady, weights_in, gradx_dims, gradx, strides, dilates, padding_l,
        padding_r, aalgorithm, apadding_kind);
  }
};

struct convolution_transpose_backward_weights : public computation,
      public utils::computation_cache<convolution_transpose_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, const tdesc_t& grady_desc, const tdesc_t& gradw_desc,
        const tdesc_t& gradb_desc, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
        const tdims_t& padding_r, algorithm aalgorithm = algorithm::deconvolution_direct,
        padding_kind apadding_kind = padding_kind::zero)
        : hint_(x_desc, gradw_desc, gradb_desc, grady_desc, strides, dilates, padding_l, padding_r) {
      utils::validate_dims(strides, padding_l, padding_r);
      mkldnn_deconvolution_desc_t data;
      auto src_any = x_desc.format_any();
      auto diff_weights_any = gradw_desc.format_any();
      auto diff_bias_any = gradb_desc.format_any();
      auto diff_dst_any = grady_desc.format_any();
      auto dilates_in = utils::get_compatible_dilates(dilates, grady_desc.ndims());

      error::wrap_c_api(
          mkldnn_dilated_deconvolution_backward_weights_desc_init(&data, convert_to_c(aalgorithm),
              &src_any, &diff_weights_any, &diff_bias_any, &diff_dst_any, &strides[0],
              &dilates_in[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not create a deconvolution backward weights descriptor");
      create_primitive_desc(data, hint_.get());
    }

   private:
    convolution_transpose_forward::descriptor hint_;
  };

 public:
  template <typename... Ts>
  convolution_transpose_backward_weights(const tdesc_t& x_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor);
  }

  /*
   * This interface require MKL-DNN fixed
   * https://github.com/intel/mkl-dnn/commit/86f152b614c947b87633062a182c57775856a348
   */
  template <class alloc, bool with_gradb, typename... Ts>
  static void compute_impl(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gbias, Ts&&... args) {
    tdesc_t gradw_desc(gradw_dims, src.get_data_type());
    tdesc_t gradb_desc;
    if (with_gradb) {
      gradb_desc = {{grady.get_dim(1)}, src.get_data_type()};
    }

    key_t key;
    check_or_create_k(key, src, with_gradb, grady, gradw_dims, args...);
    fetch_or_create_m(comp, key, src.get_descriptor(), grady.get_descriptor(), gradw_desc,
        gradb_desc, std::forward<Ts>(args)...);

    auto src_in = comp.transform_input_cache<alloc>(0, src);
    auto grady_in = comp.transform_input_cache<alloc>(1, grady);
    gradw.reinit<alloc>(comp.expected_gradw_descriptor());

    if (with_gradb) {
      gbias.reinit<alloc>(comp.expected_gradb_descriptor());
      comp.execute(src_in, grady_in, gradw, gbias);
    } else {
      comp.execute(src_in, grady_in, gradw);
    }
  }

  template<class alloc = utils::allocator, bool with_gradb = true>
  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, const tdims_t& strides, const tdims_t& padding_l,
      const tdims_t& padding_r, const tdims_t& dilates = {1, 1}, algorithm aalgorithm = algorithm::deconvolution_direct,
      padding_kind apadding_kind = padding_kind::zero, const int group = 1) {
    auto gw_dims_in = gradw_dims;
    auto src_size = grady.ndims();
    auto grouped = IDEEP_IS_GROUPED(src_size, gradw_dims);
    if (group > 1 && !grouped) {
      tensor::group_dims(gw_dims_in, group, true);
    }
    compute_impl<alloc, with_gradb>(src, grady, gw_dims_in, gradw, gradb, strides, dilates, padding_l, padding_r,
        aalgorithm, apadding_kind);
    if (group > 1 && !grouped) {
      IDEEP_ENFORCE(group == gradw.get_dim(0), "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims[1] == group * gradw.get_dim(2), "invalid dim 2 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims.size() == gradw.ndims() - 1, "invalid ndim in grouped gradw");
      gradw.reshape<alloc>(gradw_dims);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, const tdims_t& strides, const tdims_t& padding_l,
      const tdims_t& padding_r, const tdims_t& dilates = {1, 1}, algorithm aalgorithm = algorithm::deconvolution_direct,
      padding_kind apadding_kind = padding_kind::zero, const int group=1) {
    static tensor dummy_gradb;
    compute<alloc, false>(src, grady, gradw_dims, gradw, dummy_gradb, strides, padding_l, padding_r,
        dilates, aalgorithm, apadding_kind, group);
  }
};

struct lrn_forward : public computation,
  public utils::computation_cache<lrn_forward> {
  struct descriptor : public descriptor_group {
    descriptor (const tdesc_t& x_desc, int local_size, float alpha, float beta, float k = 1.0,
        algorithm aalgorithm = algorithm::lrn_across_channels, prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_lrn_desc_t data;
      auto src_data = x_desc.get_mkldnn_memory_desc_t();
      error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data, mkldnn::convert_to_c(aprop_kind),
            convert_to_c(aalgorithm), src_data, local_size, alpha, beta, k),
          "could not create a lrn forward descriptor");
      create_primitive_desc(data);
    }
  };

public:
  template<typename ...Ts>
  lrn_forward(const tdesc_t& x_desc, Ts&&... args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  void execute(const tensor& src, const tensor& dst, const tensor& workspace) {
    computation::execute(src, dst, workspace);
  }

  void execute(const tensor& src, tensor& dst) {
    if (dst.has_extra())
      computation::execute(src, dst, *dst.get_extra());
    else
      computation::execute(src, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, tensor& dst, int local_size, float alpha,
      float beta, float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels,
      prop_kind aprop_kind = prop_kind::forward_training) {

    tdesc_t src_desc;
    scale_t src_scales(IDEEP_DEF_SCALE);
    if (src.has_scale()) {
      IDEEP_ENFORCE(src.get_data_type() != tdtype_t::f32, "Incorrect data type");
      IDEEP_ENFORCE(src.get_scale().size() == 1, "Invalid scale size");
      src_desc = {src.get_dims(), tdtype_t::f32};
      src_scales[0] /= src.get_scale()[0];
    } else {
      src_desc = src.get_descriptor();
      IDEEP_ENFORCE(src.get_data_type() == tdtype_t::f32, "Incorrect src data type");
    }

    check_or_create_k(key, src, local_size, alpha, beta, k, aalgorithm, aprop_kind);
    fetch_or_create_m(comp, key, src_desc, local_size, alpha, beta, k, aalgorithm, aprop_kind);

    bool with_workspace = aprop_kind == prop_kind::forward_training;
    auto src_in = comp.transform_input_cache<alloc>(0, src, {0, src_scales});

    if (dst != src) {
      dst.reinit<alloc>(comp.expected_dst_descriptor());
      if (with_workspace)
        dst.init_extra<alloc>(comp.expected_workspace_descriptor());
    }

    comp.execute(src_in, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, tensor& dst, int local_size, float alpha, float beta,
      float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels,
      prop_kind aprop_kind = prop_kind::forward_training) {
    key_t key;
    compute<alloc>(key, src, dst, local_size, alpha, beta, k, aalgorithm, aprop_kind);
  }
};

struct lrn_backward : public computation,
  public utils::computation_cache<lrn_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, const tdesc_t& gx_desc, int local_size, float alpha,
        float beta, float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels)
      : hint_(x_desc, local_size, alpha, beta, k, aalgorithm) {
      mkldnn_lrn_desc_t data;
      error::wrap_c_api(mkldnn_lrn_backward_desc_init(
            &data, convert_to_c(aalgorithm), gx_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), local_size, alpha, beta, k),
          "could not create a lrn backward descriptor");
      create_primitive_desc(data, hint_.get());
    }

  private:
    lrn_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  lrn_backward(const tdesc_t& x_desc, Ts&&... args) {
    descriptor backward_data_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& y, const tensor& gradx) {
    if (num_of_inputs() == 2)
      computation::execute(x, grady, gradx);
    else
      computation::execute(x, grady, *y.get_extra(), gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& x, const tensor& grady, const tensor& y, tensor& gradx,
      int local_size, float alpha, float beta, float k = 1.0,
      algorithm aalgorithm = algorithm::lrn_across_channels) {
    key_t key;
    check_or_create_k(key, x, local_size, alpha, beta, k, aalgorithm);
    fetch_or_create_m(comp, key, x.get_descriptor(),
        grady.get_descriptor(), local_size, alpha, beta, k, aalgorithm);

    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    comp.execute(x, grady, y, gradx);
  }
};

struct pooling_forward : public computation,
  public utils::computation_cache<pooling_forward> {
  struct descriptor : descriptor_group {
    descriptor() = default;
    descriptor(const tdesc_t& x_desc, const tdesc_t& y_desc, const tdims_t& strides,
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
      create_primitive_desc(data);
    }
  };
public:
  template<typename ...Ts>
  pooling_forward(const tdesc_t& x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  void execute(const tensor& src, const tensor& dst, const tensor& workspace) {
    computation::execute(src, dst, workspace);
  }

  void execute(const tensor& src, tensor& dst) {
    if (dst.has_extra())
      computation::execute(src, dst, *dst.get_extra());
    else
      computation::execute(src, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, const tdims_t& dst_dims, tensor& dst,
      const tdims_t& strides, const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r,
      algorithm aalgorithm, prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
    tdesc_t dst_desc(dst_dims, src.get_data_type());
    check_or_create_k(key, src, dst_dims, strides, kernel, padding_l, padding_r, aalgorithm,
        aprop_kind, apadding_kind);
    fetch_or_create_m(comp, key, src.get_descriptor(), dst_desc, strides, kernel, padding_l,
        padding_r, aalgorithm, aprop_kind, apadding_kind);

    bool with_workspace = true && aprop_kind == prop_kind::forward_training && aalgorithm == mkldnn::pooling_max;

    if (dst != src) {
      dst.reinit<alloc>(comp.expected_dst_descriptor());
      if (with_workspace)
        dst.init_extra<alloc>(comp.expected_workspace_descriptor());
      if (src.has_scale()) {
        dst.set_scale(src.get_scale());
      }
    }

    comp.execute(src, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tdims_t& dst_dims, tensor& dst, const tdims_t& strides,
      const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r, algorithm aalgorithm,
      prop_kind aprop_kind = prop_kind::forward, padding_kind apadding_kind = padding_kind::zero) {
    key_t key;
    compute<alloc>(key, src, dst_dims, dst, strides, kernel, padding_l, padding_r,
        aalgorithm, aprop_kind, apadding_kind);
  }
};

struct pooling_backward : public computation,
  public utils::computation_cache<pooling_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& gradx_desc, const tdesc_t& grady_desc, const tdims_t& strides,
        const tdims_t& kernel, const tdims_t& padding_l, const tdims_t& padding_r,
        algorithm aalgorithm, padding_kind apadding_kind = padding_kind::zero)
      : hint_(gradx_desc, grady_desc, strides, kernel, padding_l, padding_r, aalgorithm) {
      utils::validate_dims(strides, kernel, padding_l, padding_r);
      auto gradx_data = gradx_desc.format_any();
      mkldnn_pooling_desc_t data;
      error::wrap_c_api(mkldnn_pooling_backward_desc_init(
            &data, convert_to_c(aalgorithm), &gradx_data, grady_desc.get_mkldnn_memory_desc_t(),
            &strides[0], &kernel[0], &padding_l[0], &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
          "could not init a backward pooling descriptor");
      create_primitive_desc(data, hint_.get());
    }
  private:
    pooling_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  pooling_backward(const tdesc_t& gradx_desc, Ts &&...args) {
    descriptor backward_descriptor(gradx_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor);
  }

  void execute(const tensor& grady, const tensor& y, const tensor& gradx) {
    if (num_of_inputs() == 1)
      computation::execute(grady, gradx);
    else
      computation::execute(grady, *y.get_extra(), gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& y, const tensor& x, tensor& gradx,
      const tdims_t& strides, const tdims_t& kernel, const tdims_t& padding_l,
      const tdims_t& padding_r, algorithm aalgorithm, padding_kind apadding_kind = padding_kind::zero) {
    auto grady_in = grady;
    if (grady.get_internal_format() != x.get_internal_format()) {
      grady_in.init<alloc>({grady.get_dims(), grady.get_data_type(), x.get_internal_format()});
      treorder_t::compute(grady, grady_in);
    }

    key_t key;
    check_or_create_k(key, grady_in, x.get_dims(), strides, kernel, padding_l, padding_r,
        aalgorithm, apadding_kind);
    fetch_or_create_m(comp, key, x.get_descriptor(), grady_in.get_descriptor(),
        strides, kernel, padding_l, padding_r, aalgorithm, apadding_kind);

    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, y, gradx);
  }
};

struct eltwise_forward : public computation,
  public utils::computation_cache<eltwise_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, float alpha = 0.0, float beta = 0.0,
        algorithm alg_kind = algorithm::eltwise_relu, prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_eltwise_desc_t data;
      error::wrap_c_api(mkldnn_eltwise_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), mkldnn::convert_to_c(alg_kind),
            x_desc.get_mkldnn_memory_desc_t(), alpha, beta),
          "could not create a eltwise forward descriptor");
      create_primitive_desc(data);
    }
  };

public:
  template<typename ...Ts>
  eltwise_forward(const tdesc_t& x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, tensor& dst, algorithm aalgorithm = algorithm::eltwise_relu,
      prop_kind aprop_kind = prop_kind::forward, float alpha = 0.0, float beta = 0.0) {
    auto src_in = src;
    if (aalgorithm != algorithm::eltwise_relu && src.get_data_type() != tdtype_t::f32) {
      src_in.init<alloc>({src.get_dims(), tdtype_t::f32});
      IDEEP_ENFORCE(src.has_scale(), "Can not find scales");
      IDEEP_ENFORCE(src.get_scale().size() == 1, "Incorrect scale size");
      auto scale = IDEEP_DEF_SCALE;
      scale[0] /= src.get_scale()[0];
      treorder_t::compute(src, src_in, {0, scale});
    }

    check_or_create_k(key, src_in, alpha, beta, aalgorithm, aprop_kind);
    fetch_or_create_m(comp, key, src_in.get_descriptor(),
        alpha, beta, aalgorithm, aprop_kind);

    if (dst != src) {
      dst.reinit<alloc>(src_in.get_descriptor());
      if (src_in.has_scale()) dst.set_scale(src_in.get_scale());
    }

    comp.execute(src_in, dst);
    if (dst.has_scale() && aalgorithm == algorithm::eltwise_relu && dst.get_data_type() == tdtype_t::s8)
      dst.set_descriptor({dst.get_dims(), tdtype_t::u8, dst.get_internal_format()});
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, tensor& dst, algorithm aalgorithm = algorithm::eltwise_relu,
      prop_kind aprop_kind = prop_kind::forward, float alpha = 0.0, float beta = 0.0) {
    key_t key;
    compute<alloc>(key, src, dst, aalgorithm, aprop_kind, alpha, beta);
  }
};

struct eltwise_backward : public computation,
  public utils::computation_cache<eltwise_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& grady_desc, const tdesc_t& x_desc, float alpha = 0.0,
        float beta = 0.0, algorithm alg_kind = algorithm::eltwise_relu)
      : hint_(x_desc, alg_kind) {
      mkldnn_eltwise_desc_t data;
      error::wrap_c_api(mkldnn_eltwise_backward_desc_init(
            &data, mkldnn::convert_to_c(alg_kind), grady_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), static_cast<float>(alpha), static_cast<float>(beta)),
          "could not create a eltwise backward descriptor");
      create_primitive_desc(data, hint_.get());
    }
  private:
    eltwise_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  eltwise_backward(const tdesc_t& grady_desc, Ts &&...args) {
    descriptor backward_descriptor(grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor);
  }

  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& grady, tensor& gradx,
      algorithm aalgorithm = algorithm::eltwise_relu, float alpha = 0.0, float beta = 0.0) {
    // if grady is from outside, make it ours
    tensor grady_in = grady;
    if (grady.get_internal_format() != src.get_internal_format()) {
      grady_in.init<alloc>(src.get_descriptor());
      treorder_t::compute(grady, grady_in);
      if (grady == gradx) {
        gradx.set_descriptor(grady_in.get_descriptor());
      }
    }

    key_t key;
    check_or_create_k(key, src, alpha, beta, aalgorithm);
    fetch_or_create_m(comp, key, grady_in.get_descriptor(),
        src.get_descriptor(), alpha, beta, aalgorithm);

    if (grady != gradx)
      gradx.reinit<alloc>(comp.expected_gradx_descriptor());

    comp.execute(src, grady_in, gradx);
  }
};

struct channel_shuffle_forward: public computation,
  public utils::computation_cache<channel_shuffle_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_desc, const int group_size, const int axis = 1,
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_shuffle_desc_t data;
      error::wrap_c_api(mkldnn_shuffle_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(), axis, group_size),
          "could not create a shuffle forward descriptor");
      create_primitive_desc(data);
    }
  };
public:
  template<typename ...Ts>
  channel_shuffle_forward(const tdesc_t& x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, tensor& dst, const int group, const int axis = 1,
      prop_kind aprop_kind = prop_kind::forward) {
    IDEEP_ENFORCE(src.get_dim(axis) % group == 0, "Invalid channel and group");
    IDEEP_ENFORCE(src.get_data_type() == tdtype_t::f32, "invalid data type");
    auto group_size = src.get_dim(axis) / group;

    key_t key;
    check_or_create_k(key, src, group_size, axis, aprop_kind);
    fetch_or_create_m(comp, key, src.get_descriptor(), group_size, axis, aprop_kind);

    auto src_in = comp.transform_input_cache<alloc>(0, src);
    if (dst != src) {
      dst.reinit<alloc>(comp.expected_dst_descriptor());
      if (src.has_scale()) dst.set_scale(src.get_scale());
    }

    comp.execute(src_in, dst);
  }
};

struct channel_shuffle_backward : public computation,
  public utils::computation_cache<channel_shuffle_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& grady_desc, const int group_size, const int axis = 1) {
      mkldnn_shuffle_desc_t data;
      error::wrap_c_api(mkldnn_shuffle_backward_desc_init(
            &data, grady_desc.get_mkldnn_memory_desc_t(), static_cast<int>(axis), static_cast<int>(group_size)),
          "could not create a shuffle backward descriptor");
      create_primitive_desc(data);
    }
  };
public:
  template<typename ...Ts>
  channel_shuffle_backward(const tdesc_t& grady_desc, Ts &&...args) {
    descriptor backward_descriptor(grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& grady, tensor& gradx, const int group, const int axis = 1) {
    auto group_size = grady.get_dim(axis) / group;
    key_t key;
    check_or_create_k(key, grady, group_size, axis);
    fetch_or_create_m(comp, key, grady.get_descriptor(), group_size, axis);

    auto grady_in = comp.transform_input_cache<alloc>(0, grady);
    if (gradx != grady)
      gradx.reinit<alloc>(comp.expected_gradx_descriptor());

    comp.execute(grady_in, gradx);
  }
};

struct concat : public computation,
  public utils::computation_cache<concat> {
  struct descriptor : public descriptor_group {
    descriptor(int concat_dimension, const std::vector<tdesc_t>& inputs) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = tdesc_t::convert_to_c(inputs);
      error::wrap_c_api(mkldnn_concat_primitive_desc_create(
            &result, nullptr, (int)c_api_inputs.size(), concat_dimension, &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }

    descriptor(int concat_dimension, const std::vector<tdesc_t>& inputs, const tdesc_t out_desc) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = tdesc_t::convert_to_c(inputs);
      error::wrap_c_api(mkldnn_concat_primitive_desc_create(
            &result, out_desc.get_mkldnn_memory_desc_t(), (int)c_api_inputs.size(), concat_dimension, &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }
  };
public:

  concat(int concat_dimension, const std::vector<tdesc_t>& inputs) {
    descriptor forward_descriptor (concat_dimension, inputs);
    computation::init(forward_descriptor, inputs);
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, std::vector<tensor>& inputs, int axis, tensor& output) {
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

    check_or_create_k(key, inputs_dt, inputs_dims, inputs_format, axis);

    // FIXME: currently align all inputs format with first one
    std::vector<tensor> inputs_in;
    inputs_in.push_back(inputs[0]);
    for (int i = 1; i < tdesc.size(); i++) {
      auto src_in = inputs[i];
      if (inputs_format[i] != inputs_format[0]) {
        src_in.init<alloc>({inputs_dims[i], inputs_dt[i], inputs_format[0]});
        treorder_t::compute(inputs[i], src_in);
      }
      inputs_in.push_back(src_in);
      tdesc[i] = src_in.get_descriptor();
    }

    fetch_or_create_m(comp, key, axis, tdesc);
    output.reinit<alloc>(comp.expected_dst_descriptor());

    comp.execute(inputs_in, output);
  }

  template<class alloc = utils::allocator>
  static void compute(std::vector<tensor>& inputs, int axis, tensor& output) {
    key_t key;
    compute<alloc>(key, inputs, axis, output);
  }

  template<class alloc = utils::allocator>
  static std::vector<int32_t> compute(std::vector<tensor>& inputs, int axis, bool add_axis, tensor& dst) {
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
      dst.reinit<alloc>({dst_dims, dst_data_type});
    else
      dst.reinit<alloc>({dst_dims, dst_data_type, dst_format});
    if (dst_data_type != tdtype_t::f32)
      dst.set_scale(min_scale);

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
              input_fp.reinit<alloc>({inputs[i].get_dims(), dst_data_type, inputs[i].get_internal_format()});
              treorder_t::compute(inputs[i], input_fp, {0, scales});
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
        treorder_t::compute({in_desc, inputs[i].get_data_handle()}, in_dims, offset_dims, dst, {0, scales});
      } else {
        treorder_t::compute(inputs[i], inputs[i].get_dims(), offset_dims, dst, {0, scales});
      }
      offset_dims[axis] += axis_info[i];
    }

    return axis_info;
  }
};

struct softmax_forward : public computation,
  public utils::computation_cache<softmax_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, int softmax_axis,
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_softmax_desc_t data;
      error::wrap_c_api(mkldnn_softmax_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind),
            x_desc.get_mkldnn_memory_desc_t(), softmax_axis),
          "could not create a softmax forward descriptor");
      create_primitive_desc(data);
    }
  };

public:
  template<typename ...Ts>
  softmax_forward(const tdesc_t& x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, tensor& dst,
      int softmax_axis, prop_kind aprop_kind = prop_kind::forward) {
    IDEEP_ENFORCE(!src.has_scale(), "Unsupported lowp input");
    check_or_create_k(key, src, softmax_axis, aprop_kind);
    fetch_or_create_m(comp, key, src.get_descriptor(), softmax_axis, aprop_kind);

    if (dst != src) {
      dst.reinit<alloc>(src.get_descriptor());
    }

    comp.execute(src, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, tensor& dst, int softmax_axis, prop_kind aprop_kind = prop_kind::forward) {
    key_t key;
    compute<alloc>(key, src, dst, softmax_axis, aprop_kind);
  }

};

struct softmax_backward : public computation,
  public utils::computation_cache<softmax_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& grady_desc, const tdesc_t& y_desc, int softmax_axis)
      : hint_(y_desc, softmax_axis) {
      mkldnn_softmax_desc_t data;
      error::wrap_c_api(mkldnn_softmax_backward_desc_init(
            &data, grady_desc.get_mkldnn_memory_desc_t(),
            y_desc.get_mkldnn_memory_desc_t(), softmax_axis),
          "could not create a softmax backward descriptor");
      create_primitive_desc(data, hint_.get());
    }
  private:
    softmax_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  softmax_backward(const tdesc_t& grady_desc, Ts &&...args) {
    descriptor backward_descriptor(grady_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& y, const tensor& grady, tensor& gradx, int softmax_axis) {
    // if grady is from outside, make it ours
    tensor grady_in = grady;
    if (grady.get_internal_format() != y.get_internal_format()) {
      grady_in.init<alloc>(y.get_descriptor());
      treorder_t::compute(grady, grady_in);
      if (grady == gradx) {
        gradx.set_descriptor(grady_in.get_descriptor());
      }
    }

    key_t key;
    check_or_create_k(key, y, softmax_axis);
    fetch_or_create_m(comp, key, grady_in.get_descriptor(),
        y.get_descriptor(), softmax_axis);

    if (grady != gradx)
      gradx.reinit<alloc>(comp.expected_gradx_descriptor());

    comp.execute(y, grady_in, gradx);
  }

};

struct batch_norm_forward_base : public computation {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_desc, float epsilon, unsigned flags, prop_kind aprop_kind) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(mkldnn_batch_normalization_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(), epsilon, flags),
          "could not create a batch normalization forward descriptor");
      create_primitive_desc(data);
    }

    descriptor(const tdesc_t& src_desc, float epsilon, attr_t attr, unsigned flags, prop_kind aprop_kind) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(mkldnn_batch_normalization_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(), epsilon, flags),
          "could not create a batch normalization forward descriptor");
      create_primitive_desc_v2(data, attr);
    }
  };
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
  template<class alloc = utils::allocator>
  batch_normalization_forward_inference(const tdesc_t& src_desc, float epsilon,
      unsigned flag = batch_normalization_flag::use_global_stats | batch_normalization_flag::use_scale_shift) {
    descriptor batch_norm_forward(src_desc, epsilon, flag, prop_kind::forward_scoring);
    weights_.init<alloc>(batch_norm_forward.expected_descriptor_of(query::weights_pd));
    computation::init(batch_norm_forward);
  }

  /// More functionality in this interface
  void execute(const tensor& src, const tensor& scale, const tensor& shift, const tensor& dst) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    std::memcpy((char*)weights_.get_data_handle() + scale.get_size(), shift.get_data_handle(), shift.get_size());
    computation::execute(src, weights_, dst);
  }

  /// More functionality in this interface
  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, const tensor& dst) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    std::memcpy((char*)weights_.get_data_handle() + scale.get_size(), shift.get_data_handle(), shift.get_size());
    computation::execute(src, mean, variance, weights_, dst);
  }

  // Inplace support?
  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, const tensor& scale,
      const tensor& shift, tensor& dst, float epsilon) {
    auto src_in = src;
    if (src.get_data_type() != tdtype_t::f32) {
      src_in.init<alloc>({src.get_dims(), tdtype_t::f32});
      IDEEP_ENFORCE(src.has_scale(), "Can not find scales");
      auto src_scales = IDEEP_DEF_SCALE;
      src_scales[0] /= src.get_scale()[0];
      treorder_t::compute(src, src_in, {0, src_scales});
    }

    check_or_create_k(key, src_in, epsilon, 3);
    fetch_or_create_m(comp, key, src_in.get_descriptor(), epsilon,
        batch_normalization_flag::use_scale_shift);

    if (dst != src)
      dst.reinit<alloc>(comp.expected_dst_descriptor());
    comp.execute(src_in, scale, shift, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, tensor& dst, float epsilon) {
    auto src_in = src;
    if (src.get_data_type() != tdtype_t::f32) {
      src_in.init<alloc>({src.get_dims(), tdtype_t::f32});
      IDEEP_ENFORCE(src.has_scale(), "Can not find scales");
      auto src_scales = IDEEP_DEF_SCALE;
      src_scales[0] /= src.get_scale()[0];
      treorder_t::compute(src, src_in, {0, src_scales});
    }

    check_or_create_k(key, src_in, epsilon, 5);
    fetch_or_create_m(comp, key, src_in.get_descriptor(), epsilon);

    if (dst != src) {
      dst.reinit<alloc>(comp.expected_dst_descriptor());
    }
    comp.execute(src_in, mean, variance, scale, shift, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& scale,
      const tensor& shift, tensor& dst, float epsilon) {
    key_t key;
    compute<alloc>(key, src, scale, shift, dst, epsilon);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, tensor& dst, float epsilon) {
    key_t key;
    compute<alloc>(key, src, mean, variance, scale, shift, dst, epsilon);
  }

private:
  tensor weights_;
};

struct batch_normalization_forward_training : public batch_norm_forward_base,
  public utils::computation_cache<batch_normalization_forward_training> {
  float get_epsilon() const {
    const mkldnn_batch_normalization_desc_t* p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d), 0, (void*)&p_desc),
      "could not query batch normalization descriptor");
    return p_desc->batch_norm_epsilon;
  }

public:
  template<class alloc = utils::allocator>
  batch_normalization_forward_training(const tdesc_t& src_desc, const tdesc_t& scale,
      const tdesc_t& shift, float momentum, float epsilon,
      unsigned flags = batch_normalization_flag::use_scale_shift) {
    // IDEEP_ENFORCE(scale.ndims() == 1 && shift.ndims() == 1, "Incorrect dims");
    descriptor batch_norm_forward(src_desc, epsilon, flags, prop_kind::forward_training);
    computation::init(batch_norm_forward);

    // We borrown scale and bias for the shape of mean and variance
    weights_.init<alloc>(batch_norm_forward.expected_descriptor_of(query::weights_pd));
    sum_.init({momentum, 1.f - momentum}, {scale, shift});
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
    std::memcpy((char*)weights_.get_data_handle() + scale.get_size(),
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

  tdesc_t expected_statistic_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 1);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& scale, const tensor& shift,
      tensor& dst, tensor& mean, tensor& variance, float momentum, float epsilon) {
    key_t key;
    check_or_create_k(key, src, epsilon);
    fetch_or_create_m(comp, key, src.get_descriptor(),
        scale.get_descriptor(), shift.get_descriptor(), momentum, epsilon);
    comp.eps = epsilon;

    dst.reinit<alloc>(comp.expected_dst_descriptor());
    mean.reinit<alloc>(comp.expected_statistic_descriptor());
    variance.reinit<alloc>(comp.expected_statistic_descriptor());

    comp.execute(src, scale, shift, dst, mean, variance);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& scale, const tensor& shift, tensor& dst,
      tensor& mean, tensor& variance, tensor& running_mean, tensor& running_var, float momentum, float epsilon) {
    key_t key;
    check_or_create_k(key, src, epsilon);
    fetch_or_create_m(comp, key, src.get_descriptor(),
        scale.get_descriptor(), shift.get_descriptor(), momentum, epsilon);

    // TODO: Substitue running statistics calculation with lighter version
    dst.reinit<alloc>(comp.expected_dst_descriptor());
    mean.reinit<alloc>(comp.expected_statistic_descriptor());
    variance.reinit<alloc>(comp.expected_statistic_descriptor());
    if (running_mean.get_descriptor() != comp.expected_statistic_descriptor()){
      running_mean.reinit<alloc>(comp.expected_statistic_descriptor());
      std::memset(running_mean.get_data_handle(), 0, running_mean.get_size());
    }
    if (running_var.get_descriptor() != comp.expected_statistic_descriptor()){
      running_var.reinit<alloc>(comp.expected_statistic_descriptor());
      auto p = static_cast<float*>(running_var.get_data_handle());
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
    descriptor(const tdesc_t& gradx_desc, const tdesc_t& x_desc,
        float epsilon, unsigned flags, prop_kind aprop_kind)
      : hint_(x_desc, epsilon, flags, prop_kind::forward_training) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(mkldnn_batch_normalization_backward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), gradx_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), static_cast<float>(epsilon), flags),
          "could not create a batch normalization backward descriptor");
      create_primitive_desc(data, hint_.get());
    }
  private:
    batch_normalization_forward_training::descriptor hint_;
  };

  float get_epsilon() const {
    const mkldnn_batch_normalization_desc_t* p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d), 0, (void*)&p_desc),
      "could not query batch normalization descriptor");
    return p_desc->batch_norm_epsilon;
  }

public:
  prop_kind get_prop_kind() const {
    const mkldnn_batch_normalization_desc_t* p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d), 0, (void*)&p_desc),
      "could not query batch normalization descriptor");
    return static_cast<prop_kind>(p_desc->prop_kind);
  }

  template<class alloc = utils::allocator>
  batch_normalization_backward(const tdesc_t& gradx_desc, const tdesc_t& src_desc, float epsilon,
      unsigned flags = batch_normalization_flag::use_scale_shift, prop_kind aprop_kind=prop_kind::backward) {
    descriptor batch_norm_backward(gradx_desc, src_desc, epsilon, flags, aprop_kind);
    computation::init(batch_norm_backward);
    weights_.init<alloc>(batch_norm_backward.expected_descriptor_of(query::weights_pd));
    grad_scale_shift_.init<alloc>(batch_norm_backward.expected_descriptor_of(query::weights_pd));
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx, const tensor& gradw) {
    // We can sure that only scale is matter at this place
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx, gradw);
  }

  template<class alloc>
  void execute(const tensor& src, const tensor& mean, const tensor& variance, const tensor& grady,
      const tensor& scale, const tensor& gradx, const tensor& gradw, const tensor& grad_shift) {
    // protect API integraty, should we use solid check instead of assert?
    IDEEP_ENFORCE(get_prop_kind() == prop_kind::backward, "Incorrect prop_kind");
    // We can sure that only scale is matter at this place
    // And single thread of memcpy should be fast enough
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());

    key_t key;
    check_or_create_k(key, src, get_epsilon());
    fetch_or_create_m(comp, key, src.get_descriptor(), src.get_descriptor(), get_epsilon());
    grad_scale_shift_.reinit<alloc>(comp.expected_gradw_descriptor());

    computation::execute(src, mean, variance, grady, weights_, gradx, grad_scale_shift_);
    std::memcpy(gradw.get_data_handle(), (char*)grad_scale_shift_.get_data_handle(), gradw.get_size());
    std::memcpy(grad_shift.get_data_handle(),
        (char*)grad_scale_shift_.get_data_handle() + gradw.get_size(), grad_shift.get_size());
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx) {
    IDEEP_ENFORCE(get_prop_kind() == prop_kind::backward_data, "Incorrect prop_kind");
    std::memcpy(weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, tensor& gradx, tensor& gradw, float epsilon) {
    key_t key;
    check_or_create_k(key, src, grady, epsilon);
    fetch_or_create_m(comp, key, src.get_descriptor(), src.get_descriptor(), epsilon);

    auto grady_in = comp.transform_input_cache<alloc>(3, grady);
    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    gradw.reinit<alloc>(comp.expected_gradw_descriptor());

    comp.execute(src, mean, variance, grady_in, scale, gradx, gradw);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& mean,
      const tensor& variance, const tensor& grady, const tensor& scale,
      tensor& gradx, tensor& grad_scale, tensor& grad_shift, float epsilon) {
    key_t key;
    check_or_create_k(key, src, grady, epsilon);
    fetch_or_create_m(comp, key, src.get_descriptor(), src.get_descriptor(), epsilon);

    auto grady_in = comp.transform_input_cache<alloc>(3, grady);
    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    grad_scale.reinit<alloc>(mean.get_descriptor());
    grad_shift.reinit<alloc>(mean.get_descriptor());

    comp.execute<alloc>(src, mean, variance, grady_in, scale, gradx, grad_scale, grad_shift);
  }

private:
  tensor weights_;
  tensor grad_scale_shift_;
};

struct inner_product_forward: public computation,
  public utils::computation_cache<inner_product_forward> {
  struct descriptor: public descriptor_group {
    descriptor(const tdesc_t& src_desc, const tdesc_t& weights_desc, const tdesc_t& bias_desc,
        const tdesc_t& dst_desc, const attr_t& attr = attr_t(),
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_inner_product_desc_t data;
      mkldnn_memory_desc_t weights_data;
      auto src_data = src_desc.format_any();
      if (weights_desc.get_data_type() == tdtype_t::s8 ||
          weights_desc.get_data_type() == tdtype_t::u8)
        weights_data = weights_desc.format_any();
      else
        weights_data = *weights_desc.get_mkldnn_memory_desc_t();
      auto bias_data = bias_desc.format_any();
      auto dst_data = dst_desc.format_any();
      error::wrap_c_api(mkldnn_inner_product_forward_desc_init(
            &data, mkldnn::convert_to_c(aprop_kind), &src_data, &weights_data, &bias_data, &dst_data),
          "could not create a inner product forward descriptor");
      create_primitive_desc_v2(data, attr);
    }
  };

 public:
  template<typename ...Ts>
  inner_product_forward(const tdesc_t& src_desc, Ts&&... args) {
    descriptor forward_descriptor(src_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template <class alloc, bool with_bias>
  static void compute_impl(inner_product_forward& comp, const tensor& src,
      const tensor& weights, const tensor& bias, tensor& dst) {
    auto src_in = comp.transform_input_cache<alloc>(0, src);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights.as_weights());
    if (comp.dst_exp_desc_) {
      dst.reinit<alloc>(*comp.dst_exp_desc_);
    }
    if (comp.dst_scales_) {
      dst.set_scale(*comp.dst_scales_);
    }

    if (with_bias) {
      auto bias_in = comp.transform_input_cache<alloc>(2, bias);
      comp.execute(src_in, weights_in, bias_in, dst);
    } else {
      comp.execute(src_in, weights_in, dst);
    }

    if (comp.dst_u8_desc_) {
      dst.set_descriptor(*comp.dst_u8_desc_);
    }
  }

  template <class alloc, bool with_bias, typename ...Ts>
  static inline void compute_impl(key_t& key, const tensor& src, const tensor& weights, const tensor& bias,
      tensor& dst,  const scale_t& src_scales,
      const scale_t& weights_scales, const scale_t& dst_scales, const attr_t& attr,
      const lowp_kind alowp_kind, Ts&&... args) {
    auto& post_ops = attr.get_post_ops();
    tdesc_t src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;

    scale_t dst_scales_in;
    auto dst_data_type = tdtype_t::f32;
    tensor::dims dst_dims;

    auto weights_scales_in = weights.has_scale() ? weights.get_scale() : weights_scales;

    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(alowp_kind == LOWP_U8S8 || alowp_kind == LOWP_S8S8, "Unsupported lowp kind");

      auto src_scales_in = src.has_scale() ? src.get_scale() : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);

      src_desc = {src.get_dims(), alowp_kind == LOWP_U8S8 ? tdtype_t::u8 : tdtype_t::s8};
      if (src.get_data_type() == tdtype_t::f32) {
        src_attr = {0 , src_scales_in};
      }

      dst_dims = {src_desc.get_dim(0), weights.get_dim(0)};
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;

      weights_desc = {weights.get_dims(), tdtype_t::s8};
      if (weights.get_data_type() == tdtype_t::f32) {
        weights_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), weights_scales_in};
      }

      // determine dst data type
      if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = tdtype_t::f32;
      } else if (post_ops.non_negitive_output()){
        dst_data_type = tdtype_t::u8;
      } else {
        dst_data_type = tdtype_t::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = (dst_scales.empty() || dst_data_type == tdtype_t::f32) ? IDEEP_DEF_SCALE : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
      }
      op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), op_scales);
      op_attr.set_int_output_round_mode(round_mode::round_nearest);

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
      dst_dims = {src_desc.get_dim(0), weights.get_dim(0)};
      weights_desc = weights.get_descriptor();
      IDEEP_ENFORCE(weights.get_data_type() == tdtype_t::f32, "Incorrect data type in weights");
      if (with_bias) {
        IDEEP_ENFORCE(bias.get_data_type() == tdtype_t::f32, "Incorrect data type in bias");
        bias_desc = bias.get_descriptor();
      }
    }

    auto dst_format = engine::default_format(dst_dims.size());
    tdesc_t dst_desc_in(dst_dims, dst_data_type, dst_format);

    check_or_create_k(key, src, weights, with_bias, op_attr, src_scales, dst_scales, args...);
    fetch_or_create_m(comp, key, src_desc, weights_desc, bias_desc, dst_desc_in, op_attr,
        std::forward<Ts>(args)...);

    auto src_in = comp.transform_input_cache<alloc>(0, src, src_attr);
    auto weights_in = comp.transform_input_cache<alloc>(1, weights.as_weights(), weights_attr);

    auto dst_desc = comp.expected_dst_descriptor();
    if (dst.get_descriptor() != dst_desc) {
      comp.dst_exp_desc_.reset(new tdesc_t(dst_desc));
      dst.reinit<alloc>(dst_desc);
    }

    if (!dst_scales.empty() && dst_data_type != tdtype_t::f32) {
      dst.set_scale(dst_scales_in);
      comp.dst_scales_.reset(new scale_t(dst_scales_in));
    }

    if (with_bias) {
      auto bias_in = comp.transform_input_cache<alloc>(2, bias, bias_attr);
      comp.execute(src_in, weights_in, bias_in, dst);
    } else {
      comp.execute(src_in, weights_in, dst);
    }

    if (post_ops.non_negitive_output() && dst.get_data_type() == tdtype_t::s8) {
      tdesc_t dst_u8_desc { dst.get_dims(), tdtype_t::u8, dst.get_internal_format()};
      dst.set_descriptor(dst_u8_desc);
      comp.dst_u8_desc_ = std::make_shared<tdesc_t>(dst_u8_desc);
    }

    update(comp, it);
  }

  template<class alloc = utils::allocator, bool with_bias=true>
  static inline void compute(key_t& key, const tensor& src, const tensor& weights, const tensor& bias, tensor& dst,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(), const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(), prop_kind aprop_kind = prop_kind::forward, const lowp_kind alowp_kind = LOWP_U8S8) {
    auto weights_in = weights.as_weights();
    auto src_in = src;
    auto sdim_num = src.ndims();
    auto wdim_num = weights.ndims();

    if (sdim_num != wdim_num) {
      auto ndims = src.is_public_format() ? wdim_num : sdim_num;
      if (ndims != sdim_num) {
        auto new_dims = weights.get_dims();
        new_dims[0] = src_in.get_dim(0);
        src_in.reshape<alloc>(new_dims);
      } else if (ndims != wdim_num) {
        auto new_dims = src.get_dims();
        new_dims[0] = weights_in.get_dim(0);
        weights_in.reshape<alloc>(new_dims);
        weights_in = weights_in.as_weights();
      }
    }
    IDEEP_ENFORCE(src_in.ndims() == weights_in.ndims(), "Invalid dims in src or weights");

    auto it = key.empty() ? end() : find(key);
    if (it != end()) {
      compute_impl<alloc, with_bias>(fetch(it), src_in, weights_in, bias, dst);
    } else {
      compute_impl<alloc, with_bias>(key, src_in, weights_in, bias, dst, src_scales,
          weights_scales, dst_scales, attr, alowp_kind, aprop_kind);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(key_t& key, const tensor& src, const tensor& weights, tensor& dst,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(), const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(), prop_kind aprop_kind = prop_kind::forward, const lowp_kind alowp_kind = LOWP_U8S8) {
    static tensor dummy_bias;
    compute<alloc, false>(key, src, weights, dummy_bias, dst, src_scales, weights_scales, dst_scales, attr, aprop_kind, alowp_kind);
  }

  template<class alloc = utils::allocator, bool with_bias=true>
  static inline void compute(const tensor& src, const tensor& weights, const tensor& bias, tensor& dst,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(), const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(), prop_kind aprop_kind = prop_kind::forward, const lowp_kind alowp_kind = LOWP_U8S8) {
    key_t key;
    compute<alloc, with_bias>(key, src, weights, bias, dst, src_scales, weights_scales, dst_scales, attr, aprop_kind, alowp_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& weights, tensor& dst,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(), const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(), prop_kind aprop_kind = prop_kind::forward, const lowp_kind alowp_kind = LOWP_U8S8) {
    static tensor dummy_bias;
    compute<alloc, false>(src, weights, dummy_bias, dst, src_scales, weights_scales, dst_scales, attr, aprop_kind, alowp_kind);
  }

  static tdesc_t expected_weights_descriptor(const tdims_t& weights_dims, tdtype_t dtype = tdtype_t::f32,
      tdtype_t x_dtype = tdtype_t::f32) {
    auto x_dims = weights_dims;
    x_dims[0] = 1;
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();
    auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(), "Invalid dims for data and weights");
    tdesc_t x_desc(x_dims, x_dtype, ndims == 2 ? format::nc : format::nchw);
    tdesc_t y_desc(y_dims, y_dtype, format::nc);
    tdesc_t weights_desc(weights_dims, dtype, ndims == 2 ? format::oi : format::oihw);

    inner_product_forward comp(x_desc, weights_desc, tdesc_t(), y_desc);
    return comp.dup_descriptor_of(query::weights_pd);
  }

private:
  std::shared_ptr<tdesc_t> dst_exp_desc_;
  std::shared_ptr<tdesc_t> dst_u8_desc_;
  std::shared_ptr<scale_t> dst_scales_;
};

// TODO: parameter sequence adjust?
struct inner_product_backward_data: public computation,
  public utils::computation_cache<inner_product_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& gradx_desc, const tdesc_t& weights_desc, const tdesc_t& grady_desc)
      : hint_(gradx_desc, weights_desc, tdesc_t(), grady_desc) {
      auto diff_src_data = gradx_desc.format_any();
      auto weights_data = weights_desc.get_mkldnn_memory_desc_t();
      auto diff_dst_data = grady_desc.format_any();
      mkldnn_inner_product_desc_t data;
      error::wrap_c_api(mkldnn_inner_product_backward_data_desc_init(
            &data, &diff_src_data, weights_data, &diff_dst_data),
          "could not create a inner product backward data descriptor");
      create_primitive_desc(data, hint_.get());
    }
  private:
    inner_product_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  inner_product_backward_data(const tdesc_t& gradx_desc, Ts&&... args) {
    descriptor backward_data_descriptor(gradx_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims, tensor& gradx) {
    auto weights_in = weights.as_weights();

    if (gradx_dims.size() != weights_in.ndims()) {
      auto new_dims = gradx_dims;
      new_dims[0] = weights_in.get_dim(0);
      weights_in.reshape<alloc>(new_dims);
      weights_in = weights_in.as_weights();
    }

    IDEEP_ENFORCE(gradx_dims.size() == weights_in.ndims(), "Invalid dims in src or weights");

    tdesc_t gradx_desc(gradx_dims, grady.get_data_type());

    key_t key;
    check_or_create_k(key, grady, weights_in, gradx_dims);
    fetch_or_create_m(comp, key, gradx_desc, weights_in.get_descriptor(), grady.get_descriptor());

    auto grady_in = comp.transform_input_cache<alloc>(0, grady);
    weights_in = comp.transform_input_cache<alloc>(1, weights_in);
    gradx.reinit<alloc>(comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }
};

struct inner_product_backward_weights : public computation,
  public utils::computation_cache<inner_product_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& x_desc, const tdesc_t& gradw_desc, const tdesc_t& gradb_desc, const tdesc_t& grady_desc)
      : hint_(x_desc, gradw_desc, gradb_desc, grady_desc) {
      mkldnn_inner_product_desc_t data;
      auto src_data = x_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      auto diff_weights_data = gradw_desc.format_any();
      auto diff_bias_data = gradb_desc.format_any();
      error::wrap_c_api(mkldnn_inner_product_backward_weights_desc_init(
            &data, &src_data, &diff_weights_data, &diff_bias_data, &diff_dst_data),
          "could not create a inner product backward weights descriptor");
      create_primitive_desc(data, hint_.get());
    }

  private:
    inner_product_forward::descriptor hint_;
  };

public:
  template<typename ...Ts>
  inner_product_backward_weights(const tdesc_t& x_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor);
  }

  template<class alloc = utils::allocator, bool with_gradb = true>
  static void compute(const tensor& x, const tensor& grady, tensor& gradw, tensor& gradb) {
    auto gradw_dims = x.get_dims();
    gradw_dims[0] = grady.get_dim(1);

    tdesc_t gradw_desc = tdesc_t(gradw_dims, x.get_data_type()).as_weights_format();

    tdesc_t gradb_desc;
    if (with_gradb) {
      gradb_desc = {{grady.get_dim(1)}, x.get_data_type()};
    }

    key_t key;
    check_or_create_k(key, x, gradw_dims, with_gradb, grady);
    fetch_or_create_m(comp, key, x.get_descriptor(), gradw_desc, gradb_desc, grady.get_descriptor());

    auto x_in = comp.transform_input_cache<alloc>(0, x);
    auto grady_in = comp.transform_input_cache<alloc>(1, grady);
    gradw.reinit<alloc>(comp.expected_gradw_descriptor());

    if (with_gradb) {
      gradb.reinit<alloc>(comp.expected_gradb_descriptor());
      comp.execute(x_in, grady_in, gradw, gradb);
    } else {
      comp.execute(x_in, grady_in, gradw);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& x, const tensor& grady, tensor& gradw) {
    static tensor dummy_gradb;
    compute<alloc, false>(x, grady, gradw, dummy_gradb);
  }
};

struct dropout_forward {
public:
  template<class T, class alloc = utils::allocator>
  static void compute_impl(const tensor& src, float ratio, tensor& dst, tensor& mask) {
    dropout_forward comp;
    mask.reinit<alloc>(src.get_descriptor());
    dst.reinit<alloc>(src.get_descriptor());
    if (src.has_scale()) dst.set_scale(src.get_scale());

    const auto scale = 1.0 / (1.0 - ratio);
    const auto size = src.get_size() / sizeof(T);
    std::unique_ptr<int[]> bernouli_nums(new int[size]);
    utils::bernoulli_generate(size, 1.0 - ratio, bernouli_nums.get());

    const auto src_data = static_cast<T*>(src.get_data_handle());
    const auto mask_data = static_cast<T*>(mask.get_data_handle());
    const auto dst_data = static_cast<T*>(dst.get_data_handle());
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
    for (auto i = 0; i < size; i++) {
      mask_data[i] = bernouli_nums[i] * scale;
      dst_data[i] = mask_data[i] * src_data[i];
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, float ratio,
      tensor& dst, tensor& mask) {
    switch(src.get_data_type()) {
    case tdtype_t::f32:
      compute_impl<float, alloc>(src, ratio, dst, mask);
      break;
    case tdtype_t::s32:
      compute_impl<int32_t, alloc>(src, ratio, dst, mask);
      break;
    case tdtype_t::s16:
      compute_impl<int16_t, alloc>(src, ratio, dst, mask);
      break;
    case tdtype_t::s8:
      compute_impl<int8_t, alloc>(src, ratio, dst, mask);
      break;
    case tdtype_t::u8:
      compute_impl<uint8_t, alloc>(src, ratio, dst, mask);
      break;
    default:
      throw error(mkldnn_invalid_arguments, "Unsupported mkldnn data type!");
    }
  }
};

struct dropout_backward {
public:
  template<class T, class alloc = utils::allocator>
  static void compute_impl(const tensor& mask, const tensor& gy, tensor& gx) {
    dropout_backward comp;
    gx.reinit<alloc>(gy.get_descriptor());

    const auto size = mask.get_size() /  sizeof(T);
    const auto mask_data = static_cast<T*>(mask.get_data_handle());
    const auto gy_data = static_cast<T*>(gy.get_data_handle());
    const auto gx_data = static_cast<T*>(gx.get_data_handle());
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
    for (auto i = 0; i < size; i++) {
      gx_data[i] = mask_data[i] * gy_data[i];
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& mask, const tensor& gy, tensor& gx) {
    switch(gy.get_data_type()) {
    case tdtype_t::f32:
      compute_impl<float, alloc>(mask, gy, gx);
      break;
    case tdtype_t::s32:
      compute_impl<int32_t, alloc>(mask, gy, gx);
      break;
    case tdtype_t::s16:
      compute_impl<int16_t, alloc>(mask, gy, gx);
      break;
    case tdtype_t::s8:
      compute_impl<int8_t, alloc>(mask, gy, gx);
      break;
    case tdtype_t::u8:
      compute_impl<uint8_t, alloc>(mask, gy, gx);
      break;
    default:
      throw error(mkldnn_invalid_arguments, "Unsupported mkldnn data type!");
    }
  }
};

struct eltwise_binary {
public:
  enum eltwise_binary_op {
    ELTWISE_ADD,
    ELTWISE_MUL,
    ELTWISE_DIV,
  };

  eltwise_binary() = default;

  template<class alloc = utils::allocator>
  static void compute(eltwise_binary_op op, tensor& inputA, tensor& inputB, tensor& outputC) {
    IDEEP_ENFORCE(inputA.ndims() >= inputB.ndims(), "Incorrect dims");
    outputC.reinit<alloc>(inputA.get_descriptor());
    if (inputA.get_dims() == inputB.get_dims()) {
      auto* inputB_data = inputB.get_data_handle();
      tensor scratch_tensor;
      if (inputA.get_internal_format() != inputB.get_internal_format()) {
        scratch_tensor.init<alloc>(inputA.get_descriptor());
        treorder_t::compute(inputB, scratch_tensor);
        inputB_data = scratch_tensor.get_data_handle();
      }
      float* X = static_cast<float*>(inputA.get_data_handle());
      float* Y = static_cast<float*>(inputB.get_data_handle());
      float* Z = static_cast<float*>(outputC.get_data_handle());
      unsigned N = static_cast<unsigned>(inputA.get_size() / sizeof(float));
      switch (op) {
      case ELTWISE_ADD:
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
        for (auto n = 0; n < N; n++) {
          Z[n] = X[n] + Y[n];
        }
        return;
      case ELTWISE_MUL:
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
        for (auto n = 0; n < N; n++) {
          Z[n] = X[n] * Y[n];
        }
        return;
      case ELTWISE_DIV:
      default:
        throw error(mkldnn_invalid_arguments, "Not implemented!");
      }
    } else {
      throw error(mkldnn_invalid_arguments, "Not implemented!");
    }
  }
};

struct rnn_forward: public computation,
    public utils::computation_cache<rnn_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_layer_desc, const tdesc_t& src_iter_desc,
        const tdesc_t& weights_layer_desc, const tdesc_t& weights_iter_desc, const tdesc_t& bias_desc,
        const tdesc_t& dst_layer_desc, const tdesc_t& dst_iter_desc, rnn_kind akind,
        const rnn_direction direction, prop_kind aprop_kind = prop_kind::forward_inference) {
      mkldnn_rnn_cell_desc_t rnn_cell_data;
      mkldnn_rnn_desc_t rnn_data;
      auto src_layer_data = src_layer_desc.get_mkldnn_memory_desc_t();
      auto src_iter_data = src_iter_desc.format_any();
      auto weights_layer_data = weights_layer_desc.format_any();
      auto weights_iter_data = weights_iter_desc.format_any();
      auto bias_data = bias_desc.format_any();
      auto dst_layer_data = dst_layer_desc.get_mkldnn_memory_desc_t();
      auto dst_iter_data = dst_iter_desc.format_any();

      auto algo = utils::rnn_kind_to_algorithm(akind);
      auto acti = utils::rnn_kind_to_activation(akind);

      error::wrap_c_api(
          mkldnn_rnn_cell_desc_init(&rnn_cell_data, mkldnn::convert_to_c(algo),
              mkldnn::convert_to_c(acti), 0U, 0, 0),
          "could not create a rnn cell descriptor");
      error::wrap_c_api(
          mkldnn_rnn_forward_desc_init(&rnn_data, mkldnn::convert_to_c(aprop_kind),
              &rnn_cell_data, mkldnn::convert_to_c(direction), src_layer_data, &src_iter_data,
              &weights_layer_data, &weights_iter_data, &bias_data, dst_layer_data,
              &dst_iter_data),
          "could not create a rnn forward descriptor");
      create_primitive_desc(rnn_data);
    }
  };

public:
  template<typename... Ts>
  rnn_forward(const tdesc_t& src_layer_desc, Ts&&... args) {
    descriptor forward_descriptor(src_layer_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src_layer, const tensor& src_iter,
      const tensor& weights_layer, const tensor& weights_iter, const tensor& bias,
      const tdims_t& dst_layer_dims, tensor& dst_layer,
      const tdims_t& dst_iter_dims, tensor& dst_iter,
      tensor& workspace, rnn_kind akind, rnn_direction direction,
      prop_kind aprop_kind = prop_kind::forward_training) {
    auto dtype = src_layer.get_data_type();
    auto src_layer_desc = tdesc_t(src_layer.get_dims(), dtype, format::tnc);
    auto src_iter_desc = tdesc_t(src_iter.get_dims(), dtype, format::ldsnc);
    auto dst_layer_desc = tdesc_t(dst_layer_dims, dtype, format::tnc);
    auto dst_iter_desc = tdesc_t(dst_iter_dims, dtype, format::ldsnc);

    key_t key;
    check_or_create_k(key, src_layer, src_iter, weights_layer, weights_iter, bias,
        dst_layer_dims, dst_iter_dims, akind, direction, aprop_kind);
    fetch_or_create_m(comp, key, src_layer_desc, src_iter_desc,
        weights_layer.get_descriptor(), weights_iter.get_descriptor(),
        bias.get_descriptor(), dst_layer_desc, dst_iter_desc, akind, direction, aprop_kind);

    auto src_layer_in = comp.transform_input_cache<alloc>(0, src_layer.as_rnn());
    auto src_iter_in = comp.transform_input_cache<alloc>(1, src_iter.as_rnn());
    auto weights_layer_in = comp.transform_input_cache<alloc>(2, weights_layer.as_rnn(/*is_weight*/true));
    auto weights_iter_in = comp.transform_input_cache<alloc>(3, weights_iter.as_rnn(/*is_weight*/true));
    auto bias_in = comp.transform_input_cache<alloc>(4, bias.as_rnn(/*is_weight*/true));

    dst_layer.reinit<alloc>(dst_layer_desc);
    dst_iter.reinit<alloc>(dst_iter_desc);
    if (aprop_kind == prop_kind::forward_training) {
      workspace.reinit<alloc>(comp.expected_workspace_descriptor());
      comp.execute(src_layer_in, src_iter_in, weights_layer_in, weights_iter_in,
          bias_in, dst_layer, dst_iter, workspace);
    } else {
      comp.execute(src_layer_in, src_iter_in, weights_layer_in, weights_iter_in,
          bias_in, dst_layer, dst_iter);
    }
  }
};

struct rnn_backward : public computation,
  public utils::computation_cache<rnn_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tdesc_t& src_layer_desc, const tdesc_t& src_iter_desc,
        const tdesc_t& weights_layer_desc, const tdesc_t& weights_iter_desc,
        const tdesc_t& bias_desc, const tdesc_t& dst_layer_desc, const tdesc_t& dst_iter_desc,
        const tdesc_t& diff_src_layer_desc, const tdesc_t& diff_src_iter_desc,
        const tdesc_t& diff_weights_layer_desc, const tdesc_t& diff_weights_iter_desc,
        const tdesc_t& diff_bias_desc, const tdesc_t& diff_dst_layer_desc,
        const tdesc_t& diff_dst_iter_desc, rnn_kind akind, const rnn_direction direction,
        prop_kind aprop_kind = prop_kind::backward) {
      mkldnn_rnn_cell_desc_t rnn_cell_data;
      mkldnn_rnn_desc_t data;
      auto src_layer_data = src_layer_desc.format_any();
      auto src_iter_data = src_iter_desc.format_any();
      auto weights_layer_data = weights_layer_desc.format_any();
      auto weights_iter_data = weights_iter_desc.format_any();
      auto dst_layer_data = dst_layer_desc.format_any();
      auto dst_iter_data = dst_iter_desc.format_any();
      auto diff_src_layer_data = diff_src_layer_desc.format_any();
      auto diff_src_iter_data = diff_src_iter_desc.format_any();
      auto diff_weights_layer_data = diff_weights_layer_desc.format_any();
      auto diff_weights_iter_data = diff_weights_iter_desc.format_any();
      auto diff_dst_layer_data = diff_dst_layer_desc.format_any();
      auto diff_dst_iter_data = diff_dst_iter_desc.format_any();
      auto bias_data = bias_desc.format_any();
      auto diff_bias_data = diff_bias_desc.format_any();

      auto algo = utils::rnn_kind_to_algorithm(akind);
      auto acti = utils::rnn_kind_to_activation(akind);
      error::wrap_c_api(
          mkldnn_rnn_cell_desc_init(&rnn_cell_data, mkldnn::convert_to_c(algo),
              mkldnn::convert_to_c(acti), 0U, 0, 0),
          "could not create a rnn cell descriptor");
      error::wrap_c_api(mkldnn_rnn_backward_desc_init(
          &data, mkldnn::convert_to_c(aprop_kind), &rnn_cell_data, mkldnn::convert_to_c(direction),
          &src_layer_data, &src_iter_data, &weights_layer_data, &weights_iter_data, &bias_data,
          &dst_layer_data, &dst_iter_data, &diff_src_layer_data, &diff_src_iter_data, &diff_weights_layer_data,
          &diff_weights_iter_data, &diff_bias_data, &diff_dst_layer_data, &diff_dst_iter_data),
          "could not init a backward rnn descriptor");
      create_primitive_desc(data);
    }
  };

public:
  template<typename ...Ts>
  rnn_backward(const tdesc_t& src_layer_desc, Ts &&...args) {
    descriptor backward_descriptor(src_layer_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor);
  }

  template <class alloc = utils::allocator>
  static void compute(const tensor& src_layer, const tensor& src_iter, const tensor& weights_layer,
      const tensor& weights_iter, const tensor& bias, const tensor& dst_layer, const tensor& dst_iter,
      const tensor& diff_dst_layer, const tensor& diff_dst_iter, const tensor& workspace,
      const bool with_bias, tensor& diff_src_layer, tensor& diff_src_iter, tensor& diff_weights_layer,
      tensor& diff_weights_iter, tensor& diff_bias, rnn_kind akind, rnn_direction direction,
      prop_kind aprop_kind = prop_kind::backward) {
    tdesc_t bias_desc;
    tdesc_t diff_bias_desc;
    bias_desc = bias.get_descriptor();
    diff_bias_desc = tdesc_t(bias.get_dims(), bias.get_data_type());
    tdesc_t src_layer_desc = src_layer.get_descriptor();
    tdesc_t src_iter_desc = src_iter.get_descriptor();
    tdesc_t weights_layer_desc = weights_layer.get_descriptor();
    tdesc_t weights_iter_desc = weights_iter.get_descriptor();

    key_t key;
    check_or_create_k(key, src_layer, src_iter, weights_layer, weights_iter,
        dst_layer.get_dims(), dst_iter.get_dims(), bias, akind, direction, aprop_kind);
    fetch_or_create_m(comp, key, src_layer_desc, src_iter_desc, weights_layer_desc,
        weights_iter_desc, bias_desc, dst_layer.get_descriptor(),
        dst_iter.get_descriptor(), src_layer_desc, src_iter_desc,
        weights_layer_desc, weights_iter_desc, diff_bias_desc,
        diff_dst_layer.get_descriptor(), diff_dst_iter.get_descriptor(), akind,
        direction, aprop_kind);

    auto src_layer_in = comp.transform_input_cache<alloc>(0, src_layer);
    auto src_iter_in = comp.transform_input_cache<alloc>(1, src_iter);
    auto weights_layer_in = comp.transform_input_cache<alloc>(2, weights_layer);
    auto weights_iter_in = comp.transform_input_cache<alloc>(3, weights_iter);
    auto bias_in = comp.transform_input_cache<alloc>(4, bias);
    auto dst_layer_in = comp.transform_input_cache<alloc>(5, dst_layer);
    auto dst_iter_in = comp.transform_input_cache<alloc>(6, dst_iter);
    auto diff_dst_layer_in = comp.transform_input_cache<alloc>(7, diff_dst_layer);
    auto diff_dst_iter_in = comp.transform_input_cache<alloc>(8, diff_dst_iter);
    auto workspace_in = comp.transform_input_cache<alloc>(9, workspace);

    tensor diff_src_layer_in;
    bool reorder_diff_src_layer = (
        diff_src_layer.get_descriptor() != comp.expected_descriptor_of(query::diff_src_pd, 0));

    tensor::descriptor tmp_desc1(comp.expected_descriptor_of(query::diff_src_pd, 0));
    if (reorder_diff_src_layer) {
      diff_src_layer_in.init<alloc>(comp.expected_descriptor_of(query::diff_src_pd, 0));
    } else {
      diff_src_layer_in = diff_src_layer;
    }

    tensor diff_src_iter_in;
    bool reorder_diff_src_iter = (
        diff_src_iter.get_descriptor() != comp.expected_descriptor_of(query::diff_src_pd, 1));
    if (reorder_diff_src_iter) {
      diff_src_iter_in.init<alloc>(comp.expected_descriptor_of(query::diff_src_pd, 1));
    } else {
      diff_src_iter_in = diff_src_iter;
    }

    tensor diff_weights_layer_in;
    bool reorder_diff_weights_layer = (
        diff_weights_layer.get_descriptor() != comp.expected_descriptor_of(query::diff_weights_pd, 0));

    tensor::descriptor tmp_desc(comp.expected_descriptor_of(query::diff_weights_pd, 0));
    if (reorder_diff_weights_layer) {
      diff_weights_layer_in.init<alloc>(comp.expected_descriptor_of(query::diff_weights_pd, 0));
    } else {
      diff_weights_layer_in = diff_weights_layer;
    }

    tensor diff_weights_iter_in;
    bool reorder_diff_weights_iter = (
        diff_weights_iter.get_descriptor() != comp.expected_descriptor_of(query::diff_weights_pd, 1));

    if (reorder_diff_weights_iter) {
      diff_weights_iter_in.init<alloc>(comp.expected_descriptor_of(query::diff_weights_pd, 1));
    } else {
      diff_weights_iter_in = diff_weights_iter;
    }

    tensor diff_bias_in;
    bool reorder_bias;
    reorder_bias = (
        diff_bias.get_descriptor() != comp.expected_descriptor_of(query::diff_weights_pd, 2));
    if (reorder_bias) {
      diff_bias_in.init<alloc>(comp.expected_descriptor_of(query::diff_weights_pd, 2));
    } else {
      diff_bias_in = diff_bias;
    }

    comp.execute(src_layer_in, src_iter_in, weights_layer_in, weights_iter_in,
        bias_in, dst_layer_in, dst_iter_in, diff_dst_layer_in, diff_dst_iter_in,
        workspace_in, diff_src_layer_in, diff_src_iter_in, diff_weights_layer_in,
        diff_weights_iter_in, diff_bias_in);

    if (reorder_diff_src_layer) {
      treorder_t::compute(diff_src_layer_in, diff_src_layer);
    }
    if (reorder_diff_src_iter) {
      treorder_t::compute(diff_src_iter_in, diff_src_iter);
    }
    if (reorder_diff_weights_layer) {
      treorder_t::compute(diff_weights_layer_in, diff_weights_layer);
    }
    if (reorder_diff_weights_iter) {
      treorder_t::compute(diff_weights_iter_in, diff_weights_iter);
    }
    if (with_bias) {
      if (reorder_bias) {
        treorder_t::compute(diff_bias_in, diff_bias);
      }
    }
  }

};

} // namespace ideep

#endif
