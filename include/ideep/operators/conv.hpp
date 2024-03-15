#ifndef IDEEP_OPERATORS_CONV_HPP
#define IDEEP_OPERATORS_CONV_HPP
namespace ideep {

struct convolution_forward_params {
  convolution_forward_params() {}

  convolution_forward_params(
      dnnl::convolution_forward::primitive_desc&& pd,
      dnnl::convolution_forward&& primitive,
      attr_t&& op_attr,
      int groups)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(op_attr),
        groups(groups),
        bias_attr(attr_t()),
        pd_use_threads(omp_get_max_threads()) {}

  convolution_forward_params(
      dnnl::convolution_forward::primitive_desc&& pd,
      dnnl::convolution_forward&& primitive,
      attr_t&& op_attr,
      int groups,
      attr_t&& bias_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(op_attr),
        groups(groups),
        bias_attr(std::move(bias_attr)),
        pd_use_threads(omp_get_max_threads()) {}

  dnnl::convolution_forward::primitive_desc pd;
  dnnl::convolution_forward primitive;
  attr_t op_attr;
  int groups;
  attr_t bias_attr;
  // From IPEX. Set in `do_prepare()`, only used outside ideep.
  // TO-DO: Use a better name, i.e., num_threads
  int pd_use_threads;

  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, tensor>> all_zero_points = nullptr;
};

struct conv_deconv_utils {
  // Common logic to prepare parameters for conv/deconv
  // quantization version
  static void prepare_parameters(const tensor& src,
                                 const tensor& weights,
                                 const tensor& bias,
                                 const dims& dst_dims,
                                 const tensor& dst,
                                 const dims& dilates,
                                 int groups,
                                 const scale_t& src_scales,
                                 const scale_t& weights_scales,
                                 const scale_t& dst_scales,
                                 const zero_point_t& src_zero_points,
                                 const zero_point_t& dst_zero_points,
                                 const attr_t& attr,
                                 const lowp_kind alowp_kind,
                                 bool with_bias,
                                 bool is_deconv,
                                 tensor& weight_grouped, /* Output */
                                 dims& dil_compatible, /* Output */
                                 attr_t& op_attr, /* Output */
                                 attr_t& src_attr, /* Output */
                                 attr_t& weights_attr, /* Output */
                                 attr_t& bias_attr, /* Output */
                                 tensor::desc& src_desc, /* Output */
                                 tensor::desc& weights_desc, /* Output */
                                 tensor::desc& bias_desc, /* Output */
                                 tensor::desc& dst_desc /* Output */) {
    scale_t dst_scales_in;
    data_type dst_data_type;
    op_attr.set_post_ops(attr.get_post_ops());

    // make weights and dilates compatible with DNNL
    weight_grouped = weights.make_grouped_weights(groups, is_deconv);
    dil_compatible = utils::get_compatible_dilates(dilates);

    auto& weights_scales_in =
        weight_grouped.has_scale() ? weight_grouped.get_scale() : weights_scales;
    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported lowp kind");
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
      auto& src_scales_in =
          src.has_scale() ? src.get_scale()
                          : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);

      // determine dst data type
      if (dst.get_data_type() != data_type::undef) {
        dst_data_type = dst.get_data_type();
      } else if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      dst_scales_in = dst_scales.empty() || dst_data_type == data_type::f32
                          ? IDEEP_DEF_SCALE
                          : dst_scales;
      const auto default_zero_point = zero_point_t(1);
      const auto& src_zero_point = src.has_zero_point() ? src.get_zero_point() :
                                   src_zero_points.empty() ? default_zero_point : src_zero_points;
      const auto& weights_zero_point = weight_grouped.has_zero_point() ? weight_grouped.get_zero_point() : default_zero_point;
      const auto& dst_zero_point = [&]() {
        if (attr.has_op_kind(kind::sum)) {
          // Similar logic as dst_scales_in. Since when fused with sum, the dst will be the tensor of sum,
          // In this case, the output tensor' dst_zero_points and dst_scales_in should be passed in explicitly.
          IDEEP_ENFORCE(!dst_zero_points.empty(), "When conv fused with sum, dst_zero_points must be passed in.");
          return dst_zero_points;
        } else {
          // Keep the original logic for finding dst_zero_point when not fused with sum.
          return dst.has_zero_point() ? dst.get_zero_point() :
                dst_zero_points.empty() ? default_zero_point : dst_zero_points;
        }
      }();

      const auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
      const auto dst_zero_point_size = static_cast<dim>(dst_zero_point.size());
      IDEEP_ENFORCE(src_zero_point_size == 1 && dst_zero_point_size == 1,
                    "DNNL only support 1-dim zero_point");

      if (attr.has_op_kind(kind::sum)) {
        // Here we need to recalculate the scale of sum.
        // When fused with sum, dst_scales_in is the final output tensor's scale.
        // dst.scale is the scale of sum tensor.

        float sum_scale = 1.0 / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);

        // When fused with sum, the dst tensor is same as the sum tensor.
        // So the sum_zero_point will be fetched from the dst tensor.
        int32_t sum_zero_point = dst.has_zero_point() ? dst.get_zero_point()[0] : 0;
        if (attr.has_op_kind(kind::eltwise)) {
          op_attr = attr_t::residual_with_sum_zero_point(sum_scale, sum_zero_point);
        } else {
          op_attr = attr_t::fuse_sum(sum_scale, sum_zero_point);
        }
      }
      // Set scales to op_attr
      if (src_scales_in[0] != 1.0f) {
        int src_scale_mask = 0;
        op_attr.set_scales(DNNL_ARG_SRC, src_scale_mask, {1.0f / src_scales_in[0]});
      }
      // For grouped conv, weight format is `goi...`, where g is groups and o and i are OC/IC per group.
      // `weight.get_dim(0)` returns g * o and `get_dim(1)` returns i.
      // For non-grouped conv (groups = 1), weight format is `oi...`.
      // `weight.get_dim(0)` returns o and `get_dim(1)` returns i.
      // So, it's always correct to get oc per group by `get_dim(0) / groups`
      // For deconv, weight format is `gio...` (grouped) or `io...` (non-grouped)
      // `weight.get_dim(0)` returns g * i or i and `get_dim(1)` always returns o.
      int oc_per_group = is_deconv ? weight_grouped.get_dim(1) : weight_grouped.get_dim(0) / groups;
      auto wei_scale_mask = utils::conv_weight_scale_mask(
          weights_scales_in.size(), oc_per_group, groups, is_deconv);
      auto wei_scales = weights_scales_in;
      if (!std::all_of(wei_scales.begin(), wei_scales.end(), [](float i){ return i == 1.0f; })) {
        op_attr.set_scales(DNNL_ARG_WEIGHTS, wei_scale_mask, wei_scales);
      }
      if (dst_scales_in[0] != 1.0f) {
        int dst_scale_mask = 0;
        op_attr.set_scales(DNNL_ARG_DST, dst_scale_mask, {1.0f / dst_scales_in[0]});
      }
      // Set zero points to op_attr (weight zero point is always 0. Do not set.)
      // Only set non-zero zero points otherwise there is extra oneDNN overhead
      if (src_zero_point[0] != 0) {
        auto src_zp_mask = utils::tensor_zp_mask(src_zero_point_size);
        op_attr.set_zero_points(DNNL_ARG_SRC, src_zp_mask, src_zero_point);
      }
      if (dst_zero_point[0] != 0) {
        auto dst_zp_mask = utils::tensor_zp_mask(dst_zero_point_size);
        op_attr.set_zero_points(DNNL_ARG_DST, dst_zp_mask, dst_zero_point);
      }

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8, tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      weights_desc = tensor::desc(weight_grouped.get_dims(), data_type::s8, tag::any);
      if (groups > 1) {
        weights_desc = weights_desc.to_grouped(groups);
      }
      if (weight_grouped.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, groups > 1),
                        weights_scales_in};
      }

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::f32, tag::any}; // Use f32 instead of s32 to improve accuracy
      }
    } else {
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }

      IDEEP_ENFORCE(utils::one_of(weight_grouped.get_data_type(),
                                  data_type::f32, data_type::bf16),
                    "Incorrect data type in weights");

      // align weights data type with src
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                              : data_type::f32;
      src_desc = src.get_desc().to_type(dst_data_type);
      weights_desc = tensor::desc(weight_grouped.get_dims(), dst_data_type, tag::any);
      if (groups > 1) {
        weights_desc = weights_desc.to_grouped(groups);
      }

      if (with_bias) {
        IDEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                    data_type::f32, data_type::bf16),
                      "Incorrect data type in bias");
        bias_desc = bias.get_desc();
      }
    }

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dst_desc = attr.has_op_kind(kind::sum)
                    ? dst.get_desc()
                    : tensor::desc(dst_dims, dst_data_type);
  }

  // Common logic to prepare parameters for conv/deconv.
  // non-quantization version
  static void prepare_parameters(const tensor& src,
                                 const tensor& weights,
                                 const tensor& bias,
                                 const dims& dst_dims,
                                 const tensor& dst,
                                 const dims& dilates,
                                 int groups,
                                 const attr_t& attr,
                                 bool with_bias,
                                 bool is_deconv,
                                 tensor& weight_grouped, /* Output */
                                 dims& dil_compatible, /* Output */
                                 attr_t& op_attr, /* Output */
                                 attr_t& src_attr, /* Output */
                                 attr_t& weights_attr, /* Output */
                                 attr_t& bias_attr, /* Output */
                                 tensor::desc& src_desc, /* Output */
                                 tensor::desc& weights_desc, /* Output */
                                 tensor::desc& bias_desc, /* Output */
                                 tensor::desc& dst_desc /* Output */) {
    data_type dst_data_type;
    op_attr = attr;

    // make weights and dilates compatible with DNNL
    weight_grouped = weights.make_grouped_weights(groups, is_deconv);
    dil_compatible = utils::get_compatible_dilates(dilates);

    IDEEP_ENFORCE(utils::one_of(weight_grouped.get_data_type(),
                                data_type::f32, data_type::bf16, data_type::f16),
                  "Incorrect data type in weights");

    // align weights data type with src
    dst_data_type = src.get_data_type() == data_type::bf16
        ? data_type::bf16
        : ((src.get_data_type() == data_type::f16) ? data_type::f16
                                                   : data_type::f32);
    src_desc = src.get_desc().to_type(dst_data_type);
    weights_desc = tensor::desc(weight_grouped.get_dims(), dst_data_type, tag::any);
    if (groups > 1) {
      weights_desc = weights_desc.to_grouped(groups);
    }

    if (with_bias) {
      IDEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                  data_type::f32, data_type::bf16, data_type::f16),
                    "Incorrect data type in bias");
      bias_desc = bias.get_desc();
    }

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dst_desc = attr.has_op_kind(kind::sum)
                    ? dst.get_desc()
                    : tensor::desc(dst_dims, dst_data_type);
  }

};

struct convolution_forward
    : public dnnl::convolution_forward,
#ifdef __aarch64__
      utils::computation_cache<std::pair<dnnl::convolution_forward::primitive_desc, dnnl::convolution_forward> > {
#else
      utils::computation_cache<dnnl::convolution_forward::primitive_desc> {
#endif

  using super = dnnl::convolution_forward;

  // 2-in-1 Conv computation with bias
  // reorder as true means reorder might needed for src/weight/bias/dst as
  // adapt to oneDNN recommended memory format or your simply not sure, set
  // to false if src/weight/bias/dst are pre-reorded or no reorder needed as
  // you know for sure
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      bool is_channels_last,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_dispatch<false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_dispatch<true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 2-in-1 Conv computation w/o bias
  // reorder as true means reorder might needed for src/weight/bias/dst as
  // adapt to oneDNN recommended memory format or your simply not sure, set
  // to false if src/weight/bias/dst are pre-reorded or no reorder needed as
  // you know for sure
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      bool is_channels_last,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_dispatch<false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // 2-in-1 compute for fp32 with bias
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_dispatch<false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_dispatch<true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 2-in-1 compute for fp32 without bias
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    compute_dispatch<false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // 2-in-1 Quantized Conv computation with bias
  // reorder as true means reorder might needed for src/weight/bias/dst as
  // adapt to oneDNN recommended memory format or your simply not sure, set
  // to false if src/weight/bias/dst are pre-reorded or no reorder needed as
  // you know for sure
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      bool is_channels_last,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_dispatch</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // 2-in-1 Quantized Conv computation w/o bias
  // reorder as true means reorder might needed for src/weight/bias/dst as
  // adapt to oneDNN recommended memory format or your simply not sure, set
  // to false if src/weight/bias/dst are pre-reorded or no reorder needed as
  // you know for sure
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      bool is_channels_last,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // 2-in-1 compute for int8 without bias
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, true,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_dispatch</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, true,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // 2-in-1 compute for int8 with bias
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // 2-in-1 compute (prepare & compute) for fp32 with binary fusion
  // With bias. Bias is not used if it is empty.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             const tensor &bias,
                             const dims &dst_dims,
                             tensor &dst,
                             const dims &strides,
                             const dims &dilates,
                             const dims &padding_l,
                             const dims &padding_r,
                             int groups,
                             bool is_channels_last,
                             const attr_t &attr = attr_t(),
                             algorithm aalgorithm = algorithm::convolution_direct,
                             prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_binary_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
          src, other, weights, bias, dst_dims, dst, strides, dilates, padding_l,
          padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_binary_dispatch</*with_bias=*/true, reorder_src, reorder_weight>(
          src, other, weights, bias, dst_dims, dst, strides, dilates, padding_l,
          padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 2-in-1 compute (prepare & compute) for fp32 with binary fusion
  // Without bias.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             const dims &dst_dims,
                             tensor &dst,
                             const dims &strides,
                             const dims &dilates,
                             const dims &padding_l,
                             const dims &padding_r,
                             int groups,
                             bool is_channels_last,
                             const attr_t &attr = attr_t(),
                             algorithm aalgorithm = algorithm::convolution_direct,
                             prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_binary_dispatch</*with_bias=*/false, reorder_src, reorder_weight>(
        src, other, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // Conv prepare with bias for fp32
  // params will be initialized with PD/Primitive/groups ...
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      bool is_channels_last,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_prepare</*with_bias=*/false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last,
          attr, aalgorithm, aprop_kind, aengine);
    } else {
      do_prepare</*with_bias=*/true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last,
          attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // Conv prepare w/o bias for fp32
  // params will be initialized with PD/Primitive/groups ...
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      bool is_channels_last,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last,
        attr, aalgorithm, aprop_kind, aengine);
  }

  // Prepare for fp32 with bias
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      do_prepare</*with_bias=*/false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last,
          attr, aalgorithm, aprop_kind, aengine);
    } else {
      do_prepare</*with_bias=*/true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last,
          attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // Prepare for fp32 without bias
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last,
        attr, aalgorithm, aprop_kind, aengine);
  }

  // Conv prepare with bias for int8
  // params will be initialized with PD/Primitive/groups ...
  // quant_params will be initialized with quantization related info ...
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      bool is_channels_last = false,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_prepare</*with_bias=*/false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      do_prepare</*with_bias=*/true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // Conv prepare w/o bias for int8
  // params will be initialized with PD/Primitive/groups ...
  // quant_params will be initialized with quantization related info ...
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      bool is_channels_last = false,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Prepare for int8 with bias
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      do_prepare</*with_bias=*/false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      do_prepare</*with_bias=*/true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last,
          attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // Prepare for int8 without bias
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_point,
                      const zero_point_t& dst_zero_point,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Conv computation with pre-prepared params, with bias
  //   param: pd, primitive, groups ...
  // for both fp32 and int8
  // reorder as true means reorder might needed for src/weight/bias/dst as
  // adapt to oneDNN recommended memory format or your simply not sure, set
  // to false if src/weight/bias/dst are pre-reorded or no reorder needed as
  // you know for sure
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst) {
    if (bias.is_empty()) {
      do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
          param, src, weights, bias, dst);
    } else {
      do_compute</*with_bias=*/true, reorder_src, reorder_weight>(
          param, src, weights, bias, dst);
    }
  }

  // Conv computation with pre-prepared params, w/o bias
  //   param: pd, primitive, groups ...
  // for both fp32 and int8
  // reorder as true means reorder might needed for src/weight/bias/dst as
  // adapt to oneDNN recommended memory format or your simply not sure, set
  // to false if src/weight/bias/dst are pre-reorded or no reorder needed as
  // you know for sure
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, weights, dummy_bias, dst);
  }

  // Compute for binary post-op.
  // With bias. Bias is disabled if it is empty.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const convolution_forward_params& param,
                             const tensor& src,
                             const tensor& other,
                             const tensor& weights,
                             const tensor& bias,
                             tensor& dst) {
    if (bias.is_empty()) {
      do_compute_binary</*with_bias=*/false, reorder_src, reorder_weight>(
          param, src, other, weights, bias, dst);
    } else {
      do_compute_binary</*with_bias=*/true, reorder_src, reorder_weight>(
          param, src, other, weights, bias, dst);
    }
  }

  // Compute for binary post-op.
  // Without bias.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const convolution_forward_params& param,
                             const tensor& src,
                             const tensor& other,
                             const tensor& weights,
                             tensor& dst) {
    static tensor dummy_bias;
    do_compute_binary</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, other, weights, dummy_bias, dst);
  }

  // DEPRECATED
  // 2-in-1 compute (prepare & compute) with bias
  // Bias is not used if it is empty.
  // Zero points are passed explicitly as arguments for quantization
  static void compute_v2(const tensor& src,
                         const tensor& weights,
                         const tensor& bias,
                         const dims& dst_dims,
                         tensor& dst,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         int groups,
                         const scale_t& src_scales = scale_t(),
                         const scale_t& weights_scales = scale_t(),
                         const scale_t& dst_scales = scale_t(),
                         const zero_point_t& src_zero_point = zero_point_t(),
                         const zero_point_t& dst_zero_point = zero_point_t(),
                         const attr_t& attr = attr_t(),
                         algorithm aalgorithm = algorithm::convolution_direct,
                         prop_kind aprop_kind = prop_kind::forward,
                         const lowp_kind alowp_kind = u8s8,
                         const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_dispatch</*with_bias=*/false, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_dispatch</*with_bias=*/true, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // DEPRECATED
  // 2-in-1 compute (prepare & compute) without bias
  // Zero points are passed explicitly as arguments for quantization
  static void compute_v2(const tensor& src,
                         const tensor& weights,
                         const dims& dst_dims,
                         tensor& dst,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         int groups,
                         const scale_t& src_scales = scale_t(),
                         const scale_t& weights_scales = scale_t(),
                         const scale_t& dst_scales = scale_t(),
                         const zero_point_t& src_zero_point = zero_point_t(),
                         const zero_point_t& dst_zero_point = zero_point_t(),
                         const attr_t& attr = attr_t(),
                         algorithm aalgorithm = algorithm::convolution_direct,
                         prop_kind aprop_kind = prop_kind::forward,
                         const lowp_kind alowp_kind = u8s8,
                         const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, true, true>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // 2-in-1 Conv computation with bias for fp32
  static void compute_v3(const tensor& src,
                         const tensor& weights,
                         const tensor& bias,
                         const dims& dst_dims,
                         tensor& dst,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         int groups,
                         bool is_channels_last = false,
                         const attr_t& attr = attr_t(),
                         algorithm aalgorithm = algorithm::convolution_direct,
                         prop_kind aprop_kind = prop_kind::forward,
                         const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_dispatch<false, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_dispatch<true, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 2-in-1 Conv computation w/o bias for fp32
  static void compute_v3(const tensor& src,
                         const tensor& weights,
                         const dims& dst_dims,
                         tensor& dst,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         int groups,
                         bool is_channels_last = false,
                         const attr_t& attr = attr_t(),
                         algorithm aalgorithm = algorithm::convolution_direct,
                         prop_kind aprop_kind = prop_kind::forward,
                        const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_dispatch<false, true, true>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // DEPRECATED
  // 2-in-1 compute (prepare & compute) with bias
  // Bias is not used if it is empty.
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    // Consider fp32 only for IPEX
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_dispatch</*with_bias=*/false, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last,
          attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_dispatch</*with_bias=*/true, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last,
          attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // DEPRECATED
  // 2-in-1 compute (prepare & compute) without bias
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    // Consider fp32 only for IPEX
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, true, true>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // DEPRECATED
  // Prepare with bias.
  // Bias is not used if it is empty.
  // Zero points are set to tensor for quantization
  static void prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    bool is_fp32 = src_scales.empty() && weights_scales.empty() && dst_scales.empty();
    if (is_fp32) {
      if (bias.is_empty()) {
        do_prepare</*with_bias=*/false>(
            param, src, weights, bias, dst_dims, dst, strides, dilates,
            padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
      } else {
        do_prepare</*with_bias=*/true>(
            param, src, weights, bias, dst_dims, dst, strides, dilates,
            padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
      }
    } else {
      if (bias.is_empty()) {
        do_prepare</*with_bias=*/false>(
            param, src, weights, bias, dst_dims, dst, strides, dilates,
            padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
            zero_point_t(), zero_point_t(), is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
      } else {
        do_prepare</*with_bias=*/true>(
            param, src, weights, bias, dst_dims, dst, strides, dilates,
            padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
            zero_point_t(), zero_point_t(), is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
      }
    }
  }

  // DEPRECATED
  // Prepare without bias.
  // Zero points are set to tensor for quantization
  static void prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    bool is_fp32 = src_scales.empty() && weights_scales.empty() && dst_scales.empty();
    static tensor dummy_bias;
    if (is_fp32) {
      do_prepare</*with_bias=*/false>(
          param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      do_prepare</*with_bias=*/false>(
          param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          zero_point_t(), zero_point_t(), is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // DEPRECATED
  // compute with param and primitive. With bias
  // Weight is supposed to be prepacked
  static void compute(
      const convolution_forward_params& param,
      const dnnl::convolution_forward& prim,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst) {
    (void)prim; // Mark as unused
    if (bias.is_empty()) {
      do_compute</*with_bias=*/false, /*reorder_src*/true, /*reorder_weight*/false>(
          param, src, weights, bias, dst);
    } else {
      do_compute</*with_bias=*/true, /*reorder_src*/true, /*reorder_weight*/false>(
          param, src, weights, bias, dst);
    }
  }

  // DEPRECATED
  // compute with param and primitive. Without bias
  // Weight is supposed to be prepacked
  static void compute(
      const convolution_forward_params& param,
      const dnnl::convolution_forward& prim,
      const tensor& src,
      const tensor& weights,
      tensor& dst) {
    (void)prim; // Mark as unused
    static tensor dummy_bias;
    do_compute</*with_bias=*/false, /*reorder_src*/true, /*reorder_weight*/false>(
        param, src, weights, dummy_bias, dst);
  }

  template <bool is_channels_last = false>
  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      data_type x_dtype = data_type::f32,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {
    // weights_dims.size() is 3 for conv1d, 4 for conv2d and 5 for conv3d
    auto src_size = weights_dims.size();
    auto grouped = groups > 1;
    auto weights_dims_g =
        grouped ? utils::group_dims(weights_dims, groups) : weights_dims;
    auto weights_desc = tensor::desc(weights_dims_g, dtype);

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto dilates_ = utils::get_compatible_dilates(dilates, src_size);

    IDEEP_ENFORCE(
        !(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    dims x_dims, y_dims, kernel_size;
    auto ic = groups * dims_in[1 + grouped];
    auto oc = groups * dims_in[0 + grouped];
    if (5 == src_size) {
      kernel_size.push_back(dims_in[ndims - 3]);
      kernel_size.push_back(dims_in[ndims - 2]);
    } else if (4 == src_size) {
      kernel_size.push_back(dims_in[ndims - 2]);
    }
    kernel_size.push_back(dims_in[ndims - 1]);
    if (src_dims.empty()) {
      // Construct a dummy case. Shape from resnet50 model.
      x_dims.push_back(1);
      x_dims.push_back(ic);
      y_dims.push_back(1);
      y_dims.push_back(oc);
      auto valid_x_dim = [=](int idx, int64_t scale) {
        return std::max(strides[idx] +
                            ((kernel_size[idx] - 1) * (dilates_[idx] + 1) + 1) -
                            (padding_l[idx] + padding_r[idx]),
                        scale * kernel_size[idx]);
      };
      x_dims.push_back(valid_x_dim(0, 4));
      if (4 == src_size) {
        x_dims.push_back(valid_x_dim(1, 8));
      } else if (5 == src_size) {
        x_dims.push_back(valid_x_dim(1, 8));
        x_dims.push_back(valid_x_dim(2, 8));
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
      auto out_size = (x_dims[d] - ((kernel_size[d-2] - 1) * (dilates_[d-2] + 1) + 1)
          + (padding_l[d-2] + padding_r[d-2])) / strides[d-2] + 1;
      y_dims.push_back(out_size);
    }
    x_dtype =
        (dtype == data_type::bf16 || dtype == data_type::f16) ? dtype : x_dtype;
    auto y_dtype = dtype != data_type::s8 ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    auto src_query = src_desc;
    auto dst_query = dst_desc;
    if (is_channels_last) {
      if (4 == src_size) {
        src_query = src_desc.to_format(tag::nhwc);
        dst_query = dst_desc.to_format(tag::nhwc);
      } else if (5 == src_size) {
        src_query = src_desc.to_format(tag::ndhwc);
        dst_query = dst_desc.to_format(tag::ndhwc);
      } else if (3 == src_size) {
        src_query = src_desc.to_format(tag::nwc);
        dst_query = dst_desc.to_format(tag::nwc);
      }
    }

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is
    // required by winograd primitive. Then, in the cases of variable input
    // shape, the detials of dnnl_wino_fmt will be changed. And, extra weihgts
    // reorder is inevitable each time, leading to bad performance. Here, we set
    // the prop_kind to forward, in order to reorder and cache weights as
    // blocked format, instead of dnnl_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd &&
        aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }
    attr_t op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_query, weights_desc, tensor::desc(), dst_query, strides, dilates_,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, apkind);

    // embed group info into weights_desc
#ifdef __aarch64__
    return tensor::desc(pd.first.weights_desc(), groups);
#else
    return tensor::desc(pd.weights_desc(), groups);
#endif
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      data_type x_dtype = data_type::f32,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      bool is_channels_last = false,
      const engine& aengine = engine::cpu_engine()) {
    if (is_channels_last) {
      return expected_weights_desc<true>(
          weights_dims, dtype, strides, padding_l, padding_r, dilates, groups,
          aalgorithm, aprop_kind, x_dtype, src_dims, attr, aengine);
    } else {
      return expected_weights_desc<false>(
          weights_dims, dtype, strides, padding_l, padding_r, dilates, groups,
          aalgorithm, aprop_kind, x_dtype, src_dims, attr, aengine);
    }
  }
#ifdef __aarch64__
  template <bool with_bias>
  static std::pair<primitive_desc, dnnl::convolution_forward> get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      bool is_channels_last = false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_query = src_desc;
    auto weights_desc_query = weights_desc;
    auto bias_desc_query = with_bias ? bias_desc : tensor::desc();
    auto dst_desc_query = dst_desc;

    src_desc_query = src_desc.to_format_any();
    weights_desc_query = weights_desc.to_format_any();
    bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
    dst_desc_query = dst_desc.to_format_any();

    // For nhwc path, weight uses format_tag::any,
    // while activation uses format_tag::nhwc.
    auto ndims = src_desc.get_dims().size();
    if (is_channels_last) {
      auto memory_format = tag::nhwc;
      if (3 == ndims) {
        memory_format = tag::nwc;
      } else if (5 == ndims) {
        memory_format = tag::ndhwc;
      }
      src_desc_query = src_desc.to_format(memory_format);
      dst_desc_query = dst_desc.to_format(memory_format);
    }

    auto key = utils::create_key(
        aprop_kind,
        aalgorithm,
        src_desc_query,
        weights_desc_query,
        bias_desc_query,
        dst_desc_query,
        with_bias,
        strides,
        dilates,
        padding_l,
        padding_r,
        attr,
        omp_get_max_threads());

    dnnl::convolution_forward::primitive_desc pd;
    return fetch_or_create(key, [&]() {
      if (with_bias) {
         pd = primitive_desc(
            aengine,
            aprop_kind,
            aalgorithm,
            src_desc_query,
            weights_desc_query,
            bias_desc_query,
            dst_desc_query,
            strides,
            dilates,
            padding_l,
            padding_r,
            attr
            );
      } else {
        pd = primitive_desc(
            aengine,
            aprop_kind,
            aalgorithm,
            src_desc_query,
            weights_desc_query,
            dst_desc_query,
            strides,
            dilates,
            padding_l,
            padding_r,
            attr);
      }
      return std::make_pair(pd, super(pd));
    });
  }
#else
  template <bool with_bias>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      bool is_channels_last = false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_query = src_desc;
    auto weights_desc_query = weights_desc;
    auto bias_desc_query = with_bias ? bias_desc : tensor::desc();
    auto dst_desc_query = dst_desc;

    src_desc_query = src_desc.to_format_any();
    weights_desc_query = weights_desc.to_format_any();
    bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
    dst_desc_query = dst_desc.to_format_any();

    // For nhwc path, weight uses format_tag::any,
    // while activation uses format_tag::nhwc.
    auto ndims = src_desc.get_dims().size();
    if (is_channels_last) {
      auto memory_format = tag::nhwc;
      if (3 == ndims) {
        memory_format = tag::nwc;
      } else if (5 == ndims) {
        memory_format = tag::ndhwc;
      }
      src_desc_query = src_desc.to_format(memory_format);
      dst_desc_query = dst_desc.to_format(memory_format);
    }

    auto key = utils::create_key(
        aprop_kind,
        aalgorithm,
        src_desc_query,
        weights_desc_query,
        dst_desc_query,
        with_bias,
        bias_desc_query,
        strides,
        dilates,
        padding_l,
        padding_r,
        attr,
        omp_get_max_threads());
    return fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            aengine,
            aprop_kind,
            aalgorithm,
            src_desc_query,
            weights_desc_query,
            bias_desc_query,
            dst_desc_query,
            strides,
            dilates,
            padding_l,
            padding_r,
            attr
            );
      } else {
        return primitive_desc(
            aengine,
            aprop_kind,
            aalgorithm,
            src_desc_query,
            weights_desc_query,
            dst_desc_query,
            strides,
            dilates,
            padding_l,
            padding_r,
            attr);
      }
    });
  }
#endif

private:
  static bool use_gemm(const dims& src, const dims& weight, const dims& dst,
                       int groups) {
    if (groups != 1)
      return false;

    auto product = [](const dims& v, size_t start_offset = 0) {
      return std::accumulate(
          v.begin() + start_offset, v.end(), 1, std::multiplies<size_t>());
    };

    auto ker_spatial = product(weight, 2);
    bool pointwise = ker_spatial == 1;
    if (pointwise)
      return true;

    auto im2col_cost = ker_spatial * product(src);
    auto reorder_cost = product(src) + 2 * product(weight) + 2 * product(dst);
    return im2col_cost < reorder_cost;
  }

  // For fp32
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void compute_dispatch(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      bool is_channels_last = false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {

    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    tensor weights_grouped;
    dims dil_compatible;

    conv_deconv_utils::prepare_parameters(
        src, weights, bias, dst_dims, dst, dilates, groups,
        attr, with_bias, false, weights_grouped, dil_compatible,
        op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);

    // Used for to_mkldnn() path
    auto pd = get_primitive_desc<with_bias>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dil_compatible,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, aprop_kind, aengine);

#ifdef __aarch64__
    convolution_forward_params params(std::move(pd.first), std::move(pd.second), std::move(op_attr), groups);
#else
    dnnl::convolution_forward primitive(pd);
    convolution_forward_params params(std::move(pd), std::move(primitive), std::move(op_attr), groups);
#endif
    do_compute<with_bias, reorder_src, reorder_weight>(params, src, weights, bias, dst);
  }

  // For int8
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void compute_dispatch(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      bool is_channels_last,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {

    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    tensor weights_grouped;
    dims dil_compatible;

    conv_deconv_utils::prepare_parameters(
        src, weights, bias, dst_dims, dst, dilates, groups,
        src_scales, weights_scales, dst_scales, src_zero_point, dst_zero_point,
        attr, alowp_kind, with_bias, false,
        weights_grouped, dil_compatible, op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);

    // Used for to_mkldnn() path
    auto pd = get_primitive_desc<with_bias>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dil_compatible,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, aprop_kind, aengine);

#ifdef __aarch64__
    convolution_forward_params param(
        std::move(pd.first), std::move(pd.second), std::move(op_attr), groups, std::move(bias_attr));
#else
    dnnl::convolution_forward primitive(pd);
    convolution_forward_params param(
        std::move(pd), std::move(primitive), std::move(op_attr), groups, std::move(bias_attr));
#endif

    // Prepare tensors for scales and zero points
    if (param.op_attr.has_scales()) {
      if (!param.all_scales) {
        param.all_scales.reset(new std::unordered_map<int, tensor>);
      }
      for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
        int dnnl_arg = arg_scale_pair.first;
        const scale_t& scale = arg_scale_pair.second.first;
        tensor scales_m = tensor(scale);
        param.all_scales->insert({dnnl_arg, scales_m});
      }
    }
    if (param.op_attr.has_zero_points()) {
      if (!param.all_zero_points) {
        param.all_zero_points.reset(new std::unordered_map<int, tensor>);
      }
      for (auto& arg_zp_pair : param.op_attr.get_all_zero_points()) {
        int dnnl_arg = arg_zp_pair.first;
        const zero_point_t& zp = arg_zp_pair.second.first;
        tensor zp_m = tensor(zp);
        param.all_zero_points->insert({dnnl_arg, zp_m});
      }
    }
    do_compute<with_bias, reorder_src, reorder_weight>(param, src, weights, bias, dst);
  }

  // For fp32 with binary post-op
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void compute_binary_dispatch(
      const tensor &src,
      const tensor &other,
      const tensor &weights,
      const tensor &bias,
      const dims &dst_dims,
      tensor &dst,
      const dims &strides,
      const dims &dilates,
      const dims &padding_l,
      const dims &padding_r,
      int groups,
      bool is_channels_last,
      const attr_t &attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine &aengine = engine::cpu_engine()) {
    convolution_forward_params params;
    do_prepare<with_bias>(
        params, src, weights, bias, dst_dims, dst, strides, dilates, padding_l,
        padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    do_compute_binary<with_bias, reorder_src, reorder_weight>(
        params, src, other, weights, bias, dst);
  }

  // For fp32
  template <bool with_bias>
  static void do_prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      bool is_channels_last,
      const attr_t& attr,
      algorithm aalgorithm,
      prop_kind aprop_kind,
      const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    tensor weights_grouped;
    dims dil_compatible;

    conv_deconv_utils::prepare_parameters(
        src, weights, bias, dst_dims, dst, dilates, groups,
        attr, with_bias, false, weights_grouped, dil_compatible,
        op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);

    auto pd = get_primitive_desc<with_bias>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dil_compatible,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, aprop_kind, aengine);

#ifdef __aarch64__
    param = {std::move(pd.first), std::move(pd.second), std::move(op_attr), groups};
#else
    dnnl::convolution_forward primitive(pd);

    param = {std::move(pd), std::move(primitive), std::move(op_attr), groups};
#endif
  }

  // for int8
  template <bool with_bias>
  static void do_prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      bool is_channels_last,
      const attr_t& attr,
      algorithm aalgorithm,
      prop_kind aprop_kind,
      const lowp_kind alowp_kind,
      const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    tensor weights_grouped;
    dims dil_compatible;

    conv_deconv_utils::prepare_parameters(
        src, weights, bias, dst_dims, dst, dilates, groups,
        src_scales, weights_scales, dst_scales, src_zero_point, dst_zero_point,
        attr, alowp_kind, with_bias, false,
        weights_grouped, dil_compatible, op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);

    auto pd = get_primitive_desc<with_bias>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dil_compatible,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, aprop_kind, aengine);

#ifdef __aarch64__
    param = {std::move(pd.first), std::move(pd.second), std::move(op_attr), groups, std::move(bias_attr)};
#else
    dnnl::convolution_forward primitive(pd);

    param = {std::move(pd), std::move(primitive), std::move(op_attr), groups, std::move(bias_attr)};
#endif

    // Prepare tensors for scales and zero points
    if (param.op_attr.has_scales()) {
      if (!param.all_scales) {
        param.all_scales.reset(new std::unordered_map<int, tensor>);
      }
      // get_all_scales() returns unordered_map<int, pair of scale and mask>
      for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
        int dnnl_arg = arg_scale_pair.first;
        const scale_t& scale = std::get<0>(std::get<1>(arg_scale_pair));
        tensor scales_m(scale);
        param.all_scales->insert({dnnl_arg, scales_m});
      }
    }
    if (param.op_attr.has_zero_points()) {
      if (!param.all_zero_points) {
        param.all_zero_points.reset(new std::unordered_map<int, tensor>);
      }
      // get_all_zero_points() returns unordered_map<int, pair of zp and mask>
      for (auto& arg_zp_pair : param.op_attr.get_all_zero_points()) {
        int dnnl_arg = arg_zp_pair.first;
        const zero_point_t& zp = std::get<0>(std::get<1>(arg_zp_pair));
        tensor zp_m(zp);
        param.all_zero_points->insert({dnnl_arg, zp_m});
      }
    }
  }

  // do_compute with given primitive/pd, under the precondition
  // that whether or not src/weight/bias/dst need to be reorder
  // For both fp32 and int8
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void do_compute(const convolution_forward_params& param,
                         const tensor& src,
                         const tensor& weights,
                         const tensor& bias,
                         tensor& dst) {
    auto scratchpad = tensor(param.pd.scratchpad_desc());

    auto& expected_src = reorder_src ?
        src.reorder_if_differ_in(param.pd.src_desc()) : src;
    auto&& grouped_weights = weights.make_grouped_weights(param.groups);
    auto&& expected_weights = reorder_weight ?
        grouped_weights.reorder_if_differ_in(param.pd.weights_desc()) :
        grouped_weights;
    auto& primitive = param.primitive;
    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }
    // Insert tensors of scales and zero points to args
    if (param.all_scales && !param.all_scales->empty()) {
      for (auto& arg_scale_pair : *param.all_scales) {
        int dnnl_arg = arg_scale_pair.first;
        tensor& scales_m = std::get<1>(arg_scale_pair);
        args.insert({DNNL_ARG_ATTR_SCALES | dnnl_arg, scales_m});
      }
    }
    if (param.all_zero_points && !param.all_zero_points->empty()) {
      for (auto& arg_zp_pair : *param.all_zero_points) {
        int dnnl_arg = arg_zp_pair.first;
        tensor& zp_m = std::get<1>(arg_zp_pair);
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | dnnl_arg, zp_m});
      }
    }

    // Deal with dst tensor
    if (reorder_src) {
      auto expected_dst_desc = param.pd.dst_desc();
      tensor expected_dst;
      if (dst.is_empty() || dst.get_desc() != expected_dst_desc){
        // If dst buffer is not given by user or the given buffer is not in expected format,
        // We need to init a new one
        expected_dst.init(expected_dst_desc);
        if (!dst.is_empty() && param.op_attr.has_op_kind(kind::sum)) {
          // We need to copy the content of given buffer if the op is fused with sum
          expected_dst.feed_from(dst);
        }
      } else {
        // The format of given dst buffer is expected
        expected_dst = dst;
      }
      args.insert({DNNL_ARG_DST, expected_dst});
      primitive.execute(stream::default_stream(), args);
      // reorder back to dst's buffer if needed
      if (dst.is_empty() || dst.get_desc() == expected_dst.get_desc() ||
          !dst.get_desc().has_same_shape_as(expected_dst.get_desc())) {
        dst = expected_dst;
      } else {
        dst.feed_from(expected_dst);
      }
    } else {
      args.insert({DNNL_ARG_DST, dst});
      primitive.execute(stream::default_stream(), args);
    }
  }

  // For binary post-op
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void do_compute_binary(const convolution_forward_params &param,
                                const tensor &src,
                                const tensor &other,
                                const tensor &weights,
                                const tensor &bias,
                                tensor &dst) {
    auto &pd = param.pd;
    auto& primitive = param.primitive;
    auto scratchpad = tensor(pd.scratchpad_desc());
    auto& expected_src = reorder_src ?
        src.reorder_if_differ_in(pd.src_desc()) :
        src;
    // make sure other has same format with dst.
    // TODO: other has different with dst?
    auto& expected_other = reorder_src ?
        other.reorder_if_differ_in(pd.dst_desc()) :
        other;
    auto&& grouped_weights = weights.make_grouped_weights(param.groups);
    auto&& expected_weights = reorder_weight ?
        grouped_weights.reorder_if_differ_in(pd.weights_desc()) :
        grouped_weights;
    if (reorder_src) {
      dst.reinit_if_possible(pd.dst_desc());
    }

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_DST, dst});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                 expected_other});
    // Insert tensors of scales and zero points to args
    if (param.all_scales && !param.all_scales->empty()) {
      for (auto& arg_scale_pair : *param.all_scales) {
        int dnnl_arg = arg_scale_pair.first;
        tensor& scales_m = arg_scale_pair.second;
        args.insert({DNNL_ARG_ATTR_SCALES | dnnl_arg, scales_m});
      }
    }
    if (param.all_zero_points && !param.all_zero_points->empty()) {
      for (auto& arg_zp_pair : *param.all_zero_points) {
        int dnnl_arg = arg_zp_pair.first;
        tensor& zp_m = arg_zp_pair.second;
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | dnnl_arg, zp_m});
      }
    }
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }
    primitive.execute(stream::default_stream(), args);
  }
};


struct convolution_backward_data : public dnnl::convolution_backward_data {

  using super = dnnl::convolution_backward_data;

  static void compute_v2(const tensor& diff_dst,
                         const tensor& weights,
                         const dims& diff_src_dims,
                         tensor& diff_src,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         const int groups,
                         bool is_channels_last = false,
                         algorithm aalgorithm = algorithm::convolution_direct,
                        const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);
    auto format_tag = tag::any;
    auto ndims = diff_dst.get_desc().get_dims().size();
    if (ndims == 4) {
      if (is_channels_last) {
        format_tag = tag::nhwc;
      }
    } else if (ndims == 5) {
      if (is_channels_last) {
        format_tag = tag::ndhwc;
      }
    }
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    // align weight data type with diff_dst for bf16 and f16
    auto weights_desc = tensor::desc(weights_.get_dims(), diff_dst.get_data_type(), tag::any);
    if (groups > 1) {
      weights_desc = weights_desc.to_grouped(groups);
    }

    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef __aarch64__
    auto pd = primitive_desc(
        aengine, aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
        dilates_, padding_l, padding_r, forward_hints.first, op_attr);
#else
    auto pd = primitive_desc(
        aengine, aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
        dilates_, padding_l, padding_r, forward_hints, op_attr);
#endif

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src},
                       {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
    bool is_channels_last = is_nhwc || is_ndhwc;
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    // align weight data type with diff_dst for bf16 and f16
    auto weights_desc = tensor::desc(weights_.get_dims(), diff_dst.get_data_type(), tag::any);
    if (groups > 1) {
      weights_desc = weights_desc.to_grouped(groups);
    }

    auto diff_src_desc = 
        tensor::desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last, op_attr);

#ifdef __aarch64__
    auto pd = primitive_desc(
        aengine, aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
        dilates_, padding_l, padding_r, forward_hints.first, op_attr);
#else
    auto pd = primitive_desc(
        aengine, aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
        dilates_, padding_l, padding_r, forward_hints, op_attr);
#endif

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());

    auto expected_diff_src_desc = pd.diff_src_desc();
    tensor expected_diff_src;
    // diff_src not init in FW or has same desc with expected desc.
    if (diff_src.is_empty() || diff_src.get_desc() == expected_diff_src_desc) {
      diff_src.reinit_if_possible(expected_diff_src_desc);
      expected_diff_src = diff_src;
    } else {
      expected_diff_src.init(expected_diff_src_desc);
    }

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(), 
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, expected_diff_src},
                       {DNNL_ARG_SCRATCHPAD, scratchpad}});
    // diff_src has been init in FW side, but has diff desc with
    // expected_diff_src.
    if (diff_src.get_desc() != expected_diff_src_desc) {
      if (!diff_src.get_desc().has_same_shape_as(expected_diff_src_desc)) {
        diff_src.reinit_if_possible(expected_diff_src_desc);
      }
      diff_src.feed_from(expected_diff_src);
    }
  }
};


struct convolution_backward_weights
    : public dnnl::convolution_backward_weights {

  using super = dnnl::convolution_backward_weights;

  static void compute_v2(const tensor& src,
                         const tensor& diff_dst,
                         const dims& diff_weights_dims,
                         tensor& diff_weights,
                         tensor& diff_bias,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         int groups,
                         bool is_channels_last = false,
                         const attr_t& attr = attr_t(),
                         const data_type diff_weight_type = data_type::undef,
                         algorithm aalgorithm = algorithm::convolution_direct,
                        const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last,
        attr, diff_weight_type, aalgorithm, aengine);
  }

  static void compute_v2(const tensor& src,
                         const tensor& diff_dst,
                         const dims& diff_weights_dims,
                         tensor& diff_weights,
                         const dims& strides,
                         const dims& dilates,
                         const dims& padding_l,
                         const dims& padding_r,
                         int groups,
                         bool is_channels_last = false,
                         const attr_t& attr = attr_t(),
                         const data_type diff_weight_type = data_type::undef,
                         algorithm aalgorithm = algorithm::convolution_direct,
                         const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last,
        attr, diff_weight_type, aalgorithm, aengine);
  }

  // DEPRECATED
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      const attr_t& attr = attr_t(),
                      const data_type diff_weight_type = data_type::undef,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    bool is_channels_last = is_nhwc || is_ndhwc;
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last,
        attr, diff_weight_type, aalgorithm, aengine);
  }

  // DEPRECATED
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      const attr_t& attr = attr_t(),
                      const data_type diff_weight_type = data_type::undef,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    bool is_channels_last = is_nhwc || is_ndhwc;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last,
        attr, diff_weight_type, aalgorithm, aengine);
  }

 private:
  template <bool with_diff_bias>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           const dims& diff_weights_dims,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           const int groups,
                           bool is_channels_last,
                           const attr_t& attr,
                           const data_type diff_weight_type,
                           algorithm aalgorithm,
                           const engine& aengine) {
    data_type diff_dst_type = diff_dst.get_data_type();
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst_type,
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // make diff_weights and dilates compatible with DNNL
    auto dilates_ = utils::get_compatible_dilates(dilates);
    data_type diff_weight_type_in = data_type::undef == diff_weight_type ?
                                    diff_dst_type : diff_weight_type;
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_weight_type_in, tag::any);
    if (groups > 1) {
        diff_weights_desc = diff_weights_desc.to_grouped(groups).to_format_any();
    }

    auto format_tag = tag::any;
    auto ndims = diff_dst.get_desc().get_dims().size();
    if (ndims == 4) {
      if (is_channels_last) {
        format_tag = tag::nhwc;
      }
    } else if (ndims == 5) {
      if (is_channels_last) {
        format_tag = tag::ndhwc;
      }
    }
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    auto src_desc = src.get_desc().to_format(format_tag);

    auto diff_bias_desc =     
        tensor::desc({diff_dst.get_dim(1)}, diff_weight_type_in, tag::any);

    // for forward hint, weights_desc should have same data_type
    // with other input desc, expect for bias_desc
    auto weights_desc = diff_weights_desc;
    if (diff_weight_type_in != diff_dst_type) {
      weights_desc = tensor::desc(diff_weights_desc.get_dims(), diff_dst_type, tag::any);
      if (groups > 1) {
        weights_desc = weights_desc.to_grouped(groups);
      }
    }
    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
            src_desc, weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last, op_attr, aalgorithm,
            prop_kind::forward, aengine);

#ifdef __aarch64__
    auto pd = with_diff_bias
        ? primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r, forward_hints.first, op_attr)
        : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r, forward_hints.first, op_attr);
#else
    auto pd = with_diff_bias
        ? primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                         diff_bias_desc, diff_dst_desc, strides, dilates_,
                         padding_l, padding_r, forward_hints, op_attr)
        : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                         diff_dst_desc, strides, dilates_,
                         padding_l, padding_r, forward_hints, op_attr);
#endif

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    // embed group info into diff_weights_desc
    auto expected_diff_weights_desc =
        tensor::desc(pd.diff_weights_desc(), groups);
    tensor expected_diff_weights;
    // diff_weights not init in FW or has same desc with expected desc.
    if (diff_weights.is_empty() ||
        diff_weights.get_desc() == expected_diff_weights_desc) {
      diff_weights.reinit_if_possible(expected_diff_weights_desc);
      expected_diff_weights = diff_weights;
    } else {
      expected_diff_weights.init(expected_diff_weights_desc);
    }

    tensor scratchpad(pd.scratchpad_desc());

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
                         {DNNL_ARG_DIFF_BIAS, diff_bias},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    }
    // diff_weights has been init in FW side, but has diff desc with
    // expected_diff_weights.
    if (diff_weights.get_desc() != expected_diff_weights_desc) {
      if (!diff_weights.get_desc().has_same_shape_as(expected_diff_weights_desc)) {
        diff_weights.reinit_if_possible(expected_diff_weights_desc);
      }
      diff_weights.feed_from(expected_diff_weights);
    }
  }
};
}  // namespace ideep

#endif