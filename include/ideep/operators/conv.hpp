#ifndef IDEEP_OPERATORS_CONV_HPP
#define IDEEP_OPERATORS_CONV_HPP

namespace ideep {

struct convolution_forward_params {
  dnnl::convolution_forward::primitive_desc pd;
  // bias_attr contains requantization scales for bias
  attr_t bias_attr;
  scale_t dst_scales;
  zero_point_t src_zero_point;
  int groups;
  tensor scratchpad;
};

struct conv_deconv_utils {
  /// Common logic to prepare parameters for conv/deconv.
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
    op_attr = attr;

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
      const auto& dst_zero_point = dst.has_zero_point() ? dst.get_zero_point() :
                                   dst_zero_points.empty() ? default_zero_point : dst_zero_points;
      const auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
      const auto weights_zero_point_size = 1;
      const auto dst_zero_point_size = static_cast<dim>(dst_zero_point.size());
      IDEEP_ENFORCE(src_zero_point_size == 1 && dst_zero_point_size == 1,
                    "DNNL only support 1-dim zero_point");

      scale_t bias_scales, op_scales;
      std::tie(bias_scales, op_scales) = utils::compute_scales(
          src_scales_in[0], dst_scales_in[0], weights_scales_in);

      if (attr.has_op_kind(kind::sum)) {
        float sum_scale =
            dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
        if (attr.has_op_kind(kind::eltwise)) {
          op_attr = attr_t::residual(sum_scale);
        } else {
          op_attr = attr_t::fuse_sum(sum_scale);
        }
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);
      zero_point_t src_zero_point_in_attr;
      int zp_mask = utils::tensor_zp_mask(1);
      attr.get_zero_points(DNNL_ARG_SRC, zp_mask, src_zero_point_in_attr);
      if (src_zero_point_in_attr == zero_point_t({DNNL_RUNTIME_S32_VAL})) { // runtime src zero point
        op_attr.set_zero_points(DNNL_ARG_SRC,
                                zp_mask,
                                src_zero_point_in_attr);
      } else {
        op_attr.set_zero_points(DNNL_ARG_SRC,
                                ideep::utils::tensor_zp_mask(src_zero_point_size),
                                src_zero_point);
      }
      op_attr.set_zero_points(DNNL_ARG_WEIGHTS,
                              ideep::utils::tensor_zp_mask(weights_zero_point_size),
                              zero_point_t(1, weights_zero_point[0]));
      if (dst_data_type != data_type::f32) {
        op_attr.set_zero_points(DNNL_ARG_DST,
                                ideep::utils::tensor_zp_mask(dst_zero_point_size),
                                dst_zero_point);
      }

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8, tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      weights_desc = weight_grouped.get_desc().to_type(data_type::s8);
      if (weight_grouped.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, groups > 1),
                        weights_scales_in};
      }

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::f32, tag::any}; // Use f32 instead of s32 to improve accuracy
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false),
                        bias_scales};
        }
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
      weights_desc = weight_grouped.get_desc().to_type(dst_data_type);

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

  /// Get true zero point from input tensor, specified zero point or op attr
  /// Priority: input.get_zero_point() > input_zero_point > op_attr > default
  ///
  /// @param input Get the true zero point from this tensor.
  /// @param arg_idx Parameter argument index as passed to the
  ///     primitive::execute() call. Such as DNNL_ARG_SRC.
  /// @param op_attr Attr of the conv/deconv operation.
  /// @param aengine Cpu execution engine.
  /// @param zero_point Output tensor of zero points.
  static void obtain_runtime_zero_point(const tensor& input,
                                        const zero_point_t& input_zero_point,
                                        const int& arg_idx,
                                        const dnnl::primitive_attr& op_attr,
                                        const engine& aengine,
                                        tensor& zero_point /* Output */) {
    zero_point_t src_zero_point_in_attr;
    int zp_mask = utils::tensor_zp_mask(1);
    op_attr.get_zero_points(arg_idx, zp_mask, src_zero_point_in_attr);
    dim src_zero_point_size = 1;
    const zero_point_t* zero_point_data = NULL;
    const zero_point_t default_zero_point = {0};
    if (input.has_zero_point()) {
      src_zero_point_size = static_cast<dim>(input.get_zero_point().size());
      zero_point_data = &input.get_zero_point();
    } else if (!input_zero_point.empty()) {
      src_zero_point_size = static_cast<dim>(input_zero_point.size());
      zero_point_data = &input_zero_point;
    } else if (src_zero_point_in_attr == zero_point_t({DNNL_RUNTIME_S32_VAL}) ||
        src_zero_point_in_attr.empty()) { // runtime zero point of input
      src_zero_point_size = static_cast<dim>(default_zero_point.size());
      zero_point_data = &default_zero_point;
    } else {
      src_zero_point_size = static_cast<dim>(src_zero_point_in_attr.size());
      zero_point_data = &src_zero_point_in_attr;
    }
    tensor::desc src_zero_point_desc = {{src_zero_point_size}, data_type::s32, {1}};
    zero_point.init(src_zero_point_desc, aengine);
    auto src_z = reinterpret_cast<int32_t *>(zero_point.get_data_handle());
    for (memory::dim i = 0; i < src_zero_point_size; ++i) // fill in zero point data
      src_z[i] = (*zero_point_data)[i];

  }
};

struct convolution_forward
    : public dnnl::convolution_forward,
      utils::computation_cache<dnnl::convolution_forward::primitive_desc> {

  using super = dnnl::convolution_forward;

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
      compute_dispatch_v2<false>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_dispatch_v2<true>(
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
    compute_dispatch_v2<false>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // 3-in-1 compute (prepare & compute) with bias
  // Bias is not used if it is empty.
  // This function is used to conv+binary fusion.
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
			     bool is_channels_last = false,
                             const attr_t &attr = attr_t(),
                             algorithm aalgorithm = algorithm::convolution_direct,
                             prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_binary_dispatch</*with_bias=*/false>(
          src, other, weights, bias, dst_dims, dst, strides, dilates, padding_l,
          padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_binary_dispatch</*with_bias=*/true>(
          src, other, weights, bias, dst_dims, dst, strides, dilates, padding_l,
          padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 3-in-1 compute (prepare & compute) without bias
  // Bias is not used if it is empty.
  // This function is used to conv+binary fusion.
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
                             bool is_channels_last = false,
			     const attr_t &attr = attr_t(),
                             algorithm aalgorithm = algorithm::convolution_direct,
                             prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_binary_dispatch</*with_bias=*/false>(
        src, other, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
  }

  // 2-in-1 compute (prepare & compute) with bias
  // Bias is not used if it is empty.
  // Zero points are passed explicitly as arguments for quantization
  template <bool plain_format = false>
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
    if (bias.is_empty()) {
      compute_dispatch</*with_bias=*/false, plain_format>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_dispatch</*with_bias=*/true, plain_format>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // 2-in-1 compute (prepare & compute) without bias
  // Zero points are passed explicitly as arguments for quantization
  template <bool plain_format = false>
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
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, plain_format>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

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
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_prepare</*with_bias=*/false, /*keep_format=*/false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          zero_point_t(), zero_point_t(), attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      do_prepare</*with_bias=*/true, /*keep_format=*/false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          zero_point_t(), zero_point_t(), attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

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
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false, /*keep_format=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        zero_point_t(), zero_point_t(), attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Compute with bias
  // Bias is not used if it is empty.
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst) {
    if (bias.is_empty()) {
      do_compute</*with_bias=*/false>(param, src, weights, bias, dst);
    } else {
      do_compute</*with_bias=*/true>(param, src, weights, bias, dst);
    }
  }

  // Compute without bias
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false>(param, src, weights, dummy_bias, dst);
  }

  // Compute with given primitive & src zero point with or without bias
  static void compute(const super::primitive_desc pd,
                      const super& primitive,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& expected_bias,
                      tensor& dst,
                      const tensor& src_zero_point,
                      int groups) {
    if (expected_bias.is_empty()) {
      do_compute</*with_bias=*/false>(
          pd, primitive, src, weights, expected_bias, dst, src_zero_point, groups);
    } else {
      do_compute</*with_bias=*/true>(
          pd, primitive, src, weights, expected_bias, dst, src_zero_point, groups);
    }
  }

  // Deprecated. 2-in-1 compute (prepare & compute) with bias
  // Zero points are set to tensor for quantization
  template <bool plain_format = false>
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
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_dispatch</*with_bias=*/true, plain_format>(
        src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        zero_point_t(), zero_point_t(), attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Deprecated. 2-in-1 compute (prepare & compute) without bias
  // Zero points are set to tensor for quantization
  template <bool plain_format = false>
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
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_dispatch</*with_bias=*/false, plain_format>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        zero_point_t(), zero_point_t(), attr, aalgorithm, aprop_kind, alowp_kind, aengine);
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

    auto src_size = weights_dims.size(); // weights_dims is 4 for conv2d and 5 for conv3d
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
    }
    kernel_size.push_back(dims_in[ndims - 2]);
    kernel_size.push_back(dims_in[ndims - 1]);
    if (src_dims.empty()) {
      // Construct a dummy case
      x_dims.push_back(1);
      x_dims.push_back(ic);
      y_dims.push_back(1);
      y_dims.push_back(oc);
      if (4 == src_size) {
        x_dims.push_back(4 * kernel_size[0]);
        x_dims.push_back(4 * kernel_size[1]);
      } else {
        x_dims.push_back(8 * kernel_size[0]);
        x_dims.push_back(8 * kernel_size[1]);
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
      auto out_size = (x_dims[d] - ((kernel_size[d-2] - 1) * (dilates_[d-2] + 1) + 1)
          + (padding_l[d-2] + padding_r[d-2])) / strides[d-2] + 1;
      y_dims.push_back(out_size);
    }
    x_dtype = dtype == data_type::bf16 ? dtype : x_dtype;
    auto y_dtype = dtype != data_type::s8 ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    auto src_query = src_desc;
    auto dst_query = dst_desc;
    if (is_channels_last) {
      if (src_size == 4) {
        src_query = src_desc.to_format(tag::nhwc);
        dst_query = dst_desc.to_format(tag::nhwc);
      } else if (src_size == 5) {
        src_query = src_desc.to_format(tag::ndhwc);
        dst_query = dst_desc.to_format(tag::ndhwc);
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
    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_query, weights_desc, tensor::desc(), dst_query, strides, dilates_,
        padding_l, padding_r, is_channels_last, attr, aalgorithm, apkind);

    // embed group info into weights_desc
    return tensor::desc(pd.weights_desc(), groups);
  }

  // [keep_format]
  // - Set to true would let onednn to choose the optimal
  //   blocked format for dst tensor
  // - Set to false would keep dst tensor format as it is.
  //   We used this mode in pytorch plain-in-plain-out path to force
  //   the dst to be plain as src, so that it would also instruct onednn
  //   to use gemm-based conv implementation. Apply to both NCHW and NHWC.
  template <bool with_bias, bool keep_format = false>
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
    if (!keep_format) {
      src_desc_query = src_desc.to_format_any();
      weights_desc_query = weights_desc.to_format_any();
      bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
      dst_desc_query = dst_desc.to_format_any();
    }

    // For nhwc path, weight uses format_tag::any,
    // while activation uses format_tag::nhwc.
    auto ndims = src_desc.get_dims().size();
    if (ndims == 4) {
      if (is_channels_last) {
        src_desc_query = src_desc.to_format(tag::nhwc);
        weights_desc_query = weights_desc.to_format_any();
        bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
        dst_desc_query = dst_desc.to_format(tag::nhwc);
      }
    } else if (ndims == 5) {
      if (is_channels_last) {
        src_desc_query = src_desc.to_format(tag::ndhwc);
        weights_desc_query = weights_desc.to_format_any();
        bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
        dst_desc_query = dst_desc.to_format(tag::ndhwc);
      }
    }

    auto key = utils::create_key(
        aprop_kind,
        aalgorithm,
        src_desc_query,
        weights_desc_query,
        with_bias,
        strides,
        dilates,
        padding_l,
        padding_r,
        attr,
        omp_get_max_threads());
    return fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            {aprop_kind,
             aalgorithm,
             src_desc_query,
             weights_desc_query,
             bias_desc_query,
             dst_desc_query,
             strides,
             dilates,
             padding_l,
             padding_r},
            attr,
            aengine);
      } else {
        return primitive_desc(
            {aprop_kind,
             aalgorithm,
             src_desc_query,
             weights_desc_query,
             dst_desc_query,
             strides,
             dilates,
             padding_l,
             padding_r},
            attr,
            aengine);
      }
    });
  }

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

  template <bool with_bias, bool plain_format>
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
    convolution_forward_params params;

    if (plain_format) {
      // Used for pytorch default CPU path, i.e. plain-in-plain-out
      // see [keep_format] for more details
      bool is_channels_last =
          src.get_desc().is_nhwc() || src.get_desc().is_ndhwc();
      bool use_plain_dst =
          use_gemm(src.get_dims(), weights.get_dims(), dst_dims, groups) ||
          is_channels_last;
      if (use_plain_dst) {
        do_prepare<with_bias, /*keep_format=*/true>(
            params, src, weights, bias, dst_dims, dst, strides, dilates,
            padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
            src_zero_point, dst_zero_point, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
        do_compute<with_bias>(params, src, weights, bias, dst);
      } else {
        tensor dst_blocked;
        do_prepare<with_bias, /*keep_format=*/false>(
            params, src, weights, bias, dst_dims, dst_blocked, strides, dilates,
            padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
            src_zero_point, dst_zero_point, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
        do_compute<with_bias>(params, src, weights, bias, dst_blocked);
        dst.feed_from(dst_blocked);
      }
    } else {
      // Used for to_mkldnn() path
      do_prepare<with_bias, /*keep_format=*/false>(
          params, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
      do_compute<with_bias>(params, src, weights, bias, dst);
    }
  }

  template <bool with_bias>
  static void compute_dispatch_v2(
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

    convolution_forward_params params;
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    tensor weights_grouped;
    dims dil_compatible;

    conv_deconv_utils::prepare_parameters(
        src, weights, bias, dst_dims, dst, dilates, groups,
        scale_t(), scale_t(), scale_t(), zero_point_t(), zero_point_t(),
        attr, u8s8, with_bias, false,
        weights_grouped, dil_compatible, op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);
        auto ndims = src_desc.get_dims().size();

    auto pd = get_primitive_desc<with_bias, false>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dil_compatible,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, aprop_kind, aengine);

    // allocate scratchpad
    tensor scratchpad(pd.scratchpad_desc());

    params = {std::move(pd), bias_attr, scale_t(), zero_point_t(),
             groups, std::move(scratchpad)};

    do_compute<with_bias>(params, src, weights, bias, dst);
  }

  template <bool with_bias>
  static void compute_binary_dispatch(const tensor &src,
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
                                      const attr_t &attr = attr_t(),
                                      algorithm aalgorithm = algorithm::convolution_direct,
                                      prop_kind aprop_kind = prop_kind::forward,
                                      const engine &aengine = engine::cpu_engine()) {
    convolution_forward_params params;
    do_prepare<with_bias, /*keep_format=*/false>(
        params, src, weights, bias, dst_dims, dst, strides, dilates, padding_l,
        padding_r, groups, scale_t(), scale_t(), scale_t(), zero_point_t(),
        zero_point_t(), attr, aalgorithm, aprop_kind, u8s8, aengine);
    do_compute_binary<with_bias>(params, src, other, weights, bias, dst);
  }

  template <bool with_bias, bool keep_format>
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

    auto ndims = src_desc.get_dims().size();
    bool is_channels_last = false;
    if (ndims == 4) {
      is_channels_last = src_desc.is_nhwc();
    } else if (ndims == 5) {
      is_channels_last = src_desc.is_ndhwc();
    }
    auto pd = get_primitive_desc<with_bias, keep_format>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dil_compatible,
        padding_l, padding_r, is_channels_last, op_attr, aalgorithm, aprop_kind, aengine);

    // allocate scratchpad
    tensor scratchpad(pd.scratchpad_desc());

    param = {std::move(pd), bias_attr, std::move(dst_scales), std::move(src_zero_point),
             groups, std::move(scratchpad)};
  }

  template <bool with_bias>
  static void do_compute(const convolution_forward_params& param,
                         const tensor& src, const tensor& weights,
                         const tensor& bias, tensor& dst) {
    auto& pd = param.pd;
    auto& scratchpad = param.scratchpad;
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights.make_grouped_weights(param.groups)
                                .reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (!param.dst_scales.empty() && dst.get_data_type() != data_type::f32) {
      dst.set_scale(param.dst_scales);
    }

    tensor src_zero_point_m;
    conv_deconv_utils::obtain_runtime_zero_point(
        src, param.src_zero_point, DNNL_ARG_SRC, pd.get_primitive_attr(),
        ideep::engine(pd.get_engine().get_kind()), src_zero_point_m);
    if (with_bias) {
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), param.bias_attr);
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m}});
    } else {
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m}});
    }
  }

  template <bool with_bias>
  static void do_compute_binary(const convolution_forward_params &param,
                                const tensor &src,
                                const tensor &other,
                                const tensor &weights,
                                const tensor &bias,
                                tensor &dst) {
    auto &pd = param.pd;
    auto &scratchpad = param.scratchpad;
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    // make sure other has same format with dst.
    // TODO: other has different with dst?
    auto expected_other = other.reorder_if_differ_in(pd.dst_desc());
    auto expected_weights = weights.make_grouped_weights(param.groups)
                                .reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());
    if (with_bias) {
      auto expected_bias =
          bias.reorder_if_differ_in(pd.bias_desc(), param.bias_attr);
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                          expected_other}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                          expected_other}});
    }
  }

  // Do_compute with given primitive & src zero point
  // Bias scale has been applied before passed in.
  template <bool with_bias>
  static void do_compute(const super::primitive_desc& pd,
                         const super& primitive,
                         const tensor& src,
                         const tensor& weights,
                         const tensor& expected_bias,
                         tensor& dst,
                         const tensor& src_zero_point,
                         int groups) {
    auto scratchpad = tensor(pd.scratchpad_desc());
    auto weights_grouped = weights.make_grouped_weights(groups);
    if (with_bias) {
      primitive.execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, src},
                         {DNNL_ARG_WEIGHTS, weights_grouped},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point}});
    } else {
      primitive.execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, src},
                         {DNNL_ARG_WEIGHTS, weights_grouped},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point}});
    }
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
    // align weight data type with diff_dst for bf16
    auto weights_desc =
        weights_.get_desc().to_format_any().to_type(diff_dst.get_data_type());

    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, op_attr, aengine, forward_hints);

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
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
    bool is_channels_last = is_nhwc || is_ndhwc;
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    // align weight data type with diff_dst for bf16
    auto weights_desc =
        weights_.get_desc().to_format_any().to_type(diff_dst.get_data_type());

    auto diff_src_desc = 
        tensor::desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, op_attr, aengine, forward_hints);

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
                         const data_type diff_weight_type = data_type::undef,
                         algorithm aalgorithm = algorithm::convolution_direct,
                        const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last, diff_weight_type, aalgorithm, aengine);
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
                         const data_type diff_weight_type = data_type::undef,
                         algorithm aalgorithm = algorithm::convolution_direct,
                         const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last, diff_weight_type, aalgorithm, aengine);
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
                      const data_type diff_weight_type = data_type::undef,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    bool is_channels_last = is_nhwc || is_ndhwc;
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last,
        diff_weight_type, aalgorithm, aengine);
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
        diff_weight_type, aalgorithm, aengine);
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
                           const data_type diff_weight_type,
                           algorithm aalgorithm,
                           const engine& aengine) {

    // make diff_weights and dilates compatible with DNNL
    auto dilates_ = utils::get_compatible_dilates(dilates);
    data_type diff_dst_type = diff_dst.get_data_type();
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
      weights_desc = weights_desc.to_type(diff_dst_type);
    }
    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
            src_desc, weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last, attr_t(), aalgorithm,
            prop_kind::forward, aengine);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = with_diff_bias
        ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, op_attr, aengine, forward_hints)
        : primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    // embed group info into diff_weights_desc
    auto expected_diff_weights_desc =
        tensor::desc(pd.diff_weights_desc(), groups);
    diff_weights.reinit_if_possible(expected_diff_weights_desc);

    tensor scratchpad(pd.scratchpad_desc());

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights},
                         {DNNL_ARG_DIFF_BIAS, diff_bias},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    }
  }
};
}  // namespace ideep

#endif
