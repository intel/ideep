#ifndef IDEEP_OPERATORS_DECONV_HPP
#define IDEEP_OPERATORS_DECONV_HPP

namespace ideep {

struct deconv_forward_params {
  deconv_forward_params() {}

  deconv_forward_params(
      dnnl::deconvolution_forward::primitive_desc&& pd,
      dnnl::deconvolution_forward&& primitive,
      int groups,
      attr_t&& op_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        groups(groups),
        op_attr(std::move(op_attr)),
        bias_attr(attr_t()) {}

  deconv_forward_params(
      dnnl::deconvolution_forward::primitive_desc&& pd,
      dnnl::deconvolution_forward&& primitive,
      int groups,
      attr_t&& op_attr,
      attr_t&& bias_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        groups(groups),
        op_attr(std::move(op_attr)),
        bias_attr(std::move(bias_attr)) {}

  dnnl::deconvolution_forward::primitive_desc pd;
  dnnl::deconvolution_forward primitive;
  int groups;
  attr_t op_attr;
  attr_t bias_attr;
  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, tensor>> all_zero_points = nullptr;
};

struct convolution_transpose_forward
    : public dnnl::deconvolution_forward,
      utils::computation_cache<dnnl::deconvolution_forward::primitive_desc> {

  using super = dnnl::deconvolution_forward;

  // 2-in-1 Compute for fp32
  // With bias. Bias is disabled if it is empty.
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered.
  // Set them to False if you are sure they don't need reordering.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_v3(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      bool is_channels_last=false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 2-in-1 Compute for fp32
  // Without bias.
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered.
  // Set them to False if you are sure they don't need reordering.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_v3(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      bool is_channels_last=false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    static const tensor dummy_bias;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // 2-in-1 Compute for fp32
  // With bias. Bias is disabled if it is empty.
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // 2-in-1 Compute for fp32
  // Without bias.
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static const tensor dummy_bias;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }
  
  // 2-in-1 Compute for int8
  // With bias. Bias is disabled if it is empty.
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered.
  // Set them to False if you are sure they don't need reordering.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_v3(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      bool is_channels_last,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // 2-in-1 Compute for int8
  // Without bias.
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered.
  // Set them to False if you are sure they don't need reordering.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_v3(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      bool is_channels_last,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static const tensor dummy_bias;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // 2-in-1 Compute for int8
  // With bias. Bias is disabled if it is empty.
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // 2-in-1 Compute for int8
  // Without bias.
  // This one does not have `is_channels_last` argument for compatibility
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static const tensor dummy_bias;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Prepare for fp32
  // With bias. Bias is disabled if it is empty.
  static void prepare_v3(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      bool is_channels_last=false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_prepare<false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      do_prepare<true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // Prepare for fp32
  // Without bias.
  static void prepare_v3(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      bool is_channels_last=false,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    static const tensor dummy_bias;
    do_prepare<false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // Prepare for fp32
  // With bias. Bias is disabled if it is empty.
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      do_prepare<false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    } else {
      do_prepare<true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    }
  }

  // Prepare for fp32
  // Without bias.
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates = {1, 1},
      int groups = 1,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static const tensor dummy_bias;
    do_prepare<false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
  }

  // Prepare for int8
  // With bias. Bias is disabled if it is empty.
  static void prepare_v3(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates, // default = {1, 1}
      int groups, // default = 1
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      bool is_channels_last,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_prepare<false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      do_prepare<true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // Prepare for int8
  // Without bias.
  static void prepare_v3(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates, // default = {1, 1}
      int groups, // default = 1
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      bool is_channels_last,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static const tensor dummy_bias;
    do_prepare<false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Prepare for int8
  // With bias. Bias is disabled if it is empty.
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates, // default = {1, 1}
      int groups, // default = 1
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      do_prepare<false>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      do_prepare<true>(
          param, src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // Prepare for int8
  // Without bias.
  // This one does not have `is_channels_last` argument for compatibility
  static void prepare(
      deconv_forward_params& param,
      const tensor& src,
      const tensor& weights, // dim: {o, i[, d], h, w}
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& padding_l,
      const dims& padding_r,
      const dims& dilates, // default = {1, 1}
      int groups, // default = 1
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_point,
      const zero_point_t& dst_zero_point,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static const tensor dummy_bias;
    do_prepare<false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // Compute with prepared params. For both fp32 and int8
  // With bias. Bias is disabled if it is empty.
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered.
  // Set them to False if you are sure they don't need reordering.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const deconv_forward_params& param,
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

  // Compute with prepared params. For both fp32 and int8
  // Without bias.
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered.
  // Set them to False if you are sure they don't need reordering.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const deconv_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static const tensor dummy_bias;
    do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, weights, dummy_bias, dst);
  }

  // Deprecated
  // With bias. Zero points are passed explicitly as arguments for quantization
  // Bias is not used if it is empty.
  static void compute_v2(const tensor& src,
                         const tensor& weights, // dim: {o, i[, d], h, w}
                         const tensor& bias,
                         const dims& dst_dims,
                         tensor& dst,
                         const dims& strides,
                         const dims& padding_l,
                         const dims& padding_r,
                         const dims& dilates = {1, 1},
                         int groups = 1,
                         const scale_t& src_scales = scale_t(),
                         const scale_t& weights_scales = scale_t(),
                         const scale_t& dst_scales = scale_t(),
                         const zero_point_t& src_zero_point = zero_point_t(),
                         const zero_point_t& dst_zero_point = zero_point_t(),
                         const attr_t& attr = attr_t(),
                         algorithm aalgorithm = algorithm::deconvolution_direct,
                         prop_kind aprop_kind = prop_kind::forward,
                         const lowp_kind alowp_kind = u8s8,
                         const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, true, true>(
          src, weights, bias, dst_dims, dst, strides, dilates,
          padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    }
  }

  // Deprecated
  // Without bias. Zero points are passed explicitly as arguments for quantization
  static void compute_v2(const tensor& src,
                         const tensor& weights, // dim: {o, i[, d], h, w}
                         const dims& dst_dims,
                         tensor& dst,
                         const dims& strides,
                         const dims& padding_l,
                         const dims& padding_r,
                         const dims& dilates = {1, 1},
                         int groups = 1,
                         const scale_t& src_scales = scale_t(),
                         const scale_t& weights_scales = scale_t(),
                         const scale_t& dst_scales = scale_t(),
                         const zero_point_t& src_zero_point = zero_point_t(),
                         const zero_point_t& dst_zero_point = zero_point_t(),
                         const attr_t& attr = attr_t(),
                         algorithm aalgorithm = algorithm::deconvolution_direct,
                         prop_kind aprop_kind = prop_kind::forward,
                         const lowp_kind alowp_kind = u8s8,
                         const engine& aengine = engine::cpu_engine()) {
    bool is_channels_last = src.get_desc().is_channels_last() || weights.get_desc().is_channels_last();
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false, true, true>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  /// @param is_channels_last Indicate whether weight is channels-last or not.
  /// @param transposed If true, return weight desc in [i, o, ...] or [g, i/g, o, ...]
  ///                   format, which is used by PyTorch/Aten. If false, return desc in
  ///                   [o, i, ...] or [g, o, i/g, ...] format used by oneDNN. By default
  ///                   it is true for backward compatibility (e.g. Caffe2)
  template <bool is_channels_last = false, bool transposed = true>
  static tensor::desc expected_weights_desc(
      const dims& weights_dims,   // [i, o, ...]
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {

    auto src_size = weights_dims.size(); // weights_dims is 4 for conv2d and 5 for conv3d
    auto grouped = groups > 1;
    auto weights_dims_g =
        grouped ? utils::group_dims(weights_dims, groups) : weights_dims;
    // (g)iohw -> (g)oihw
    std::swap(weights_dims_g[grouped + 0], weights_dims_g[grouped + 1]);
    auto weights_desc = tensor::desc(weights_dims_g, dtype);

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto g = grouped ? dims_in[0] : 1;
    auto dilates_ = utils::get_compatible_dilates(dilates);

    dims x_dims, y_dims, kernel_size;
    auto ic = g * dims_in[1 + grouped];
    auto oc = g * dims_in[0 + grouped];
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
      auto valid_x_dim = [=](int idx) {
          return std::max((padding_l[idx] + padding_r[idx] - (1 + (kernel_size[idx] - 1) * dilates[idx])) / strides[idx] + 2,
                          2 * kernel_size[idx]);
      };
      if (4 == src_size) {
        x_dims.push_back(valid_x_dim(0));
        x_dims.push_back(valid_x_dim(1));
      } else {
        x_dims.push_back(valid_x_dim(0));
        x_dims.push_back(valid_x_dim(1));
        x_dims.push_back(valid_x_dim(2));
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
      auto out_size = (x_dims[d] - 1) * strides[d-2] + (1 + (kernel_size[d-2] - 1) * (dilates[d-2])) - padding_l[d-2] - padding_r[d-2];
      y_dims.push_back(out_size);
    }
    auto x_dtype = (dtype != data_type::s8) ? dtype : data_type::u8;
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);


    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
        padding_l, padding_r, is_channels_last, attr, aalgorithm, aprop_kind);

    // embed group info into weights_desc
    if (grouped) {
      if (transposed) {
        // [g, o, i/g, ...] -> [g, i/g, o, ...]
        return tensor::desc(pd.weights_desc(), groups).transpose(1, 2);
      } else {
        return tensor::desc(pd.weights_desc(), groups);
      }
    } else {
      if (transposed) {
        // [o, i, ...] -> [i, o, ...]
        return tensor::desc(pd.weights_desc(), groups).transpose(0, 1);
      } else {
        return tensor::desc(pd.weights_desc(), groups);
      }
    } 
  }

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
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_query = src_desc.to_format_any();
    auto weights_desc_query = weights_desc.to_format_any();
    auto bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
    auto dst_desc_query = dst_desc.to_format_any();

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
            attr);
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

 private:
  // For 2-in-1 compute. For fp32
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void compute_impl(const tensor& src,
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
    deconv_forward_params param;
    do_prepare<with_bias>(param, src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aprop_kind, aengine);
    do_compute<with_bias, reorder_src, reorder_weight>(param, src, weights, bias, dst);
  }

  // For 2-in-1 compute. For int8
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void compute_impl(const tensor& src,
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
    deconv_forward_params param;
    do_prepare<with_bias>(param, src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        src_zero_point, dst_zero_point, is_channels_last, attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    do_compute<with_bias, reorder_src, reorder_weight>(param, src, weights, bias, dst);
  }

  // For fp32
  template <bool with_bias>
  static void do_prepare(
      deconv_forward_params& param,
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
        attr, with_bias, /*is_deconv=*/true, weights_grouped,
        dil_compatible, op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);

    param.pd = get_primitive_desc<with_bias>(
                  src_desc, weights_desc, bias_desc, dst_desc,
                  strides, dil_compatible, padding_l, padding_r, is_channels_last, op_attr, aalgorithm,
                  aprop_kind, aengine);

    param.primitive = std::move(super(param.pd));
    param.op_attr = std::move(op_attr);
    param.bias_attr = std::move(bias_attr);
    param.groups = groups;
  }

  // For int8
  template <bool with_bias>
  static void do_prepare(deconv_forward_params& param,
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
        attr, alowp_kind, with_bias, /*is_deconv=*/true,
        weights_grouped, dil_compatible, op_attr, src_attr, weights_attr, bias_attr,
        src_desc, weights_desc, bias_desc, dst_desc);

    param.pd = get_primitive_desc<with_bias>(
                  src_desc, weights_desc, bias_desc, dst_desc,
                  strides, dil_compatible, padding_l, padding_r, is_channels_last, op_attr, aalgorithm,
                  aprop_kind, aengine);

    param.primitive = std::move(super(param.pd));
    param.op_attr = std::move(op_attr);
    param.bias_attr = std::move(bias_attr);
    param.groups = groups;

    // Prepare tensors for scales and zero points
    if (param.op_attr.has_scales()) {
      if (!param.all_scales) {
        param.all_scales.reset(new std::unordered_map<int, tensor>);
      }
      for (auto& arg_scale_pair : param.op_attr.get_all_scales()) {
        int dnnl_arg = arg_scale_pair.first;
        const scale_t& scale = arg_scale_pair.second.first;
        tensor scales_m(scale);
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
        tensor zp_m(zp);
        param.all_zero_points->insert({dnnl_arg, zp_m});
      }
    }
  }

  // For fp32 and int8
  // `reorder_src` and `reorder_weight` indicate whether
  // src/dst and weight/bias should be checked and possibly reordered
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static void do_compute(const deconv_forward_params& param,
                         const tensor& src,
                         const tensor& weights,
                         const tensor& bias,
                         tensor& dst) {
    auto scratchpad = tensor(param.pd.scratchpad_desc());
    auto& expected_src = reorder_src ?
        src.reorder_if_differ_in(param.pd.src_desc()) : src;
    auto&& grouped_weights = weights.make_grouped_weights(param.groups, /*is_deconv=*/true);
    auto&& expected_weights = reorder_weight ?
        grouped_weights.reorder_if_differ_in(param.pd.weights_desc()) :
        grouped_weights;
    if (reorder_src) {
      dst.reinit_if_possible(param.pd.dst_desc());
    }
    auto& primitive = param.primitive;

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_DST, dst});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }
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
    primitive.execute(stream::default_stream(), args);
  }

};

struct convolution_transpose_backward_data
    : public dnnl::deconvolution_backward_data {

  using super = dnnl::deconvolution_backward_data;

  // This one does not have `is_channels_last` argument for compatibility
  static void compute(const tensor& diff_dst,
                      const tensor& weights, // dim: {i, o[, d], h, w}
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups, true);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
    bool is_channels_last = is_nhwc || is_ndhwc;
    auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    auto weights_desc = weights_.get_desc().to_format_any();

    tensor::desc diff_src_desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);

    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints =
        convolution_transpose_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last, op_attr);

    auto pd = primitive_desc(
        aengine, aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
        dilates_, padding_l, padding_r, forward_hints, op_attr);

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

  static void compute_v3(const tensor& diff_dst,
                      const tensor& weights, // dim: {i, o[, d], h, w}
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      bool is_channels_last = false,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups, true);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    // bool is_nhwc = diff_dst.get_desc().is_nhwc();
    // bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    // auto format_tag = is_nhwc ? tag::nhwc : (is_ndhwc ? tag::ndhwc : tag::any);
    // auto diff_dst_desc = diff_dst.get_desc().to_format(format_tag);
    // auto weights_desc = weights_.get_desc().to_format_any();
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

    auto weights_desc =
        weights_.get_desc().to_format_any().to_type(diff_dst.get_data_type());

    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst_desc.get_data_type(), format_tag);
    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints =
        convolution_transpose_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last, op_attr);

    auto pd = primitive_desc(
        aengine, aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
        dilates_, padding_l, padding_r, forward_hints, op_attr);

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

struct convolution_transpose_backward_weights
    : public dnnl::deconvolution_backward_weights {

  using super = dnnl::deconvolution_backward_weights;

  static void compute_v3(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      bool is_channels_last = false,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
   compute_impl</*with_diff_bias=*/true>(
       src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
       strides, dilates, padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aengine);
  }

  // This one does not have `is_channels_last` argument for compatibility
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
  bool is_nhwc = diff_dst.get_desc().is_nhwc();
  bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
  bool is_channels_last = is_nhwc || is_ndhwc;
   compute_impl</*with_diff_bias=*/true>(
       src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
       strides, dilates, padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aengine);
  }

  static void compute_v3(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      bool is_channels_last = false,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aengine);
  }

  // This one does not have `is_channels_last` argument for compatibility
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    bool is_nhwc = diff_dst.get_desc().is_nhwc();
    bool is_ndhwc = diff_dst.get_desc().is_ndhwc();
    bool is_channels_last = is_nhwc || is_ndhwc;
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, is_channels_last, attr, aalgorithm, aengine);
  }
private:
  template <bool with_diff_bias>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           const dims& diff_weights_dims, // [i, o, ...]
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           const int groups,
                           bool is_channels_last,
                           const attr_t& attr,
                           algorithm aalgorithm,
                           const engine& aengine) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // make diff_weights and dilates compatible with DNNL
    auto dilates_ = utils::get_compatible_dilates(dilates);

    // dim: [i, o, ...]
    auto diff_weights_desc = 
        tensor::desc(diff_weights_dims, diff_dst.get_data_type(), tag::any);

    if (groups > 1) {
      // dim: [g, o, i/g, ...]
      diff_weights_desc = diff_weights_desc.to_grouped(groups).transpose(1, 2);
    } else {
      // dim: [o, i, ...]
      diff_weights_desc = diff_weights_desc.transpose(0, 1);
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

    auto diff_bias_desc = with_diff_bias
        ? tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type())
              .to_format_any()
        : tensor::desc();

    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints =
        convolution_transpose_forward::get_primitive_desc<with_diff_bias>(
            src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, is_channels_last, op_attr, aalgorithm,
            prop_kind::forward, aengine);

    auto pd = with_diff_bias
        ? primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                         diff_bias_desc, diff_dst_desc, strides, dilates_,
                         padding_l, padding_r, forward_hints, op_attr)
        : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r, forward_hints, op_attr);

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

    diff_weights.feed_from(expected_diff_weights, /*is_deconv_weights=*/true);

    // recover output dims to align with pytorch
    if (groups > 1) {
      // [g, o, i/g, ...] -> [g, i/g, o, ...]
      diff_weights.transpose_(1, 2);
    } else {
      // [o, i, ...] -> [i, o, ...]
      diff_weights.transpose_(0, 1);
    }
  }
};
}  // namespace ideep

#endif