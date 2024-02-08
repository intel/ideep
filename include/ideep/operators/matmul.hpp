#ifndef IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP

namespace ideep {

// Parameters for dynamic quantization
struct matmul_forward_dyn_quant_params {
  tensor::desc src_desc; // to create src tensor
  dnnl::reorder::primitive src_reorder; // to reorder src

  matmul_forward_dyn_quant_params() {}

  matmul_forward_dyn_quant_params(
      tensor::desc&& src_desc,
      dnnl::reorder::primitive&& src_reorder)
      : src_desc(std::move(src_desc)),
        src_reorder(std::move(src_reorder)) {}
};

// Common parameters for computation
struct matmul_forward_params {
  dnnl::matmul::primitive_desc pd;
  dnnl::matmul primitive;
  attr_t op_attr;
  attr_t src_attr;
  attr_t weights_attr;
  attr_t bias_attr; // contains requantization scales for bias
  std::shared_ptr<matmul_forward_dyn_quant_params> dq_param_ptr;
  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive execution
  std::shared_ptr<std::unordered_map<int, tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, tensor>> all_zero_points = nullptr;

  matmul_forward_params() {}

  matmul_forward_params(
      dnnl::matmul::primitive_desc&& pd,
      attr_t&& op_attr,
      attr_t&& src_attr,
      attr_t&& weights_attr,
      attr_t&& bias_attr)
      : pd(std::move(pd)),
        op_attr(std::move(op_attr)),
        src_attr(std::move(src_attr)),
        weights_attr(std::move(weights_attr)),
        bias_attr(std::move(bias_attr)) {
    primitive = dnnl::matmul(pd);
  }
};

struct matmul_forward : public dnnl::matmul,
#ifdef __aarch64__
                        utils::computation_cache<std::pair<dnnl::matmul::primitive_desc, dnnl::matmul> > {
#else
                        utils::computation_cache<dnnl::matmul::primitive_desc> {
#endif
  using super = dnnl::matmul;

  // 2-in-1 compute for fp32 op with bias. Bias is disabled if it is empty.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    static lowp_kind dummy_lowp_kind = u8s8;
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst,
          IDEEP_EMPTY_SCALE, IDEEP_EMPTY_SCALE, IDEEP_EMPTY_SCALE,
          IDEEP_EMPTY_ZP, IDEEP_EMPTY_ZP,
          dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
          dst_type, dummy_lowp_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst,
          IDEEP_EMPTY_SCALE, IDEEP_EMPTY_SCALE, IDEEP_EMPTY_SCALE,
          IDEEP_EMPTY_ZP, IDEEP_EMPTY_ZP,
          dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
          dst_type, dummy_lowp_kind,  aengine);
    }
  }

  // 2-in-1 compute for fp32 op without bias.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    static lowp_kind dummy_lowp_kind = u8s8;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst,
        IDEEP_EMPTY_SCALE, IDEEP_EMPTY_SCALE, IDEEP_EMPTY_SCALE,
        IDEEP_EMPTY_ZP, IDEEP_EMPTY_ZP,
        dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
        dst_type, dummy_lowp_kind, aengine);
  }

  // 2-in-1 compute for fp32 op with bias. Bias is disabled if it is empty.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute_binary(
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_binary_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, other, weights, bias, dst,
          dst_coeff, attr, dst_type, aengine);
    } else {
      compute_binary_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, other, weights, bias, dst,
          dst_coeff, attr, dst_type, aengine);
    }
  }

  // 2-in-1 compute for fp32 op without bias.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute_binary(
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_binary_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, other, weights, dummy_bias, dst,
        dst_coeff, attr, dst_type, aengine);
  }

  // 2-in-1 compute for int8 op with bias. Bias is not used if it is empty.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, weights, bias, dst,
          src_scales, weights_scales, dst_scales,
          src_zero_points, dst_zero_points,
          dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
          dst_type, alowp_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, weights, bias, dst,
          src_scales, weights_scales, dst_scales,
          src_zero_points, dst_zero_points,
          dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
          dst_type, alowp_kind, aengine);
    }
  }

  // 2-in-1 compute for int8 op with bias. Bias is not used if it is empty.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst,
        src_scales, weights_scales, dst_scales,
        src_zero_points, dst_zero_points,
        dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
        dst_type, alowp_kind, aengine);
  }

  // 2-in-1 compute for int8 op with bias. Bias is disabled if it is empty.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute_binary(
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_binary_impl</*with_bias=*/false, reorder_src, reorder_weight>(
          src, other, weights, bias, dst,
          src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          dst_coeff, attr, dst_type, aengine);
    } else {
      compute_binary_impl</*with_bias=*/true, reorder_src, reorder_weight>(
          src, other, weights, bias, dst,
          src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          dst_coeff, attr, dst_type, aengine);
    }
  }

  // 2-in-1 compute for int8 op without bias.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute_binary(
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_binary_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, other, weights, dummy_bias, dst,
        src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
        dst_coeff, attr, dst_type, aengine);
  }

  // Prepare for fp32 op
  // With bias. Bias is disabled if it is empty.
  static inline void prepare(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f, // for post-op sum
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_prepare</*with_bias=*/false>(param, src, weights, bias, dst, dst_coeff, sum_coeff,
          attr, dst_type, aengine);
    } else {
      do_prepare</*with_bias=*/true>(param, src, weights, bias, dst, dst_coeff, sum_coeff,
          attr, dst_type, aengine);
    }
  }

  // Prepare for fp32 op
  // Without bias.
  static inline void prepare(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const float dst_coeff, // default = 1.0f
      const float sum_coeff, // for post-op sum, default = 1.0f
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(param, src, weights, dummy_bias, dst, dst_coeff, sum_coeff,
        attr, dst_type, aengine);
  }

  // Prepare for int8 op with bias. Bias is not used if it is empty.
  // Static: int8 * int8 -> int8. Dynamic: fp32 * int8 -> fp32
  template <bool is_dynamic = false>
  static inline void prepare(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f, // for post-op sum
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::u8,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (is_dynamic) {
      if (bias.is_empty()) {
        do_prepare_dynamic_quant</*with_bias=*/false>(
            param, src, weights, bias, dst, weights_scales,
            sum_coeff, attr, data_type::f32, aengine);
      } else {
        do_prepare_dynamic_quant</*with_bias=*/true>(
            param, src, weights, bias, dst, weights_scales,
            sum_coeff, attr, data_type::f32, aengine);
      }
    } else {
      if (bias.is_empty()) {
        do_prepare_static_quant</*with_bias=*/false>(
            param, src, weights, bias, dst,
            src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
            dst_coeff, sum_coeff, attr, dst_type, alowp_kind, aengine);
      } else {
        do_prepare_static_quant</*with_bias=*/true>(
            param, src, weights, bias, dst,
            src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
            dst_coeff, sum_coeff, attr, dst_type, alowp_kind, aengine);
      }
    }
  }

  // Prepare for int8 op without bias.
  // Static: int8 * int8 -> int8. Dynamic: fp32 * int8 -> fp32
  template <bool is_dynamic = false>
  static inline void prepare(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f, // for post-op sum
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::u8,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    if (is_dynamic) {
      do_prepare_dynamic_quant</*with_bias=*/false>(
          param, src, weights, dummy_bias, dst, weights_scales,
          sum_coeff, attr, data_type::f32, aengine);
    } else {
      do_prepare_static_quant</*with_bias=*/false>(
          param, src, weights, dummy_bias, dst,
          src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          dst_coeff, sum_coeff, attr, dst_type, alowp_kind, aengine);
    }
  }

  // Compute for fp32 and static int8 (int8 * int8 -> int8)
  // With bias. Bias is not used if it is empty.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute(const matmul_forward_params& param,
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

  // Compute for fp32 and static int8 (int8 * int8 -> int8)
  // Without bias.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static inline void compute(const matmul_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, weights, dummy_bias, dst);
  }

  // Compute for dynamic int8 (fp32 * int8 -> fp32)
  // With bias. Bias is not used if it is empty.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_weight = true>
  static inline void compute(
      const matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const zero_point_t& src_zero_points,
      const float dst_coeff = 1.0f,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
          param, src, weights, bias, dst,
          src_scales, src_zero_points, dst_coeff, aengine);
    } else {
      do_compute_dynamic_quant</*with_bias=*/true, reorder_weight>(
          param, src, weights, bias, dst,
          src_scales, src_zero_points, dst_coeff, aengine);
    }
  }

  // Compute for dynamic int8 (fp32 * int8 -> fp32)
  // Without bias.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_weight = true>
  static inline void compute(
      const matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const scale_t& src_scales,
      const zero_point_t& src_zero_points,
      const float dst_coeff = 1.0f,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
        param, src, weights, dummy_bias, dst,
        src_scales, src_zero_points, dst_coeff, aengine);
  }

  // Deprecated 2-in-1 compute
  // With bias. Zero points are passed explicitly as arguments for quantization
  // Bias is not used if it is empty.
  static void compute_v2(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const zero_point_t& src_zero_points = zero_point_t(),
      const zero_point_t& dst_zero_points = zero_point_t(),
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      compute_impl</*with_bias=*/false, /*reorder_src*/true, /*reorder_weight*/true>(
          src, weights, bias, dst,
          src_scales, weights_scales, dst_scales,
          src_zero_points, dst_zero_points,
          dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
          dst_type, alowp_kind, aengine);
    } else {
      compute_impl</*with_bias=*/true, /*reorder_src*/true, /*reorder_weight*/true>(
          src, weights, bias, dst,
          src_scales, weights_scales, dst_scales,
          src_zero_points, dst_zero_points,
          dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
          dst_type, alowp_kind, aengine);
    }
  }

  // Deprecated 2-in-1 compute
  // Without bias. Zero points are passed explicitly as arguments for quantization
  static void compute_v2(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const zero_point_t& src_zero_points = zero_point_t(),
      const zero_point_t& dst_zero_points = zero_point_t(),
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false, /*reorder_src*/true, /*reorder_weight*/true>(
        src, weights, dummy_bias, dst,
        src_scales, weights_scales, dst_scales,
        src_zero_points, dst_zero_points,
        dst_coeff, sum_coeff, attr, /*bin_post_params=*/{},
        dst_type, alowp_kind, aengine);
  }

  // Deprecated 2-in-1 compute. With bias. Set zero points to tensors for quantization.
  static void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const std::vector<tensor>& bin_post_params = {},
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    // Consider fp32 only for IPEX
    compute_impl</*with_bias=*/true, /*reorder_src*/true, /*reorder_weight*/true>(
        src, weights, bias, dst,
        src_scales, weights_scales, dst_scales,
        IDEEP_EMPTY_ZP, IDEEP_EMPTY_ZP,
        dst_coeff, sum_coeff, attr, bin_post_params,
        dst_type, alowp_kind, aengine);
  }

  // Deprecated 2-in-1 compute. Without bias. Set zero points to tensors for quantization.
  static void compute(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const std::vector<tensor>& bin_post_params = {},
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    // Consider fp32 only for IPEX
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false, /*reorder_src*/true, /*reorder_weight*/true>(
        src, weights, dummy_bias, dst,
        src_scales, weights_scales, dst_scales,
        IDEEP_EMPTY_ZP, IDEEP_EMPTY_ZP,
        dst_coeff, sum_coeff, attr, bin_post_params,
        dst_type, alowp_kind, aengine);
  }

  // Deprecated. Prepare for int8 op with bias. Bias is not used if it is empty.
  // Static: int8 * int8 -> int8. Dynamic: fp32 * int8 -> fp32
  template <bool is_dynamic = false>
  static void prepare(matmul_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      const float dst_coeff, // default = 1.0f
                      const float sum_coeff, // for post-op sum, default = 1.0f
                      const scale_t& src_scales,
                      const scale_t& weights_scales,
                      const scale_t& dst_scales,
                      const zero_point_t& src_zero_points,
                      const zero_point_t& dst_zero_points,
                      const attr_t& attr = attr_t(),
                      const data_type dst_type = data_type::u8,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    if (is_dynamic) {
      if (bias.is_empty()) {
        do_prepare_dynamic_quant</*with_bias=*/false>(
            param, src, weights, bias, dst, weights_scales,
            sum_coeff, attr, data_type::f32, aengine);
      } else {
        do_prepare_dynamic_quant</*with_bias=*/true>(
            param, src, weights, bias, dst, weights_scales,
            sum_coeff, attr, data_type::f32, aengine);
      }
    } else {
      if (bias.is_empty()) {
        do_prepare_static_quant</*with_bias=*/false>(
            param, src, weights, bias, dst,
            src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
            dst_coeff, sum_coeff, attr, dst_type, alowp_kind, aengine);
      } else {
        do_prepare_static_quant</*with_bias=*/true>(
            param, src, weights, bias, dst,
            src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
            dst_coeff, sum_coeff, attr, dst_type, alowp_kind, aengine);
      }
    }
  }

  // Deprecated.
  // Bias is not used if it is empty.
  static void compute_dynamic(
      const matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const zero_point_t& src_zero_points = zero_point_t(),
      const zero_point_t& dst_zero_points = zero_point_t(),
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_compute_dynamic_quant</*with_bias=*/false>(
          param, src, weights, bias, dst,
          src_scales, src_zero_points, dst_coeff, aengine);
    } else {
      do_compute_dynamic_quant</*with_bias=*/true>(
          param, src, weights, bias, dst,
          src_scales, src_zero_points, dst_coeff, aengine);
    }
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      const dims& src_dims = dims(),
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {
    auto ndims = weights_dims.size();
    auto x_dims = weights_dims;
    x_dims[ndims-2] = src_dims.size() > 0 && src_dims.size() == ndims ? src_dims[ndims-2] : 1;
    x_dims[ndims-1] = weights_dims[ndims-2];
    dims y_dims = (ndims == 3) ? dims({x_dims[0], x_dims[1], weights_dims[2]})
                               : dims({x_dims[0], weights_dims[1]});
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc x_desc(x_dims, x_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc y_desc(y_dims, y_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc weights_desc(weights_dims , dtype, tag::any);
    auto pd = primitive_desc(aengine, x_desc, weights_desc, y_desc, attr);
    return pd.weights_desc();
  }

 private:
  // For 2-in-1 compute: prepare + compute
  // Supports fp32, static int8 and dynamic int8
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static inline void compute_impl(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const zero_point_t& src_zero_points = zero_point_t(),
      const zero_point_t& dst_zero_points = zero_point_t(),
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const std::vector<tensor>& bin_post_params = {},
      const data_type dst_type = data_type::undef,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    matmul_forward_params param;
    auto& weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;
    if (!weights_scales_in.empty()) { // for int8
      if (src.get_data_type() == data_type::f32) { // invoke dynamic quant
        // prepare
        do_prepare_dynamic_quant<with_bias>(param, src, weights,
                              bias, dst, weights_scales, sum_coeff,
                              attr, data_type::f32, aengine);
        // compute
        if (bias.is_empty()) {
          do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
                              param, src, weights, bias, dst,
                              src_scales, src_zero_points, dst_coeff, aengine);
        } else {
          do_compute_dynamic_quant</*with_bias=*/true, reorder_weight>(
                              param, src, weights, bias, dst,
                              src_scales, src_zero_points, dst_coeff, aengine);
        }
        return;
      } else { // invoke static quant
        do_prepare_static_quant<with_bias>(
            param, src, weights, bias, dst,
            src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
            dst_coeff, sum_coeff, attr, dst_type, alowp_kind, aengine);
      }
    } else {
      do_prepare<with_bias>(param, src, weights, bias, dst, dst_coeff, sum_coeff,
                 attr, dst_type, aengine);
    }
    do_compute<with_bias, reorder_src, reorder_weight>(
        param, src, weights, bias, dst, bin_post_params);
  }

  // For 2-in-1 compute with binary post-op: prepare + compute
  // Supports fp32.
  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static inline void compute_binary_impl(
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    matmul_forward_params param;
    do_prepare<with_bias>(param, src, weights, bias, dst, dst_coeff, 1.0f,
        attr, dst_type, aengine);
    do_compute_binary<with_bias, reorder_src, reorder_weight>(
        param, src, other, weights, bias, dst);
  }

  // For 2-in-1 compute with binary post-op: prepare + compute
  // Supports int8.
  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static inline void compute_binary_impl(
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::undef,
      const engine& aengine = engine::cpu_engine()) {
    matmul_forward_params param;
    do_prepare<with_bias>(param, src, weights, bias, dst,
        src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
        dst_coeff, 1.0f, attr, dst_type, aengine);
    do_compute_binary<with_bias, reorder_src, reorder_weight>(
        param, src, other, weights, bias, dst);
  }

  // For fp32 op
  template <bool with_bias>
  static inline void do_prepare(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f, // for post-op sum
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(src.ndims() == weights.ndims(), "Invalid dims in src or weights");

    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t& op_attr = param.op_attr;
    attr_t& bias_attr = param.bias_attr;
    op_attr = attr;
    auto dst_data_type = data_type::f32;

    tensor::dims src_dims = src.get_dims();
    tensor::dims dst_dims = {src_dims[0]};
    auto ndims = weights.ndims();
    for (auto i = 1; i < ndims - 1; i++) {
      dst_dims.push_back(src_dims[i]);
    }
    dst_dims.push_back(weights.get_dim(ndims - 1));

    // We intentionally didn't set weight desc to format `any` so DNNL wouldn't
    // have to determine weight format for us. Because the weight tensor from
    // pytorch may have a transposed format (say `ba`). However, DNNL would
    // choose plain format for it by default (`ab` in this case), which would
    // introduces *an extra reorder* afterwards. Here we keep the weight format
    // untouched thanks to optimizations for both plain and transposed formats
    // in DNNL.
    IDEEP_ENFORCE(weights.get_data_type() == data_type::f32 ||
                  weights.get_data_type() == data_type::bf16 ||
                  weights.get_data_type() == data_type::f16,
                  "Incorrect data type in weights");
    dst_data_type = src.get_data_type() == data_type::bf16 ?
                    data_type::bf16 :
                    ((src.get_data_type() == data_type::f16) ?
                      data_type::f16 : data_type::f32);
    src_desc = src.get_desc().to_type(dst_data_type);

#ifdef __aarch64__
    // for aarch64 ACL backend with fixed format kernels, the weights are
    // always in blocked layout, so, set the descriptor to tag::any for the backend
    // to decide the format
    weights_desc = tensor::desc(weights.get_dims(), dst_data_type, tag::any);
#else

    // For fp32 matmul, weight (2nd input) is usually not in blocked layout
    // Plain layout runs faster as of oneDNN 3.0
    // Should use tag::any to query blocked layout if there is perf gain later
    weights_desc = weights.get_desc().is_plain() ?
                   weights.get_desc().to_type(dst_data_type) :
                   tensor::desc(weights.get_dims(), dst_data_type, tag::any);
#endif

    if (with_bias) {
      IDEEP_ENFORCE(bias.get_data_type() == data_type::f32 ||
                    bias.get_data_type() == data_type::bf16 ||
                    bias.get_data_type() == data_type::f16,
                    "Incorrect data type in bias");
      bias_desc = bias.get_desc().to_format_any();
    }

    if (attr.has_op_kind(kind::sum) && attr.get_post_ops().len() == 1) {
      op_attr = attr_t::fuse_sum(sum_coeff);
    }
    if (dst_coeff != 1.0f) {
      // Since onednn 3.0, dst scales are applied after all post ops
      // But here the dst_coeff should be applied before post ops
      // So we set it as src scales here. This keeps the semantics.
      int scale_size = 1;
      op_attr.set_scales(DNNL_ARG_SRC, utils::op_scale_mask(scale_size),
                         scale_t(1, dst_coeff));
    }

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dst_data_type = dst_type == data_type::undef ? dst_data_type : dst_type;
    /* rely on tensor descriptor class to se the format appropriately
     * based on the tensor dimensions.
     */
    tensor::desc dst_desc = tensor::desc(dst_dims, dst_data_type);
    if (!dst.is_empty()) {
      dst_desc = dst.get_desc().to_type(dst_data_type);
    }

    auto key = utils::create_key(
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        op_attr,
        with_bias,
        omp_get_max_threads());

#ifdef __aarch64__
    auto pd_pair = fetch_or_create(key, [&]() {
      if (with_bias) {
        param.pd = primitive_desc(
            aengine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
      } else {
        param.pd = primitive_desc(
            aengine, src_desc, weights_desc, dst_desc, op_attr);
      }
      return std::make_pair(param.pd, super(param.pd));
    });
    param.pd = std::move(pd_pair.first);
    param.primitive = std::move(pd_pair.second);
#else
    param.pd = fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            aengine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
      } else {
        return primitive_desc(
            aengine, src_desc, weights_desc, dst_desc, op_attr);
      }
    });
    param.primitive = std::move(super(param.pd));
#endif
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
  }

  // For static int8 op (int8 * int8 -> int8)
  template <bool with_bias>
  static inline void do_prepare_static_quant(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f, // for post-op sum
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::u8,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(src.ndims() == weights.ndims(), "Invalid dims in src or weights");

    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t& op_attr = param.op_attr;
    attr_t& src_attr = param.src_attr;
    attr_t& weights_attr = param.weights_attr;
    attr_t& bias_attr = param.bias_attr;
    op_attr.set_post_ops(attr.get_post_ops());
    scale_t dst_scales_in;
    auto dst_data_type = data_type::u8;

    tensor::dims src_dims = src.get_dims();
    tensor::dims dst_dims = {src_dims[0], weights.get_dim(1)};
    auto ndims = weights.ndims();
    if (ndims == 3) {
      dst_dims = {src_dims[0], src.get_dim(1), weights.get_dim(2)};
    }

    auto& weights_scales_in = weights_scales.empty() ? IDEEP_DEF_SCALE : weights_scales;

    IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                  "Unsupported lowp kind");

    auto src_scales_in = src_scales.empty() ? IDEEP_DEF_SCALE : src_scales;
    auto src_data_type = (alowp_kind == u8s8) ? data_type::u8 : data_type::s8;
    std::vector<int64_t> src_strides = (ndims == 3) ?
        std::vector<int64_t>({src_dims[1] * src_dims[2], src_dims[1], 1}) :
        std::vector<int64_t>({src_dims[1], 1});
    src_desc = tensor::desc(src_dims, src_data_type, tag::any);
    for (auto& s : src_scales_in) {
      s = 1.0 / s;
    }
    if (src.get_data_type() == data_type::f32) {
      src_attr.set_scales(DNNL_ARG_DST, /* mask */ 0, src_scales_in);
    }

    int scale_size = (weights_scales_in.size() > 1) ? weights.get_dim(1) : 1;
    weights_desc = weights.get_desc();
    if (weights.get_data_type() == data_type::f32) {
      weights_attr = {utils::tensor_scale_mask(scale_size, false),
                      weights_scales_in};
    }

    // determine dst data type
    if (dst.get_data_type() != data_type::undef) {
      dst_data_type = dst.get_data_type();
    } else if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
      dst_data_type = data_type::f32;
    } else {
      dst_data_type = data_type::u8;
    }

    // fill primitive attr
    dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32)
        ? IDEEP_DEF_SCALE
        : dst_scales;
    const auto& src_zero_point = src.has_zero_point() ? src.get_zero_point() :
                                 src_zero_points.empty() ? IDEEP_DEF_ZP : src_zero_points;
    const auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
    const auto& dst_zero_point = dst.has_zero_point() ? dst.get_zero_point() :
        dst_zero_points.empty() ? IDEEP_DEF_ZP : dst_zero_points;
    const auto dst_zero_point_size = static_cast<dim>(dst_zero_point.size());
    IDEEP_ENFORCE(src_zero_point_size == 1 && dst_zero_point_size == 1,
                  "DNNL only support 1-dim zero_point");
    const auto& wei_zero_point = weights.has_zero_point() ?
                                 weights.get_zero_point() : IDEEP_DEF_ZP;

    if (attr.has_op_kind(kind::sum)) {
      float sum_scale =
          sum_coeff / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
      op_attr = attr_t::fuse_sum(sum_scale);
    }

    if (src_scales_in[0] != 1.0f) {
      op_attr.set_scales(DNNL_ARG_SRC, utils::op_scale_mask(src_scales_in.size()), src_scales_in);
    }
    auto wei_scales = weights_scales_in;
    if (!std::all_of(wei_scales.begin(), wei_scales.end(), [](float i){ return i == 1.0f; })) {
      for (auto& s : wei_scales) {
        s = 1.0 / s;
      }
      op_attr.set_scales(DNNL_ARG_WEIGHTS, utils::op_scale_mask(wei_scales.size()), wei_scales);
    }
    for (auto& s : dst_scales_in) {
      s = dst_coeff / s;
    }
    if (dst_scales_in[0] != 1.0f) {
      op_attr.set_scales(DNNL_ARG_DST, utils::op_scale_mask(dst_scales_in.size()), dst_scales_in);
    }
    if (src_zero_point[0] != 0) {
      op_attr.set_zero_points(DNNL_ARG_SRC,
                              utils::tensor_zp_mask(src_zero_point.size()), src_zero_point);
    }
    if (src.get_data_type() == data_type::f32) {
      // Set zero point for src reorder (fp32 -> int8).
      // First arg should be DNNL_ARG_DST rather than DNNL_ARG_SRC
      src_attr.set_zero_points(DNNL_ARG_DST,
                               utils::tensor_zp_mask(src_zero_point.size()),
                               src_zero_point);
    }
    if (dst_data_type != data_type::f32 && dst_zero_point[0] != 0) {
      op_attr.set_zero_points(DNNL_ARG_DST,
                              utils::tensor_zp_mask(dst_zero_point.size()), dst_zero_point);
    }

    if (with_bias) {
      tag bia_tag = bias.get_dims().size() == 2 ? tag::ab : tag::abc;
      bias_desc = {bias.get_dims(), data_type::f32, bia_tag}; // Use f32 instead of s32 to improve accuracy
    }

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dst_data_type = dst_type == data_type::undef ? dst_data_type : dst_type;
    tensor::desc dst_desc = tensor::desc(dst_dims, dst_data_type, tag::any);
    if (!dst.is_empty()) {
      dst_desc = dst.get_desc().to_type(dst_data_type);
    }
    auto key = utils::create_key(
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        op_attr,
        with_bias,
        omp_get_max_threads());
#ifdef __aarch64__
    auto pd_pair = fetch_or_create(key, [&]() {
      if (with_bias) {
        param.pd =  primitive_desc(
            aengine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
      } else {
        param.pd =  primitive_desc(
            aengine, src_desc, weights_desc, dst_desc, op_attr);
      }
      return std::make_pair(param.pd, super(param.pd));
    });
    param.pd = std::move(pd_pair.first);
    param.primitive = std::move(pd_pair.second);
#else
    param.pd = fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            aengine, src_desc, weights_desc, bias_desc, dst_desc, op_attr);
      } else {
        return primitive_desc(
            aengine, src_desc, weights_desc, dst_desc, op_attr);
      }
    });
    param.primitive = std::move(super(param.pd));
#endif
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

  // For dynamic int8 op (fp32 * int8 -> fp32)
  template <bool with_bias>
  static inline void do_prepare_dynamic_quant(
      matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& weights_scales,
      const float sum_coeff = 1.0f, // for post-op sum
      const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    /* This function does the following things:
     * - Determine expected descs of src/weight/dst
     * - Use runtime values for op attributes
     * - Create matmul primitive desc and primitive
     * - Create reorder primitive for src (fp32 -> int8)
     */

    IDEEP_ENFORCE(src.ndims() == weights.ndims(), "Invalid dims in src or weights");
    if (!param.dq_param_ptr) {
      param.dq_param_ptr = std::make_shared<matmul_forward_dyn_quant_params>();
    }
    IDEEP_ENFORCE(param.dq_param_ptr, "Failed to allocate memory for parameters");

    tensor::desc &src_desc = param.dq_param_ptr->src_desc;
    attr_t& op_attr = param.op_attr;
    attr_t src_attr;

    tensor::dims src_dims = src.get_dims();
    tensor::dims dst_dims = {src_dims[0], weights.get_dim(1)};
    auto ndims = weights.ndims();
    if (ndims == 3)
      dst_dims = {src_dims[0], src.get_dim(1), weights.get_dim(2)};

    auto& weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;

    auto src_data_type = data_type::u8;
    std::vector<int64_t> src_strides = (ndims == 3) ?
        std::vector<int64_t>({src_dims[1] * src_dims[2], src_dims[1], 1}) :
        std::vector<int64_t>({src_dims[1], 1});
    src_desc = tensor::desc(src_dims, src_data_type, src_strides);

    // Post-ops
    auto pops = attr.get_post_ops();
    dnnl::post_ops new_pops;
    for (int i = 0; i < pops.len(); ++i) {
      // Only sum and eltwise is supported now
      if (kind::sum == pops.kind(i)) {
        // The parameter sum_coeff is passed in explicitly now due to legacy code.
        // TO-DO:
        // Remove the argument 'sum_coeff'.
        // User should prepare all post-ops in argument 'attr'.
        new_pops.append_sum(sum_coeff);
      } else if (kind::eltwise == pops.kind(i)) {
        float alpha = 1.0, beta = 0.0;
        dnnl::algorithm alg;
        pops.get_params_eltwise(i, alg, alpha, beta);
        new_pops.append_eltwise(alg, alpha, beta);
      }
    }
    op_attr.set_post_ops(new_pops);

    // Scales and zero points of src are obtained at runtime. So only set mask here
    // Weight zero point is always 0. Do not set.
    // Dst is in fp32, no scales or zero points.
    op_attr.set_scales_mask(DNNL_ARG_SRC, utils::op_scale_mask(1/* scale_size */));
    op_attr.set_zero_points_mask(DNNL_ARG_SRC, utils::tensor_zp_mask(1));
    auto wei_scales = weights_scales_in;
    for (auto& s : wei_scales) {
      s = 1.0 / s;
    }
    op_attr.set_scales(DNNL_ARG_WEIGHTS,
                       utils::op_scale_mask(wei_scales.size()),
                       wei_scales);
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Src attr for reorder
    src_attr.set_scales_mask(DNNL_ARG_DST, utils::op_scale_mask(1));
    src_attr.set_zero_points_mask(DNNL_ARG_DST, utils::tensor_zp_mask(1));

    // Dst desc
    std::vector<int64_t> dst_strides = (ndims == 3) ?
        std::vector<int64_t>({dst_dims[2]* dst_dims[1], dst_dims[1], 1}) :
        std::vector<int64_t>({dst_dims[1], 1});
    tensor::desc dst_desc = tensor::desc(dst_dims, dst_type, dst_strides);
    if (!dst.is_empty()) {
      dst_desc = dst.get_desc().to_type(dst_type);
    }
    tensor::desc bias_desc;
    if (with_bias) {
      tag bia_tag = bias.get_dims().size() == 2 ? tag::ab : tag::abc;
      bias_desc = {bias.get_dims(), data_type::f32, bia_tag};
    }

    auto key = utils::create_key(
        src_desc,
        weights.get_desc(),
        bias_desc,
        dst_desc,
        op_attr,
        with_bias,
        omp_get_max_threads());

    // Create pd and primitive
#ifdef __aarch64__
    auto pd_pair = fetch_or_create(key, [&]() {
      if (with_bias) {
        param.pd = primitive_desc(
            aengine, src_desc, weights.get_desc(), bias_desc, dst_desc, op_attr);
      } else {
        param.pd = primitive_desc(
            aengine, src_desc, weights.get_desc(), dst_desc, op_attr);
      }
      return std::make_pair(param.pd, super(param.pd));
    });
    param.pd = std::move(pd_pair.first);
    param.primitive = std::move(pd_pair.second);
#else
    param.pd = fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            aengine, src_desc, weights.get_desc(), bias_desc, dst_desc, op_attr);
      } else {
        return primitive_desc(
            aengine, src_desc, weights.get_desc(), dst_desc, op_attr);
      }
    });
    param.primitive = super(param.pd);
#endif

    // Create src reorder primitive with runtime scales/zero point
    auto src_reorder_pd = dnnl::reorder::primitive_desc(aengine, src.get_desc(), aengine, src_desc, src_attr);
    param.dq_param_ptr->src_reorder = dnnl::reorder(src_reorder_pd);

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
  }

  // For fp32 and static int8 op (int8 * int8 -> int8)
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static inline void do_compute(
      const matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const std::vector<tensor>& bin_post_params = {}) {
    auto& pd = param.pd;
    auto& primitive = param.primitive;
    auto& op_attr = param.op_attr;
    auto& src_attr = param.src_attr;
    auto& weights_attr = param.weights_attr;
    auto& bias_attr = param.bias_attr;

    auto expected_src_desc = pd.src_desc();
    auto expected_wei_desc = pd.weights_desc();
    auto expected_dst_desc = pd.dst_desc();

    auto& expected_src = reorder_src ?
                         src.reorder_if_differ_in(expected_src_desc, src_attr) :
                         src;
    auto& expected_weights = reorder_weight ?
                             weights.reorder_if_differ_in(expected_wei_desc, weights_attr) :
                             weights;
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }
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
    // Do not reorder these params. They may have different shapes as dst
    for (int i = 0; i < bin_post_params.size(); i++) {
      args.insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
            bin_post_params[i]});
    }
    if (reorder_src) {
      tensor expected_dst;
      if (dst.is_empty() || dst.get_desc() != expected_dst_desc){
        // If dst buffer are not given by user or user given dst buffer are not under expected format
        // We need init a new one
        expected_dst.init(expected_dst_desc);
        if (!dst.is_empty() && op_attr.has_op_kind(kind::sum)) {
          // We need copy the content of given buffer if matmul is fused with sum
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

  // For fp32. Set reorder flags to false if you are sure the memory layout
  // aligns with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static inline void do_compute_binary(
      const matmul_forward_params& param,
      const tensor& src,
      const tensor& other,
      const tensor& weights,
      const tensor& bias,
      tensor& dst) {
    auto& pd = param.pd;
    auto& primitive = param.primitive;
    auto& bias_attr = param.bias_attr;

    auto expected_src_desc = pd.src_desc();
    auto expected_wei_desc = pd.weights_desc();
    auto expected_dst_desc = pd.dst_desc();

    auto& expected_src = reorder_src ?
                         src.reorder_if_differ_in(expected_src_desc) :
                         src;
    
    auto& expected_other = reorder_src ?
                           other.reorder_if_differ_in(expected_dst_desc) :
                           other;
    
    auto& expected_weights = reorder_weight ?
                             weights.reorder_if_differ_in(expected_wei_desc) :
                             weights;
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    args.insert(
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, expected_other});
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }
    if (param.all_scales && !param.all_scales->empty()) {
      for (auto& arg_scale_pair : *param.all_scales) {
        int dnnl_arg = arg_scale_pair.first;
        tensor& scales_m = arg_scale_pair.second;
        args.insert({DNNL_ARG_ATTR_SCALES | dnnl_arg, scales_m});
      }
    }
    if (reorder_src) {
      tensor expected_dst;
      if (dst.is_empty() || dst.get_desc() != expected_dst_desc){
        // If dst buffer are not given by user or user given dst buffer are not under expected format
        // We need init a new one
        expected_dst.init(expected_dst_desc);
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

  // For dynamic int8 op (fp32 * int8 -> fp32)
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_weight = true>
  static inline void do_compute_dynamic_quant(
      const matmul_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales,
      const zero_point_t& src_zero_points,
      const float dst_coeff = 1.0f,
      const engine& aengine = engine::cpu_engine()) {
    /* Compute for dynamic quantized linear. This function does the following things:
     * - Get matmul primitive from param
     * - Get reorder primitive for src from param.dq_param_ptr
     * - Prepare tensors of output scales and zero points.
     * - Compute by executing matmul primitive
     */

    // Get primitive, etc. from param
    IDEEP_ENFORCE(param.dq_param_ptr, "Parameters for dynamic quantization not found");
    auto& pd = param.pd;
    auto& primitive = param.primitive;
    auto& weights_attr = param.weights_attr;
    auto& expected_src_desc = param.dq_param_ptr->src_desc;
    auto &src_reorder = param.dq_param_ptr->src_reorder;

    // Prepare tensor of output scales
    auto src_scales_in =
        src.has_scale() ? src.get_scale()
                        : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);
    auto& dst_scales_in = IDEEP_DEF_SCALE;

    const dim scale_zp_stride = 1;

    // Prepare tensor of src scales
    for (auto& s : src_scales_in) {
      s = 1.0 / s;
    }
    int src_scale_size = src_scales_in.size();
    tensor::desc src_scales_desc = {{src_scale_size}, data_type::f32, {scale_zp_stride}};
    tensor src_scales_m(src_scales_desc, reinterpret_cast<float*>(src_scales_in.data()), aengine);

    // Prepare tensor of src zero point
    auto src_zero_point = src.has_zero_point() ? src.get_zero_point() :
                              src_zero_points.empty() ? IDEEP_DEF_ZP : src_zero_points;
    const auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
    IDEEP_ENFORCE(src_zero_point_size == 1,
                  "DNNL only support 1-dim zero_point");
    tensor::desc src_zero_point_desc = {{src_zero_point_size}, data_type::s32, {scale_zp_stride}};
    tensor src_zero_point_m(src_zero_point_desc, reinterpret_cast<int32_t*>(src_zero_point.data()), aengine);

    // Reroder src (f32 -> u8)
    tensor expected_src(expected_src_desc);
    src_reorder.execute(stream::default_stream(),
                        {{DNNL_ARG_FROM, src},
                        {DNNL_ARG_TO, expected_src},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, src_scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, src_zero_point_m}});

    // Check weight desc
    auto& expected_weights = reorder_weight ?
                             weights.reorder_if_differ_in(pd.weights_desc(), weights_attr) :
                             weights;
    tensor &expected_dst = dst;

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_DST, expected_dst});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }
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
    // Src scales and zero point are obtained at runtime thus set separately
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_m});
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m});

    primitive.execute(stream::default_stream(), args);
  }

};

}  // namespace ideep

#endif