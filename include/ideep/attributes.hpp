#ifndef IDEEP_ATTRIBUTES_HPP
#define IDEEP_ATTRIBUTES_HPP

#include "abstract_types.hpp"
#include "utils.hpp"

namespace ideep {

using post_ops = dnnl::post_ops;
using scale_mask_pair = std::pair<scale_t, int>;
using zp_mask_pair = std::pair<zero_point_t, int>;
// Map DNNL arg to pair of scales and mask
using scale_map = std::unordered_map<int, scale_mask_pair>;
// Map DNNL arg to pair of zero points and mask
using zero_point_map = std::unordered_map<int, zp_mask_pair>;
static const scale_map empty_scale_map;
static const zero_point_map empty_zp_map;

/// Attribute class for extra information into computations
struct attr_t : public dnnl::primitive_attr {
  attr_t() {}

  attr_t(const attr_t& other) : dnnl::primitive_attr(other) {
    *this = other;
  }

  attr_t(const dnnl::primitive_attr& other) : dnnl::primitive_attr(other) {}

  /// @param arg Parameter argument index
  attr_t(int arg, int mask, const scale_t& scales) {
    set_scales_mask(arg, mask);
    if (!scales_) {
      scales_.reset(new scale_map());
    }
    (*scales_)[arg] = std::make_pair(scales, mask);
  }

  // Defualt arg = DNNL_ARG_DST
  attr_t(int mask, const scale_t& scales) {
    set_scales_mask(DNNL_ARG_DST, mask);
    if (!scales_) {
      scales_.reset(new scale_map());
    }
    (*scales_)[DNNL_ARG_DST] = std::make_pair(scales, mask);
  }

  attr_t(dnnl_fpmath_mode_t fpmath_mode,
         dnnl::scratchpad_mode sp_mode = dnnl::scratchpad_mode::user) {
    set_fpmath_mode(fpmath_mode);
    set_scratchpad_mode(sp_mode);
  }

  attr_t& set_fpmath_mode(dnnl_fpmath_mode_t mode) {
    error::wrap_c_api(
        dnnl_primitive_attr_set_fpmath_mode(get(), mode),
        "could not set fpmath mode primitive attribute");
    return *this;
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  void set_scales(int arg, int mask, const scale_t& scales) {
    set_scales_mask(arg, mask);
    if (!scales_) {
      scales_.reset(new scale_map());
    }
    (*scales_)[arg] = std::make_pair(scales, mask);
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  scale_mask_pair& get_scales(int arg) const {
    IDEEP_ENFORCE(has_scales_for(arg),
                  "Scales for arg not found!");
    return (*scales_)[arg];
  }

  const scale_map& get_all_scales() const {
    return scales_ ? *scales_ : empty_scale_map;
  }

  bool has_scales() const {
    return (scales_ && !(*scales_).empty());
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  bool has_scales_for(int arg) const {
    return has_scales() && scales_->count(arg);
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  void set_zero_points(int arg, int mask, const zero_point_t& zero_points) {
    set_zero_points_mask(arg, mask);
    if (!zero_points_) {
      zero_points_.reset(new zero_point_map());
    }
    (*zero_points_)[arg] = std::make_pair(zero_points, mask);
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  zp_mask_pair& get_zero_points(int arg) const {
    IDEEP_ENFORCE(has_zero_points_for(arg),
                  "Zero point for arg not found!");
    return (*zero_points_)[arg];
  }

  const zero_point_map& get_all_zero_points() const {
    return zero_points_ ? *zero_points_ : empty_zp_map;
  }

  bool has_zero_points() const {
    return (zero_points_ && !(*zero_points_).empty());
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  bool has_zero_points_for(int arg) const {
    return has_zero_points() && zero_points_->count(arg);
  }

  // Helper factory
  static inline attr_t fuse_eltwise(algorithm alg, float alpha, float beta) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(alg, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_sum(float scale = 1.0, int32_t sum_zero_point = 0) {
    attr_t attr;
    post_ops po;
    po.append_sum(scale, sum_zero_point);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_swish_sum(
      float sum_scale = 1.0,
      float swish_alpha = 1.0,
      float swish_beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, swish_alpha, swish_beta);
    po.append_sum(sum_scale);
    attr.set_post_ops(po);
    return attr;
  }

  // Deprecated
  static attr_t fuse_relu(
      float scale = 1.0f, // unused since onednn 3.0
      float alpha = 0.f,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_relu, alpha, beta);
  }

  static attr_t fuse_relu_v2(
      float alpha = 0.f,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_relu, alpha, beta);
  }

  // Deprecated
  static attr_t fuse_gelu(
      float scale, // unused since onednn 3.0
      float alpha = 0.f,
      float beta = 0.f,
      algorithm gelu_type = algorithm::eltwise_gelu_erf) {
    return fuse_eltwise(gelu_type, alpha, beta);
  }

  static attr_t fuse_gelu_v2(
      float alpha = 0.f,
      float beta = 0.f,
      algorithm gelu_type = algorithm::eltwise_gelu_erf) {
    return fuse_eltwise(gelu_type, alpha, beta);
  }

  // Deprecated
  static attr_t fuse_elu(
      float scale, // unused since onednn 3.0
      float alpha = 0.f,
      float beta = 1.0) {
    return fuse_eltwise(algorithm::eltwise_elu, alpha, beta);
  }

  static attr_t fuse_elu_v2(
      float alpha = 0.f,
      float beta = 1.0) {
    return fuse_eltwise(algorithm::eltwise_elu, alpha, beta);
  }

  static attr_t fuse_sigmoid(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_logistic, alpha, beta);
  }

  static attr_t fuse_swish(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_swish, alpha, beta);
  }

  static attr_t fuse_tanh(
      float alpha = 0.f,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_tanh, alpha, beta);
  }

  static attr_t fuse_mish(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_mish, alpha, beta);
  }

  static attr_t residual(
      float sum_scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_sum(sum_scale);
    po.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t residual_with_sum_zero_point(
      float sum_scale = 1.0,
      int32_t sum_zero_point = 0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_sum(sum_scale, sum_zero_point);
    po.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_clamp(float lower_bound = -1.0, float upper_bound = 1.0) {
    return fuse_eltwise(algorithm::eltwise_clip, lower_bound, upper_bound);
  }

  static attr_t fuse_hardswish(
      float alpha = 1.0f / 6.0f,
      float beta = 0.5f) {
    return fuse_eltwise(algorithm::eltwise_hardswish, alpha, beta);
  }

  static attr_t fuse_abs(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_abs, alpha, beta);
  }

  static attr_t fuse_exp(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_exp, alpha, beta);
  }

  static attr_t fuse_square(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_square, alpha, beta);
  }

  static attr_t fuse_log(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_log, alpha, beta);
  }

  static attr_t fuse_round(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_round, alpha, beta);
  }

  static attr_t fuse_sqrt(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_sqrt, alpha, beta);
  }

  // Deprecated
  static attr_t fuse_pow(
      float scale, // unused since onednn 3.0
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_pow, alpha, beta);
  }

  static attr_t fuse_pow_v2(
      float alpha = 1.0,
      float beta = 0.f) {
    return fuse_eltwise(algorithm::eltwise_pow, alpha, beta);
  }

  static attr_t fuse_hardsigmoid() {
    constexpr float alpha = 1.0f / 6.0f;
    constexpr float beta = 1.0f / 2.0f;
    return fuse_eltwise(algorithm::eltwise_hardsigmoid, alpha, beta);
  }

  static attr_t fuse_binary(algorithm alg, memory::desc src_desc) {
    attr_t attr;
    post_ops po;
    po.append_binary(alg, src_desc);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t attr_post_ops(post_ops po) {
    attr_t attr;
    attr.set_post_ops(po);
    return attr;
  }

  bool has_op_kind(kind op_kind) const {
    auto po = get_post_ops();
    for (int i = 0; i < po.len(); i++)
      if (op_kind == po.kind(i))
        return true;
    return false;
  }

  bool has_post_op() const {
    auto po = get_post_ops();
    return po.len() > 0;
  }

  std::tuple<kind, float, float, float, algorithm, int32_t> get_params(int index) const {
    auto po = get_post_ops();
    IDEEP_ENFORCE(index < po.len(), "post_ops index is out of range");

    algorithm alg = algorithm::undef;
    float scale = 1.0, alpha = 1.0, beta = 0.0;
    memory::desc binary_src_desc;
    int32_t zero_point = 0;
    memory::data_type post_op_sum_dtype;

    auto akind = po.kind(index);
    switch (akind) {
      case kind::sum:
        po.get_params_sum(index, scale, zero_point, post_op_sum_dtype);
        break;
      case kind::eltwise:
        po.get_params_eltwise(index, alg, alpha, beta);
        break;
      case kind::binary:
        po.get_params_binary(index, alg, binary_src_desc);
        break;
      default:
        error::wrap_c_api(dnnl_invalid_arguments, "could not get params");
        break;
    }

    return std::make_tuple(akind, scale, alpha, beta, alg, zero_point);
  }

  bool non_negitive_output() const {
    auto po = get_post_ops();
    auto last = po.len() - 1;
    if (last < 0) {
      return false;
    }

    auto params = get_params(last);
    if (std::get<0>(params) != kind::eltwise || std::get<1>(params) <= 0.f ||
        std::get<2>(params) != 0.f || std::get<3>(params) != 0.f ||
        std::get<4>(params) != algorithm::eltwise_relu)
      return false;

    return true;
  }

  // deep data copy
  attr_t& operator=(const attr_t& rhs) {
    if (this == &rhs) {
      return *this;
    }
    dnnl_primitive_attr_t result;
    error::wrap_c_api(
        dnnl_primitive_attr_clone(&result, rhs.get()),
        "could not clone primitive attributes");
    this->reset(result);
    if (rhs.has_scales()) {
      if (scales_) {
        *scales_ = rhs.get_all_scales();
      } else {
        scales_.reset(new scale_map(rhs.get_all_scales()));
      }
    }
    if (rhs.has_zero_points()) {
      if (zero_points_) {
        *zero_points_ = rhs.get_all_zero_points();
      } else {
        zero_points_.reset(new zero_point_map(rhs.get_all_zero_points()));
      }
    }
    return *this;
  }

  bool has_same_postop_as(const attr_t& rhs) const {
    auto l_po = get_post_ops();
    auto r_po = rhs.get_post_ops();
    if (l_po.len() != r_po.len()) {
      return false;
    }
    for (auto index = 0; index < l_po.len(); index++) {
      kind l_akind, r_akind;
      algorithm l_alg, r_alg;
      float l_scale = 1.0, l_alpha = 1.0, l_beta = 0.0;
      float r_scale = 1.0, r_alpha = 1.0, r_beta = 0.0;
      int32_t l_zp = 0, r_zp = 0;
      std::tie(l_akind, l_scale, l_alpha, l_beta, l_alg, l_zp) = get_params(index);
      std::tie(r_akind, r_scale, r_alpha, r_beta, r_alg, r_zp) =
          rhs.get_params(index);
      if (l_akind != r_akind || l_alg != r_alg || l_scale != r_scale ||
          l_alpha != r_alpha || l_beta != r_beta || l_zp != r_zp) {
        return false;
      }
    }
    return true;
  }

  void to_bytes(utils::bytestring& bytes) const {
    // encode post ops
    auto num_ops = get_post_ops().len();
    for (int i = 0; i < num_ops; i++) {
      kind akind;
      algorithm alg = algorithm::undef;
      float scale = 1.0, alpha = 1.0, beta = 0.0;
      int32_t zp = 0;
      std::tie(akind, scale, alpha, beta, alg, zp) = get_params(i);

      switch (akind) {
        case kind::sum:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, scale);
          bytes.append(1, '.');
          utils::to_bytes(bytes, zp);
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
        case kind::binary:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alg);
        default:
          break;
      }
    }

    // encode output scales
    if (has_scales()) {
      for (auto& scales : get_all_scales()) {
        utils::to_bytes(bytes, scales.first); // dnnl arg index
        utils::to_bytes(bytes, scales.second.first); // scale vector
        utils::to_bytes(bytes, scales.second.second); // mask
      }
    }

    // encode zero points
    if (has_zero_points()) {
      for (auto& zp : get_all_zero_points()) {
        utils::to_bytes(bytes, zp.first); // dnnl arg index
        utils::to_bytes(bytes, zp.second.first); // zero point vector
        utils::to_bytes(bytes, zp.second.second); // mask
      }
    }

    // Note: depthwise/binary post op, zero points, scales, rnn params are
    // not encoded so far. PD cache is supposed to use in convolution only
    // as a temporary workaround for gemm-based conv pd overhead
  }

private:
  std::shared_ptr<scale_map> scales_;
  std::shared_ptr<zero_point_map> zero_points_;
};

} // namespace ideep

#endif