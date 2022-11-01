#ifndef IDEEP_ATTRIBUTES_HPP
#define IDEEP_ATTRIBUTES_HPP

#include "abstract_types.hpp"
#include "utils.hpp"

namespace ideep {

using post_ops = dnnl::post_ops;
using zero_point_map = std::unordered_map<int, zero_point_t>;
static const zero_point_map empty_zp_map;

/// Attribute class for extra information into computations
struct attr_t : public dnnl::primitive_attr {
  attr_t() {}

  attr_t(const attr_t& other) : dnnl::primitive_attr(other) {
    *this = other;
  }

  attr_t(const dnnl::primitive_attr& other) : dnnl::primitive_attr(other) {}

  attr_t(int mask, const scale_t& scales) {
    // set_output_scales(mask, scales);
    set_output_scales_mask(mask);
    scales_.reset(new scale_t(scales));
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

  void set_output_scales(int mask, const scale_t& scales) {
    set_output_scales_mask(mask);
    if (!scales_) {
      scales_.reset(new scale_t(scales));
    } else {
      *scales_ = scales;
    }
  }

  std::pair<scale_t, int> get_output_scales() const {
    if (!scales_) {
      return std::make_pair(scale_t(), 0);
    }
    int c_mask = utils::op_scale_mask(scales_->size());
    return std::make_pair(*scales_, c_mask);
  }

  bool has_output_scales() const {
    return (scales_ && !(*scales_).empty());
  }

  // @param arg DNNL_ARGS
  void set_zero_points(int arg, int mask, const zero_point_t& zero_points) {
    set_zero_points_mask(arg, mask);
    if (!zero_points_) {
      zero_points_.reset(new zero_point_map());
    }
    (*zero_points_)[arg] = zero_points;
  }

  std::pair<zero_point_t, int> get_zero_points(int arg) const {
    if (!zero_points_ || !zero_points_->count(arg)) {
      return std::make_pair(zero_point_t(), 0);
    }
    auto& zp = (*zero_points_)[arg];
    int mask = utils::tensor_zp_mask(zp.size());
    return std::make_pair(zp, mask);
  }

  const zero_point_map& get_all_zero_points() const {
    if (!zero_points_) {
      return empty_zp_map;
    }
    return *zero_points_;
  }

  bool has_zero_points() const {
    return (zero_points_ && !(*zero_points_).empty());
  }

  // Helper factory
  static attr_t fuse_sum(float scale = 1.0) {
    attr_t attr;
    post_ops po;
    po.append_sum(scale);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_swish_sum(
      float sum_scale = 1.0,
      float swish_scale = 1.0,
      float swish_alpha = 1.0,
      float swish_beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(swish_scale, algorithm::eltwise_swish, swish_alpha, swish_beta);
    po.append_sum(sum_scale);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_relu(
      float scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_gelu(
      float scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f,
      algorithm gelu_type = algorithm::eltwise_gelu_erf) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, gelu_type, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_elu(
      float scale = 1.0,
      float alpha = 0.f,
      float beta = 1.0) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_elu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_sigmoid(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_logistic, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_swish(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_swish, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_tanh(
      float scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_tanh, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_mish(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_mish, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t residual(
      float sum_scale = 1.0,
      float relu_scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_sum(sum_scale);
    po.append_eltwise(relu_scale, algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_clamp(float lower_bound = -1.0, float upper_bound = 1.0) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(1.0, algorithm::eltwise_clip, lower_bound, upper_bound);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_hardswish(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_hardswish, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_abs(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_abs, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_exp(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_exp, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_square(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_square, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_log(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_log, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_round(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_round, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_sqrt(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_sqrt, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_pow(
      float scale = 1.0,
      float alpha = 1.0,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_pow, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_hardsigmoid() {
    constexpr float scale = 1.0f;
    constexpr float alpha = 1.0f / 6.0f;
    constexpr float beta = 1.0f / 2.0f;

    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_hardsigmoid, alpha, beta);
    attr.set_post_ops(po);
    return attr;
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

  std::tuple<kind, float, float, float, algorithm> get_params(int index) const {
    auto po = get_post_ops();
    IDEEP_ENFORCE(index < po.len(), "post_ops index is out of range");

    algorithm alg = algorithm::undef;
    float scale = 1.0, alpha = 1.0, beta = 0.0;
    memory::desc binary_src_desc;

    auto akind = po.kind(index);
    switch (akind) {
      case kind::sum:
        po.get_params_sum(index, scale);
        break;
      case kind::eltwise:
        po.get_params_eltwise(index, scale, alg, alpha, beta);
        break;
      case kind::binary:
        po.get_params_binary(index, alg, binary_src_desc);
        break;
      default:
        error::wrap_c_api(dnnl_invalid_arguments, "could not get params");
        break;
    }

    return std::make_tuple(akind, scale, alpha, beta, alg);
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

  attr_t& operator=(const attr_t& rhs) {
    if (this == &rhs) {
      return *this;
    }
    dnnl_primitive_attr_t result;
    error::wrap_c_api(
        dnnl_primitive_attr_clone(&result, rhs.get()),
        "could not clone primitive attributes");
    this->reset(result);
    int c_mask, z_mask;
    scale_t scales;
    zero_point_t zero_points;
    std::tie(scales, c_mask) = rhs.get_output_scales();
    if (scales_) {
      *scales_ = scales;
    } else {
      scales_.reset(new scale_t(scales));
    }
    if (zero_points_) {
      *zero_points_ = rhs.get_all_zero_points();
    } else {
      zero_points_.reset(new zero_point_map(rhs.get_all_zero_points()));
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
      std::tie(l_akind, l_scale, l_alpha, l_beta, l_alg) = get_params(index);
      std::tie(r_akind, r_scale, r_alpha, r_beta, r_alg) =
          rhs.get_params(index);
      if (l_akind != r_akind || l_alg != r_alg || l_scale != r_scale ||
          l_alpha != r_alpha || l_beta != r_beta) {
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
      std::tie(akind, scale, alpha, beta, alg) = get_params(i);

      switch (akind) {
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
        case kind::binary:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alg);
        default:
          break;
      }
    }

    // encode output scales
    if (has_output_scales()) {
      auto scales = get_output_scales();
      utils::to_bytes(bytes, scales.first);
      utils::to_bytes(bytes, scales.second);
    }

    // encode zero points
    if (has_zero_points()) {
      for (auto& zp : get_all_zero_points()) {
        utils::to_bytes(bytes, zp.first);
        utils::to_bytes(bytes, zp.second);
      }
    }

    // Note: depthwise/binary post op, zero points, scales, rnn params are
    // not encoded so far. PD cache is supposed to use in convolution only
    // as a temporary workaround for gemm-based conv pd overhead
  }

private:
  std::shared_ptr<scale_t> scales_;
  // Map key: DNNL ARG (e.g. DNNL_ARG_SRC)
  std::shared_ptr<std::unordered_map<int, zero_point_t>> zero_points_;
};

} // namespace ideep

#endif
