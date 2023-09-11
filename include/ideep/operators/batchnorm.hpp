#ifndef IDEEP_OPERATORS_BATCHNORM_HPP
#define IDEEP_OPERATORS_BATCHNORM_HPP
#include "sum.hpp"

namespace ideep {

struct batch_normalization_forward_inference
    : public dnnl::batch_normalization_forward {
  using super = dnnl::batch_normalization_forward;

  static void compute(
      const tensor& src,
      const tensor& scale,
      const tensor& shift,
      tensor& dst,
      float epsilon,
      const batch_normalization_flag flags =
          batch_normalization_flag::use_scale |
          batch_normalization_flag::use_shift,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy;
    compute_impl</*use_stats=*/false>(
        src, dummy, dummy, scale, shift, dst, epsilon, flags, aengine);
  }

  static void compute(
      const tensor& src,
      const tensor& mean,
      const tensor& variance,
      const tensor& scale,
      const tensor& shift,
      tensor& dst,
      float epsilon,
      const batch_normalization_flag flags =
          batch_normalization_flag::use_scale |
          batch_normalization_flag::use_shift,
      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*use_stats=*/true>(
        src, mean, variance, scale, shift, dst, epsilon, flags, aengine);
  }

 private:
  template <bool use_stats>
  static void compute_impl(
      const tensor& src,
      const tensor& mean,
      const tensor& variance,
      const tensor& scale,
      const tensor& shift,
      tensor& dst,
      float epsilon,
      const batch_normalization_flag flags,
      const engine& aengine) {
    auto pd_flags = batch_normalization_flag::use_scale |
                    batch_normalization_flag::use_shift;
    if (use_stats)
      pd_flags |= batch_normalization_flag::use_global_stats;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    bool fuse_norm_relu =
        (bool)(flags & batch_normalization_flag::fuse_norm_relu);
    attr_t attr = fuse_norm_relu ? attr_t::fuse_relu() : attr_t();
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = primitive_desc(
        aengine, prop_kind::forward_inference,
        src_desc, src_desc, epsilon, pd_flags, attr);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());

    if (use_stats) {
      auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
      auto expected_var = variance.reorder_if_differ_in(pd.variance_desc());
      super(pd).execute(
          stream::default_stream(),
          {{DNNL_ARG_SRC, expected_src},
           {DNNL_ARG_SCALE, scale},
           {DNNL_ARG_SHIFT, shift},
           {DNNL_ARG_VARIANCE, expected_var},
           {DNNL_ARG_MEAN, expected_mean},
           {DNNL_ARG_DST, dst},
           {DNNL_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(
          stream::default_stream(),
          {{DNNL_ARG_SRC, expected_src},
           {DNNL_ARG_SCALE, scale},
           {DNNL_ARG_SHIFT, shift},
           {DNNL_ARG_DST, dst},
           {DNNL_ARG_SCRATCHPAD, scratchpad}});
    }
  }
};

struct batch_normalization_forward_training
    : public dnnl::batch_normalization_forward {
  using super = dnnl::batch_normalization_forward;

  static void compute(
      const tensor& src,
      const tensor& scale,
      const tensor& shift,
      tensor& dst,
      tensor& mean,
      tensor& variance,
      float momentum,
      float epsilon,
      const batch_normalization_flag flags =
          batch_normalization_flag::use_scale |
          batch_normalization_flag::use_shift,
      const engine& aengine = engine::cpu_engine()) {
    auto pd_flags = flags | batch_normalization_flag::use_scale |
                    batch_normalization_flag::use_shift;
    bool with_workspace =
        (bool)(flags & batch_normalization_flag::fuse_norm_relu);

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine,
        prop_kind::forward_training, src_desc, src_desc, epsilon, pd_flags,
        op_attr);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    mean.reinit_if_possible(pd.mean_desc());
    variance.reinit_if_possible(pd.variance_desc());
    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_SRC, expected_src},
        {DNNL_ARG_SCALE, scale},
        {DNNL_ARG_SHIFT, shift},
        {DNNL_ARG_MEAN, mean},
        {DNNL_ARG_VARIANCE, variance},
        {DNNL_ARG_DST, dst},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);
  }

  static void compute(
      const tensor& src,
      const tensor& scale,
      const tensor& shift,
      tensor& dst,
      tensor& mean,
      tensor& variance,
      tensor& running_mean,
      tensor& running_var,
      float momentum,
      float epsilon,
      const batch_normalization_flag flags =
          batch_normalization_flag::use_scale |
          batch_normalization_flag::use_shift) {
    compute(src, scale, shift, dst, mean, variance, momentum, epsilon, flags);
    ideep::sum::compute(
        {momentum, 1 - momentum}, {running_mean, mean}, running_mean);
    ideep::sum::compute(
        {momentum, 1 - momentum}, {running_var, variance}, running_var);
  }
};

struct batch_normalization_backward
    : public dnnl::batch_normalization_backward {
  using super = dnnl::batch_normalization_backward;

  static void compute(
      const tensor& src,
      const tensor& mean,
      const tensor& variance,
      const tensor& diff_dst,
      const tensor& scale,
      tensor& diff_src,
      tensor& diff_scale,
      tensor& diff_shift,
      float epsilon,
      const tensor& dst = tensor(),
      const batch_normalization_flag flags =
          batch_normalization_flag::use_scale |
          batch_normalization_flag::use_shift,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(src.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    // TODO: support no-affine model
    auto pd_flags = flags | batch_normalization_flag::use_scale
                          | batch_normalization_flag::use_shift;
    bool with_workspace =
        (bool)(flags & batch_normalization_flag::fuse_norm_relu);
    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
        aengine, prop_kind::forward_training, src_desc, src_desc, epsilon, pd_flags);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine,
        prop_kind::backward,
        forward_hints.src_desc(),
        forward_hints.dst_desc(),
        src_desc,
        epsilon,
        pd_flags,
        forward_hints,
        op_attr);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
    auto expected_variance = variance.reorder_if_differ_in(pd.variance_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    diff_scale.reinit_if_possible(scale.get_desc());
    diff_shift.reinit_if_possible(scale.get_desc());

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_SRC, expected_src},
        {DNNL_ARG_DIFF_DST, expected_diff_dst},
        {DNNL_ARG_SCALE, scale},
        {DNNL_ARG_MEAN, expected_mean},
        {DNNL_ARG_VARIANCE, expected_variance},
        {DNNL_ARG_DIFF_SRC, diff_src},
        {DNNL_ARG_DIFF_SCALE, diff_scale},
        {DNNL_ARG_DIFF_SHIFT, diff_shift},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};
    if (with_workspace) {
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);
  }

};

} // namespace ideep

#endif
