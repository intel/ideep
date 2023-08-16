#ifndef IDEEP_OPERATORS_POOL_HPP
#define IDEEP_OPERATORS_POOL_HPP

namespace ideep {
// pooling_v2_forward/backward supports dilation,
// while pooling_forward/backward does not.

struct pooling_forward : public dnnl::pooling_forward {
  using super = dnnl::pooling_forward;

  static void compute(
      const tensor& src,
      const dims& output_sizes,
      tensor& dst,
      const dims& strides,
      const dims& kernel,
      const dims& padding_l,
      const dims& padding_r,
      algorithm aalgorithm,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    bool with_workspace = aprop_kind == prop_kind::forward_training &&
        aalgorithm == dnnl::algorithm::pooling_max;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    tensor::desc dst_desc(output_sizes, src.get_data_type(), tag::any);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const dims dilation(strides.size(), 0);

    auto pd = primitive_desc(
        aengine,
        aprop_kind,
        aalgorithm,
        src_desc,
        dst_desc,
        strides,
        kernel,
        dilation,
        padding_l,
        padding_r,
        op_attr);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_SRC, expected_src},
        {DNNL_ARG_DST, dst},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct pooling_v2_forward : public dnnl::pooling_forward {
  using super = dnnl::pooling_forward;

  static void compute(
      const tensor& src,
      const dims& output_sizes,
      tensor& dst,
      const dims& strides,
      const dims& kernel,
      const dims& dilation,
      const dims& padding_l,
      const dims& padding_r,
      algorithm aalgorithm,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    bool with_workspace = aprop_kind == prop_kind::forward_training &&
        aalgorithm == dnnl::algorithm::pooling_max;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    tensor::desc dst_desc(output_sizes, src.get_data_type(), tag::any);

    auto dil_compatible = utils::get_compatible_dilates(dilation);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine,
        aprop_kind,
        aalgorithm,
        src_desc,
        dst_desc,
        strides,
        kernel,
        dil_compatible,
        padding_l,
        padding_r,
        op_attr);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_SRC, expected_src},
        {DNNL_ARG_DST, dst},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};

    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct pooling_backward : public dnnl::pooling_backward {
  using super = dnnl::pooling_backward;

  static void compute(
      const tensor& diff_dst,
      const tensor& dst,
      const tensor& src,
      tensor& diff_src,
      const dims& strides,
      const dims& kernel,
      const dims& padding_l,
      const dims& padding_r,
      algorithm aalgorithm,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(src.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    auto src_desc = src.get_desc();
    auto dst_desc = dst.get_desc();

    const dims dilation(strides.size(), 0);

    auto forward_hints = pooling_forward::primitive_desc(
        aengine,
        prop_kind::forward,
        aalgorithm,
        src_desc,
        dst_desc,
        strides,
        kernel,
        dilation,
        padding_l,
        padding_r);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine, aalgorithm, src_desc, dst_desc, strides, kernel,
        dilation, padding_l, padding_r, forward_hints, op_attr);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_DIFF_DST, expected_diff_dst},
        {DNNL_ARG_DIFF_SRC, diff_src},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};

    if (dst.has_workspace()) {
      auto expected_workspace =
          dst.get_workspace().reorder_if_differ_in(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, expected_workspace});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct pooling_v2_backward : public dnnl::pooling_backward {
  using super = dnnl::pooling_backward;

  static void compute(
      const tensor& diff_dst,
      const tensor& dst,
      const tensor& src,
      tensor& diff_src,
      const dims& strides,
      const dims& kernel,
      const dims& dilation,
      const dims& padding_l,
      const dims& padding_r,
      algorithm aalgorithm,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(src.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    auto src_desc = src.get_desc();
    auto dst_desc = dst.get_desc();
    auto dil_compatible = utils::get_compatible_dilates(dilation);

    auto forward_hints = pooling_forward::primitive_desc(
        aengine,
        prop_kind::forward,
        aalgorithm,
        src_desc,
        dst_desc,
        strides,
        kernel,
        dil_compatible,
        padding_l,
        padding_r);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine,
        aalgorithm,
        src_desc,
        dst_desc,
        strides,
        kernel,
        dil_compatible,
        padding_l,
        padding_r,
        forward_hints,
        op_attr);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_DIFF_DST, expected_diff_dst},
        {DNNL_ARG_DIFF_SRC, diff_src},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};
    if (dst.has_workspace()) {
      auto expected_workspace =
          dst.get_workspace().reorder_if_differ_in(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, expected_workspace});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

} // namespace ideep

#endif
