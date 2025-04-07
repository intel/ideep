#ifndef IDEEP_OPERATORS_ELTWISE_HPP
#define IDEEP_OPERATORS_ELTWISE_HPP

namespace ideep {

struct eltwise_forward : public dnnl::eltwise_forward {
  using super = dnnl::eltwise_forward;

  static void compute(
      const tensor& src,
      tensor& dst,
      algorithm aalgorithm = algorithm::eltwise_relu,
      prop_kind aprop_kind = prop_kind::forward,
      float alpha = 0.0,
      float beta = 0.0,
      const engine& aengine = engine::cpu_engine()) {
    auto src_in = src;
    // we should leave dequantization to the framework
    if (aalgorithm != algorithm::eltwise_relu &&
        utils::one_of(src.get_data_type(), data_type::s8, data_type::u8)) {
      src_in = src_in.dequantize();
    }
    auto src_desc = src_in.get_desc();

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine, aprop_kind, aalgorithm, src_desc, src_desc, alpha, beta, op_attr);

    dst.reinit_if_possible(pd.dst_desc());
    if (src_in.has_scale()) {
      dst.set_scale(src_in.get_scale());
    }
    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC, src_in},
         {DNNL_ARG_DST, dst},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});

    // xpz: ???
    if (dst.has_scale() && aalgorithm == algorithm::eltwise_relu &&
        dst.get_data_type() == data_type::s8) {
      dst.to_type(data_type::u8);
    }
  }
};

struct eltwise_backward : public dnnl::eltwise_backward {
  using super = dnnl::eltwise_backward;
  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  static void compute(
      const tensor& src,
      const tensor& diff_dst,
      tensor& diff_src,
      algorithm aalgorithm = algorithm::eltwise_relu,
      float alpha = 0.0,
      float beta = 0.0,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(src.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    auto src_desc = src.get_desc();

    auto forward_hints = eltwise_forward::primitive_desc(
        aengine, prop_kind::forward, aalgorithm, src_desc, src_desc, alpha, beta);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine, aalgorithm, forward_hints.src_desc(), forward_hints.dst_desc(),
        src_desc, alpha, beta, forward_hints, op_attr);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    tensor expected_diff_src;
    if (diff_src.is_empty() || diff_src.get_desc() != pd.diff_src_desc()){
      // If diff_src buffer are not given by user or user given diff_src buffer are not under expected format
      // We need init a new one
      expected_diff_src.init(pd.diff_src_desc());
    } else {
      // The format of given diff_src buffer is expected
      expected_diff_src = diff_src;
    }

    auto use_dst = utils::one_of(
        aalgorithm,
        algorithm::eltwise_relu_use_dst_for_bwd,
        algorithm::eltwise_tanh_use_dst_for_bwd,
        algorithm::eltwise_elu_use_dst_for_bwd,
        algorithm::eltwise_sqrt_use_dst_for_bwd,
        algorithm::eltwise_logistic_use_dst_for_bwd,
        algorithm::eltwise_exp_use_dst_for_bwd);
    auto src_dst_arg = use_dst ? DNNL_ARG_DST : DNNL_ARG_SRC;
    auto expected_src_dst_desc = use_dst ? pd.dst_desc() : pd.src_desc();
    auto expected_src_dst = src.reorder_if_differ_in(expected_src_dst_desc);
    tensor scratchpad(pd.scratchpad_desc());
    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {src_dst_arg, expected_src_dst},
         {DNNL_ARG_DIFF_SRC, expected_diff_src},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});

    // reorder back to diff_src's buffer if needed
    if (diff_src != expected_diff_src) {
      if (!diff_src.is_empty() && diff_src.get_desc().has_same_shape_as(expected_diff_src.get_desc())) {
        // When diff_src buffer is given by user, and expected_diff_src has same shape
        // and different stride with diff_src, then expected_diff_src need to reorder
        // back to diff_src's buffer.
        diff_src.feed_from(expected_diff_src);
      } else {
        diff_src = expected_diff_src;
      }
    }
  }
};
} // namespace ideep

#endif
