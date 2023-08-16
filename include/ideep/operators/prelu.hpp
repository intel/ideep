#ifndef IDEEP_OPERATORS_PRELU_HPP
#define IDEEP_OPERATORS_PRELU_HPP

namespace ideep {

struct prelu_forward : public dnnl::prelu_forward {
  using super = dnnl::prelu_forward;

  static void compute(
      const tensor& src,
      const tensor& weight,
      tensor& dst,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_in = src;
    auto weight_in = weight;

    // Reshape weight to src dimension
    auto new_dims = src.get_dims();
    if (src.ndims() != weight.ndims()) {
      std::vector<dim> dim_w(src.ndims(), 1);
      dim_w[1] = weight.get_dim(0);
      weight_in.reshape(dim_w);
    }

    auto src_desc = src_in.get_desc();
    auto weight_desc = weight_in.get_desc().to_format_any();

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd =
        primitive_desc(aengine, aprop_kind, src_desc, weight_desc, src_desc, op_attr);
    auto expected_weights = weight_in.reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC, src_in},
         {DNNL_ARG_WEIGHTS, expected_weights},
         {DNNL_ARG_DST, dst},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

struct prelu_backward : public dnnl::prelu_backward {
  using super = dnnl::prelu_backward;

  static void compute(
      const tensor& src,
      const tensor& weight,
      const tensor& diff_dst,
      tensor& diff_src,
      tensor& diff_weight,
      prop_kind aprop_kind = prop_kind::backward,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    auto src_in = src;
    auto weight_in = weight;
    auto diff_dst_in = diff_dst;
    auto weight_dims = weight_in.get_dims();

    // Reshape wieght to src dimension
    auto new_dims = src.get_dims();
    if (src.ndims() != weight.ndims()) {
      std::vector<dim> dim_w(src.ndims(), 1);
      dim_w[1] = weight.get_dim(0);
      weight_in.reshape(dim_w);
    }

    auto src_desc = src_in.get_desc();
    auto weight_desc = weight_in.get_desc().to_format_any();
    auto diff_dst_desc = diff_dst_in.get_desc();
    auto diff_weights_desc =
        tensor::desc(
            weight_in.get_dims(), diff_dst_in.get_data_type(), tag::any)
            .to_format_any();
    auto forward_hints = prelu_forward::primitive_desc(
        aengine, prop_kind::forward, src_desc, weight_desc, src_desc);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        aengine, src_desc, weight_desc, diff_dst_desc, diff_weights_desc, diff_dst_desc,
        forward_hints, op_attr);

    auto expected_diff_dst =
        diff_dst_in.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src_in.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weight_in.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    diff_weight.reinit_if_possible(pd.diff_weights_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_WEIGHTS, expected_weights},
         {DNNL_ARG_DIFF_SRC, diff_src},
         {DNNL_ARG_DIFF_WEIGHTS, diff_weight},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});

    // Reshape weight back to original dimension
    if (diff_weight.get_dims() != weight_dims) {
      diff_weight.reshape(weight_dims);
    }
  }
};

} // namespace ideep

#endif
