#ifndef IDEEP_OPERATORS_PRELU_HPP
#define IDEEP_OPERATORS_PRELU_HPP

namespace ideep {

struct prelu_forward : public dnnl::prelu_forward {

  using super = dnnl::prelu_forward;

  // weights: [num_channel/1] => ndims: 1, dims[0] = num_channels/1
  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_in = src;
    // we should leave dequantization to the framework
    // if (utils::one_of(src.get_data_type(), data_type::s8, data_type::u8)) {
    //   src_in = src_in.dequantize();
    // }
    auto weights_ = weights;
    if(src.ndims() != weights.ndims()) {
      std::vector<dim> dim_w(src.ndims(), 1);
      dim_w[1] = weights.get_dim(0);
      weights_ = weights_.reshape(dim_w);
    }
    auto src_desc = src_in.get_desc();
    auto weights_desc = weights_.get_desc();

    auto pd = primitive_desc({prop_kind::forward_training, src_desc, weights_desc}, aengine);

    dst.reinit_if_possible(pd.dst_desc());
    if (src_in.has_scale()) {
      dst.set_scale(src_in.get_scale());
    }

    auto expected_src = src_in.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());
    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_WEIGHTS, expected_weights},
                      {DNNL_ARG_DST, dst}});

    // xpz: ???
    // if (dst.has_scale() && dst.get_data_type() == data_type::s8) {
    //   dst.to_type(data_type::u8);
    // }
  }
};

struct prelu_backward : public dnnl::prelu_backward {
  using super = dnnl::prelu_backward;
  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& diff_dst,
                      tensor& diff_src,
                      tensor& diff_weights,
                      const engine& aengine = engine::cpu_engine()) {
    auto weights_ = weights;
    bool reshape = false;
    if(src.ndims() != weights.ndims()) {
      std::vector<dim> dim_w(src.ndims(), 1);
      dim_w[1] = weights.get_dim(0);
      weights_ = weights_.reshape(dim_w);
      reshape = true;
    }

    auto src_desc = src.get_desc();
    auto weights_desc = weights_.get_desc();
    auto diff_dst_desc = diff_dst.get_desc();
    auto diff_src_desc = tensor::desc(src.get_dims(), src.get_data_type(), tag::any);
    auto diff_weights_desc = tensor::desc(weights_.get_dims(), weights_.get_data_type(), tag::any);

    auto forward_hints = prelu_forward::primitive_desc(
            {prop_kind::forward_training, src_desc, weights_desc},
            aengine);
    auto pd = primitive_desc(
            {src_desc, weights_desc, diff_dst_desc, diff_weights_desc},
            aengine,
            forward_hints);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    diff_weights.reinit_if_possible(pd.diff_weights_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_WEIGHTS, expected_weights},
                      {DNNL_ARG_DIFF_DST, expected_diff_dst},
                      {DNNL_ARG_DIFF_SRC, diff_src},
                      {DNNL_ARG_DIFF_WEIGHTS, diff_weights}});

    if(reshape)
      diff_weights = diff_weights.reshape(weights.get_dims());
  }
};
}  // namespace ideep

#endif
