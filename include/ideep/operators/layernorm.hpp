#ifndef IDEEP_OPERATORS_LAYERNORM_HPP
#define IDEEP_OPERATORS_LAYERNORM_HPP

namespace ideep {

struct layer_normalization_forward : public dnnl::layer_normalization_forward {
  using super = dnnl::layer_normalization_forward;

  static void compute(
      const tensor& src,
      const tensor& scale,
      const tensor& shift,
      tensor& dst,
      tensor& mean,
      tensor& variance,
      float epsilon,
      const engine& aengine = engine::cpu_engine()) {
    auto flags = batch_normalization_flag::use_scale |
                 batch_normalization_flag::use_shift;
    auto src_desc = src.get_desc();
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = primitive_desc(
        aengine, prop_kind::forward_training, src_desc, src_desc,
        epsilon, flags, op_attr);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    mean.reinit_if_possible(pd.mean_desc());
    variance.reinit_if_possible(pd.variance_desc());
    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_SCALE, scale},
         {DNNL_ARG_SHIFT, shift},
         {DNNL_ARG_MEAN, mean},
         {DNNL_ARG_VARIANCE, variance},
         {DNNL_ARG_DST, dst},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

struct layer_normalization_backward
    : public dnnl::layer_normalization_backward {
  static void compute() {}
};

} // namespace ideep

#endif
