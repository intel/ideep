#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

namespace ideep {

struct inner_product_forward_params {
  dnnl::inner_product_forward::primitive_desc pd;
  dnnl::inner_product_forward primitive;
  attr_t op_attr;

  inner_product_forward_params() {}

  inner_product_forward_params(
      dnnl::inner_product_forward::primitive_desc&& pd,
      dnnl::inner_product_forward&& primitive,
      attr_t&& op_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(std::move(op_attr)) {}
};

struct inner_product_forward
    : public dnnl::inner_product_forward,
      utils::computation_cache<dnnl::inner_product_forward::primitive_desc> {
  using super = dnnl::inner_product_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(src, weights, bias, dst, src_scales,
                                     weights_scales, dst_scales, attr,
                                     aprop_kind, alowp_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(src, weights, dummy_bias, dst, src_scales,
                                      weights_scales, dst_scales, attr,
                                      aprop_kind, alowp_kind, aengine);
  }

  // 2-in-1 compute, with bias
  static void compute_binary(const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             const tensor &bias,
                             tensor &dst,
                             const attr_t &attr = attr_t(),
                             const prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    inner_product_forward_params param;
    do_prepare<true>(param, src, weights, bias, dst, attr, aprop_kind, aengine);
    do_compute_binary<true>(param, src, other, weights, bias, dst);
  }

  // 2-in-1 compute, without bias
  static void compute_binary(const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             tensor &dst,
                             const attr_t &attr = attr_t(),
                             const prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    inner_product_forward_params param;
    do_prepare<false>(param, src, weights, dummy_bias, dst, attr, aprop_kind, aengine);
    do_compute_binary<false>(param, src, other, weights, dummy_bias, dst);
  }

  // Compute with prepared param, with bias
  static void compute_binary(const inner_product_forward_params &param,
                             const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             const tensor &bias,
                             tensor &dst) {
    do_compute_binary</*with_bias=*/true>(param, src, other, weights, bias, dst);
  }

  // Compute with prepared param, without bias
  static void compute_binary(const inner_product_forward_params &param,
                             const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             tensor &dst) {
    static tensor dummy_bias;
    do_compute_binary</*with_bias=*/false>(param, src, other, weights, dummy_bias, dst);
  }

  // Prepare with bias
  static void prepare(inner_product_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    do_prepare</*with_bias=*/true>(param, src, weights, bias, dst, attr,
                                   aprop_kind, aengine);
  }

  // Prepare without bias
  static void prepare(inner_product_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(param, src, weights, dummy_bias, dst, attr,
                                   aprop_kind, aengine);
  }


  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      const dims& src_dims = dims(),
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto x_dims = weights_dims;
    x_dims[0] = src_dims.empty() ? 1 : src_dims[0];
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc src_desc(x_dims, x_dtype, tag::any);
    tensor::desc dst_desc(y_dims, y_dtype, tag::any);
    tensor::desc weights_desc(weights_dims, dtype, tag::any);
    auto pd =
        primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc}, aengine);
    return pd.weights_desc();
  }

  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& dst_desc,
      const tensor::desc& bias_desc = tensor::desc(),
      const bool with_bias = false,
      const attr_t& attr = attr_t(),
      const prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto key = utils::create_key(
        aprop_kind,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        attr,
        with_bias,
        omp_get_max_threads());
    return fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc(
            {aprop_kind, src_desc, weights_desc, bias_desc, dst_desc},
            attr,
            aengine);
      } else {
        return primitive_desc(
            {aprop_kind, src_desc, weights_desc, dst_desc}, attr, aengine);
      }
    });
  };

private:
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           tensor& dst,
                           const scale_t& src_scales,
                           const scale_t& weights_scales,
                           const scale_t& dst_scales,
                           const attr_t& attr,
                           const prop_kind aprop_kind,
                           const lowp_kind alowp_kind,
                           const engine& aengine) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    auto src_ = src;
    if (src.ndims() != weights.ndims()) {
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
    }
    compute_impl_<with_bias>(src_, weights, bias, dst, src_scales,
                             weights_scales, dst_scales, attr, aprop_kind,
                             alowp_kind, aengine);
  }

  template <bool with_bias>
  static void compute_impl_(const tensor& src,
                            const tensor& weights,
                            const tensor& bias,
                            tensor& dst,
                            const scale_t& src_scales,
                            const scale_t& weights_scales,
                            const scale_t& dst_scales,
                            const attr_t& attr,
                            const prop_kind aprop_kind,
                            const lowp_kind alowp_kind,
                            const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    scale_t dst_scales_in;
    data_type dst_data_type;
    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};

    auto weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;

    // TODO(xpz): Remove int8 inner product implementation. We are switching to
    // matmul for quantized *mm ops
    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported lowp kind");

      auto src_scales_in =
          src.has_scale() ? src.get_scale()
                          : src_scales.empty() ? IDEEP_DEF_SCALE : src_scales;

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8,
                  tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      int scale_size = weights_scales_in.size() > 1 ? weights.get_dim(0) : 1;

      weights_desc = {weights.get_dims(), data_type::s8, tag::any};
      if (weights.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, false),
                        weights_scales_in};
      }

      // determine dst data type
      if (dst.get_data_type() != data_type::undef) {
        dst_data_type = dst.get_data_type();
      } else if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = dst_scales.empty() || dst_data_type == data_type::f32
                          ? IDEEP_DEF_SCALE
                          : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32, format_tag::any};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false),
                       bias_scales};
        }
      }
    } else {
      op_attr = attr;
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.f / src_scale[0];
        src_attr = {0, src_scale};
      }

      IDEEP_ENFORCE(utils::one_of(weights.get_data_type(),
                                  data_type::f32, data_type::bf16),
              "Incorrect data type in weights");

      // align weights data type with src
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                             : data_type::f32;
      src_desc = {src.get_dims(), dst_data_type, format_tag::any};
      weights_desc = {weights.get_dims(), dst_data_type, format_tag::any};
      if (with_bias) {
        IDEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                    data_type::f32, data_type::bf16),
                      "Incorrect data type in bias");
        bias_desc = bias.get_desc().to_format_any();
      }
    }

    tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = get_primitive_desc(
        src_desc,
        weights_desc,
        dst_desc,
        bias_desc,
        with_bias,
        op_attr,
        aprop_kind);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
  
    tensor expected_dst;
    if (dst.is_empty() || dst.get_desc() != pd.dst_desc()){
      // If dst buffer are not given by user or user given dst buffer are not under expected format
      // We need init a new one. "dst.get_desc() != pd.dst_desc()" conditional is setting for 
      // caffe2 caller, it might given a non-empty but uncorrect dst (maybe the size is uncorrect) 
      expected_dst.init(pd.dst_desc());
      if (!dst.is_empty() && op_attr.has_op_kind(kind::sum)) {
        // We need copy the content of given buffer if ip is fused with sum
        expected_dst.feed_from(dst);
      }
    } else {
      // The format of given dst buffer is expected
      expected_dst = dst;
    }

    if (!dst_scales.empty() && utils::one_of(dst.get_data_type(), data_type::u8, data_type::s8)) {  
      expected_dst.set_scale(dst_scales_in);
    }

    tensor scratchpad(pd.scratchpad_desc());

    if (with_bias){
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, expected_dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, expected_dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    }

    if (attr.non_negitive_output() && expected_dst.get_data_type() == data_type::s8) {
      expected_dst.to_type(data_type::u8);
    }
    // reorder back to dst's buffer if needed
    if (dst.is_empty() || 
        // when dst is empty, expect return buffer allocate by ideep  
        dst.get_desc() == expected_dst.get_desc() || 
        // dst and expected_dst is the same under this case
        !dst.get_desc().has_same_shape_as(expected_dst.get_desc())){
        // for caffe2 caller, get an uncorrect size dst from caller, can return buffer allocate by ideep
      dst =  expected_dst;
    } else {
      dst.feed_from(expected_dst);
    }
  }

  template <bool with_bias>
  static void do_prepare(
      inner_product_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const attr_t& attr,
      const prop_kind aprop_kind,
      const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t& op_attr = param.op_attr;
    data_type dst_data_type;
    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};

    op_attr = attr;
    IDEEP_ENFORCE(utils::one_of(weights.get_data_type(),
                                data_type::f32, data_type::bf16),
            "Incorrect data type in weights");

    // align weights data type with src
    dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                           : data_type::f32;
    src_desc = {src.get_dims(), dst_data_type, format_tag::any};
    weights_desc = {weights.get_dims(), dst_data_type, format_tag::any};
    if (with_bias) {
      IDEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                  data_type::f32, data_type::bf16),
                    "Incorrect data type in bias");
      bias_desc = bias.get_desc().to_format_any();
    }

    tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    param.pd = get_primitive_desc(
        src_desc,
        weights_desc,
        dst_desc,
        bias_desc,
        with_bias,
        op_attr,
        aprop_kind);
    param.primitive = std::move(super(param.pd));
  }

  template <bool with_bias>
  static void do_compute_binary(const inner_product_forward_params &param,
                                const tensor &src,
                                const tensor &other,
                                const tensor &weights,
                                const tensor &bias,
                                tensor &dst) {
    auto &pd = param.pd;
    auto &primitive = param.primitive;
    auto &op_attr = param.op_attr;

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    // make sure other has same format with dst.
    // TODO: other has different with dst?
    auto expected_other = other.reorder_if_differ_in(pd.dst_desc());
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc());
    tensor scratchpad(pd.scratchpad_desc());
    tensor expected_dst;
    if (dst.is_empty() || dst.get_desc() != pd.dst_desc()) {
      // If dst buffer are not given by user or user given dst buffer are not
      // under expected format We need init a new one.
      expected_dst.init(pd.dst_desc());
    } else {
      // The format of given dst buffer is expected
      expected_dst = dst;
    }

    if (with_bias) {
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
      primitive.execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, expected_dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                          expected_other}});
    } else {
      primitive.execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, expected_dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad},
                         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                          expected_other}});
    }
   // reorder back to dst's buffer if needed
   if (dst.is_empty() ||
       dst.get_desc() == expected_dst.get_desc() ||
       !dst.get_desc().has_same_shape_as(expected_dst.get_desc())){
     dst =  expected_dst;
   } else {
     dst.feed_from(expected_dst);
   }
  }

};

struct inner_product_backward_data : public dnnl::inner_product_backward_data {

  using super = dnnl::inner_product_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const engine& aengine = engine::cpu_engine()) {
    auto weights_ = weights;
    if (diff_dst.get_data_type() == data_type::bf16) {
      weights_.init(weights.get_desc().to_type(data_type::bf16));
      weights_.reorder_from(weights);
    }

    // workaround: diff_src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (diff_src_dims.size() != weights.ndims()) {
      auto new_dims = diff_src_dims;
      new_dims[0] = weights.get_dim(0);
      weights_.reshape(new_dims);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc();
    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst.get_data_type(), tag::any);

    auto forward_hints = inner_product_forward::get_primitive_desc(
        diff_src_desc, weights_desc, diff_dst_desc);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {diff_src_desc, weights_desc, diff_dst_desc}, op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    tensor expected_diff_src;
    if (diff_src.is_empty() || diff_src.get_desc() != pd.diff_src_desc()){
      // If diff_src buffer are not given by user or user given diff_src buffer are not under expected format
      // We need init a new one
      expected_diff_src.init(pd.diff_src_desc());
    } else {
      // The format of given diff_src buffer is expected
      expected_diff_src = diff_src;
    }

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, expected_diff_src},
                       {DNNL_ARG_SCRATCHPAD, scratchpad}});
    // reorder back to diff_src's buffer if needed
    if (diff_src.is_empty() || 
         diff_src.get_desc() == expected_diff_src.get_desc() || 
         !diff_src.get_desc().has_same_shape_as(expected_diff_src.get_desc())){
      diff_src = expected_diff_src;
    } else {
      diff_src.feed_from(expected_diff_src);
    }
  }
};

struct inner_product_backward_weights
    : public dnnl::inner_product_backward_weights {

  using super = dnnl::inner_product_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const data_type diff_weight_type = data_type::undef,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights, diff_bias, diff_weight_type);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const data_type diff_weight_type = data_type::undef,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights, dummy_diff_bias, diff_weight_type);
  }

private:
  template<bool with_diff_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const data_type diff_weight_type,
                           const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto diff_weights_dims = src.get_dims();
    diff_weights_dims[0] = diff_dst.get_dim(1);
    data_type diff_dst_type = diff_dst.get_data_type();
    data_type diff_weight_type_in = data_type::undef== diff_weight_type ?
                                    diff_dst_type : diff_weight_type;
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_weight_type_in, tag::any);

    auto diff_bias_desc =
        tensor::desc({diff_dst.get_dim(1)}, diff_weight_type_in, tag::any);

    // for forward hint, weights_desc should have same data_type
    // with other input desc, expect for bias_desc
    auto weights_desc = diff_weights_desc;
    if (diff_weight_type_in != diff_dst_type) {
      weights_desc = weights_desc.to_type(diff_dst_type);
    }
    auto forward_hints = inner_product_forward::get_primitive_desc(
        src_desc, weights_desc, diff_dst_desc, diff_bias_desc, with_diff_bias);

    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = with_diff_bias
        ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc,
                          diff_dst_desc}, op_attr, aengine, forward_hints)
        : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc},
                          op_attr, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    tensor expected_diff_weights;
    if (diff_weights.is_empty() || diff_weights.get_desc() != pd.diff_weights_desc()){
      // If diff_weights buffer are not given by user or user given diff_weights buffer are not under expected format
      // We need init a new one
      expected_diff_weights.init(pd.diff_weights_desc());
    } else {
      // The format of given diff_weights buffer is expected
      expected_diff_weights = diff_weights;
    }

    tensor scratchpad(pd.scratchpad_desc());

    exec_args args {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DIFF_WEIGHTS ,expected_diff_weights},
                    {DNNL_ARG_SCRATCHPAD, scratchpad}};

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      args.insert({DNNL_ARG_DIFF_BIAS, diff_bias});
    }

    super(pd).execute(stream::default_stream(), args);
      // reorder back to diff_weights's buffer if needed
    if (diff_weights.is_empty() || 
         diff_weights.get_desc() == expected_diff_weights.get_desc() || 
         !diff_weights.get_desc().has_same_shape_as(expected_diff_weights.get_desc())){
      diff_weights = expected_diff_weights;
    } else {
      diff_weights.feed_from(expected_diff_weights);
    }
  }
};

}  // namespace ideep

#endif
