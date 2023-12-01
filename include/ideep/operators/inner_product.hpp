#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

namespace ideep {

// Note:
// Inner product does not support quantization.
// Please switch to matmul for quantized *mm ops

struct inner_product_forward_params {
  dnnl::inner_product_forward::primitive_desc pd;
  dnnl::inner_product_forward primitive;
  attr_t op_attr;
  attr_t src_attr;
  attr_t weights_attr;
  attr_t bias_attr;

  inner_product_forward_params() {}

  inner_product_forward_params(
      dnnl::inner_product_forward::primitive_desc&& pd,
      dnnl::inner_product_forward&& primitive,
      attr_t&& op_attr,
      attr_t&& src_attr,
      attr_t&& weights_attr,
      attr_t&& bias_attr)
      : pd(std::move(pd)),
        primitive(std::move(primitive)),
        op_attr(std::move(op_attr)),
        src_attr(std::move(src_attr)),
        weights_attr(std::move(weights_attr)),
        bias_attr(std::move(bias_attr)) {}
};

struct inner_product_forward
    : public dnnl::inner_product_forward,
#ifdef __aarch64__
      utils::computation_cache<std::pair<dnnl::inner_product_forward::primitive_desc, dnnl::inner_product_forward>> {
#else
      utils::computation_cache<dnnl::inner_product_forward::primitive_desc> {
#endif
  using super = dnnl::inner_product_forward;

  // 2-in-1 compute, with bias
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor &src,
                      const tensor &weights,
                      const tensor &bias,
                      tensor &dst,
                      const attr_t &attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const engine &aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
        src, weights, bias, dst, attr, aprop_kind, aengine);
  }

  // 2-in-1 compute, without bias
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const tensor &src,
                      const tensor &weights,
                      tensor &dst,
                      const attr_t &attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const engine &aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
        src, weights, dummy_bias, dst, attr, aprop_kind, aengine);
  }

  // 2-in-1 compute, with bias
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             const tensor &bias,
                             tensor &dst,
                             const attr_t &attr = attr_t(),
                             const prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    inner_product_forward_params param;
    do_prepare_<true, reorder_src, reorder_weight>(param, src, weights, bias, 
                                                   dst, attr, aprop_kind, aengine);
    do_compute_binary<true, reorder_src, reorder_weight>(
        param, src, other, weights, bias, dst);
  }

  // 2-in-1 compute, without bias
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             tensor &dst,
                             const attr_t &attr = attr_t(),
                             const prop_kind aprop_kind = prop_kind::forward,
                             const engine &aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    inner_product_forward_params param;
    do_prepare_<false, reorder_src, reorder_weight>(param, src, weights, dummy_bias,
                                                    dst, attr, aprop_kind, aengine);
    do_compute_binary<false, reorder_src, reorder_weight>(
        param, src, other, weights, dummy_bias, dst);
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

  // Compute with prepared param, with bias
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const inner_product_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst) {
    do_compute</*with_bias=*/true, reorder_src, reorder_weight>(
        param, src, weights, bias, dst);
  }

  // Compute with prepared param, without bias
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const inner_product_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, weights, dummy_bias, dst);
  }

  // Compute with prepared param, with bias
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const inner_product_forward_params &param,
                             const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             const tensor &bias,
                             tensor &dst) {
    do_compute_binary</*with_bias=*/true, reorder_src, reorder_weight>(
        param, src, other, weights, bias, dst);
  }

  // Compute with prepared param, without bias
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute_binary(const inner_product_forward_params &param,
                             const tensor &src,
                             const tensor &other,
                             const tensor &weights,
                             tensor &dst) {
    static tensor dummy_bias;
    do_compute_binary</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, other, weights, dummy_bias, dst);
  }

  // DEPRECATED
  // 2-in-1 compute. With bias.
  static void compute(
      const tensor& src,
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
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst, attr, aprop_kind, aengine);
  }

  // DEPRECATED
  // 2-in-1 compute. Without bias.
  static void compute(
      const tensor& src,
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
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst, attr, aprop_kind, aengine);
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      const dims& src_dims = dims(),
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto x_dims = weights_dims;
    // 128 is default batch size for inner product
    x_dims[0] = src_dims.empty() ? 128 : src_dims[0];
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc src_desc(x_dims, x_dtype, tag::any);
    tensor::desc dst_desc(y_dims, y_dtype, tag::any);
    tensor::desc weights_desc(weights_dims, dtype, tag::any);
    auto pd =
        primitive_desc(aengine, aprop_kind, src_desc, weights_desc, dst_desc);
    return pd.weights_desc();
  }

#ifdef __aarch64__
  static std::pair<dnnl::inner_product_forward::primitive_desc, dnnl::inner_product_forward> get_primitive_desc(
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
      dnnl::inner_product_forward::primitive_desc pd;
      if (with_bias) {
        pd = primitive_desc(
            aengine, aprop_kind, src_desc, weights_desc, bias_desc, dst_desc, attr);
      } else {
        pd = primitive_desc(
            aengine, aprop_kind, src_desc, weights_desc, dst_desc, attr);
      }
      return std::make_pair(pd, super(pd));
    });
  }
#else
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
            aengine, aprop_kind, src_desc, weights_desc, bias_desc, dst_desc, attr);
      } else {
        return primitive_desc(
            aengine, aprop_kind, src_desc, weights_desc, dst_desc, attr);
      }
    });
  };
#endif

private:
  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static void compute_impl(const tensor &src,
                           const tensor &weights,
                           const tensor &bias,
                           tensor &dst,
                           const attr_t &attr,
                           const prop_kind aprop_kind,
                           const engine &aengine) {
    inner_product_forward_params param;
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (src.ndims() != weights.ndims()) {
      auto src_ = src;
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
      do_prepare_<with_bias, reorder_src, reorder_weight>(param, src_, weights, bias, 
                                                          dst, attr, aprop_kind, aengine);
      do_compute_<with_bias, reorder_src, reorder_weight>(param, src_, weights,
                                                          bias, dst);
    } else {
      do_prepare_<with_bias, reorder_src, reorder_weight>(param, src, weights, bias, 
                                                          dst, attr, aprop_kind, aengine);
      do_compute_<with_bias, reorder_src, reorder_weight>(param, src, weights,
                                                          bias, dst);
    }
  }

  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static void do_prepare(
      inner_product_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const attr_t& attr,
      const prop_kind aprop_kind,
      const engine& aengine) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (src.ndims() != weights.ndims()) {
      auto src_ = src;
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
      do_prepare_<with_bias, reorder_src, reorder_weight>(param, src_, weights, bias, dst, attr, aprop_kind, aengine);
    } else {
      do_prepare_<with_bias, reorder_src, reorder_weight>(param, src, weights, bias, dst, attr, aprop_kind, aengine);
    }
  }

  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static void do_prepare_(
      inner_product_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const attr_t& attr,
      const prop_kind aprop_kind,
      const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    attr_t& op_attr = param.op_attr;
    attr_t& src_attr = param.src_attr;
    data_type dst_data_type;
    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};

    op_attr = attr;
    // Below is used by int8 FC in Caffe2
    if (src.has_scale()) {
      auto src_scale = src.get_scale();
      src_scale[0] = 1.f / src_scale[0];
      src_attr = {0, src_scale};
    }

    IDEEP_ENFORCE(utils::one_of(weights.get_data_type(),
                                data_type::f32, data_type::bf16, data_type::f16),
            "Incorrect data type in weights");
    // align weights data type with src
    dst_data_type = src.get_data_type() == data_type::bf16
        ? data_type::bf16
        : ((src.get_data_type() == data_type::f16) ? data_type::f16
                                                   : data_type::f32);
    if (!reorder_weight)  {
      IDEEP_ENFORCE(weights.get_data_type() == src.get_data_type(),
                  "weights' data type should be same with input's data type when reorder_weight is false");
    }
    src_desc = reorder_src ? tensor::desc(src.get_dims(), dst_data_type, format_tag::any)
                           : src.get_desc();
    weights_desc = reorder_weight ? tensor::desc(weights.get_dims(), dst_data_type, format_tag::any)
                                  : weights.get_desc();
    if (with_bias) {
      IDEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                  data_type::f32, data_type::bf16, data_type::f16),
                    "Incorrect data type in bias");
      bias_desc = reorder_weight ? bias.get_desc().to_format_any()
                                 : bias.get_desc();
    }

    if (!reorder_src)  {
      IDEEP_ENFORCE(!dst.is_empty() && dst.get_data_type() == src.get_data_type(),
                  "dst can't be a empty tensor and data type should be same with input when reorder_src is false");
    }
    dst_desc = reorder_src ? tensor::desc(dst_dims, dst_data_type, format_tag::any)
                           : dst.get_desc();

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef __aarch64__
    auto pd_pair = get_primitive_desc(
        src_desc,
        weights_desc,
        dst_desc,
        bias_desc,
        with_bias,
        op_attr,
        aprop_kind);
    param.pd = std::move(pd_pair.first);
    param.primitive = std::move(pd_pair.second);
#else
    param.pd = get_primitive_desc(
        src_desc,
        weights_desc,
        dst_desc,
        bias_desc,
        with_bias,
        op_attr,
        aprop_kind);
    param.primitive = std::move(super(param.pd));
#endif
  }

  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static void do_compute(
      const inner_product_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (src.ndims() != weights.ndims()) {
      auto src_ = src;
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
      do_compute_<with_bias, reorder_src, reorder_weight>(param, src_, weights, bias, dst);
    } else {
      do_compute_<with_bias, reorder_src, reorder_weight>(param, src, weights, bias, dst);
    }
  }

  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed. Used for binary fusion
  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static void do_compute_binary(const inner_product_forward_params &param,
                                const tensor &src,
                                const tensor &other,
                                const tensor &weights,
                                const tensor &bias,
                                tensor &dst) {
    auto &pd = param.pd;
    auto &primitive = param.primitive;
    auto &op_attr = param.op_attr;
    auto &src_attr = param.src_attr;
    auto &weights_attr = param.weights_attr;
    auto &bias_attr = param.bias_attr;

    auto &expected_src =
        reorder_src ? src.reorder_if_differ_in(pd.src_desc(), src_attr) : src;
    // make sure other has same format with dst.
    // TODO: other has different with dst?
    auto &expected_other =
        reorder_src ? other.reorder_if_differ_in(pd.dst_desc()) : other;
    auto &expected_weights =
        reorder_weight
            ? weights.reorder_if_differ_in(pd.weights_desc(), weights_attr)
            : weights;
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    if (with_bias) {
      auto& expected_bias = reorder_weight
          ? bias.reorder_if_differ_in(pd.bias_desc(), bias_attr)
          : bias;
      args.insert({DNNL_ARG_BIAS, expected_bias});
    }
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    args.insert(
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, expected_other});
    if (reorder_src) {
      tensor expected_dst;
      if (dst.is_empty() || dst.get_desc() != pd.dst_desc()) {
        // If dst buffer are not given by user or user given dst buffer are not
        // under expected format We need init a new one. "dst.get_desc() !=
        // pd.dst_desc()" conditional is setting for caffe2 caller, it might
        // given a non-empty but incorrect dst (maybe the size is incorrect)
        expected_dst.init(pd.dst_desc());
      } else {
        // The format of given dst buffer is expected
        expected_dst = dst;
      }
      args.insert({DNNL_ARG_DST, expected_dst});
      primitive.execute(stream::default_stream(), args);

      // reorder back to dst's buffer if needed
      if (dst.is_empty() ||
          // when dst is empty, expect return buffer allocate by ideep
          dst.get_desc() == expected_dst.get_desc() ||
          // dst and expected_dst is the same under this case
          !dst.get_desc().has_same_shape_as(expected_dst.get_desc())) {
        // for caffe2 caller, get an incorrect size dst from caller, can return
        // buffer allocate by ideep
        dst = expected_dst;
      } else {
        dst.feed_from(expected_dst);
      }
    } else { // reorder_src
      if (!reorder_src)  {
          IDEEP_ENFORCE(!dst.is_empty(),
                      "dst can't be a empty tensor when reorder_src is false");
      }
      args.insert({DNNL_ARG_DST, dst});
      primitive.execute(stream::default_stream(), args);
    }
  }

  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_src = true, bool reorder_weight = true>
  static void do_compute_(
      const inner_product_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst) {
    auto& pd = param.pd;
    auto& primitive = param.primitive;
    auto& op_attr = param.op_attr;
    auto& src_attr = param.src_attr;
    auto& weights_attr = param.weights_attr;
    auto& bias_attr = param.bias_attr;

    auto& expected_src = reorder_src ?
        src.reorder_if_differ_in(pd.src_desc(), src_attr) :
        src;
    auto& expected_weights = reorder_weight ?
        weights.reorder_if_differ_in(pd.weights_desc(), weights_attr) :
        weights;
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    if (with_bias) {
      auto& expected_bias = reorder_weight
          ? bias.reorder_if_differ_in(pd.bias_desc(), bias_attr)
          : bias;
      args.insert({DNNL_ARG_BIAS, expected_bias});
    }
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    if (reorder_src) {
      tensor expected_dst;
      if (dst.is_empty() || dst.get_desc() != pd.dst_desc()){
        // If dst buffer are not given by user or user given dst buffer are not
        // under expected format We need init a new one. "dst.get_desc() !=
        // pd.dst_desc()" conditional is setting for caffe2 caller, it might
        // given a non-empty but incorrect dst (maybe the size is incorrect)
        expected_dst.init(pd.dst_desc());
        if (!dst.is_empty() && op_attr.has_op_kind(kind::sum)) {
          // We need copy the content of given buffer if ip is fused with sum
          expected_dst.feed_from(dst);
        }
      } else {
        // The format of given dst buffer is expected
        expected_dst = dst;
      }
      args.insert({DNNL_ARG_DST, expected_dst});
      primitive.execute(stream::default_stream(), args);

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
    } else { // reorder_src
      if (!reorder_src)  {
          IDEEP_ENFORCE(!dst.is_empty(),
                      "dst can't be a empty tensor when reorder_src is false");
      }
      args.insert({DNNL_ARG_DST, dst});
      primitive.execute(stream::default_stream(), args);
    }
  }
};


struct inner_product_backward_data : public dnnl::inner_product_backward_data {

  using super = dnnl::inner_product_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const attr_t& attr = attr_t(),
                      const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
    auto weights_ = weights;

    // workaround: diff_src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (diff_src_dims.size() != weights.ndims()) {
      auto new_dims = diff_src_dims;
      new_dims[0] = weights.get_dim(0);
      weights_.reshape(new_dims);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = tensor::desc(weights_.get_dims(), diff_dst.get_data_type(), tag::any);
    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst.get_data_type(), tag::any);

    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints = inner_product_forward::get_primitive_desc(
        diff_src_desc, weights_desc, diff_dst_desc, tensor::desc(), false, op_attr);

#ifdef __aarch64__
    auto pd = primitive_desc(
        aengine, diff_src_desc, weights_desc, diff_dst_desc, forward_hints.first, op_attr);
#else
    auto pd = primitive_desc(
        aengine, diff_src_desc, weights_desc, diff_dst_desc, forward_hints, op_attr);
#endif

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
                      const attr_t& attr = attr_t(),
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights, diff_bias, diff_weight_type, attr);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const data_type diff_weight_type = data_type::undef,
                      const attr_t& attr = attr_t(),
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights, dummy_diff_bias, diff_weight_type, attr);
  }

private:
  template<bool with_diff_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const data_type diff_weight_type,
                           const attr_t& attr = attr_t(),
                           const engine& aengine = engine::cpu_engine()) {
    IDEEP_CHECK(!(check_isa_is_avx2_vnni_2() &&
                  utils::one_of(diff_dst.get_data_type(),
                                data_type::bf16, data_type::f16)),
                  "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
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
    auto op_attr = attr;
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_hints = inner_product_forward::get_primitive_desc(
        src_desc, weights_desc, diff_dst_desc, diff_bias_desc, with_diff_bias, op_attr);

#ifdef __aarch64__
    auto pd = with_diff_bias
        ? primitive_desc(aengine, src_desc, diff_weights_desc, diff_bias_desc,
                         diff_dst_desc, forward_hints.first, op_attr)
        : primitive_desc(aengine, src_desc, diff_weights_desc, diff_dst_desc,
                         forward_hints.first, op_attr);
#else
    auto pd = with_diff_bias
        ? primitive_desc(aengine, src_desc, diff_weights_desc, diff_bias_desc,
                         diff_dst_desc, forward_hints, op_attr)
        : primitive_desc(aengine, src_desc, diff_weights_desc, diff_dst_desc,
                         forward_hints, op_attr);
#endif

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