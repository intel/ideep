#ifndef IDEEP_OPERATORS_GROUPED_MATMUL_HPP
#define IDEEP_OPERATORS_GROUPED_MATMUL_HPP

namespace ideep {

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

struct grouped_matmul_forward_params {
  dnnl::matmul::primitive_desc pd;
  dnnl::matmul primitive;

  grouped_matmul_forward_params() {}

  explicit grouped_matmul_forward_params(dnnl::matmul::primitive_desc&& pd)
      : pd(std::move(pd)) {
    primitive = dnnl::matmul(this->pd);
  }
};

// Grouped matmul for MoE: dst[M, N] = src[M, K] x weights[G, K, N], src and dst
// split by row into G groups in oneDNN's grouped memory (values plus s32 row
// offsets). The pd is cached, keyed on shapes, dtype, weights and thread count.
struct grouped_matmul_forward : public dnnl::matmul,
#ifdef __aarch64__
                                utils::placeholder_computation_cache {
#else
                                utils::computation_cache<dnnl::matmul::primitive_desc> {
#endif
  using super = dnnl::matmul;

  // The pd for `m` rows in `groups` groups. `weights_desc` is used as is, so
  // packed weights stay valid (`any` lets oneDNN pick); `allow_empty` returns
  // false, uncached, if unsupported. No attr, user scratchpad, CPU engine only.
  static bool prepare(
      grouped_matmul_forward_params& param,
      dim m,
      dim k,
      dim n,
      dim groups,
      const tensor::desc& weights_desc,
      data_type dtype,
      const engine& aengine = engine::cpu_engine(),
      bool allow_empty = false) {
    param = grouped_matmul_forward_params();
    IDEEP_CHECK(m > 0 && groups > 0, "grouped_matmul: rows and groups have to be positive");
    const auto wdims = weights_desc.get_dims();
    IDEEP_CHECK(
        wdims.size() == 3 && wdims[0] == groups && wdims[1] == k && wdims[2] == n,
        "grouped_matmul: weights have to be {groups, K, N}, one {K, N} slice per group");
    IDEEP_CHECK(
        aengine.get() == engine::cpu_engine().get(),
        "grouped_matmul: only the CPU engine is supported");

    attr_t attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    const auto src_desc = memory::desc::grouped({m, k}, dtype, /*variable_dim_idx=*/0, groups);
    const auto dst_desc = memory::desc::grouped({m, n}, dtype, /*variable_dim_idx=*/0, groups);

#ifdef __aarch64__
    param.pd = primitive_desc(aengine, src_desc, weights_desc, dst_desc, attr, allow_empty);
    if (!param.pd) {
      return false;
    }
#else
    auto key = utils::create_key(
        m, k, n, groups, static_cast<int>(dtype), weights_desc,
        omp_get_max_threads());

    // With `allow_empty`, a miss is probed outside the cache first, since
    // caching an empty descriptor would make the failure sticky for every
    // later call sharing the key; a non-empty probe is what gets cached.
    primitive_desc probe;
    if (allow_empty && find(key) == end()) {
      probe = primitive_desc(
          aengine, src_desc, weights_desc, dst_desc, attr, /*allow_empty=*/true);
      if (!probe) {
        return false;
      }
    }
    param.pd = fetch_or_create(key, [&]() {
      return probe ? probe
                   : primitive_desc(aengine, src_desc, weights_desc, dst_desc, attr);
    });
#endif
    param.primitive = super(param.pd);
    return true;
  }

  // Run the primitive. src and dst share `offsets` because they partition the
  // same rows. The caller owns the scratchpad (prepare() sets
  // `scratchpad_mode::user`), so the primitive can be shared between threads.
  static void compute(
      const grouped_matmul_forward_params& param,
      const void* src,
      const void* offsets,
      const void* weights,
      void* dst,
      const tensor& scratchpad,
      const dnnl::stream& astream,
      const engine& aengine = engine::cpu_engine()) {
    void* offs = const_cast<void*>(offsets);
    const memory src_mem(param.pd.src_desc(), aengine, {const_cast<void*>(src), offs});
    const memory wei_mem(param.pd.weights_desc(), aengine, const_cast<void*>(weights));
    const memory dst_mem(param.pd.dst_desc(), aengine, {dst, offs});
    param.primitive.execute(
        astream,
        {{DNNL_ARG_SRC, src_mem},
         {DNNL_ARG_WEIGHTS, wei_mem},
         {DNNL_ARG_DST, dst_mem},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

} // namespace ideep

#endif
