#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>
#include "abstract_types.hpp"
#include "allocators.hpp"

namespace ideep {
struct computation;

/// @addtogroup api_tensor Tensor
//
/// Param class describes parameters internal to operators
class param: public c_wrapper<mkldnn_primitive_t> {
public:
  using dims = mkldnn::memory::dims;
  using dim_t = dims::value_type;
  using data_type = mkldnn::memory::data_type;

  /// A param descriptor.
  struct descriptor : public c_wrapper<mkldnn_primitive_desc_t> {
    friend class param;
    inline static mkldnn_primitive_kind_t convert_to_c(kind akind) {
      return static_cast<mkldnn_primitive_kind_t>(akind);
    }
    inline static mkldnn_data_type_t convert_to_c(data_type adata_type) {
      return static_cast<mkldnn_data_type_t>(adata_type);
    }
    inline static mkldnn_memory_format_t convert_to_c(format aformat) {
      return static_cast<mkldnn_memory_format_t>(aformat);
    }

    static inline void fill_param(mkldnn_memory_desc_t &md,
        const dims &adims, data_type adata_type, format aformat) {
      md.primitive_kind = convert_to_c(kind::memory);
      md.ndims = static_cast<int>(adims.size());
      std::copy(adims.begin(), adims.end(), md.dims);
      md.data_type = convert_to_c(adata_type);
      md.format = convert_to_c(aformat);
    }

    // borrowed from memory_desc_wrapper
    static inline void set_default_strides(dims &strides, const dims &adims,
        const int *perm = NULL) {
      static const int id_perm[]
        = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      if (perm == NULL)
          perm = id_perm;

      auto ndims = adims.size();
      strides[(unsigned)perm[ndims - 1]] = 1;
      for (unsigned d = 1; d < ndims; ++d) {
          const int prev_idx = perm[ndims - d];
          const int curr_idx = perm[ndims - 1 - d];

          strides[(unsigned)curr_idx] = adims[(unsigned)curr_idx] == 0
              ? 1
              : strides[(unsigned)prev_idx]
              * std::max(1, adims[(unsigned)prev_idx]);
      }
    }

    static inline void fill_blocking(mkldnn_memory_desc_t &md, const dims adims,
        const dims &block_dims, const dims &stride, const dims &stride_inner) {
      mkldnn_blocking_desc_t &blk = md.layout_desc.blocking;
      std::copy(block_dims.begin(), block_dims.end(), blk.block_dims);
      std::copy(stride.begin(), stride.end(), &blk.strides[0][0]);
      std::copy(stride_inner.begin(), stride_inner.end(), &blk.strides[1][0]);
      std::copy(adims.begin(), adims.end(), blk.padding_dims);
      auto ndims = adims.size();
      std::fill(blk.offset_padding_to_data, blk.offset_padding_to_data + ndims, 0);
      blk.offset_padding = 0;
    }

  public:
    /// Initiate a param descriptor, specifying all details.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    /// @param extra block information for data.
    /// @param perm  permutation for layout sequence
    descriptor(const dims adims, data_type adata_type, const dims stride,
        const dims block_dims, const dims stride_inner = dims(12, 1))
      : c_wrapper([&adims, adata_type, &block_dims,
          &stride, &stride_inner] {
      mkldnn_memory_desc_t data;
      fill_param(data, adims, adata_type, format::blocked);
      fill_blocking(data, adims, block_dims, stride, stride_inner);

      mkldnn_primitive_desc_t result;
      mkldnn::error::wrap_c_api(
          mkldnn_memory_primitive_desc_create(&result, &data
            , engine::cpu_engine().get()),
          "could not initialize a memory descriptor");
      return result;
    }()), public_format_(format::blocked) {}

    /// Initiate a param descriptor, specifying format.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    /// @param aformat Data layout format.
    descriptor(dims adims, data_type adata_type, format aformat)
      :c_wrapper([&adims, adata_type, aformat]() {
        mkldnn::memory::validate_dims(adims);

        mkldnn_memory_desc_t data;
        if (adims.size() == 3) {
          fill_param(data, adims, adata_type, aformat);
          dims strides(3);
          set_default_strides(strides, adims);
          fill_blocking(data, adims, dims(3, 1), strides, dims(3, 1));
        } else {
          error::wrap_c_api(
              mkldnn_memory_desc_init(&data, (int)adims.size(),
                adims.size() == 0 ? nullptr : &adims[0],
                convert_to_c(adata_type), convert_to_c(aformat)),
              "could not initialize a memory descriptor");
        }

        mkldnn_primitive_desc_t result;
        mkldnn::error::wrap_c_api(
            mkldnn_memory_primitive_desc_create(&result, &data
              , engine::cpu_engine().get()),
            "could not initialize a memory descriptor");

        return result;
      }()), public_format_(public_format(aformat)) {}

    /// Initiate a param descriptor, specifying no format.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    descriptor(dims adims, data_type adata_type)
      : descriptor(adims, adata_type,
          engine::default_format((int)adims.size())) {

      // TODO: Do we need this checking?
      if (adims.size() == 4 || adims.size() == 2)
        public_format_ = format::format_undef;
    }

    /// Initiate a tensor descriptor from primitive_desc_t struct
    ///
    /// @param adesc Pointer to a primitive_desct_t C struct
    /// @param aformat Specify a format for current descriptor
    descriptor(mkldnn_primitive_desc_t adesc, format aformat)
      :c_wrapper(adesc), public_format_(aformat) {
    }

    /// Initiate a tensor descriptor from primitive_desc_t struct
    ///
    /// @param adesc Pointer to a primitive_desct_t C struct
    descriptor(mkldnn_primitive_desc_t adesc) : descriptor(adesc,
      public_format(
          static_cast<format>(
            mkldnn_primitive_desc_query_memory_d(adesc)->format))) {
    }

    /// Initiate a tensor descriptor from another one, share resource
    ///
    /// @param adesc is a reference to another descriptor
    descriptor(const descriptor &adesc): c_wrapper(adesc),
      public_format_ (adesc.public_format_) {
    }

    /// Empty initiate a tensor decriptor
    ///
    descriptor():descriptor(dims(0), data_type::f32, format::format_undef) {
    }

    /// Copy a tensor descriptor from another, share resource
    descriptor &operator=(const descriptor& adesc) {
      c_wrapper::operator=(adesc);
      public_format_ = adesc.public_format_;
      return *this;
    }

    /// Returns the number of bytes required to allocate the memory
    /// described including the padding area.
    inline size_t get_size() const {
      return mkldnn_memory_primitive_desc_get_size(get());
    }

    inline int ndims() const {
      return get_mkldnn_memory_desc_t()->ndims;
    }

    inline dims get_dims() const {
      auto *internal = get_mkldnn_memory_desc_t();
      return dims(internal->dims, &internal->dims[internal->ndims]);
    }

    inline data_type get_data_type() const {
      auto *internal = get_mkldnn_memory_desc_t();
      return static_cast<data_type>(internal->data_type);
    }

    template<typename T>
    inline static param::data_type type_to_id() {
      return data_type::data_undef;
    }

    /// Returns C API mkldnn_memory_desc_t structure which had same
    /// dimension and data type but without format constrain.
    mkldnn_memory_desc_t format_any() const {
      mkldnn_memory_desc_t any;
      const mkldnn_memory_desc_t *origin = get_mkldnn_memory_desc_t();

      error::wrap_c_api(
          mkldnn_memory_desc_init(&any, origin->ndims,
            origin->dims, origin->data_type,
            convert_to_c(format::any)),
          "could not initialize a memory descriptor");

      return any;
    }

    /// Returns a new descriptor which had same dimension and data type
    /// but different public format.
    /// Format protocol:
    /// pre-condition. 4-dimension only
    /// 1. (format_undef, nchw) for all unknown format creation
    /// 2. (format_undef, <internel>) compatible with all public correspondent
    descriptor format_to(format expected) const {
      mkldnn_memory_desc_t adesc;
      const mkldnn_memory_desc_t *origin = get_mkldnn_memory_desc_t();
      auto aformat = static_cast<format>(origin->format);

      if (public_format_ == format::format_undef) {
        if (public_format(aformat) != format::format_undef) {
          aformat = expected;
        }
      } else {
        if (public_format_ != expected)
          throw error(mkldnn_runtime_error, "format_to errors");
      }

      error::wrap_c_api(
          mkldnn_memory_desc_init(&adesc, origin->ndims,
            origin->dims, origin->data_type,
            convert_to_c(aformat)),
          "could not initialize a memory descriptor");

      mkldnn_primitive_desc_t result;
      mkldnn::error::wrap_c_api(
          mkldnn_memory_primitive_desc_create(&result, &adesc
            , engine::cpu_engine().get()),
          "could not initialize a memory descriptor");

      return descriptor(result, expected);
    }

    descriptor as_weights_format(format expected) const {
      switch(expected) {
      case format::nc:
      case format::io:
        return format_to(format::oi);
      case format::nchw:
      case format::oihw:
        return format_to(format::oihw);
      case format::nhwc:
      case format::ihwo:
        return format_to(format::ihwo);
      case format::chwn:
        return format_to(format::hwio);
      default:
        return format_to(format::format_undef);
      }
    }

    descriptor as_data_format(format expected) const {
      return format_to(expected);
    }

    bool is_shape_compatible(dims next) const {
      auto origin = get_mkldnn_memory_desc_t();

      auto volume_old = std::accumulate(origin->dims,
          &origin->dims[origin->ndims], 1, std::multiplies<int>());
      auto volume_new = std::accumulate(next.begin(), next.end(), 1
          , std::multiplies<dims::value_type>());

      return volume_old == volume_new;
    }

    descriptor reshape(dims adims) {
      if (!is_shape_compatible(adims)) {
        throw error(mkldnn_runtime_error, "reshape to incompatible shape");
      }
      const mkldnn_memory_desc_t *origin = get_mkldnn_memory_desc_t();
      return descriptor(adims, static_cast<data_type>(origin->data_type));
    }

    /// Returns C API mkldnn_memory_desc_t structure
    const mkldnn_memory_desc_t *get_mkldnn_memory_desc_t() const {
      return mkldnn_primitive_desc_query_memory_d(get());
    }

    inline bool operator ==(const descriptor &other) const {
      // TODO: (format_undef, *) == (nhwc, *) like
      return mkldnn_memory_primitive_desc_equal(get(), other.get());
    }

    inline bool operator !=(const descriptor &other) const {
      return !operator==(other);
    }

    format get_internal_format() const {
      return static_cast<format>(this->get_mkldnn_memory_desc_t()->format);
    }

    // oi, nc, oihw, nchw
    // TODO: other public compatible format, eg. iohw, nhwc.
    static inline format public_compatible_format(const descriptor &desc)
    {
      format ret;
      switch(desc.get_mkldnn_memory_desc_t()->format) {
      case mkldnn_x:
        ret = format::x;
        break;
      case mkldnn_oi:
      case mkldnn_io:
        ret = format::oi;
        break;
      case mkldnn_nc:
        ret = format::nc;
        break;
      case mkldnn_nchw:
      case mkldnn_nhwc:
      case mkldnn_chwn:
      case mkldnn_nChw8c:
      case mkldnn_nChw16c:
        ret = format::nchw;
        break;
      case mkldnn_ncdhw:
      case mkldnn_ndhwc:
      case mkldnn_nCdhw16c:
        ret = format::ncdhw;
        break;
      case mkldnn_oihw:
      case mkldnn_ihwo:
      case mkldnn_hwio:
      case mkldnn_OIhw8i8o:
      case mkldnn_OIhw16i16o:
      case mkldnn_OIhw8o8i:
      case mkldnn_OIhw16o16i:
      case mkldnn_OIhw8i16o2i:
      case mkldnn_OIhw8o16i2o:
      case mkldnn_Oihw8o:
      case mkldnn_Oihw16o:
      case mkldnn_Ohwi8o:
      case mkldnn_Ohwi16o:
      case mkldnn_OhIw16o4i:
        ret = format::oihw;
        break;
      default:
        ret = format::format_undef;
        break;
      }
      return ret;
    }

  private:
    // format that perceived by user
    format public_format_;

    /// Helper function: if aformat is public format, then returns it, else
    /// returns format_undef.
    static inline format public_format(format aformat) {
      switch(aformat) {
        // Public format
        case format::x:
        case format::nc:
        case format::io:
        case format::oi:
        case format::nchw:
        case format::nhwc:
        case format::chwn:
        case format::oihw:
        case format::ihwo:
        case format::hwio:
        case format::goihw:
          return aformat;
        default:
          return format::format_undef;
      }
    }

    inline bool format_compatible_with(format aformat) {
      if ( public_format_ == format::format_undef
          && public_format_ == aformat ) {
          return true;
      } else {
        switch(public_format_) {
        case format::nc:
          if (aformat == oi) return true;
          break;
        case format::nchw:
          if (aformat == oihw) return true;
          break;
        case format::nhwc:
          if (aformat == ihwo) return true;
          break;
        case format::chwn:
          if (aformat == hwio) return true;
          break;
        default:
          break;
        }
      }

      return false;
    }
  };

  struct view : public c_wrapper<mkldnn_primitive_desc_t> {
    view (const descriptor& host, dims volume, dims start) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_view_primitive_desc_create(&result,
            host.get(), &volume[0], &start[0]),
          "could not create a view primitive descriptor");

      auto desc_closer = [] (mkldnn_primitive_desc_t res) {
        mkldnn_primitive_desc_destroy(res);
      };

      std::unique_ptr<
        std::remove_pointer<mkldnn_primitive_desc_t>::type,
        decltype(desc_closer)> guard(result, desc_closer);

      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(result,
              mkldnn::convert_to_c(query::dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
          "could not clone a src primititve descriptor");

      reset(cdesc);
    }

    descriptor expected_dst_descriptor() const {
      auto internal = mkldnn_primitive_desc_query_memory_d(get());
      dims adims (internal->dims, &internal->dims[internal->ndims]);
      data_type adata_type = static_cast<data_type>(internal->data_type);
      // Care about 3D senario
      format inner_format = static_cast<format>(internal->format);
      return descriptor(adims, adata_type, inner_format);
    }
  };

  struct reorder: public c_wrapper<mkldnn_primitive_t> {
    struct descriptor : public c_wrapper<mkldnn_primitive_desc_t> {
      descriptor(const param::descriptor &input
          , const param::descriptor &output) {
        mkldnn_primitive_desc_t result;
        error::wrap_c_api(mkldnn_reorder_primitive_desc_create(
              &result, input.get(), output.get()),
            "could not create a reorder primitive descriptor");
        reset(result);
      }
    };

    reorder() {}

    void execute(const param& input, const param& output) {
      auto input_d = input.get_descriptor();
      auto output_d = output.get_descriptor();

      auto reorder_d = descriptor(input_d, output_d);

      mkldnn_primitive_t result;
      mkldnn_primitive_at_t inputs[] = { {input.get(), 0} };
      const_mkldnn_primitive_t outputs[] = { output.get() };
      error::wrap_c_api(mkldnn_primitive_create(&result,
            reorder_d.get(), inputs, outputs),
          "could not create a reorder primitive");
      reset(result);

      std::vector<mkldnn_primitive_t> execution_sequence = {result};
      mkldnn_primitive_t c_api_error_primitive;

      error::wrap_c_api(
          mkldnn_stream_submit(stream::default_stream().get()
            , execution_sequence.size(), &execution_sequence[0]
            , &c_api_error_primitive)
          , "could not execute reorder", &c_api_error_primitive);
    }
  };

  template<class alloc = utils::allocator, class computation_t = computation>
  void init(const descriptor &adesc) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
      mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr)
      , "could not create a memory primitive");

    reset(result);
    // TODO: lazy buffer allocation
    buffer_.reset(alloc::template malloc<computation_t>(
        adesc.get_size()), alloc::template free<computation_t>);
    set_data_handle(buffer_.get());
    public_format_ = adesc.public_format_;
  }

  void init(const descriptor &adesc, void *ahandle) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
      mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
      "could not create a memory primitive");

    reset(result);
    set_data_handle(ahandle);
    buffer_.reset();
    public_format_ = adesc.public_format_;
  }

  void init(const descriptor &adesc) {
    init<utils::allocator, computation>(adesc);
  }

  /// Function that refill tensor with new description or buffer
  //
  template<class alloc = utils::allocator, class computation_t = computation>
  void reinit(const descriptor &adesc) {
    auto curr_size = get_size();
    auto new_size = adesc.get_size();

    if (curr_size >= new_size ||
        (buffer_ == nullptr && get_data_handle() != nullptr)) {
      // We don't have to allocate new buffer or we don't manage the buffer
      // either way, we don't allocate new buffer
      // People who manage buffer provide enough space
      set_descriptor(adesc);
    } else {
      // re-allocate new room
      init<alloc, computation_t>(adesc);
    }
  }

  /// Function that refill tensor with new description or buffer
  void reinit(const descriptor &adesc) {
    reinit<utils::allocator, computation>(adesc);
  }

  template<class alloc = utils::allocator, class computation_t = computation>
  void reinit_like(const param &aparam) {
    reinit<alloc, computation_t>(aparam.get_descriptor());
  }

  void reinit_like(const param &aparam) {
    reinit<utils::allocator, computation>(aparam.get_descriptor());
  }

  /// Empty construction
  ///
  param() {
    init(descriptor(), nullptr);
  }

  /// Constructs a param and allocating internal buffer.
  ///
  /// @param adesc param descriptor.
  param(const descriptor &adesc) {
    init(adesc);
  }

  param(const descriptor &adesc, void *ahandle) {
    init(adesc, ahandle);
  }

  /// Recreate a param with completely different content from old one
  /// but reuse the param shell. Notice that after resize, its format
  /// is undefined
  void resize(dims adims, data_type adata_type) {
    descriptor adesc(adims, adata_type);
    init(adesc);
  }

  param &reshape(dims new_dims) {
    if (!get_descriptor().is_shape_compatible(new_dims)) {
      throw error(mkldnn_runtime_error, "reshape to incompatible shape");
    } else if (new_dims != get_dims()) {
      // XXX: format is an issue here, only default format considered
      // Lock buffer by auto _buff temporarily
      std::shared_ptr<char> _buff = buffer_;
      descriptor new_desc(new_dims, get_data_type());
      void *handle = get_data_handle();
      init(new_desc, handle);
      buffer_ = _buff;
    }

    return *this;
  }

  param &_reshape(dims new_dims) {
    return reshape(new_dims);
  }

  /// Returns the internal structure of primitive descriptor.
  const_mkldnn_primitive_desc_t get_mkldnn_primitive_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(),
                &cdesc),
            "could not get primitive descriptor from a memory primitive");
    return cdesc;
  }

  const mkldnn_memory_desc_t *get_mkldnn_memory_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(),
          &cdesc),
        "could not get primitive descriptor from a param");

    return mkldnn_primitive_desc_query_memory_d(cdesc);
  }

  descriptor get_descriptor() const {
    mkldnn_primitive_desc_t clone;
    error::wrap_c_api(mkldnn_primitive_desc_clone(&clone,
          get_mkldnn_primitive_desc_t()),
        "could not clone a primitive descriptor");

    return descriptor(clone, public_format_);
  }

  // Force a descriptor into param
  void set_descriptor(const descriptor& new_desc) {
    // Keep the original management
    auto buf = std::move(buffer_);
    init(new_desc, get_data_handle());
    buffer_ = std::move(buf);
  }

  view create_view(dims view_dims, dims offsets) const {
    return view(get_descriptor(), view_dims, offsets);
  }

  inline data_type get_data_type() const {
    const mkldnn_memory_desc_t *adesc = get_mkldnn_memory_desc_t();
    return static_cast<data_type>(adesc->data_type);
  }

  inline dim_t get_dim(int index) const {
    if (index < 0 || index >= ndims())
      return static_cast<dim_t>(0);
    const mkldnn_memory_desc_t *mdesc = get_mkldnn_memory_desc_t();
    return mdesc->dims[index];
  }

  inline dims get_dims() const {
    const mkldnn_memory_desc_t *mdesc = get_mkldnn_memory_desc_t();
    return dims (mdesc->dims, &mdesc->dims[mdesc->ndims]);
  }

  inline int ndims() const {
    return get_mkldnn_memory_desc_t()->ndims;
  }

  inline bool is_empty() const {
    return ndims() == 0 && get_data_handle() == 0;
  }

  inline size_t get_size() const {
    return mkldnn_memory_primitive_desc_get_size(get_mkldnn_primitive_desc_t());
  }

  inline dim_t get_nelems() const {
    const mkldnn_memory_desc_t *mdesc = get_mkldnn_memory_desc_t();
    return std::accumulate(
        mdesc->dims, &mdesc->dims[mdesc->ndims], 1, std::multiplies<dim_t>());
  }

  /// Returns a handle of the data contained in the param. On
  /// the CPU engine, this is a pointer to the allocated memory.
  inline void *get_data_handle() const {
      void *handle;
      error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
              "could not get native handle");
      return handle;
  }

  inline void set_data_handle(void *handle) {
      error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
              "could not set native handle");
  }

  /// Materialize a param. For specific scenario param will allocate
  /// internal buffer and manage it. As if it created with buffers.
  /// Materialize a materialied param cause no effect at all.
  void materialize() {
    if (!materialized()) {
      auto adesc = get_descriptor();

      buffer_.reset(utils::allocator::template malloc<computation>(
          adesc.get_size()), utils::allocator::template free<computation>);
      // set_data_handle will generate exception if malloc fail
      set_data_handle(buffer_.get());
    }
  }

  /// Materialize API used internal only, we should deal with it
  inline bool materialized() const {
    return (get_data_handle() != nullptr);
  }

  void dematerialize() {
    if (get_data_handle() != nullptr) {
      buffer_.reset();
      set_data_handle(nullptr);
    }
  }

  // Must go away or be private:
  static mkldnn_data_type_t convert_to_c(data_type adata_type) {
      return static_cast<mkldnn_data_type_t>(adata_type);
  }
  static mkldnn_memory_format_t convert_to_c(format aformat) {
      return static_cast<mkldnn_memory_format_t>(aformat);
  }

  inline format get_internal_format() const {
    return static_cast<format>(get_mkldnn_memory_desc_t()->format);
  }

  inline bool need_reorder() const {
    return get_internal_format() != public_format_;
  }

  // TODO: param rvalue reference
  void reorder_from(const param &src) {
    reorder().execute (src, *this);
  }

  void reorder_from(const dims adims, data_type adata_type, const void *array) {
    if (public_format_ == format::format_undef)
      reorder_from({{adims, adata_type,
          engine::default_format(ndims())},const_cast<void *>(array)});
    else
      reorder_from({{adims, adata_type, public_format_},
          const_cast<void *>(array)});
  }

  // TODO: param rvalue reference
  void reorder_to(const param &dst) const {
    reorder().execute (*this, dst);
  }

  void reorder_to(void *array) const {
    if (public_format_ == format::format_undef)
      reorder_to({
          {get_dims(), get_data_type(), engine::default_format(ndims())},
          array});
    else
      reorder_to({{get_dims(), get_data_type(), public_format_}, array});
  }

  inline int canonical_axis_index(int axis_index) const {
    assert(axis_index >= -ndims());
    assert(axis_index < ndims());
    if (axis_index < 0) {
      return axis_index + ndims();
    }
    return axis_index;
  }

  bool is_shape_compatible(dims next) const {
    const mkldnn_memory_desc_t *adesc
      = mkldnn_primitive_desc_query_memory_d(get_descriptor().get());

    auto origin = adesc->dims;

    auto volume_old = std::accumulate(origin, &origin[adesc->ndims], 1
        , std::multiplies<int>());
    auto volume_new = std::accumulate(next.begin(), next.end(), 1
        , std::multiplies<dims::value_type>());

    // More check than just volume
    return volume_old == volume_new;
  }

  inline bool is_public_format() const {
    auto desc = get_descriptor();
    return desc.get_mkldnn_memory_desc_t()->format ==
           convert_to_c(descriptor::public_compatible_format(desc));
  }

  inline std::shared_ptr<char> get_tensor_buffer() const { return buffer_; }
private:
  // mirror descriptor's same information
  format public_format_;
  std::shared_ptr<char> buffer_;
};

/// Tensor that describes the data and its explanation.
class tensor : public param {
public:
  using param::param;

  template<class alloc = utils::allocator, class computation_t = computation>
  void init_extra(const descriptor &workspace) {
    auto twin = new tensor();
    twin->init<alloc, computation_t>(workspace);
    twin_.reset(twin);
  }

  void init_extra(const descriptor &workspace, void *handle) {
    twin_.reset(new tensor(workspace, handle));
  }

  // for gcc4.8
  tensor() : param() {}

  tensor(const descriptor &major, const descriptor &workspace)
    : tensor(major) {
    init_extra(workspace);
  }

  tensor(const descriptor &major, void *h_major, const descriptor &workspace)
    : tensor(major, h_major) {
    init_extra(workspace);
  }

  tensor(const descriptor &major, void *h_major,
      const descriptor &workspace, void *h_workspace)
    : tensor(major, h_major) {
    init_extra(workspace, h_workspace);
  }

  tensor (const tensor& t) : param(t) {
    twin_ = t.twin_;
  }

  tensor (tensor&& movable) : param(std::move(movable)) {
    twin_ = std::move(movable.twin_);
  }

  tensor &operator = (const tensor& t) {
    param::operator = (t);
    twin_ = t.twin_;
    return *this;
  }

  tensor &operator = (tensor&& movable) {
    param::operator = (std::move(movable));
    twin_ = std::move(movable.twin_);
    return *this;
  }

  tensor *get_extra() {
    return twin_.get();
  }

  const tensor *get_extra() const {
    return twin_.get();
  }

  bool has_extra() const {
    return twin_ != nullptr;
  }
protected:
  std::shared_ptr<tensor> twin_;
};

static inline tensor make_output(void *buf = nullptr) {
  tensor ret;
  ret.set_data_handle(buf);
  return ret;
}

/*
static inline tensor alloc_output(tensor::dims adims) {
  tensor ret;
  return ret;
}*/

}
#endif
