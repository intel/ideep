#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include <algorithm>
#include <numeric>
#include <functional>
#include "abstract_types.hpp"
#include "allocators.hpp"
#include "web.hpp"

namespace ideep {
struct computation;

/// @addtogroup api_tensor Tensor
///
/// Param class handles operands to computations' internal, it wrappers MKL-DNN
/// memory primitive and provides utilities to manipulate underlying object.
/// It's also the base class of tensor, handles major tensor services.
class param: public c_wrapper<mkldnn_primitive_t> {
public:
  using super = c_wrapper<mkldnn_primitive_t>;
  using dims = mkldnn::memory::dims;
  using dim_t = dims::value_type;
  using data_type = mkldnn::memory::data_type;

  /// Param descriptor class wrappers MKL-DNN memory primitive descriptor
  /// and provides utilities to manipulate underlying object
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
    /// Initiate a param descriptor, specifying blocking details.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    /// @param extra block information for data.
    /// @param perm permutation for layout sequence
    descriptor(const dims adims, data_type adata_type, const dims stride,
        const dims block_dims, const dims stride_inner = dims(12, 1))
      : c_wrapper([&adims, adata_type, &block_dims,
          &stride, &stride_inner] {
      mkldnn_memory_desc_t data;
      fill_param(data, adims, adata_type, format::blocked);
      fill_blocking(data, adims, block_dims, stride, stride_inner);

      mkldnn_primitive_desc_t result;
      mkldnn::error::wrap_c_api(
          mkldnn_memory_primitive_desc_create(&result, &data,
           engine::cpu_engine().get()),
          "could not initialize a memory descriptor");
      return result;
    }()), public_format_(format::blocked) {}

    /// Initiate a param descriptor, using format for blocking initialization.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    /// @param aformat Data layout format.
    descriptor(dims adims, data_type adata_type, format aformat)
      :c_wrapper([&adims, adata_type, aformat]() {
        mkldnn::memory::validate_dims(adims);

        // XXX: out of range enum might result unspecified behavior
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
            mkldnn_memory_primitive_desc_create(&result, &data,
             engine::cpu_engine().get()),
            "could not initialize a memory descriptor");

        return result;
      }()), public_format_(public_format(aformat)) {}

    /// Initiate a param descriptor, assume nature format.
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

    /// Initiate a descriptor from primitive_desc_t struct
    ///
    /// @param adesc Pointer to a primitive_desct_t C struct
    /// @param aformat Specify public format for current descriptor
    descriptor(mkldnn_primitive_desc_t adesc, format aformat)
      :c_wrapper(adesc), public_format_(aformat) {
    }

    /// Initiate a descriptor from primitive_desc_t struct
    ///
    /// @param adesc Pointer to a primitive_desct_t C struct
    descriptor(mkldnn_primitive_desc_t adesc) : descriptor(adesc,
      public_format(
          convert_to_public_format(
            mkldnn_primitive_desc_query_memory_d(adesc)->format))) {
    }

    /// Initiate a descriptor from another, share resource
    ///
    /// @param adesc is a reference to another descriptor
    descriptor(const descriptor &adesc): c_wrapper(adesc),
      public_format_ (adesc.public_format_) {
    }

    /// Empty decriptor constructor
    ///
    descriptor():descriptor(dims(0), data_type::f32, format::format_undef) {
    }

    /// Share a descriptor from another, share resource
    descriptor &operator=(const descriptor& adesc) {
      c_wrapper::operator=(adesc);
      public_format_ = adesc.public_format_;
      return *this;
    }

    /// Returns the number of bytes required to allocate the memory
    /// described including the padding area.
    ///
    inline size_t get_size() const {
      return mkldnn_memory_primitive_desc_get_size(get());
    }

    /// Returns number of dimensions
    ///
    inline int ndims() const {
      return get_mkldnn_memory_desc_t()->ndims;
    }

  /// Return size of specified dimension
  /// @param index Dimension index
    inline dim_t get_dim(int index) const {
      if (index < 0 || index >= ndims())
        return static_cast<dim_t>(0);
      auto *internal = get_mkldnn_memory_desc_t();
      return internal->dims[index];
    }

    /// Returns dimension vector
    ///
    inline dims get_dims() const {
      auto *internal = get_mkldnn_memory_desc_t();
      return dims(internal->dims, &internal->dims[internal->ndims]);
    }

    /// Returns descriptor data type
    ///
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
    /// @param expected Expected format to transform to
    descriptor format_to(format expected) const {
      mkldnn_memory_desc_t adesc;
      const mkldnn_memory_desc_t *origin = get_mkldnn_memory_desc_t();
      auto aformat = static_cast<format>(origin->format);

      if (public_format_ == format::format_undef) {
        if (public_format(aformat) != format::format_undef) {
          aformat = expected;
        }
      } else {
        if (format_compatible_with(expected))
          aformat = expected;
        else
          throw error(mkldnn_runtime_error, "format_to errors");
      }

      error::wrap_c_api(
          mkldnn_memory_desc_init(&adesc, origin->ndims,
            origin->dims, origin->data_type,
            convert_to_c(aformat)),
          "could not initialize a memory descriptor");

      mkldnn_primitive_desc_t result;
      mkldnn::error::wrap_c_api(
          mkldnn_memory_primitive_desc_create(&result, &adesc,
           engine::cpu_engine().get()),
          "could not initialize a memory descriptor");

      return descriptor(result, expected);
    }

    /// Change format from data representation to weights, only nature formats
    /// were supported.
    /// Example: from nchw to oihw
    descriptor as_weights_format() const {
      switch(get_internal_format()) {
      case format::nc:
        return format_to(format::oi);
      case format::nchw:
        return format_to(format::oihw);
      case format::nhwc:
        return format_to(format::ihwo);
      case format::chwn:
        return format_to(format::hwio);
      default:
        return *this;
      }
    }

    descriptor as_data_format(format expected) const {
      return format_to(expected);
    }

    bool is_shape_compatible(dims next) const {
      auto origin = get_mkldnn_memory_desc_t();

      auto volume_old = std::accumulate(origin->dims,
          &origin->dims[origin->ndims], 1, std::multiplies<int>());
      auto volume_new = std::accumulate(next.begin(), next.end(), 1,
         std::multiplies<dims::value_type>());

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

    /// Operator ==
    inline bool operator ==(const descriptor &other) const {
      return mkldnn_memory_primitive_desc_equal(get(), other.get());
    }

    /// Operator !=
    inline bool operator !=(const descriptor &other) const {
      return !operator==(other);
    }

    /// Return format generated by MKL-DNN
    // XXX: format might be out of range.
    format get_internal_format() const {
      return static_cast<format>(this->get_mkldnn_memory_desc_t()->format);
    }

    static inline format convert_to_public_format(const mkldnn_memory_format_t mformat) {
      format ret;
      switch(mformat) {
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
      case mkldnn_OIhw4i16o4i:
      case mkldnn_IOhw16o16i:
        ret = format::oihw;
        break;
      case mkldnn_goihw:
      case mkldnn_hwigo:
      case mkldnn_gOIhw8i8o:
      case mkldnn_gOIhw16i16o:
      case mkldnn_gOIhw4i16o4i:
      case mkldnn_gOIhw8i16o2i:
      case mkldnn_gOIhw8o16i2o:
      case mkldnn_gOIhw8o8i:
      case mkldnn_gOIhw16o16i:
      case mkldnn_gIOhw16o16i:
      case mkldnn_gOihw8o:
      case mkldnn_gOihw16o:
      case mkldnn_gOhwi8o:
      case mkldnn_gOhwi16o:
      case mkldnn_Goihw8g:
      case mkldnn_Goihw16g:
      case mkldnn_gOhIw16o4i:
        ret = format::goihw;
        break;
      default:
        ret = format::format_undef;
        break;
      }
      return ret;
    }

    // oi, nc, oihw, nchw
    // TODO: other public compatible format, eg. iohw, nhwc.
    static inline format public_compatible_format(const descriptor &desc) {
      return convert_to_public_format(desc.get_mkldnn_memory_desc_t()->format);
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

    inline bool format_compatible_with(format aformat) const {
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

  /// View is for describing a subregion from a param
  ///
  struct view : public c_wrapper<mkldnn_primitive_desc_t> {
    /// Create view by specifying starting coordinate and size of each dimension
    /// @param host From which the view was created
    /// @param volume Size of each dimension of the subregion
    /// @param start Start coordinates
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
    class attr_t : public c_wrapper<mkldnn_primitive_attr_t> {
    public:
      attr_t() : c_wrapper([]() {
        mkldnn_primitive_attr_t result;
        error::wrap_c_api(mkldnn_primitive_attr_create(&result),
            "could not create a primitive attr");
        return result;
      }()) {}

      attr_t(int mask, scale_t &scales,
          round_mode mode = round_mode::round_nearest)
        : c_wrapper([]() {
        mkldnn_primitive_attr_t result;
        error::wrap_c_api(mkldnn_primitive_attr_create(&result),
            "could not create a primitive attr");
        return result; }()) {
        set_output_scales(mask, scales);
        set_int_output_round_mode(round_mode::round_nearest);
      }

      void set_int_output_round_mode(round_mode mode) {
        error::wrap_c_api(mkldnn_primitive_attr_set_int_output_round_mode(
              get(), mkldnn::convert_to_c(mode)),
            "could not set int output round mode");
      }

      void set_output_scales(int mask, const scale_t &scales) {
        error::wrap_c_api(mkldnn_primitive_attr_set_output_scales(get(),
              (int)scales.size(), mask, &scales[0]),
            "could not set int output scales");
      }
    };

    struct descriptor : public c_wrapper<mkldnn_primitive_desc_t> {
      descriptor(const param::descriptor &input,
          const param::descriptor &output,
          const attr_t attr = attr_t()) {
        IDEEP_ENFORCE(!(input.get_data_type() == data_type::s8
              && output.get_data_type() == data_type::u8),
            "Not support the reorder of s8 to u8 to avoid overflow.");
        mkldnn_primitive_desc_t result;
        error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
              &result, input.get(), output.get(), attr.get()),
            "could not create a reorder primitive descriptor");
        reset(result);
      }
    };

    reorder() {}

    void execute(const param& input,
        const param& output,
        const attr_t attr = attr_t()) {
      auto input_d = input.get_descriptor();
      auto output_d = output.get_descriptor();

      auto reorder_d = descriptor(input_d, output_d, attr);

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
          mkldnn_stream_submit(stream::default_stream().get(),
           execution_sequence.size(), &execution_sequence[0],
           &c_api_error_primitive),
         "could not execute reorder");
    }
  };

  /// The template initialize param with a descriptor, allocate and manage
  /// buffer automatically. A customized allocator can be specified to override
  /// default implementation.
  /// @param adesc Descriptor for the param
  template<class alloc = utils::allocator, class computation_t = computation>
  void init(const descriptor &adesc) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
      mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
     "could not create a memory primitive");

    reset(result);
    // TODO: lazy buffer allocation
    scale_.reset();
    buffer_.reset(alloc::template malloc<computation_t>(
        adesc.get_size()), alloc::template free<computation_t>);
    set_data_handle(buffer_.get());
    public_format_ = adesc.public_format_;
  }

  /// The template initialize param with a descriptor. Specifiy extra buffer.
  /// @param adesc Descriptor for the param
  /// @param ahandle Buffer of the param
  void init(const descriptor &adesc, void *ahandle) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
      mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
      "could not create a memory primitive");

    reset(result);
    scale_.reset();
    buffer_.reset();
    set_data_handle(ahandle);
    public_format_ = adesc.public_format_;
  }

  /// The template initialize param with a descriptor, allocate and manage
  /// buffer automatically. A customized allocator can be specified to override
  /// default implementation.
  /// @param adesc Descriptor for the param
  void init(const descriptor &adesc) {
    init<utils::allocator, computation>(adesc);
  }

  /// Function that refill tensor with new description or buffer
  template<class alloc = utils::allocator, class computation_t = computation>
  void reinit(const descriptor &adesc) {
    auto curr_size = get_size();
    auto new_size = adesc.get_size();

    if (curr_size >= new_size && buffer_.get() == get_data_handle()) {
      // We don't have to allocate new buffer or we don't manage the buffer
      // either way, we don't allocate new buffer
      // People who manage buffer provide enough space
      scale_.reset();
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
  /// @param adesc Descriptor for the param
  param(const descriptor &adesc) {
    init(adesc);
  }

  /// Constructs a param and allocating internal buffer.
  ///
  /// @param adesc Descriptor for the param.
  /// @param ahandle Buffer for the param.
  param(const descriptor &adesc, void *ahandle) {
    init(adesc, ahandle);
  }

  /// Constructs a param and allocating internal buffer.
  ///
  /// @param adesc Descriptor for the param.
  /// @param ahandle Buffer for the param.
  /// @param ascale Scale for the param.
  param(const descriptor &adesc, void *ahandle, const scale_t &ascale) {
    init(adesc, ahandle);
    scale_.reset(new scale_t(ascale));
  }

  /// Copy constructor
  param(const param& p) : super(p) {
    public_format_ = p.public_format_;
    buffer_ = p.buffer_;
    scale_ = p.scale_;
  }

  /// Move constructor
  param(param&& movable) : super(std::move(movable)) {
    public_format_ = movable.public_format_;
    buffer_ = std::move(movable.buffer_);
    scale_ = std::move(movable.scale_);
  }

  /// Assignment operator
  param &operator = (const param& p) {
    super::operator = (p);
    public_format_ = p.public_format_;
    buffer_ = p.buffer_;
    scale_ = p.scale_;
    return *this;
  }

  /// Move assignment operator
  param &operator = (param&& movable) {
    super::operator = (std::move(movable));
    public_format_ = movable.public_format_;
    buffer_ = std::move(movable.buffer_);
    scale_ = std::move(movable.scale_);
    return *this;
  }

  /// Operator "==" override
  ///
  /// @param right operand.
  bool operator ==(const param& p) {
    return get_descriptor() == p.get_descriptor() &&
        get_data_handle() == p.get_data_handle() ? true : false;
  }

  /// Recreate a param with completely different content from old one
  /// but reuse the param shell. Notice that after resize, its format
  /// is undefined
  /// @param adims New dimension
  /// @param adata_type New data type
  void resize(dims adims, data_type adata_type) {
    descriptor adesc(adims, adata_type);
    init(adesc);
  }

  /// Returns pointer to structure of primitive descriptor.
  const_mkldnn_primitive_desc_t get_mkldnn_primitive_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(),
                &cdesc),
            "could not get primitive descriptor from a memory primitive");
    return cdesc;
  }

  /// Return pointer to memory descriptor structure
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

  /// Set a descriptor into param to replace the older one, keep buffer
  /// It is caller's responsibility to make sure the original buffer is large
  /// enough for specified descriptor
  /// @param new_desc New descriptor
  void set_descriptor(const descriptor& new_desc) {
    // Keep the original management
    auto buf = std::move(buffer_);
    auto scale = std::move(scale_);
    init(new_desc, get_data_handle());
    buffer_ = std::move(buf);
    scale_ = std::move(scale);
    public_format_ = new_desc.public_format_;
  }

  /// Create a view from current param
  /// @param view_dims Size of each dimension of the view
  /// @param offsets Start cooridinate of the view
  view create_view(dims view_dims, dims offsets) const {
    return view(get_descriptor(), view_dims, offsets);
  }

  /// Reture param's data type
  inline data_type get_data_type() const {
    const mkldnn_memory_desc_t *adesc = get_mkldnn_memory_desc_t();
    return static_cast<data_type>(adesc->data_type);
  }

  /// Return size of specified dimension
  /// @param index Dimension index
  inline dim_t get_dim(int index) const {
    if (index < 0 || index >= ndims())
      return static_cast<dim_t>(0);
    const mkldnn_memory_desc_t *mdesc = get_mkldnn_memory_desc_t();
    return mdesc->dims[index];
  }

  /// Return dimensions' size vector
  inline dims get_dims() const {
    const mkldnn_memory_desc_t *mdesc = get_mkldnn_memory_desc_t();
    return dims (mdesc->dims, &mdesc->dims[mdesc->ndims]);
  }

  /// Return number of dimensions
  inline int ndims() const {
    return get_mkldnn_memory_desc_t()->ndims;
  }

  /// Return whether the tensor is empty
  inline bool is_empty() const {
    return ndims() == 0 && get_data_handle() == 0;
  }

  /// Return buffer size required by the param
  inline size_t get_size() const {
    return mkldnn_memory_primitive_desc_get_size(get_mkldnn_primitive_desc_t());
  }

  /// Return element number of the param
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

  /// Set new buffer handle into param
  /// @param handle Buffer handle
  inline void set_data_handle(void *handle) {
    if (buffer_.get() != handle && buffer_ != nullptr) buffer_.reset();
    error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
            "could not set native handle");
  }

  /// Return the scale of this param.
  const scale_t &get_scale() const {
    return *scale_.get();
  }

  /// Set new scale into param
  /// @param scale New scale
  void set_scale(const scale_t& ascale) {
    scale_.reset(new scale_t(ascale));
  }

  /// Return whether the param has a scale
  bool has_scale() const {
    return (scale_ != nullptr) && (!scale_->empty());
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

  /// Return internal format of the param
  inline format get_internal_format() const {
    return static_cast<format>(get_mkldnn_memory_desc_t()->format);
  }

  /// Need reorder if current param used by non MKL-DNN routines.
  inline bool need_reorder() const {
    return get_internal_format() != public_format_
      || get_data_type() != data_type::f32;
  }

  inline int canonical_axis_index(int axis_index) const {
    IDEEP_ENFORCE((axis_index >= -ndims()) && (axis_index < ndims()),
        "Invalid axis index");
    if (axis_index < 0) {
      return axis_index + ndims();
    }
    return axis_index;
  }

  bool is_shape_compatible(dims next) const {
    const mkldnn_memory_desc_t *adesc
      = mkldnn_primitive_desc_query_memory_d(get_descriptor().get());

    auto origin = adesc->dims;

    auto volume_old = std::accumulate(origin, &origin[adesc->ndims], 1,
       std::multiplies<int>());
    auto volume_new = std::accumulate(next.begin(), next.end(), 1,
       std::multiplies<dims::value_type>());

    // More check than just volume
    return volume_old == volume_new;
  }

  inline bool is_public_format() const {
    auto desc = get_descriptor();
    return desc.get_mkldnn_memory_desc_t()->format ==
           convert_to_c(descriptor::public_compatible_format(desc));
  }

  inline bool is_weights() const {
    return get_internal_format() > format::oi;
  }

  inline bool is_grouped() const {
    return public_format_ == format::goihw;
  }

  static inline void group_dims(dims& adims, const int group) {
    adims.insert(adims.begin(), group);
    adims[1] /= group;
  }

  static inline int ungroup_dims(dims& adims) {
    int group = adims[0];
    adims[1] *= group;
    adims.erase(adims.begin());
    return group;
  }

  void make_group(int group) {
    if (group > 1 && !is_grouped()) {
      IDEEP_ENFORCE(is_public_format(),
          "can not make grouped with internal format");
      auto adims = get_dims();
      group_dims(adims, group);
      set_descriptor({adims, get_data_type(), format::goihw});
    }
  }

  void make_ungroup() {
    if (is_grouped()) {
      IDEEP_ENFORCE(is_public_format(),
          "can not make ungrouped with internal format");
      auto adims = get_dims();
      ungroup_dims(adims);
      set_descriptor({adims, get_data_type(), format::oihw});
    }
  }

  inline std::shared_ptr<char> get_tensor_buffer() const { return buffer_; }
  inline void set_tensor_buffer(
      const std::shared_ptr<char>& buffer) {buffer_ = buffer;}

  // Internal use only
  // Please use feed_from, instead.
  void reorder_from(const param &src) {
    reorder().execute (src, *this);
  }

  // Internal use only
  // Please use feed_from, instead.
  void reorder_from(const dims adims, data_type adata_type, const void *array) {
    if (public_format_ == format::format_undef)
      reorder_from({{adims, adata_type,
          engine::default_format(ndims())},const_cast<void *>(array)});
    else
      reorder_from({{adims, adata_type, public_format_},
          const_cast<void *>(array)});
  }

  // Internal use only
  // Please use to_public, instead.
  void reorder_to(const param &dst) const {
    reorder().execute (*this, dst);
  }

  // Internal use only
  // Please use to_public, instead.
  void reorder_to(void *array) const {
    if (public_format_ == format::format_undef)
      reorder_to({
          {get_dims(), get_data_type(), engine::default_format(ndims())},
          array});
    else
      reorder_to({{get_dims(), get_data_type(), public_format_}, array});
  }

protected:
  // mirror descriptor's same information
  format public_format_;
  std::shared_ptr<char> buffer_;
  std::shared_ptr<scale_t> scale_;
};

/// Tensor that describes data buffer and its explanation.
/// It also integrates an optional tensor as an intemediate results, used in
/// Pooling/LRN
class tensor : public param,
  public utils::computation_web::parameter<tensor> {
public:
  using param::param;
  virtual ~tensor() { twin_.reset(); }

  /// Pack an extra tensor into current one, allocate buffer using specified
  /// allocator.
  /// @param descriptor Descriptor of the extra tensor
  template<class alloc = utils::allocator, class computation_t = computation>
  void init_extra(const descriptor &workspace) {
    auto twin = new tensor();
    twin->init<alloc, computation_t>(workspace);
    twin_.reset(twin);
  }

  /// Pack an extra tensor into current one
  ///
  /// @param descriptor Descriptor of the extra tensor
  /// @param handle Buffer handle
  void init_extra(const descriptor &workspace, void *handle) {
    twin_.reset(new tensor(workspace, handle));
  }

  /// Pack an extra tensor into current one
  ///
  /// @param tensor Extra tensor to pack in
  void init_extra(const tensor &ws) {
    twin_.reset();
    twin_ = std::make_shared<tensor>(ws);
  }

  // for gcc4.8
  /// Empty construction
  tensor()
    : param() {}

  tensor(const descriptor &major)
    : param(major) {}

  tensor(const descriptor &major, void *h_major)
    : param(major, h_major) {}

  tensor(const descriptor &major, void *h_major, const scale_t &scale)
    : param(major, h_major, scale) {}

  /// Construct tensor
  ///
  /// @param major Descriptor for the tensor
  /// @param workspace Extra descriptor of the tensor which will be packed in
  tensor(const descriptor &major, const descriptor &workspace)
    : tensor(major) {
    init_extra(workspace);
  }

  /// Construct tensor
  ///
  /// @param major Descriptor of the tensor
  /// @param h_major Buffer handle of the tensor
  /// @param workspace Descriptor of the extra tensor
  /// @param h_workspace Buffer handle of the extra tensor
  tensor(const descriptor &major, void *h_major,
      const descriptor &workspace, void *h_workspace,
      const scale_t &scale)
    : tensor(major, h_major, scale) {
    init_extra(workspace, h_workspace);
  }

  /// Construct tensor
  ///
  /// @param major Descriptor of the tensor
  /// @param h_major Buffer handle of the tensor
  /// @param workspace Descriptor of the extra tensor
  tensor(const descriptor &major, void *h_major,
      const descriptor &workspace)
    : tensor(major, h_major) {
    init_extra(workspace);
  }

  /// Construct tensor
  ///
  /// @param major Descriptor of the tensor
  /// @param h_major Buffer handle of the tensor
  /// @param workspace Descriptor of the extra tensor
  /// @param scale Scale for the tensor.
  tensor(const descriptor &major, void *h_major,
      const descriptor &workspace, const scale_t &scale)
    : tensor(major, h_major, scale) {
    init_extra(workspace);
  }

  /// Construct tensor
  ///
  /// @param major Descriptor of the tensor
  /// @param h_major Buffer handle of the tensor
  /// @param workspace Descriptor of the extra tensor
  /// @param h_workspace Buffer handle of the extra tensor
  tensor(const descriptor &major, void *h_major,
      const descriptor &workspace, void *h_workspace)
    : tensor(major, h_major) {
    init_extra(workspace, h_workspace);
  }

  /// Copy constructor
  tensor(const tensor& t) : param(t),
    utils::computation_web::parameter<tensor>(t) {
    twin_ = t.twin_;
  }

  /// Move constructor
  tensor(tensor&& movable) : param(std::move(movable)),
    utils::computation_web::parameter<tensor>(std::move(movable)) {
    twin_ = std::move(movable.twin_);
  }

  /// Assignment operator
  tensor &operator = (const tensor& t) {
    param::operator = (t);
    parameter<tensor>::operator = (t);
    twin_ = t.twin_;
    return *this;
  }

  /// Move assignment operator
  tensor &operator = (tensor&& movable) {
    param::operator = (std::move(movable));
    parameter<tensor>::operator = (std::move(movable));
    twin_ = std::move(movable.twin_);
    return *this;
  }

  /// Return extra packed tensor
  tensor *get_extra() {
    return twin_.get();
  }

  /// Return extra packed tensor
  const tensor *get_extra() const {
    return twin_.get();
  }

  /// Decide wether there is an extra tensor packed in
  bool has_extra() const {
    return twin_ != nullptr;
  }

  // XXX: ???
  tensor as_weights() const {
    tensor ret = *this;
    if (!is_weights())
      ret.set_descriptor(get_descriptor().as_weights_format());
    return ret;
  }

  /// Returns a handle of the data contained in the param. On
  /// the CPU engine, this is a pointer to the allocated memory.
  template<bool data_materialized = true>
  inline void *get_data_handle() const {
    if (data_materialized == true)
      // computation_param_materialize();
      utils::computation_web::template parameter<tensor>::
          computation_param_materialize(*this);
    void *handle;
    error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
            "could not get native handle");
    return handle;
  }

  /// Reshape a param, reorder might happen if its format is internal
  /// @param new_dims New dimension
  /// @result Return new param reference
  tensor& reshape(dims new_dims) {
    if (!get_descriptor().is_shape_compatible(new_dims)) {
      throw error(mkldnn_runtime_error, "reshape to incompatible shape");
    } else if (new_dims != get_dims()) {
      if (!is_public_format()) {
        utils::computation_web::template parameter<tensor>::
            computation_param_materialize(*this);
        tensor p;
        p.init<utils::scratch_allocator>({get_dims(), get_data_type()});
        reorder_to(p);
        set_data_handle(p.get_data_handle());
        set_tensor_buffer(p.get_tensor_buffer());
      }

      set_descriptor({new_dims, get_data_type()});
    }

    return *this;
  }

  // XXX: ???
  tensor& _reshape(dims new_dims) {
    return reshape(new_dims);
  }

  /// Fill the tensor with a src tensor
  /// @param src Source tensor
  inline void feed_from(const tensor &src) {
    auto dst_scale = has_scale() ? get_scale() : IDEEP_DEF_SCALE;
    auto src_scale = src.has_scale() ? src.get_scale() : IDEEP_DEF_SCALE;
    IDEEP_ENFORCE(dst_scale.size() == src_scale.size(),
        "Invalid tensor scales");
    IDEEP_ENFORCE(src.get_dims() == get_dims(), "Incorrect tesnor dims");

    scale_t scales(dst_scale.size());
    for (int i = 0; i < dst_scale.size(); i++) {
      scales[i] = dst_scale[i] / src_scale[i];
    }
    int mask = IDEEP_TENSOR_SCALE_MASK(src_scale.size(), src.is_grouped());
    reorder().execute(src, *this, {mask, scales});
  }

  /// Fill the tensor with parameters
  /// @param adims The dims input
  /// @param adata_type The data type input
  /// @param array The data buffer
  inline void feed_from(const dims adims,
      data_type adata_type, const void *array) {
    feed_from({{adims, adata_type,
        engine::default_format(adims.size())}, const_cast<void *>(array)});
  }

  /// Convert the tensor to public format and data type
  /// @param array The data buffer to convert to
  inline tensor to_public(void *array = nullptr) const {
    tensor ret;
    auto dst_format = (public_format_ == format::format_undef)
      ? engine::default_format(ndims()) : public_format_;
    if (array == nullptr)
      ret.init({get_dims(), data_type::f32, dst_format});
    else
      ret.init({get_dims(), data_type::f32, dst_format}, array);

    if (!has_scale()) {
      reorder().execute(*this, ret);
    } else {
      auto &src_scale = get_scale();
      scale_t scales(src_scale.size());
      for (int i = 0 ; i < src_scale.size(); i++) {
        scales[i] = 1.0f / src_scale[i];
      }
      int mask = IDEEP_TENSOR_SCALE_MASK(src_scale.size(), is_grouped());
      reorder().execute(*this, ret, {mask, scales});
    }

    return ret;
  }

  virtual bool computation_param_own_of_memory() const {
    if (get_tensor_buffer().get() == nullptr)
      return false;
    return true;
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
