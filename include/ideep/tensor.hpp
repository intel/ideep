#ifndef IDEEP_TENSOR_HPP
#define IDEEP_TENSOR_HPP

#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

#include "abstract_types.hpp"
#include "lru_cache.hpp"
#include "utils.hpp"
#include "allocators.hpp"
#include "primitives.hpp"

namespace ideep {


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
    inline static std::vector<const_mkldnn_primitive_desc_t> convert_to_c(const std::vector<descriptor>& inputs) {
      std::vector<const_mkldnn_primitive_desc_t> c_api_inputs;
      c_api_inputs.reserve(inputs.size());
      auto convert_to_c = [](const descriptor& d) { return d.get(); };
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(c_api_inputs), convert_to_c);
      return c_api_inputs;
    }

    static inline void fill_param(mkldnn_memory_desc_t& md, const dims& adims, data_type adata_type, format aformat) {
      md.primitive_kind = convert_to_c(kind::memory);
      md.ndims = static_cast<int>(adims.size());
      std::copy(adims.begin(), adims.end(), md.dims);
      md.data_type = convert_to_c(adata_type);
      md.format = convert_to_c(aformat);
    }

    static inline void set_default_strides(dims& strides, const dims& adims, const int* perm = NULL) {
      static const int id_perm[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      if (perm == NULL) perm = id_perm;

      auto ndims = adims.size();
      strides[(unsigned)perm[ndims - 1]] = 1;
      for (unsigned d = 1; d < ndims; ++d) {
          const int prev_idx = perm[ndims - d];
          const int curr_idx = perm[ndims - 1 - d];
          strides[(unsigned)curr_idx] = adims[(unsigned)curr_idx] ==
            0 ? 1 : strides[(unsigned)prev_idx] * std::max(1, adims[(unsigned)prev_idx]);
      }
    }

    static inline void fill_blocking(mkldnn_memory_desc_t& md, const dims& adims,
        const dims& block_dims, const dims& stride, const dims& stride_inner) {
      mkldnn_blocking_desc_t& blk = md.layout_desc.blocking;
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
    descriptor(const dims& adims, data_type adata_type, const dims& stride,
        const dims block_dims = dims(12, 1), const dims stride_inner = dims(12, 1))
      : c_wrapper([&adims, adata_type, &block_dims, &stride, &stride_inner] {
      mkldnn_memory_desc_t data;
      fill_param(data, adims, adata_type, format::blocked);
      fill_blocking(data, adims, block_dims, stride, stride_inner);

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_memory_primitive_desc_create(
            &result, &data, engine::cpu_engine().get()),
          "could not initialize a memory descriptor");
      return result;
    }()), public_format_(format::blocked) {}

    /// Initiate a param descriptor, using format for blocking initialization.
    descriptor(const dims& adims, data_type adata_type, format aformat)
      :c_wrapper([&adims, adata_type, aformat]() {
        mkldnn::memory::validate_dims(adims);

        // XXX: out of range enum might result unspecified behavior
        mkldnn_memory_desc_t data;
        error::wrap_c_api(mkldnn_memory_desc_init(
              &data, (int)adims.size(), adims.size() == 0 ? nullptr : &adims[0],
              convert_to_c(adata_type), convert_to_c(aformat)),
            "could not initialize a memory descriptor");

        mkldnn_primitive_desc_t result;
        error::wrap_c_api(mkldnn_memory_primitive_desc_create(
              &result, &data, engine::cpu_engine().get()),
            "could not initialize a memory descriptor");
        return result;
      }()), public_format_(public_format(aformat)) {}

    /// Initiate a param descriptor, assume nature format.
    descriptor(const dims& adims, data_type adata_type)
      : descriptor(adims, adata_type, engine::default_format((int)adims.size())) {}

    /// Initiate a descriptor from primitive_desc_t struct
    descriptor(mkldnn_primitive_desc_t adesc, format aformat)
      : c_wrapper(adesc), public_format_(aformat) {}

    /// Initiate a descriptor from primitive_desc_t struct
    descriptor(mkldnn_primitive_desc_t adesc) : descriptor(adesc,
      public_format(convert_to_public_format(mkldnn_primitive_desc_query_memory_d(adesc)->format))) {}

    /// Initiate a descriptor from primitive_desc_t struct
    descriptor(const_mkldnn_primitive_desc_t adesc, format aformat)
      : c_wrapper(const_cast<mkldnn_primitive_desc_t>(adesc), true), public_format_(aformat) {}

    /// Initiate a descriptor from primitive_desc_t struct
    descriptor(const_mkldnn_primitive_desc_t adesc)
      : descriptor(adesc, public_format(convert_to_public_format(
          mkldnn_primitive_desc_query_memory_d(adesc)->format))) {}

    /// Initiate a descriptor from another, share resource
    descriptor(const descriptor& adesc)
      : c_wrapper(adesc), public_format_ (adesc.public_format_) {}

    /// Empty decriptor constructor
    descriptor():descriptor(dims(0), data_type::data_undef, format::format_undef) {}

    /// Share a descriptor from another, share resource
    descriptor& operator=(const descriptor& adesc) {
      c_wrapper::operator=(adesc);
      public_format_ = adesc.public_format_;
      return *this;
    }

    inline void to_bytes(utils::bytestring& bytes) const {
      auto* desc = get_mkldnn_memory_desc_t();
      utils::to_bytes(bytes, desc->data_type);
      utils::to_bytes(bytes, desc->format);
      for (int i = 0; i < desc->ndims; i++) {
        utils::to_bytes(bytes, desc->dims[i]);
      }

      if (desc->format == format::blocked) {
        for (int i = 0; i < desc->ndims; i++) {
          utils::to_bytes(bytes, static_cast<uint64_t>(desc->layout_desc.blocking.strides[0][i]));
          utils::to_bytes(bytes, static_cast<uint64_t>(desc->layout_desc.blocking.strides[1][i]));
          utils::to_bytes(bytes, desc->layout_desc.blocking.block_dims[i]);
          utils::to_bytes(bytes, desc->layout_desc.blocking.padding_dims[i]);
          utils::to_bytes(bytes, desc->layout_desc.blocking.offset_padding_to_data[i]);
        }
        utils::to_bytes(bytes, static_cast<uint64_t>(desc->layout_desc.blocking.offset_padding));
      }
    }

    /// Returns the number of bytes required to allocate the memory
    /// described including the padding area.
    inline size_t get_size() const {
      return mkldnn_memory_primitive_desc_get_size(get());
    }

    /// Returns number of dimensions
    inline int ndims() const {
      return get_mkldnn_memory_desc_t()->ndims;
    }

  /// Return size of specified dimension
    inline dim_t get_dim(int index) const {
      if (index < 0 || index >= ndims()) return static_cast<dim_t>(0);
      auto* internal = get_mkldnn_memory_desc_t();
      return internal->dims[index];
    }

    /// Returns dimension vector
    inline dims get_dims() const {
      auto* internal = get_mkldnn_memory_desc_t();
      return dims(internal->dims, &internal->dims[internal->ndims]);
    }

    /// Returns descriptor data type
    inline data_type get_data_type() const {
      auto* internal = get_mkldnn_memory_desc_t();
      return static_cast<data_type>(internal->data_type);
    }

    /// Returns C API mkldnn_memory_desc_t structure which had same
    /// dimension and data type but without format constrain.
    mkldnn_memory_desc_t format_any() const {
      mkldnn_memory_desc_t any;
      const mkldnn_memory_desc_t* origin = get_mkldnn_memory_desc_t();

      error::wrap_c_api(mkldnn_memory_desc_init(
            &any, origin->ndims, origin->dims, origin->data_type, convert_to_c(format::any)),
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
      const mkldnn_memory_desc_t* origin = get_mkldnn_memory_desc_t();
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

      error::wrap_c_api(mkldnn_memory_desc_init(
            &adesc, origin->ndims, origin->dims, origin->data_type, convert_to_c(aformat)),
          "could not initialize a memory descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_memory_primitive_desc_create(
            &result, &adesc, engine::cpu_engine().get()),
          "could not initialize a memory descriptor");
      return descriptor(result, expected);
    }

    /// Change format from data representation to weights, only nature formats
    /// were supported. Example: from nchw to oihw
    descriptor as_weights_format() const {
      switch(get_internal_format()) {
      case format::nc:
        return format_to(format::oi);
      case format::ncw:
        return format_to(format::oiw);
      case format::nchw:
        return format_to(format::oihw);
      case format::nhwc:
        return format_to(format::ihwo);
      case format::chwn:
        return format_to(format::hwio);
      case format::ncdhw:
        return format_to(format::oidhw);
      case format::ndhwc:
        return format_to(format::dhwio);
      default:
        return *this;
      }
    }

    // Change format to rnn
    descriptor as_rnn_format(bool is_weight) const {
      switch(ndims()) {
      case 3:
        return {get_dims(), get_data_type(), format::tnc};
      case 4:
        return {get_dims(), get_data_type(), format::ldgo};
      case 5:
        if (is_weight) {
          return {get_dims(), get_data_type(), format::ldigo};
        } else {
          return {get_dims(), get_data_type(), format::ldsnc};
        }
      default:
        return *this;
      }
    }

    bool is_shape_compatible(const dims& next) const {
      auto origin = get_mkldnn_memory_desc_t();
      auto volume_old = std::accumulate(
          origin->dims, &origin->dims[origin->ndims], 1, std::multiplies<int>());
      auto volume_new = std::accumulate(
          next.begin(), next.end(), 1, std::multiplies<dims::value_type>());
      return volume_old == volume_new;
    }

    descriptor reshape(const dims& adims) {
      if (!is_shape_compatible(adims)) {
        throw error(mkldnn_runtime_error, "reshape to incompatible shape");
      }
      const mkldnn_memory_desc_t* origin = get_mkldnn_memory_desc_t();
      return descriptor(adims, static_cast<data_type>(origin->data_type));
    }

    /// Returns C API mkldnn_memory_desc_t structure
    const mkldnn_memory_desc_t* get_mkldnn_memory_desc_t() const {
      return mkldnn_primitive_desc_query_memory_d(get());
    }

    inline bool operator ==(const descriptor& other) const {
      return mkldnn_memory_primitive_desc_equal(get(), other.get());
    }

    inline bool operator !=(const descriptor& other) const {
      return !operator==(other);
    }

    /// Return format generated by MKL-DNN
    // XXX: format might be out of range.
    format get_internal_format() const {
      return static_cast<format>(get_mkldnn_memory_desc_t()->format);
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
      case mkldnn_ncw:
      case mkldnn_nwc:
      case mkldnn_nCw4c:
      case mkldnn_nCw8c:
      case mkldnn_nCw16c:
        ret = format::ncw;
        break;
      case mkldnn_nhwc:
        ret = format::nhwc;
        break;
      case mkldnn_nchw:
      case mkldnn_chwn:
      case mkldnn_nChw4c:
      case mkldnn_nChw8c:
      case mkldnn_nChw16c:
        ret = format::nchw;
        break;
      case mkldnn_ncdhw:
      case mkldnn_ndhwc:
      case mkldnn_nCdhw4c:
      case mkldnn_nCdhw8c:
      case mkldnn_nCdhw16c:
        ret = format::ncdhw;
        break;
      case mkldnn_oihw:
      case mkldnn_ihwo:
      case mkldnn_hwio:
      case mkldnn_ohwi:
      case mkldnn_iohw:
      case mkldnn_hwio_s8s8:
      case mkldnn_oIhw8i:
      case mkldnn_oIhw16i:
      case mkldnn_OIhw4i4o:
      case mkldnn_OIhw8i8o:
      case mkldnn_OIhw16i16o:
      case mkldnn_OIhw4i16o4i:
      case mkldnn_OIhw4i16o4i_s8s8:
      case mkldnn_OIhw8i16o2i:
      case mkldnn_IOhw8i16o2i:
      case mkldnn_OIhw8o16i2o:
      case mkldnn_IOhw8o16i2o:
      case mkldnn_OIhw8o8i:
      case mkldnn_OIhw16o16i:
      case mkldnn_IOhw16o16i:
      case mkldnn_Oihw8o:
      case mkldnn_Oihw4o:
      case mkldnn_Oihw16o:
      case mkldnn_Ohwi8o:
      case mkldnn_Ohwi4o:
      case mkldnn_Ohwi16o:
      case mkldnn_OhIw16o4i:
        ret = format::oihw;
        break;
      case mkldnn_oidhw:
      case mkldnn_dhwio:
      case mkldnn_oIdhw8i:
      case mkldnn_oIdhw16i:
      case mkldnn_OIdhw4i4o:
      case mkldnn_Odhwi4o:
      case mkldnn_OIdhw8i8o:
      case mkldnn_OIdhw8o8i:
      case mkldnn_Odhwi8o:
      case mkldnn_OIdhw16i16o:
      case mkldnn_OIdhw16o16i:
      case mkldnn_Oidhw4o:
      case mkldnn_Oidhw16o:
      case mkldnn_Odhwi16o:
      case mkldnn_OIdhw8i16o2i:
      case mkldnn_OIdhw8o16i2o:
      case mkldnn_IOdhw8o16i2o:
        ret = format::oidhw;
        break;
      case mkldnn_goihw:
      case mkldnn_hwigo:
      case mkldnn_giohw:
      case mkldnn_hwigo_s8s8:
      case mkldnn_gOIhw4i4o:
      case mkldnn_gOIhw8i8o:
      case mkldnn_gOIhw16i16o:
      case mkldnn_gOIhw4i16o4i:
      case mkldnn_gOIhw4i16o4i_s8s8:
      case mkldnn_gOIhw2i8o4i:
      case mkldnn_gOIhw2i8o4i_s8s8:
      case mkldnn_gOIhw8i16o2i:
      case mkldnn_gIOhw8i16o2i:
      case mkldnn_gOIhw8o16i2o:
      case mkldnn_gIOhw8o16i2o:
      case mkldnn_gOIhw4o4i:
      case mkldnn_gOIhw4o4i_s8s8:
      case mkldnn_gOIhw8o8i:
      case mkldnn_gOIhw16o16i:
      case mkldnn_gIOhw16o16i:
      case mkldnn_gOihw8o:
      case mkldnn_gOihw4o:
      case mkldnn_gOihw16o:
      case mkldnn_gOhwi8o:
      case mkldnn_gOhwi4o:
      case mkldnn_gOhwi16o:
      case mkldnn_Goihw8g:
      case mkldnn_Goihw16g:
      case mkldnn_Goihw16g_s8s8:
      case mkldnn_gOhIw16o4i:
        ret = format::goihw;
        break;
      case mkldnn_goidhw:
      case mkldnn_gOIdhw4i4o:
      case mkldnn_gOdhwi4o:
      case mkldnn_gOIdhw8i8o:
      case mkldnn_gOIdhw8o8i:
      case mkldnn_gOdhwi8o:
      case mkldnn_gOIdhw8i16o2i:
      case mkldnn_gOIdhw8o16i2o:
      case mkldnn_gIOdhw8o16i2o:
      case mkldnn_gOIdhw16i16o:
      case mkldnn_gOIdhw16o16i:
      case mkldnn_gOidhw4o:
      case mkldnn_gOidhw16o:
      case mkldnn_gOdhwi16o:
        ret = format::goidhw;
        break;
      case mkldnn_ntc:
      case mkldnn_tnc:
        ret = format::tnc;
        break;
      case mkldnn_ldigo:
        ret = format::ldigo;
        break;
      case mkldnn_ldgoi:
        ret = format::ldgoi;
        break;
      case mkldnn_ldgo:
        ret = format::ldgo;
        break;
      case mkldnn_ldsnc:
        ret = format::ldsnc;
        break;
      case mkldnn_rnn_packed:
        ret = format::rnn_packed;
        break;
      case mkldnn_blocked:
      case mkldnn_wino_fmt:
      case mkldnn_format_undef:
        ret = format::format_undef;
        break;
      default:
        //std::cout<<"Unsupported MKL-DNN memory format: "<<mformat<<std::endl;
        throw error(mkldnn_runtime_error, "unsupported mkldnn memory format!");
      }
      return ret;
    }

    // oi, nc, oihw, nchw
    // TODO: other public compatible format, eg. iohw, nhwc.
    static inline format public_compatible_format(const descriptor& desc) {
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
        case format::ncw:
        case format::nwc:
        case format::oiw:
        case format::wio:
        case format::tnc:
        case format::ldigo:
        case format::ldgoi:
        case format::ldgo:
        case format::ldsnc:
        case format::rnn_packed:
        case format::nchw:
        case format::nhwc:
        case format::chwn:
        case format::oihw:
        case format::ihwo:
        case format::hwio:
        case format::ncdhw:
        case format::ndhwc:
        case format::oidhw:
        case format::dhwio:
        case format::goihw:
        case format::goidhw:
          return aformat;
        default:
          return format::format_undef;
      }
    }

    inline bool format_compatible_with(format aformat) const {
      if (public_format_ == format::format_undef && public_format_ == aformat ) {
          return true;
      } else if (aformat == tnc || aformat == ldgo || aformat == ldsnc || aformat == ldigo ||
          aformat == ldgoi || aformat == rnn_packed) {
          return true;
      } else {
        switch(public_format_) {
        case format::nc:
          if (aformat == oi) return true;
          break;
        case format::ncw:
          if (aformat == oiw) return true;
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
        case format::ncdhw:
          if (aformat == oidhw) return true;
          break;
        default:
          break;
        }
      }
      return false;
    }
  };

  /// View is for describing a subregion from a param
  struct view : public c_wrapper<mkldnn_primitive_desc_t> {
    /// Create view by specifying starting coordinate and size of each dimension
    view (const descriptor& host, const dims& volume, const dims& start) {
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_view_primitive_desc_create(
            &result, host.get(), &volume[0], &start[0]),
          "could not create a view primitive descriptor");

      auto desc_closer = [] (mkldnn_primitive_desc_t res) {
        mkldnn_primitive_desc_destroy(res);
      };

      std::unique_ptr<std::remove_pointer<mkldnn_primitive_desc_t>::type, decltype(desc_closer)>
        guard(result, desc_closer);

      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc = mkldnn_primitive_desc_query_pd(
          result, mkldnn::convert_to_c(query::dst_pd), 0);
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

  /// The template initialize param with a descriptor.
  template<class alloc = utils::allocator>
  void init(const descriptor& adesc) {
    mkldnn_primitive_t result;
    error::wrap_c_api(mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
        "could not create a memory primitive");

    reset(result);
    // TODO: lazy buffer allocation
    scale_.reset();
    buffer_.reset(alloc::malloc(adesc.get_size()), alloc::free);
    set_data_handle(buffer_.get());
    public_format_ = adesc.public_format_;
  }

  /// The template initialize param with a descriptor. Specifiy extra buffer.
  void init(const descriptor& adesc, void* ahandle) {
    mkldnn_primitive_t result;
    error::wrap_c_api(mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
        "could not create a memory primitive");

    reset(result);
    scale_.reset();
    buffer_.reset();
    set_data_handle(ahandle);
    public_format_ = adesc.public_format_;
  }

  /// Function that refill tensor with new description or buffer
  template<class alloc = utils::allocator>
  void reinit(const descriptor& adesc) {
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
      init<alloc>(adesc);
    }
  }

  template<class alloc = utils::allocator>
  void reinit_like(const param& aparam) {
    reinit<alloc>(aparam.get_descriptor());
  }

  /// Empty construction
  param() {
    init(descriptor(), nullptr);
  }

  /// Constructs a param and allocating internal buffer.
  param(const descriptor& adesc, void* ahandle) {
    init(adesc, ahandle);
  }

  /// Constructs a param and allocating internal buffer.
  param(const descriptor& adesc, void* ahandle, const scale_t& ascale) {
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
  param& operator = (const param& p) {
    super::operator = (p);
    public_format_ = p.public_format_;
    buffer_ = p.buffer_;
    scale_ = p.scale_;
    return *this;
  }

  /// Move assignment operator
  param& operator = (param&& movable) {
    super::operator = (std::move(movable));
    public_format_ = movable.public_format_;
    buffer_ = std::move(movable.buffer_);
    scale_ = std::move(movable.scale_);
    return *this;
  }

  /// Operator "==" override
  bool operator ==(const param& p) {
    return get_descriptor() == p.get_descriptor() &&
        get_data_handle() == p.get_data_handle() ? true : false;
  }

  inline void to_bytes(utils::bytestring& bytes) const {
    get_descriptor().to_bytes(bytes);
  }

  /// Recreate a param with completely different content from old one
  /// but reuse the param shell. Notice that after resize, its format
  /// is undefined
  template<class alloc = utils::allocator>
  void resize(const dims& adims, data_type adata_type) {
    descriptor adesc(adims, adata_type);
    init<alloc>(adesc);
  }

  /// Returns pointer to structure of primitive descriptor.
  const_mkldnn_primitive_desc_t get_mkldnn_primitive_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &cdesc),
            "could not get primitive descriptor from a memory primitive");
    return cdesc;
  }

  /// Return pointer to memory descriptor structure
  const mkldnn_memory_desc_t* get_mkldnn_memory_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &cdesc),
        "could not get primitive descriptor from a param");
    return mkldnn_primitive_desc_query_memory_d(cdesc);
  }

  descriptor get_descriptor() const {
    return descriptor(get_mkldnn_primitive_desc_t(), public_format_);
  }

  descriptor dup_descriptor() const {
    mkldnn_primitive_desc_t clone;
    error::wrap_c_api(mkldnn_primitive_desc_clone(&clone, get_mkldnn_primitive_desc_t()),
        "could not clone a primitive descriptor");
    return descriptor(clone, public_format_);
  }


  /// Set a descriptor into param to replace the older one, keep buffer
  /// It is caller's responsibility to make sure the original buffer is large
  /// enough for specified descriptor
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
  view create_view(const dims& view_dims, const dims& offsets) const {
    return view(get_descriptor(), view_dims, offsets);
  }

  /// Reture param's data type
  inline data_type get_data_type() const {
    const mkldnn_memory_desc_t* adesc = get_mkldnn_memory_desc_t();
    return static_cast<data_type>(adesc->data_type);
  }

  /// Return size of specified dimension
  inline dim_t get_dim(int index) const {
    if (index < 0 || index >= ndims()) return static_cast<dim_t>(0);
    const mkldnn_memory_desc_t* mdesc = get_mkldnn_memory_desc_t();
    return mdesc->dims[index];
  }

  /// Return dimensions' size vector
  inline dims get_dims() const {
    const mkldnn_memory_desc_t* mdesc = get_mkldnn_memory_desc_t();
    return dims (mdesc->dims, &mdesc->dims[mdesc->ndims]);
  }

  /// Return public format dimensions' size vector
  inline dims get_public_format_dims() const {
    if (public_format_ == format::iohw) {
      return {get_dim(1), get_dim(0), get_dim(2), get_dim(3)};
    } else if (public_format_ == format::nhwc) {
      return {get_dim(0), get_dim(2), get_dim(3), get_dim(1)};
    } else {
      return get_dims();
    }
  }

  /// Return number of dimensions
  inline int ndims() const {
    return get_mkldnn_memory_desc_t()->ndims;
  }

  inline dims get_block_dims() const {
    const mkldnn_memory_desc_t* mdesc = get_mkldnn_memory_desc_t();
    const auto block_dims = mdesc->layout_desc.blocking.block_dims;
    return dims (block_dims, &block_dims[mdesc->ndims]);
  }

  inline dims get_block_stride() const {
    const mkldnn_memory_desc_t* mdesc = get_mkldnn_memory_desc_t();
    const auto block_stride = mdesc->layout_desc.blocking.strides[0];
    return dims (block_stride, &block_stride[mdesc->ndims]);
  }

  /// Return whether the tensor is empty
  inline bool is_empty() const {
    return ndims() == 0 && get_data_handle() == nullptr;
  }

  /// Return buffer size required by the param
  inline size_t get_size() const {
    return mkldnn_memory_primitive_desc_get_size(get_mkldnn_primitive_desc_t());
  }

  /// Return element number of the param.
  /// The number is the meaning values for a tensor, instead of whole buffer.
  /// It is the number without counting in paddings.
  inline dim_t get_nelems() const {
    const mkldnn_memory_desc_t* mdesc = get_mkldnn_memory_desc_t();
    return std::accumulate(mdesc->dims, &mdesc->dims[mdesc->ndims], 1, std::multiplies<dim_t>());
  }

  /// Returns a handle of the data contained in the param. On
  /// the CPU engine, this is a pointer to the allocated memory.
  inline void* get_data_handle() const {
    void* handle;
    error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle), "could not get native handle");
    return handle;
  }

  /// Set new buffer handle into param
  inline void set_data_handle(void* handle) {
    if (buffer_.get() != handle && buffer_ != nullptr) buffer_.reset();
    error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle), "could not set native handle");
  }

  /// Return the scale of this param.
  const scale_t& get_scale() const {
    return *scale_.get();
  }

  /// Set new scale into param
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
      buffer_.reset(utils::allocator::malloc(adesc.get_size()), utils::allocator::free);
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

  inline format get_public_format() const {
    return public_format_;
  }

  inline format set_public_format(format aformat) {
    return public_format_ = aformat;
  }

  /// Need reorder if current param used by non MKL-DNN routines.
  inline bool need_reorder() const {
    return (!is_public_format() || get_data_type() != data_type::f32);
  }

  inline int canonical_axis_index(int axis_index) const {
    IDEEP_ENFORCE((axis_index >= -ndims()) && (axis_index < ndims()), "Invalid axis index");
    if (axis_index < 0) {
      return axis_index + ndims();
    }
    return axis_index;
  }

  bool is_shape_compatible(const dims& next) const {
    const mkldnn_memory_desc_t* adesc = mkldnn_primitive_desc_query_memory_d(get_descriptor().get());
    auto origin = adesc->dims;
    auto volume_old = std::accumulate(origin, &origin[adesc->ndims], 1, std::multiplies<int>());
    auto volume_new = std::accumulate(next.begin(), next.end(), 1, std::multiplies<dims::value_type>());

    // More check than just volume
    return volume_old == volume_new;
  }

  inline bool is_public_format() const {
    auto desc = get_descriptor();
    return desc.get_mkldnn_memory_desc_t()->format == convert_to_c(descriptor::public_compatible_format(desc));
  }

  inline bool is_weights() const {
    auto fmt = convert_to_c(get_internal_format());
    return (fmt >= mkldnn_oi && fmt < mkldnn_ntc)
      || (fmt > mkldnn_ldsnc && fmt < mkldnn_nCw8c)
      || fmt > mkldnn_nCdhw16c;
  }

  inline bool is_grouped() const {
    return public_format_ == format::goihw || public_format_ == format::goidhw;
  }

  static inline void group_dims(dims& adims, const int group, const bool deconv = false) {
    adims.insert(adims.begin(), group);
    if (deconv) {
      adims[2] /= group;
    } else {
      adims[1] /= group;
    }
  }


  static inline int ungroup_dims(dims& adims, const bool deconv = false) {
    int group = adims[0];
    if (deconv) {
      adims[2] *= group;
    } else {
      adims[1] *= group;
    }
    adims.erase(adims.begin());
    return group;
  }

  void make_group(int group, const bool deconv = false) {
    // for deconv with g > 1, the format is go(i/g)hw given oihw and group,
    // for conv with g > 1, the format is g(o/g)ihw given oihw and group
    if (group > 1 && !is_grouped()) {
      IDEEP_ENFORCE(is_public_format(), "can not make grouped with internal format");
      auto adims = get_dims();
      group_dims(adims, group, deconv);
      auto ndims = adims.size();
      if (ndims == 5) {
        set_descriptor({adims, get_data_type(), format::goihw});
      } else {
        set_descriptor({adims, get_data_type(), format::goidhw});
      }
    }
  }

  void make_ungroup(const bool deconv = false) {
    if (is_grouped()) {
      IDEEP_ENFORCE(is_public_format(), "can not make ungrouped with internal format");
      auto adims = get_dims();
      ungroup_dims(adims, deconv);
      auto ndims = adims.size();
      if (ndims == 4) {
        set_descriptor({adims, get_data_type(), format::oihw});
      } else {
        set_descriptor({adims, get_data_type(), format::oidhw});
      }
    }
  }

  inline std::shared_ptr<char> get_tensor_buffer() const { return buffer_; }

  inline void set_tensor_buffer(const std::shared_ptr<char>& buffer) {buffer_ = buffer;}

protected:
  // mirror descriptor's same information
  format public_format_;
  std::shared_ptr<char> buffer_;
  std::shared_ptr<scale_t> scale_;
};

/// Tensor that describes data buffer and its explanation.
/// It also integrates an optional tensor as an intemediate results, used in
/// Pooling/LRN
class IDEEP_EXPORT tensor : public param {
public:
  using param::param;
  using attr_t = descriptor_group::attr_t;

  struct reorder: public primitive_group,
      public utils::computation_cache<reorder> {
    struct reorder_desc : public descriptor_group {
      reorder_desc(const c_wrapper<mkldnn_primitive_desc_t>& input,
          const c_wrapper<mkldnn_primitive_desc_t>& output, const attr_t& attr = attr_t()) {
        mkldnn_primitive_desc_t result;
        error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
              &result, input.get(), output.get(), attr.get()),
            "could not create a reorder primitive reorder descriptor");
        reset(result);
      }
    };

  public:
    reorder() = default;

    void init(const reorder_desc& desc, const descriptor& src_desc, const descriptor& dst_desc) {
      in_.init(src_desc, nullptr);
      out_.init(dst_desc, nullptr);

      mkldnn_primitive_at_t inputs[] = { {in_.get(), 0} };
      const_mkldnn_primitive_t outputs[] = { out_.get() };
      create_primitive(desc, inputs, outputs);
    }

    reorder(const descriptor& src_desc, const descriptor& dst_desc, const attr_t& attr = attr_t()) {
      reorder_desc desc(src_desc, dst_desc, attr);
      init(desc, src_desc, dst_desc);
    }

    reorder(const view& aview, const descriptor& src_desc, const descriptor& dst_desc, const attr_t& attr = attr_t()) {
      reorder_desc desc(aview, dst_desc, attr);
      init(desc, src_desc, dst_desc);
    }

    reorder(const descriptor& src_desc, const view& aview, const descriptor& dst_desc, const attr_t& attr = attr_t()) {
      reorder_desc desc(src_desc, aview, attr);
      init(desc, src_desc, dst_desc);
    }

    void operator() (const tensor& input, const tensor& output) {
      IDEEP_ENFORCE(!(input.get_data_type() == data_type::s8 && output.get_data_type() == data_type::u8),
          "Not support the reorder of s8 to u8 to avoid overflow.");
      IDEEP_ENFORCE(input.get_descriptor() == in_.get_descriptor() && output.get_descriptor() == out_.get_descriptor(),
          "Unmatch tensor reorder descriptor in reorder");

      in_.set_data_handle(input.get_data_handle());
      out_.set_data_handle(output.get_data_handle());

      stream parallel_control = stream::default_stream();
      primitive_group::execute(parallel_control);
    }

    static void compute(const dims& volume, const dims& offset, const tensor& input,
        tensor& output, const attr_t& attr = attr_t()) {
      if (input.is_empty() || output.is_empty()) { return; }

      key_t key;
      check_or_create_k(key, volume, offset, input, output, attr);
      auto view = input.create_view(volume, offset);
      fetch_or_create_m(comp, key, view, input.get_descriptor(), output.get_descriptor(), attr);
      comp(input, output);
    }

    static void compute(const tensor& input, const dims& volume, const dims& offset,
        tensor& output, const attr_t& attr = attr_t()) {
      if (input.is_empty() || output.is_empty()) { return; }

      key_t key;
      check_or_create_k(key, input, volume, offset, output, attr);
      auto view = output.create_view(volume, offset);
      fetch_or_create_m(comp, key, input.get_descriptor(), view, output.get_descriptor(), attr);
      comp(input, output);
    }

    static void compute(const tensor& input, tensor& output, const attr_t& attr = attr_t()) {
      if (input.is_empty() || output.is_empty()) { return; }

      key_t key;
      check_or_create_k(key, input, output, attr);
      fetch_or_create_m(comp, key, input.get_descriptor(), output.get_descriptor(), attr);
      comp(input, output);
    }

  protected:
    param in_, out_;
  };

  /// Pack an extra tensor into current one, allocate buffer using specified allocator.
  template<class alloc = utils::allocator>
  void init_extra(const descriptor& workspace) {
    auto twin = new tensor();
    twin->init<alloc>(workspace);
    twin_.reset(twin);
  }

  /// Pack an extra tensor into current one
  void init_extra(const descriptor& workspace, void* handle) {
    twin_.reset(new tensor(workspace, handle));
  }

  /// Pack an extra tensor into current one
  void init_extra(const tensor& ws) {
    twin_.reset();
    twin_ = std::make_shared<tensor>(ws);
  }

  tensor() : param() {}

  tensor(const descriptor& major, void* h_major) : param(major, h_major) {}

  tensor(const descriptor& major, void* h_major, const scale_t& scale)
    : param(major, h_major, scale) {}

  tensor(const descriptor& major, void* h_major, const descriptor& workspace, void* h_workspace,
      const scale_t& scale)
    : tensor(major, h_major, scale) {
    init_extra(workspace, h_workspace);
  }

  tensor(const descriptor& major, void* h_major, const descriptor& workspace, void* h_workspace)
    : tensor(major, h_major) {
    init_extra(workspace, h_workspace);
  }

  /// Copy constructor
  tensor(const tensor& t) : param(t) {
    twin_ = t.twin_;
  }

  /// Move constructor
  tensor(tensor&& movable) : param(std::move(movable)) {
    twin_ = std::move(movable.twin_);
  }

  /// Assignment operator
  tensor& operator = (const tensor& t) {
    param::operator = (t);
    twin_ = t.twin_;
    return *this;
  }

  /// Move assignment operator
  tensor& operator = (tensor&& movable) {
    param::operator = (std::move(movable));
    twin_ = std::move(movable.twin_);
    return *this;
  }

  template<class alloc = utils::allocator>
  void init(const descriptor& adesc) {
    param::init<alloc>(adesc);
    twin_.reset();
  }

  void init(const descriptor& adesc, void* ahandle) {
    param::init(adesc, ahandle);
    twin_.reset();
  }

  template<class alloc = utils::allocator>
  void reinit(const descriptor& adesc) {
    param::reinit<alloc>(adesc);
    twin_.reset();
  }

  template<class alloc = utils::allocator>
  void reinit_like(const param& aparam) {
    param::reinit<alloc>(aparam.get_descriptor());
    twin_.reset();
  }

  /// Return extra packed tensor
  tensor* get_extra() {
    return twin_.get();
  }

  /// Return extra packed tensor
  const tensor* get_extra() const {
    return twin_.get();
  }

  /// Decide wether there is an extra tensor packed in
  bool has_extra() const {
    return twin_ != nullptr;
  }

  tensor as_weights() const {
    tensor ret = *this;
    if (!is_weights())
      ret.set_descriptor(get_descriptor().as_weights_format());
    return ret;
  }

  tensor as_rnn(bool is_weight = false) const {
    tensor ret = *this;
    ret.set_descriptor(get_descriptor().as_rnn_format(is_weight));
    return ret;
  }

  /// Returns a handle of the data contained in the param. On
  /// the CPU engine, this is a pointer to the allocated memory.
  inline void* get_data_handle() const {
    void* handle;
    error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle), "could not get native handle");
    return handle;
  }

  /// Reshape a param, reorder might happen if its format is internal
  template<class alloc = utils::allocator>
  tensor& reshape(const dims& new_dims) {
    if (!get_descriptor().is_shape_compatible(new_dims)) {
      throw error(mkldnn_runtime_error, "reshape to incompatible shape");
    } else if (new_dims != get_dims()) {
      if (!is_public_format()) {
        tensor p;
        p.init<alloc>({get_dims(), get_data_type()});
        reorder::compute(*this, p);
        set_data_handle(p.get_data_handle());
        set_tensor_buffer(p.get_tensor_buffer());
      }

      set_descriptor({new_dims, get_data_type()});
    }

    return *this;
  }

  template<class alloc = utils::allocator>
  inline tensor permute(const std::vector<int>& permute_axes = {}) const {
    if (ndims() <= 1) {
      return to_public_format<alloc>();
    }

    auto axes = permute_axes;
    if (axes.empty()) {
      axes.resize(ndims());
      std::iota(axes.rbegin(), axes.rend(), 0);
    } else {
      IDEEP_ENFORCE(static_cast<int>(axes.size()) == ndims(),
          "Axes should be size like source tensor.");
      auto axes_sorted = axes;
      std::sort(axes_sorted.begin(), axes_sorted.end());
      for (auto i = 0; i < axes_sorted.size(); ++i) {
        IDEEP_ENFORCE(static_cast<float>(axes_sorted[i]) == i,
            "Axes should be a permutation of 0 to ndim.");
      }
      if (axes_sorted == axes) {
        return to_public_format<alloc>();
      }
    }

    auto src = *this;
    if (!is_public_format()) {
      src = to_public_format<alloc>();
    }

    auto src_dims = src.get_dims();
    dims dst_dims(src_dims.size());
    for (int i = 0; i < src_dims.size(); i++) {
      dst_dims[i] = src_dims[axes[i]];
    }

    tensor dst;
    dst.init<alloc>({dst_dims, src.get_data_type(), src.get_public_format()});
    auto dst_stride = dst.get_block_stride();
    dims stride (dst_stride.size(), 1);
    for (int i = stride.size() - 2; i >= 0; i--) {
      stride[axes[i]] = dst_stride[i];
    }

    tensor mask_dst;
    mask_dst.init({src.get_dims(), src.get_data_type(), stride}, dst.get_data_handle());
    reorder::compute(src, mask_dst);
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }

    return dst;
  }

  template<class alloc = utils::allocator>
  void transpose_from(const tensor& src, const std::vector<int>& axes = {}) {
    *this = src.permute<alloc>(axes);
  }

  /// Fill the tensor with a src tensor
  inline void feed_from(const tensor& src) {
    scale_t dst_scale, src_scale;
    if (has_scale() && src.has_scale()) {
      dst_scale = get_scale();
      src_scale = src.get_scale();
    } else if (has_scale()) {
      dst_scale = get_scale();
      src_scale.assign(dst_scale.size(), 1.0f);
    } else if (src.has_scale()) {
      src_scale = src.get_scale();
      dst_scale.assign(src_scale.size(), 1.0f);
    } else {
      dst_scale = IDEEP_DEF_SCALE;
      src_scale = IDEEP_DEF_SCALE;
    }

    IDEEP_ENFORCE(dst_scale.size() == src_scale.size(), "Invalid tensor scales");
    auto src_in = src;
    if (src_in.is_iohw_public_layout()) {
      iohw_definedby_blocked(src_in);
    } else {
      IDEEP_ENFORCE(src_in.get_dims() == get_dims(), "Incorrect tesnor dims");
    }

    scale_t scales(dst_scale.size());
    for (int i = 0; i < dst_scale.size(); i++) {
      scales[i] = dst_scale[i] / src_scale[i];
    }
    int mask = IDEEP_TENSOR_SCALE_MASK(src_scale.size(), src_in.is_grouped());
    reorder::compute(src_in, *this, {mask, scales});
  }

  /// Fill the tensor with parameters
  inline void feed_from(const dims& adims, data_type adata_type, const void* array) {
    feed_from({{adims, adata_type, engine::default_format(adims.size())}, const_cast<void*>(array)});
  }

  /// Convert the tensor to public format, and f32 data type by default
  template<class alloc = utils::allocator>
  inline tensor to_public(void* array = nullptr, bool scale_out = true) const {
    tensor ret;
    auto dst_dtype = scale_out ? data_type::f32 : get_data_type();
    auto dst_format = ((public_format_ == format::format_undef) || (public_format_ == format::iohw))
      ? engine::default_format(ndims()) : public_format_;

    dims iohw_dims;
    // TODO:it will be remove when deconvolution in mkl-dnn support iohw format.
    if (public_format_ == format::iohw) {
      iohw_dims = get_public_format_dims();
      if (array == nullptr)
        ret.init<alloc>({iohw_dims, dst_dtype, format::oihw});
      else
        ret.init({iohw_dims, dst_dtype, format::oihw}, array);
      iohw_definedby_blocked(ret);
    } else {
      if (array == nullptr)
        ret.init<alloc>({get_dims(), dst_dtype, dst_format});
      else
        ret.init({get_dims(), dst_dtype, dst_format}, array);
    }

    if (scale_out && has_scale()) {
      auto& src_scale = get_scale();
      scale_t scales(src_scale.size());
      for (int i = 0 ; i < src_scale.size(); i++) {
        scales[i] = 1.0f / src_scale[i];
      }
      int mask = IDEEP_TENSOR_SCALE_MASK(src_scale.size(), is_grouped());
      reorder::compute(*this, ret, {mask, scales});
    } else {
      reorder::compute(*this, ret);
      if (has_scale()) {
        ret.set_scale(get_scale());
      }
    }

    // TODO:it will be remove when deconvolution in mkl-dnn support iohw format.
    if (!iohw_dims.empty()) {
      ret.set_descriptor({iohw_dims, dst_dtype, dst_format});
    }

    return ret;
  }

  /// Convert the tensor to public format and keep original data type
  template<class alloc = utils::allocator>
  inline tensor to_public_format(void* array = nullptr) const {
    return to_public<alloc>(array, false /* scale out */);
  }

  bool is_nchw_channel_blocking() const {
    auto aformat = get_internal_format();
    return aformat == static_cast<format>(mkldnn_nchw)
      || aformat == static_cast<format>(mkldnn_nChw8c)
      || aformat == static_cast<format>(mkldnn_nChw16c);
  }

  bool is_nhwc_format() const {
    auto aformat = get_internal_format();
    return aformat == static_cast<format>(mkldnn_nhwc);
  }

  bool is_iohw_public_layout() const {
    return (get_public_format() == format::iohw && get_internal_format() != format::blocked);
  }

  bool is_limited_blockable() {
    auto& blocking = get_mkldnn_memory_desc_t()->layout_desc.blocking.block_dims;
    for (auto i = 0; i < ndims(); i++) {
      if (get_dim(i) < blocking[i]) continue;
      if (get_dim(i) % blocking[i] == 0) continue;
      return false;
    }
    return true;
  }

  // TODO:it will be remove when deconvolution in mkl-dnn support iohw format.
  static void iohw_definedby_blocked(tensor& atensor) {
    IDEEP_ENFORCE(atensor.ndims() == 4, "Only support 4 dims tensor");
    dims oihw_dims {atensor.get_dim(1), atensor.get_dim(0), atensor.get_dim(2), atensor.get_dim(3)};
    descriptor desc(oihw_dims, atensor.get_data_type(), format::oihw);

    auto oi_primitive_desc = desc.get_mkldnn_memory_desc_t();
    auto oi_blk = oi_primitive_desc->layout_desc.blocking;
    oi_blk.strides[0][0] = oi_blk.strides[0][1];
    oi_blk.strides[0][1] = oi_blk.strides[0][0] * oi_blk.padding_dims[0];

    dims stride(oi_blk.strides[0], oi_blk.strides[0] + oi_primitive_desc->ndims);
    dims stride_inner(oi_blk.strides[1], oi_blk.strides[1] + oi_primitive_desc->ndims);
    dims block_dims(oi_blk.block_dims, oi_blk.block_dims + oi_primitive_desc->ndims);

    descriptor io_desc(oihw_dims, atensor.get_data_type(), stride, block_dims, stride_inner);
    atensor.set_descriptor(io_desc);
  }

  template<typename T>
  inline void fill_all_elems(T val) {
    utils::fast_memset(static_cast<T*>(get_data_handle()), val, get_size() / sizeof(T));
  }

protected:
  std::shared_ptr<tensor> twin_;
};

}
#endif
