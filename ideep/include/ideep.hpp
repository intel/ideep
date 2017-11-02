/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef IDEEP_HPP
#define IDEEP_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <assert.h>
#include <stdlib.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>

#include "mkldnn.h"
#include "mkldnn.hpp"
#endif

namespace ideep {

using error = mkldnn::error;

/// Same class for resource management, except public default constructor
template <typename T> class handle: public mkldnn::handle<T>{
  public:
    using mkldnn::handle<T>::handle;
};

using primitive = mkldnn::primitive;

/// cpu execution engine only.
struct engine: public mkldnn::engine {
  using format = mkldnn::memory::format;

  explicit engine(const mkldnn_engine_t& aengine) = delete;
  engine(engine const &) = delete;
  void operator =(engine const &) = delete;

  /// Singleton CPU engine for all primitives
  static engine &cpu_engine() {
    static engine cpu_engine;
    return cpu_engine;
  }

  format expected_format() const {
    return default_format_;
  }

private:
  /// Constructs an engine.
  ///
  /// @param akind The kind of engine to construct.
  /// @param dformat The default data type of the engine.

  engine(kind akind = kind::cpu, format dformat = format::nchw)
    :mkldnn::engine(akind, 0), default_format_(dformat) {
  }
    
  format default_format_;
};

/// @}

/// A default stream
struct stream: public mkldnn::stream {
  using mkldnn::stream::stream;
  static mkldnn_stream_t default_stream() {
      static stream def(mkldnn::stream::kind::eager);
      return def.get();
  }
};

/// @addtogroup api_tensor Tensor
/// @{

/// Tensor that describes the data/tensor/blob. It resembles MKLDNN's memory
/// But with less burden interface
class tensor: public primitive  {
private:
  std::shared_ptr<char> _handle;

public:
  using dims = mkldnn::memory::dims;
  using data_type = mkldnn::memory::data_type;
  using format = mkldnn::memory::format;

  constexpr static const long invalid_buffer = 42;

  /// A tensor descriptor.
  struct descriptor : public handle<mkldnn_primitive_desc_t> {
    descriptor() {}

    /// The underlying C API data structure.

    /// Constructs a memory primitive descriptor.
    ///
    /// @param adims Data dimensions
    /// @param adata_type Data precision/type.
    /// @param aformat Data layout format.
    descriptor(dims adims, data_type adata_type, format aformat) {
      mkldnn::memory::validate_dims(adims);

      mkldnn_memory_desc_t data;
      error::wrap_c_api(
              mkldnn_memory_desc_init(&data, (int)adims.size(),
                  adims.size() == 0 ? nullptr : &adims[0],
                  convert_to_c(adata_type), convert_to_c(aformat)),
              "could not initialize a memory descriptor");

      mkldnn_primitive_desc_t result;
      mkldnn::error::wrap_c_api(
          mkldnn_memory_primitive_desc_create(&result, &data
            , engine::cpu_engine().get()),
          "could not initialize a memory descriptor");

      reset(result);
    }

    descriptor(dims adims, data_type adata_type)
      : descriptor(adims, adata_type
        , engine::cpu_engine().expected_format()) {}

    /// Returns the number of bytes required to allocate the memory
    /// described including the padding area.
    size_t get_size() const {
      return mkldnn_memory_primitive_desc_get_size(get());
    }

    /// Returns C API mkldnn_memory_desc_t structure which had same 
    /// dimension and data type but without format constrain.
    mkldnn_memory_desc_t format_any() const {
      mkldnn_memory_desc_t any;
      // TODO: Check possible resource leak
      const mkldnn_memory_desc_t *origin
        = mkldnn_primitive_desc_query_memory_d(get());

      error::wrap_c_api(
          mkldnn_memory_desc_init(&any, origin->ndims,
            origin->dims, origin->data_type,
            convert_to_c(format::any)),
          "could not initialize a memory descriptor");

      return any;
    }
  };

  /// Constructs a tensor and allocating internal buffer.
  ///
  /// @param adesc tensor descriptor.
  tensor(const descriptor &adesc) {
    mkldnn_primitive_t result;
    error::wrap_c_api(
      mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr)
      , "could not create a memory primitive");

    reset(result);
    auto _malloc = [](size_t size, size_t alignment) {
      void *ptr;
#ifdef _WIN32
      ptr = _aligned_malloc(size, alignment);
      int rc = ((ptr)? 0 : errno);
#else
      int rc = ::posix_memalign(&ptr, alignment, size);
#endif /* _WIN32 */
      return (rc == 0) ? (char*)ptr : nullptr;
    };

    auto _free = [](char* p) {
#ifdef _WIN32
      _aligned_free((void*)p);
#else
      ::free((void*)p);
#endif /* _WIN32 */
    };
    _handle.reset(_malloc(adesc.get_size(), 4096), _free);
    set_data_handle(_handle.get());
  }

  tensor(const descriptor &adesc, void *ahandle) {
      mkldnn_primitive_t result;
      error::wrap_c_api(
              mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
              "could not create a memory primitive");
      reset(result);
      set_data_handle(ahandle);
  }
    
  /// Allow empty construction
  tensor () {}

  /// Returns the descriptor of the memory primitive.
  descriptor get_descriptor() const {
    descriptor adesc;
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(),
                &cdesc),
            "could not get primitive descriptor from a memory primitive");

    adesc.reset(const_cast<mkldnn_primitive_desc_t>(cdesc), true);
    return adesc;
  }

  /// Returns a handle of the data contained in the tensor. On
  /// the CPU engine, this is a pointer to the allocated memory.
  inline void *get_data_handle() const {
      void *handle;
      error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
              "could not get native handle");
      return handle;
  }

  inline void set_data_handle(void *handle) const {
      error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
              "could not set native handle");
  }

  // Must go away or be private:
  static mkldnn_data_type_t convert_to_c(data_type adata_type) {
      return static_cast<mkldnn_data_type_t>(adata_type);
  }
  static mkldnn_memory_format_t convert_to_c(format aformat) {
      return static_cast<mkldnn_memory_format_t>(aformat);
  }
};

using prop_kind = mkldnn::prop_kind;
using algorithm = mkldnn::algorithm;
using reorder = mkldnn::reorder;
using padding_kind = mkldnn::padding_kind;

template <typename T>
class handles:public std::vector<handle<T>> {
public:
  using size_type = typename std::vector<handle<T>>::size_type;

  handles(): auxiliaries_() {}

  handles(size_type num_of_aux)
    : auxiliaries_(num_of_aux) {}

  template <typename hT>
  void push_auxiliary(hT &&handle) {
    auxiliaries_.push_back(std::forward<hT>(handle));
  }

  void reset(T handle) {
    major_.reset(handle);
  }

  T get() const {
    return major_.get();
  }

protected:
  handle<T> major_;
  std::vector<handle<T>> auxiliaries_;
};

class primitive_group: public handles<mkldnn_primitive_t> {
public:
  primitive_group()
    : handles() {}
};

class descriptor_group: public handles<mkldnn_primitive_desc_t> {
public:
  descriptor_group()
    : handles() {}
};

struct convolution_forward: public primitive_group {
  struct descriptor : public descriptor_group {
    descriptor(prop_kind aprop_kind, algorithm aalgorithm
        , const tensor::descriptor &src_desc
        , const tensor::descriptor &weights_desc
        , const tensor::descriptor &bias_desc
        , const tensor::descriptor &dst_desc
        , const tensor::dims strides
        , const tensor::dims padding_l
        , const tensor::dims padding_r
        , const padding_kind apadding_kind) {

      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);

      mkldnn_convolution_desc_t data;

      mkldnn_memory_desc_t src_data = src_desc.format_any();
      mkldnn_memory_desc_t weights_data = weights_desc.format_any();
      mkldnn_memory_desc_t bias_data = bias_desc.format_any();
      mkldnn_memory_desc_t dst_data = dst_desc.format_any();

      error::wrap_c_api(mkldnn_convolution_forward_desc_init(&data,
                  mkldnn::convert_to_c(aprop_kind), 
                  convert_to_c(aalgorithm),
                  &src_data, &weights_data, &bias_data,
                  &dst_data, &strides[0], &padding_l[0],
                  &padding_r[0],
                  mkldnn::convert_to_c(apadding_kind)),
              "could not create a convolution forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr)
          , "could not create a convolution forward primitive descriptor");

      reset(result);
    }
    descriptor(prop_kind aprop_kind, algorithm aalgorithm
        , const tensor::descriptor &src_desc
        , const tensor::descriptor &weights_desc
        , const tensor::descriptor &dst_desc
        , const tensor::dims strides
        , const tensor::dims padding_l
        , const tensor::dims padding_r
        , const padding_kind apadding_kind) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);

      mkldnn_convolution_desc_t data;

      mkldnn_memory_desc_t src_data = src_desc.format_any();
      mkldnn_memory_desc_t weights_data = weights_desc.format_any();
      mkldnn_memory_desc_t dst_data = dst_desc.format_any();

      error::wrap_c_api(mkldnn_convolution_forward_desc_init(&data,
                  mkldnn::convert_to_c(aprop_kind),
                  convert_to_c(aalgorithm),
                  &src_data, &weights_data, nullptr,
                  &dst_data, &strides[0], &padding_l[0],
                  &padding_r[0],
                  mkldnn::convert_to_c(apadding_kind)),
              "could not create a convolution forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
                  &result, &data, engine::cpu_engine().get(), nullptr),
              "could not create a convolution forward primitive descriptor");

      reset(result);
    }
    descriptor(prop_kind aprop_kind, algorithm aalgorithm
        , const tensor::descriptor &src_desc
        , const tensor::descriptor &weights_desc
        , const tensor::descriptor &bias_desc
        , const tensor::descriptor &dst_desc
        , const tensor::dims strides
        , const tensor::dims dilates
        , const tensor::dims padding_l
        , const tensor::dims padding_r
        , const padding_kind apadding_kind) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(dilates);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t src_data = src_desc.format_any();
      mkldnn_memory_desc_t weights_data = weights_desc.format_any();
      mkldnn_memory_desc_t bias_data = bias_desc.format_any();
      mkldnn_memory_desc_t dst_data = dst_desc.format_any();
      error::wrap_c_api(
          mkldnn_dilated_convolution_forward_desc_init(&data,
              mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                  &src_data, &weights_data, &bias_data,
                  &dst_data, &strides[0], &dilates[0],
                  &padding_l[0], &padding_r[0],
                  mkldnn::convert_to_c(apadding_kind)),
              "could not create a dilated convolution forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
        &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a convolution forward primitive descriptor");
      reset(result);
    }
    descriptor(prop_kind aprop_kind, algorithm aalgorithm
        , const tensor::descriptor &src_desc
        , const tensor::descriptor &weights_desc
        , const tensor::descriptor &dst_desc
        , const tensor::dims strides
        , const tensor::dims dilates
        , const tensor::dims padding_l
        , const tensor::dims padding_r
        , const padding_kind apadding_kind) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(dilates);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t src_data = src_desc.format_any();
      mkldnn_memory_desc_t weights_data = weights_desc.format_any();
      mkldnn_memory_desc_t dst_data = dst_desc.format_any();
      error::wrap_c_api(
        mkldnn_dilated_convolution_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_data, &weights_data, nullptr,
                &dst_data, &strides[0], &dilates[0],
                &padding_l[0], &padding_r[0],
                mkldnn::convert_to_c(apadding_kind)),
            "could not create a dilated convolution forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
        &result, &data, engine::cpu_engine().get(), nullptr),
        "could not create a convolution forward primitive descriptor");

      reset(result);
    }

    tensor::descriptor expected_src_descriptor() const {
      tensor::descriptor adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(get()
              , mkldnn::convert_to_c(mkldnn::src_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc
            , const_cdesc)
          , "could not clone a src primititve descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    tensor::descriptor expected_weights_descriptor() const {
      tensor::descriptor adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(get(),
                         mkldnn::convert_to_c(mkldnn::weights_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
              "could not clone a weights primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    // Do we really need this?
    tensor::descriptor expected_bias_descriptor() const {
      tensor::descriptor adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(get(),
                         mkldnn::convert_to_c(mkldnn::weights_pd), 1);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
              "could not clone a bias primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    tensor::descriptor expected_dst_descriptor() const {
      tensor::descriptor adesc;
      mkldnn_primitive_desc_t cdesc;
      const_mkldnn_primitive_desc_t const_cdesc =
          mkldnn_primitive_desc_query_pd(get(),
                         mkldnn::convert_to_c(mkldnn::dst_pd), 0);
      error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
              "could not clone a dst primitive descriptor");
      adesc.reset(cdesc);
      return adesc;
    }

    int num_of_inputs() const {
        return mkldnn_primitive_desc_query_s32(get(),
            mkldnn::convert_to_c(mkldnn::num_of_inputs_s32), 0);
    }
  };

  convolution_forward(const descriptor &adesc)
    : src_(adesc.expected_src_descriptor()
      , reinterpret_cast<void *>(tensor::invalid_buffer))
    , weights_(adesc.expected_weights_descriptor()
      , reinterpret_cast<void *>(tensor::invalid_buffer))
    , dst_(adesc.expected_dst_descriptor()
      , reinterpret_cast<void *>(tensor::invalid_buffer)) {

    mkldnn_primitive_t result;
    const_mkldnn_primitive_t outputs[] = { dst_.get() };

    if (adesc.num_of_inputs() == 2) {
      mkldnn_primitive_at_t inputs[] = { { src_.get(), 0 }
        , { weights_.get(), 0 } };
      error::wrap_c_api(mkldnn_primitive_create(&result,
              adesc.get(), inputs, outputs),
          "could not create a convolution forward bias primitive");
    } else if (adesc.num_of_inputs() == 3) {
      tensor bias_tmp(adesc.expected_bias_descriptor()
          , reinterpret_cast<void *>(tensor::invalid_buffer));

      // TODO: operator = for memory primitive
      bias_ = bias_tmp;
      bias_.set_data_handle(bias_tmp.get_data_handle());

      mkldnn_primitive_at_t inputs[] = { { src_.get(), 0 }
        , { weights_.get(), 0 }, { bias_.get(), 0 } };
      error::wrap_c_api(mkldnn_primitive_create(&result,
              adesc.get(), inputs, outputs),
          "could not create a convolution forward bias primitive");
    }

    reset(result);
  }

  stream operator() (tensor &src, tensor &weights, tensor &dst
      , stream &parallel_control) {
      auto actual_src = src;

      if (src.get_descriptor() != src_.get_descriptor()) {
          actual_src = memory(src_.get_descriptor());
          reorder(src, actual_src);
      }

      auto actual_weights = weights;
      if (weights.get_desc() != weights_.get_desc()) {
          actual_weights = memory(weights_.get_desc());
          reorder(weights, actual_weights);
      }

      auto actual_dst = dst;
      if (dst.get_desc() != dst_.get_desc()) {
          actual_dst = memory(dst_.get_desc());
      }

      src_.set_data_handle(actual_src.get_data_handle());
      weights_.set_data_handle(actual_weights.get_data_handle());
      dst_.set_data_handle(actual_dst.get_data_handle());

      std::vector<mkldnn_primitive_t> exe {get()};
      mkldnn_primitive_t c_api_error_primitive;

      error::wrap_c_api(
          mkldnn_stream_submit(parallel_control.get()
            , exe.size(), &exe[0], &c_api_error_primitive)
          , "could not execute convolution_forward"
          , &c_api_error_primitive);

      if (dst.get_desc() != dst_.get_desc()) {
          reorder(actual_dst, dst);
      }
      
      return parallel_control;
  }

  stream operator() (memory &src, memory &weights, memory &bias
      , memory &dst, stream &parallel_control) {
      auto actual_src = src;

      if (src.get_desc() != src_.get_desc()) {
          actual_src = memory(src_.get_desc());
          reorder(src, actual_src);
      }

      auto actual_weights = weights;
      if (weights.get_desc() != weights_.get_desc()) {
          actual_weights = memory(weights_.get_desc());
          reorder(weights, actual_weights);
      }

      auto actual_dst = dst;
      if (dst.get_desc() != dst_.get_desc()) {
          actual_dst = memory(dst_.get_desc());
      }

      src_.set_data_handle(actual_src.get_data_handle());
      weights_.set_data_handle(actual_weights.get_data_handle());
      bias_.set_data_handle(bias.get_data_handle());
      dst_.set_data_handle(actual_dst.get_data_handle());

      std::vector<mkldnn_primitive_t> exe {get()};
      mkldnn_primitive_t c_api_error_primitive;

      error::wrap_c_api(
          mkldnn_stream_submit(parallel_control.get()
            , exe.size(), &exe[0], &c_api_error_primitive)
          , "could not execute convolution_forward"
          , &c_api_error_primitive);

      if (dst.get_desc() != dst_.get_desc()) {
          reorder(actual_dst, dst);
      }

      return parallel_control;
  }
private:
    // Placeholders for further buffer handlers
    tensor src_, weights_, dst_, bias_;
};

#if 0
struct convolution_backward_data : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_backward_data_desc_init(
                    &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                    &weights_desc.data, &diff_dst_desc.data,
                    &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                    mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }
    };
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a convolution backward data primitive descriptor");
            reset(result);
        }
        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    convolution_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at &weights,
            const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { diff_dst.data, weights.data  };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution backward data primitive");
        reset(result);
    }
};

struct convolution_backward_weights : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &dilates[0],  &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }

    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a convolution backward weights primitive descriptor");
            reset(result);
        }
        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_bias_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 1);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights, const memory &diff_bias) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_weights.get(),
                    diff_bias.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution backward weights primitive");
        reset(result);
    }
    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_weights.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution backward weights primitive");
        reset(result);
    }
};

struct convolution_relu_forward : public primitive {
    struct desc {
        mkldnn_convolution_relu_desc_t data;
        desc(const convolution_forward::desc conv_desc,
                const float negative_slope)
        {
            error::wrap_c_api(mkldnn_convolution_relu_desc_init(&data,
                        &conv_desc.data, negative_slope),
                    "could not create a convolution_relu_forward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                    &result, &adesc.data, aengine.get(), nullptr),
                "could not create a convolution relu forward descriptor");
            reset(result);
        }

        engine get_engine() { return engine::query(*this); }
    };

    convolution_relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, weights.data,
                bias.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a convolution relu forward primitive");
        reset(result);
    }

    convolution_relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a convolution relu forward primitive");
        reset(result);
    }
};
struct lrn_forward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, float alpha, float beta, float k)
        {
            error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, local_size, alpha, beta, k),
                "could not create a lrn forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, float alpha, float beta)
        {
            error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, local_size, alpha, beta, float(1.0)),
                "could not create a lrn forward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                    &result, &adesc.data, aengine.get(), nullptr),
                "could not create a lrn forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t ldesc;
            const_mkldnn_primitive_desc_t const_ldesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                    "could not clone a workspace primitive descriptor");
            adesc.reset(ldesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &workspace,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(),
                workspace.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn forward primitive");
        reset(result);
    }

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn forward primitive");
        reset(result);
    }
};

struct lrn_backward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, float alpha, float beta, float k)
        {
            error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
                convert_to_c(aalgorithm), &diff_data_desc.data,
                &data_desc.data, local_size, alpha, beta, k),
                "could not create a lrn backward descriptor");
        }
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, float alpha, float beta)
        {
            error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
                convert_to_c(aalgorithm), &diff_data_desc.data,
                &data_desc.data, local_size, alpha, beta, float(1.0)),
                "could not create a lrn backward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const lrn_forward::primitive_desc &hint_fwd_primitive_desc) {
        mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a backward lrn primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t ldesc;
            const_mkldnn_primitive_desc_t const_ldesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                    "could not clone a workspace primitive descriptor");
            adesc.reset(ldesc);
            return adesc;
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data,
                workspace.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn backward primitive");
        reset(result);
    }

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn backward primitive");
        reset(result);
    }
};

struct pooling_forward : public primitive {
    struct desc {
        mkldnn_pooling_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims kernel,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_pooling_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm),
                        &src_desc.data, &dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a forward pooling descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
        mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a forward pooling primitive descriptor");
            reset(result);
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a workspace primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    pooling_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(), nullptr };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling forward primitive");
        reset(result);
    }

    pooling_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst, const memory &workspace) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(), workspace.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling forward primitive");
        reset(result);
    }
};

struct pooling_backward : public primitive {
    struct desc {
        mkldnn_pooling_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims &strides,
                const memory::dims &kernel,
                const memory::dims &padding_l,
                const memory::dims &padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_pooling_backward_desc_init(&data,
                        convert_to_c(aalgorithm),
                        &diff_src_desc.data, &diff_dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a backward pooling descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const pooling_forward::primitive_desc &hint_fwd_primitive_desc) {
        mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a backward pooling primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    pooling_backward(const primitive_desc &aprimitive_desc, const primitive::at &diff_dst,
            const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling backward primitive");
        reset(result);
    }

    pooling_backward(const primitive_desc &aprimitive_desc, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { diff_dst.data, workspace.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling backward primitive");
        reset(result);
    }
};

struct eltwise_forward : public primitive {
    struct desc {
        mkldnn_eltwise_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, algorithm alg_kind,
                const memory::desc &src_desc, T alpha = 0, T beta = 0) {
            error::wrap_c_api(mkldnn_eltwise_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        mkldnn::convert_to_c(alg_kind), &src_desc.data,
                        static_cast<float>(alpha), static_cast<float>(beta)),
                    "could not create a eltwise forward descriptor");
        }

        /** @deprecated: api backward compatibility for relu */
        template <typename T>
        MKLDNN_DEPRECATED
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                T negative_slope)
        : desc(aprop_kind, eltwise_relu, src_desc, negative_slope) {}
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a eltwise forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(
                    mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    eltwise_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a eltwise forward primitive");
        reset(result);
    }
};

typedef eltwise_forward relu_forward;

struct eltwise_backward : public primitive {
    struct desc {
        mkldnn_eltwise_desc_t data;

        template <typename T>
        desc(algorithm alg_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T alpha = 0, T beta = 0) {
            error::wrap_c_api(mkldnn_eltwise_backward_desc_init(&data,
                        mkldnn::convert_to_c(alg_kind), &diff_data_desc.data,
                        &data_desc.data, static_cast<float>(alpha),
                        static_cast<float>(beta)),
                    "could not create a eltwise backward descriptor");
        }

        /** @deprecated: api backward compatibility for relu */
        template <typename T>
        MKLDNN_DEPRECATED
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
            T negative_slope): desc(eltwise_relu, diff_data_desc, data_desc,
                negative_slope) {}
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const eltwise_forward::primitive_desc &hint_fwd_primitive_desc) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a eltwise backward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    eltwise_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a eltwise backward primitive");
        reset(result);
    }
};

typedef eltwise_backward relu_backward;

struct softmax_forward : public primitive {
    struct desc {
        mkldnn_softmax_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int softmax_axis) {
            error::wrap_c_api(mkldnn_softmax_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), &data_desc.data,
                    softmax_axis),
                "could not create a softmax forward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                    &result, &adesc.data, aengine.get(), nullptr),
                "could not create a softmax forward primitive descriptor");
            reset(result);
        }

        engine get_engine() { return engine::query(*this); }
    };

    softmax_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a softmax forward primitive");
        reset(result);
    }
};

struct batch_normalization_forward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &src_desc, T epsilon,
                unsigned flags) {
            error::wrap_c_api(
                    mkldnn_batch_normalization_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        static_cast<float>(epsilon), flags),
                "could not create a batch normalization forward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                &result, &adesc.data, aengine.get(), nullptr),
        "could not create a batch normalization forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t bndesc;
            const_mkldnn_primitive_desc_t const_bndesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc mean_primitive_desc() const {
            memory::primitive_desc aprimitive_desc;
            mkldnn_primitive_desc_t bndesc;
            mkldnn_batch_normalization_desc_t *p;
            error::wrap_c_api(mkldnn_primitive_desc_query(
                    get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                    "could not get a batch-normalization descriptor");
            const_mkldnn_primitive_desc_t const_bndesc =
                (p->flags & use_global_stats) ?
                    mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(src_pd), 1) :
                    mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(dst_pd), 1);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a mean primitive descriptor");
            aprimitive_desc.reset(bndesc);
            return aprimitive_desc;
        }

        memory::primitive_desc variance_primitive_desc() const {
            memory::primitive_desc aprimitive_desc;
            mkldnn_primitive_desc_t bndesc;
            mkldnn_batch_normalization_desc_t *p;
            error::wrap_c_api(mkldnn_primitive_desc_query(
                    get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                    "could not get a batch-normalization descriptor");
            const_mkldnn_primitive_desc_t const_bndesc =
                (p->flags & use_global_stats) ?
                    mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(src_pd), 2) :
                    mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(dst_pd), 2);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a variance primitive descriptor");
            aprimitive_desc.reset(bndesc);
            return aprimitive_desc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc,
                        const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &weights,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst, const memory &mean, const memory &variance) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(),
            mean.get(), variance.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst, const memory &mean,
            const memory &variance) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(),
            mean.get(), variance.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }
};

struct batch_normalization_backward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T epsilon, unsigned flags) {
            error::wrap_c_api(
                    mkldnn_batch_normalization_backward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &diff_data_desc.data, &data_desc.data,
                        static_cast<float>(epsilon), flags),
                "could not create a batch normalization backward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const batch_normalization_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                &result, &adesc.data, aengine.get(),
                hint_fwd_primitive_desc.get()),
        "could not create a batch normalization backward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t bndesc;
            const_mkldnn_primitive_desc_t const_bndesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc diff_weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t bndesc;
            const_mkldnn_primitive_desc_t const_bndesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a diff_weights primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc mean_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t bndesc;
            const_mkldnn_primitive_desc_t const_bndesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 1);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a mean primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc variance_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t bndesc;
            const_mkldnn_primitive_desc_t const_bndesc =
                    mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 2);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a variance primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc,
                        const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    // Prop_kind == backward
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const primitive::at &weights, const memory &diff_src,
            const memory &diff_weights) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, diff_dst.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get(),
                diff_weights.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization backward primitive");
        reset(result);
    }

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance,const primitive::at &diff_dst,
            const primitive::at &weights,  const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, diff_dst.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization backward primitive");
        reset(result);
    }

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization backward primitive");
        reset(result);
    }
};

struct inner_product_forward: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                &result, &adesc.data, aengine.get(), nullptr),
        "could not create a inner product forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc bias_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 1);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const primitive::at &bias, const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, weights.data,
                bias.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product forward primitive");
        reset(result);
    }

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product forward primitive");
        reset(result);
    }
};

struct inner_product_backward_data: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_data_desc_init(&data,
                        &diff_src_desc.data, &weights_desc.data,
                        &diff_dst_desc.data),
                "could not create a inner product backward data descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(&result,
                    &adesc.data, aengine.get(), hint_fwd_primitive_desc.get()),
        "could not create a inner product backward data primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff dst primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    inner_product_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at weights,
            const memory &diff_src) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { diff_dst.data, weights.data };
        const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product backward data primitive");
        reset(result);
    }
};

struct inner_product_backward_weights: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        &diff_bias_desc.data, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        nullptr, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(&result,
                    &adesc.data, aengine.get(), hint_fwd_primitive_desc.get()),
        "could not create a inner product backward weights primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff dst primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_weights_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_bias_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 1);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] = { diff_weights.get() };
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product backward weights primitive");
        reset(result);
    }

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights, const memory &diff_bias) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        const_mkldnn_primitive_t outputs[] =
                { diff_weights.get(), diff_bias.get()};
        error::wrap_c_api(mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product backward weights primitive");
        reset(result);
    }
};
#endif
} // namespace mkldnn

#endif
