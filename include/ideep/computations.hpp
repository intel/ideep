/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


#ifndef IDEEP_HPP
#define IDEEP_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <iterator>
#include <string>
#include <cstring>
#include <numeric>
#include <functional>
#include <iostream>
#include <immintrin.h>
#include <random>
#include <atomic>
#include <chrono>

#include "abstract_types.hpp"
#include "fast_math.hpp"
#include "tensor.hpp"
#include "lru_cache.hpp"
#include "scope_guard.hpp"
#include "instruments.hpp"
#include <mkl_vsl.h>
#endif

namespace ideep {

template<>
inline tensor::data_type tensor::descriptor::type_to_id<float>() {
  return tensor::data_type::f32;
}

template<>
inline tensor::data_type tensor::descriptor::type_to_id<int>() {
  return tensor::data_type::s32;
}

template<>
inline tensor::data_type tensor::descriptor::type_to_id<unsigned char>() {
  return tensor::data_type::u8;
}

template<>
inline tensor::data_type tensor::descriptor::type_to_id<signed char>() {
  return tensor::data_type::s8;
}

/// Descriptor group, create relative descriptors all in one
class descriptor_group: public c_wrapper_complex<mkldnn_primitive_desc_t> {
  friend class primitive_group;
public:
  class post_ops : public c_wrapper<mkldnn_post_ops_t> {
  public:
    post_ops() : c_wrapper([]() {
      mkldnn_post_ops_t result;
      error::wrap_c_api(mkldnn_post_ops_create(&result),
          "could not create post operation sequence");
      return result;
    }()) {}


    int num_ops() const {
      return mkldnn_post_ops_len(get());
    }

    kind op_kind(int index) const {
      IDEEP_ENFORCE(index < num_ops(), "post_ops index is out of range");
      return static_cast<kind>(mkldnn_post_ops_get_kind(get(), index));
    }

    void append(kind op_kind,
        float scale, float alpha, float beta, algorithm alg) {
      switch(op_kind) {
        case kind::sum:
          error::wrap_c_api(
              mkldnn_post_ops_append_sum(get(), scale),
              "could not append sum");
          break;
        case kind::eltwise:
          error::wrap_c_api(mkldnn_post_ops_append_eltwise(get(), scale,
                convert_to_c(alg), alpha, beta), "could not append eltwise");
          break;
        default:
          // TODO: throw?
          break;
      }
    }

    std::tuple<kind, float, float, float, algorithm>
      get_params(int index) const {
      mkldnn_alg_kind_t c_alg = mkldnn_eltwise_relu;
      float scale, alpha = 1.0, beta = 0.0;

      auto akind = op_kind(index);
      switch(akind) {
        case kind::sum:
          error::wrap_c_api(mkldnn_post_ops_get_params_sum(get(), index, &scale),
              "could not get sum params");
          break;
        case kind::eltwise:
          error::wrap_c_api(mkldnn_post_ops_get_params_eltwise(get(), index,
                &scale, &c_alg, &alpha, &beta), "could not get eltwise params");
          break;
        default:
          error::wrap_c_api(mkldnn_invalid_arguments, "could not get params");
          break;
      }

      return std::make_tuple(
          akind, scale, alpha, beta, static_cast<algorithm>(c_alg));
    }

    utils::bytestring to_bytes() const {
      utils::bytestring ret;

      for (int i = 0; i < num_ops(); i ++) {
        kind akind;
        float scale, alpha, beta;
        algorithm alg;
        std::tie(akind, scale, alpha, beta, alg) = get_params(i);

        switch(akind) {
          case kind::sum:
            ret += utils::to_bytes(akind) + '.' + utils::to_bytes(scale);
            break;
          case kind::eltwise:
            ret += utils::to_bytes(akind) + '.' + utils::to_bytes(scale)
              + '.' + utils::to_bytes(alpha) + '.' + utils::to_bytes(beta)
              + '.' + utils::to_bytes(alg);
          default:
            break;
        }
      }

      return ret;
    }

  public:
    // Helper factory
    static post_ops sum(float scale = 1.0) {
      post_ops ret;
      ret.append(kind::sum, scale,
          /* meanless dummies */1.0, 0.0, algorithm::eltwise_relu);
      return ret;
    }

    static post_ops relu(float scale = 1.f,
        float alpha = 0.f, float beta = 0.f) {
      post_ops ret;
      ret.append(kind::eltwise, scale, alpha, beta, algorithm::eltwise_relu);
      return ret;
    }

    static post_ops residual(float scale = 1.0,
        float alpha = 0.f, float beta = 0.f) {
      post_ops ret;

      ret.append(kind::sum, scale, 1.0, 0.0, algorithm::eltwise_relu);
      ret.append(kind::eltwise, scale, alpha, beta, algorithm::eltwise_relu);

      return ret;
    }
  };

  class attr_t : public c_wrapper<mkldnn_primitive_attr_t> {
  public:
    attr_t() : c_wrapper([]() {
      mkldnn_primitive_attr_t result;
      error::wrap_c_api(mkldnn_primitive_attr_create(&result),
          "could not create a primitive attr");
      return result;
    }()) {}

    round_mode get_int_output_round_mode() const {
      mkldnn_round_mode_t result;
      error::wrap_c_api(mkldnn_primitive_attr_get_int_output_round_mode(
            get(), &result), "could not get int output round mode");
      return round_mode(result);
    }

    void set_int_output_round_mode(round_mode mode) {
      error::wrap_c_api(mkldnn_primitive_attr_set_int_output_round_mode(
            get(), mkldnn::convert_to_c(mode)),
          "could not set int output round mode");
    }

    std::pair<std::vector<float>, int> get_output_scales() const {
      int count, c_mask;
      const float *c_scales;
      error::wrap_c_api(mkldnn_primitive_attr_get_output_scales(get(),
            &count, &c_mask, &c_scales), "could not get int output scales");

      return std::make_pair(
          std::vector<float>(c_scales, c_scales + count), c_mask);
    }

    void set_output_scales(int mask, std::vector<float> scales) {
      error::wrap_c_api(mkldnn_primitive_attr_set_output_scales(get(),
            (int)scales.size(), mask, &scales[0]),
          "could not set int output scales");
    }

    const post_ops get_post_ops() const {
      const_mkldnn_post_ops_t c_result;
      error::wrap_c_api(mkldnn_primitive_attr_get_post_ops(get(), &c_result),
          "could not get post operatoion sequence");

      // XXX: resource management OK?
      post_ops result;
      result.reset(const_cast<mkldnn_post_ops_t>(c_result), true);
      return result;
    }

    void set_post_ops(post_ops ops) {
      error::wrap_c_api(mkldnn_primitive_attr_set_post_ops(get(), ops.get()),
            "could not set post operation sequence");
    }

    utils::bytestring to_bytes() const {
      auto bytes = get_post_ops().to_bytes();
      auto scales = get_output_scales();

      bytes += utils::to_bytes(scales.first) + utils::to_bytes(scales.second);
      return bytes;
    }

  public:
    // Helper factory
    //
    static inline attr_t fuse_sum(float scale = 1.0) {
      attr_t attr;
      attr.set_post_ops(post_ops::sum(scale));
      return attr;
    }

    static inline attr_t fuse_relu(float scale = 1.0,
        float alpha = 0.f, float beta = 0.f) {
      attr_t attr;
      attr.set_post_ops(post_ops::relu(scale, alpha, beta));
      return attr;
    }

    static inline attr_t residual(float scale = 1.0,
        float alpha = 0.f, float beta = 0.f) {
      attr_t attr;
      attr.set_post_ops(post_ops::residual(scale, alpha, beta));
      return attr;
    }

    static inline attr_t attr_post_ops(post_ops post) {
      attr_t attr;
      attr.set_post_ops(post);
      return attr;
    }
  };

protected:
  std::vector<const_mkldnn_primitive_desc_t> cpp_to_c(
      const std::vector<tensor::descriptor> &inputs) {
    std::vector<const_mkldnn_primitive_desc_t> c_api_inputs;
    c_api_inputs.reserve(inputs.size());

    auto convert_to_c = [](const tensor::descriptor &d) {
      return d.get();
    };

    std::transform(inputs.begin(), inputs.end(),
        std::back_inserter(c_api_inputs), convert_to_c);

    return c_api_inputs;
  }

public:
  descriptor_group()
    : c_wrapper_complex() {}

  tensor::descriptor expected_descriptor_of(mkldnn::query q
      , int index = 0) const {
    mkldnn_primitive_desc_t cdesc;
    const_mkldnn_primitive_desc_t const_cdesc =
        mkldnn_primitive_desc_query_pd(get()
            , mkldnn::convert_to_c(q), index);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc
          , const_cdesc)
        , "could not clone a src primititve descriptor");
    return param::descriptor(cdesc);
  }

  tensor::descriptor expected_input_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::input_pd, index);
  }

  tensor::descriptor expected_output_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::output_pd, index);
  }

  tensor::descriptor expected_src_descriptor() const {
    return expected_descriptor_of(mkldnn::src_pd);
  }

  tensor::descriptor expected_weights_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd);
  }

  tensor::descriptor expected_bias_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd, 1);
  }

  tensor::descriptor expected_dst_descriptor() const {
    return expected_descriptor_of(mkldnn::dst_pd, 0);
  }

  tensor::descriptor expected_workspace_descriptor() const {
    return expected_descriptor_of(mkldnn::workspace_pd, 0);
  }

  tensor::descriptor expected_gradx_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_src_pd, 0);
  }

  tensor::descriptor expected_grady_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_dst_pd, 0);
  }

  tensor::descriptor expected_gradw_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 0);
  }

  tensor::descriptor expected_gradb_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 1);
  }

  int num_of_inputs() const {
      return mkldnn_primitive_desc_query_s32(get()
          , mkldnn::convert_to_c(mkldnn::num_of_inputs_s32), 0);
  }

  int num_of_outputs() const {
      return mkldnn_primitive_desc_query_s32(get()
          , mkldnn::convert_to_c(mkldnn::num_of_outputs_s32), 0);
  }

protected:
  void create_reorder_pds(std::vector<tensor::descriptor> descriptors) {
    for (unsigned i = 0; i < descriptors.size(); i ++) {
      assert((int)i < num_of_inputs());
      auto &provided = descriptors[i];
      auto expected = expected_input_descriptor((int)i);
      if (expected != provided) {
        mkldnn_primitive_desc_t result;
        error::wrap_c_api(mkldnn_reorder_primitive_desc_create(
              &result, provided.get(), expected.get()),
            "could not create reorder primitive descriptor");
        auxiliaries_[i].reset(result);
      }
    }
  }
};

class primitive_group: public c_wrapper_complex<mkldnn_primitive_t> {
public:
  primitive_group()
    : c_wrapper_complex() {}

  /// Returns the internal structure of primitive descriptor.
  const_mkldnn_primitive_desc_t get_mkldnn_primitive_desc_t() const {
    const_mkldnn_primitive_desc_t cdesc;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(),
                &cdesc),
            "could not get primitive descriptor from a memory primitive");
    return cdesc;
  }

  tensor::descriptor expected_descriptor_of(mkldnn::query q,
      int index = 0) const {
    mkldnn_primitive_desc_t cdesc;
    const_mkldnn_primitive_desc_t const_cdesc =
        mkldnn_primitive_desc_query_pd(get_mkldnn_primitive_desc_t(),
            mkldnn::convert_to_c(q), index);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc
          , const_cdesc)
        , "could not clone a src primititve descriptor");
    return tensor::descriptor(cdesc);
  }

protected:
  void create_reorder_for(unsigned index
      , const descriptor_group &g, param& in, param& out) {
    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out.get() };

    error::wrap_c_api(mkldnn_primitive_create(&result
          , g.auxiliaries_[index].get(), inputs, outputs),
        "could not create a reorder");

    auxiliaries_[index].reset(result);
  }

  tensor::descriptor expected_input_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::input_pd, index);
  }

  tensor::descriptor expected_output_descriptor(int index) const {
    return expected_descriptor_of(mkldnn::output_pd, index);
  }

  tensor::descriptor expected_src_descriptor() const {
    return expected_descriptor_of(mkldnn::src_pd);
  }

  tensor::descriptor expected_weights_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd);
  }

  tensor::descriptor expected_bias_descriptor() const {
    return expected_descriptor_of(mkldnn::weights_pd, 1);
  }

  tensor::descriptor expected_dst_descriptor() const {
    return expected_descriptor_of(mkldnn::dst_pd, 0);
  }

  tensor::descriptor expected_workspace_descriptor() const {
    return expected_descriptor_of(mkldnn::workspace_pd, 0);
  }

  tensor::descriptor expected_gradx_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_src_pd, 0);
  }

  tensor::descriptor expected_grady_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_dst_pd, 0);
  }

  tensor::descriptor expected_gradw_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 0);
  }

  tensor::descriptor expected_gradb_descriptor() const {
    return expected_descriptor_of(mkldnn::diff_weights_pd, 1);
  }

  void execute(stream &parallel_control) {
    std::vector<mkldnn_primitive_t> execution_sequence;
    mkldnn_primitive_t c_api_error_primitive;

    // TODO: varadic needed
    if (need_reorder_input(0))
      execution_sequence.push_back(auxiliaries_[0].get());

    if (need_reorder_input(1))
      execution_sequence.push_back(auxiliaries_[1].get());

    // Operator
    execution_sequence.push_back(get());

    // if (need_reorder_input(3))
    //   execution_sequence.push_back(auxiliaries_[3].get());

    __itt_frame_begin_v3(instruments::domain::ideep(), nullptr);
    error::wrap_c_api(
        mkldnn_stream_submit(parallel_control.get()
          , execution_sequence.size(), &execution_sequence[0]
          , &c_api_error_primitive)
        , "could not execute the computation"
        , &c_api_error_primitive);
    __itt_frame_end_v3(instruments::domain::ideep(), nullptr);
  }
};

struct reorder: public c_wrapper<mkldnn_primitive_t>,
  public utils::computation_cache<reorder> {
  struct descriptor : public c_wrapper<mkldnn_primitive_desc_t> {
    using attr_t = descriptor_group::attr_t;
    using post_ops = descriptor_group::post_ops;

    descriptor(const c_wrapper<mkldnn_primitive_desc_t> &input,
        const tensor::descriptor &output,
        const attr_t attr = attr_t()) {
      // TODO: check to make sure primitive_desc is memory/view
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
            &result, input.get(), output.get(), attr.get()),
          "could not create a reorder primitive descriptor");
      reset(result);
    }
  };

  reorder() = default;

  void init(const tensor::descriptor& src_desc,
      const tensor::descriptor& dst_desc,
      const descriptor::attr_t attr = descriptor::attr_t()) {
    mkldnn_primitive_desc_t desc;
    error::wrap_c_api(mkldnn_reorder_primitive_desc_create_v2(
          &desc, src_desc.get(), dst_desc.get(), attr.get()),
        "could not create a reorder primitive descriptor");
    c_wrapper<mkldnn_primitive_desc_t> sg(desc);

    in.init(src_desc, nullptr);
    out.init(dst_desc, nullptr);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out.get() };
    error::wrap_c_api(mkldnn_primitive_create(&result, desc, inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }

  void init(const tensor::view& view, const tensor::descriptor& src_desc,
      const tensor::descriptor& dst_desc) {
    mkldnn_primitive_desc_t desc;
    error::wrap_c_api(mkldnn_reorder_primitive_desc_create(
          &desc, view.get(), dst_desc.get()),
        "could not create a reorder primitive descriptor");
    c_wrapper<mkldnn_primitive_desc_t> sg(desc);

    in.init(src_desc, nullptr);
    out.init(dst_desc, nullptr);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out.get() };
    error::wrap_c_api(mkldnn_primitive_create(&result, desc, inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }

  void init(const tensor::descriptor& src_desc, const tensor::view& view,
      const tensor::descriptor& dst_desc) {
    mkldnn_primitive_desc_t desc;
    error::wrap_c_api(mkldnn_reorder_primitive_desc_create(
          &desc, src_desc.get(), view.get()),
        "could not create a reorder primitive descriptor");
    c_wrapper<mkldnn_primitive_desc_t> sg(desc);

    in.init(src_desc, nullptr);
    out.init(dst_desc, nullptr);

    mkldnn_primitive_t result;
    mkldnn_primitive_at_t inputs[] = { {in.get(), 0} };
    const_mkldnn_primitive_t outputs[] = { out.get() };
    error::wrap_c_api(mkldnn_primitive_create(&result, desc, inputs, outputs),
        "could not create a reorder primitive");
    reset(result);
  }

  template<typename T, typename... Ts>
  reorder(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void operator() (const tensor &input, const tensor &output) {
    assert(input.get_descriptor() == in.get_descriptor()
        && output.get_descriptor() == out.get_descriptor());
    in.set_data_handle(input.get_data_handle());
    out.set_data_handle(output.get_data_handle());

    std::vector<mkldnn_primitive_t> execution_sequence = {get()};
    mkldnn_primitive_t c_api_error_primitive;

    __itt_frame_begin_v3(instruments::domain::ideep(), nullptr);
    error::wrap_c_api(
        mkldnn_stream_submit(stream::default_stream().get(),
          execution_sequence.size(), &execution_sequence[0],
          &c_api_error_primitive),
        "could not execute reorder", &c_api_error_primitive);
    __itt_frame_end_v3(instruments::domain::ideep(), nullptr);
  }

  static void compute(
      const tensor& input, const tensor& output,
      const descriptor::attr_t attr = descriptor::attr_t()) {
    if (input.is_empty() || output.is_empty())
      return;

    auto key = utils::create_key(input.get_dims(), input.get_data_type(),
        input.get_internal_format(), output.get_dims(), output.get_data_type(),
        output.get_internal_format(), attr);

    auto op = fetch_or_create_m(key, input.get_descriptor(),
        output.get_descriptor(), attr);

    op(input, output);
  }

  // TODO: make this right
  template <typename alloc = utils::allocator>
  static tensor compute(
      const tensor &input, const tensor::dims &volume, const tensor::dims &start) {
    auto key = utils::create_key(input.get_dims(), input.get_data_type(),
        input.get_internal_format(), volume, start);

    auto view = input.create_view(volume, start);
    tensor gx;
    gx.init<alloc, reorder>(view.expected_dst_descriptor());

    auto op = fetch_or_create_m(key, view, input.get_descriptor(),
        gx.get_descriptor());

    op(input, gx);
    return gx;
  }

protected:
  param in, out;
};

struct direct_copy : public reorder {
public:
  using reorder::reorder;

  template <typename alloc = utils::allocator>
  static void compute(const tensor& input, tensor& output) {
    if (input.is_empty())
      return;

    output.reinit<alloc, direct_copy>(input.get_descriptor());
    reorder::compute(input, output);
  }
};

struct spliter : public reorder {
public:
  using reorder::reorder;

  static std::vector<tensor> compute(
      tensor input, std::vector<int32_t> axis_info, int axis, bool add_axis) {
    reorder reorder_;
    std::vector<tensor> outputs;
    tensor::dims output_dims(input.get_dims());
    tensor::dims offset_dims(output_dims.size(), 0);
    IDEEP_ENFORCE(axis < input.ndims(), "invalid axis in split");

    for (unsigned i = 0; i < axis_info.size(); ++i) {
      output_dims[axis] = axis_info[i];
      auto view = input.create_view(output_dims, offset_dims);
      tensor output(view.expected_dst_descriptor());
      reorder_.init(view, input.get_descriptor(), output.get_descriptor());
      reorder_(input, output);

      if (add_axis) {
        tensor::dims out_dims(output_dims);
        out_dims.erase(out_dims.begin() + axis);
        output.reshape(out_dims);
      }

      outputs.emplace_back(output);
      offset_dims[axis] += axis_info[i];
    }

    return outputs;
  }
};

struct computation : public primitive_group {
  computation() = default;

  void connect_reorder_for(const descriptor_group& adesc,
      const std::vector<tensor::descriptor>& args) {
    for (int i = 0; (unsigned)i < args.size(); i ++) {
      connect_reorder_for(i, adesc, args[(unsigned)i]);
    }
  }

  void connect_reorder_for(int, const descriptor_group&) {
    // dummy, do nothing
  }

  template <typename... Ts>
  void connect_reorder_for(int index, const descriptor_group &adesc,
      const tensor::descriptor& first, const Ts&... rest) {
    connect_reorder_for(index, adesc, first);
    connect_reorder_for(index + 1, adesc, rest...);
  }

  void connect_reorder_for(int index, const descriptor_group &adesc,
      const tensor::descriptor &desc) {
    if (adesc.need_reorder_input(index)) {
      inouts_[index] = param { desc, nullptr };
      create_reorder_for(
          (unsigned)index, adesc, inouts_[(unsigned)index],
          primitive_inputs_[(unsigned)index]);
    }
  }

  inline void init_internal(
      const descriptor_group &adesc, int n_inputs, int n_outputs) {
    // init contents
    primitive_inputs_ = std::vector<param>((unsigned)n_inputs);
    inouts_ = std::vector<param>((unsigned)(n_inputs + n_outputs));

    mkldnn_primitive_at_t inputs[n_inputs];
    for (int i =0; i < n_inputs; i ++) {
      primitive_inputs_[i] = {
        adesc.expected_input_descriptor(i), nullptr };
      // connect real inputs and primitive inputs
      inouts_[i] = primitive_inputs_[i];
      inputs[i] = { primitive_inputs_[i].get(), 0 };
    }

    const_mkldnn_primitive_t outputs[n_outputs];
    for (int i = 0; i < n_outputs; i ++) {
      inouts_[i + n_inputs] = {
        adesc.expected_output_descriptor(i), nullptr };
      outputs[i] = inouts_[i + n_inputs].get();
    }

    mkldnn_primitive_t result;
    error::wrap_c_api(mkldnn_primitive_create(&result,
          adesc.get(), inputs, outputs),
        "could not create a computation primitive");

    reset(result);
  }

  void init(const descriptor_group& adesc,
      const std::vector<tensor::descriptor> &args) {
    assert(adesc.num_of_inputs() == (int)args.size());
    auto n_inputs = (int)args.size();
    auto n_outputs = adesc.num_of_outputs();
    init_internal(adesc, n_inputs, n_outputs);
    connect_reorder_for(adesc, args);
  }

  template <typename... Ts>
  void init(const descriptor_group &adesc, const Ts&... args) {
    auto n_inputs = adesc.num_of_inputs();
    auto n_outputs = adesc.num_of_outputs();
    init_internal(adesc, n_inputs, n_outputs);
    connect_reorder_for(0, adesc, args...);
  }

  void connect_handle_for(int index, const param& atensor) {
    if ((unsigned)index < primitive_inputs_.size() &&
        inouts_[index] != primitive_inputs_[index]) {
      // Connect inputs
      if (inouts_.at((unsigned)index).get_descriptor()
          == atensor.get_descriptor()) {
        inouts_[(unsigned)index].set_data_handle(atensor.get_data_handle());
        primitive_inputs_[(unsigned)index].materialize();
      } else if(primitive_inputs_.at((unsigned)index).get_descriptor()
          == atensor.get_descriptor()) {
        // Destructional move, assume we never change back
        primitive_inputs_[(unsigned)index].dematerialize();
        primitive_inputs_[(unsigned)index].set_data_handle(
            atensor.get_data_handle());

        // We throw the reorder away.
        auxiliaries_[index].reset(nullptr);
      } else
        throw error(mkldnn_runtime_error, "Cannot accept incompatible input");
    } else {
      // Connect outputs
      assert(inouts_.at((unsigned)index).get_descriptor()
          == atensor.get_descriptor());
      inouts_.at((unsigned)index).set_data_handle(atensor.get_data_handle());
    }
  }

  void connect_handle_for(const std::vector<tensor>& inputs,
      const param& output) {
    int i = 0;
    for(; (unsigned)i < inputs.size(); i++){
      connect_handle_for(i, inputs[(unsigned)i]);
    }
    connect_handle_for(i, output);
  }

  template <typename ...Params>
  void connect_handle_for(int index, const param& first,
      const Params&... rest) {
    connect_handle_for(index, first);
    connect_handle_for(index + 1, rest...);
  }

  void execute(const std::vector<tensor>& inputs, const tensor& outputs) {
    connect_handle_for(inputs, outputs);
    stream parallel_control = stream::default_stream();
    primitive_group::execute(parallel_control);
  }

  template<typename ...Params>
  void execute(const param& arg0, const Params&... args) {
    connect_handle_for(0, arg0, args...);
    stream parallel_control = stream::default_stream();
    primitive_group::execute(parallel_control);
  }

  int num_of_inputs() const {
    return primitive_inputs_.size();
  }

  int num_of_outputs() const {
    return inouts_.size() - primitive_inputs_.size();
  }

private:
  // outputs after inputs
  // TODO: turn in into share_ptr
  std::vector<param> inouts_;
  std::vector<param> primitive_inputs_;
};

struct convolution_forward: public computation,
  public utils::computation_cache<convolution_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &src_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &bias_desc,
        const tensor::descriptor &dst_desc,
        const tensor::dims strides,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        const attr_t attr = attr_t(),
        algorithm aalgorithm = algorithm::convolution_direct,
        prop_kind aprop_kind = prop_kind::forward,
        const padding_kind apadding_kind = padding_kind::zero) {
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
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
            &result, &data, attr.get(), engine::cpu_engine().get(), nullptr)
          , "could not create a convolution forward primitive descriptor");

      reset(result);
      create_reorder_pds({src_desc, weights_desc});
    }
    descriptor(const tensor::descriptor &src_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &dst_desc,
        const tensor::dims strides,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        const attr_t attr = attr_t(),
        algorithm aalgorithm = algorithm::convolution_direct,
        prop_kind aprop_kind = prop_kind::forward,
        const padding_kind apadding_kind = padding_kind::zero) {
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
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
            &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
            "could not create a convolution forward primitive descriptor");

      reset(result);
      create_reorder_pds({src_desc, weights_desc});
    }
    descriptor(const tensor::descriptor &src_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &bias_desc,
        const tensor::descriptor &dst_desc,
        const tensor::dims strides,
        const tensor::dims dilates,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        const attr_t attr = attr_t(),
        algorithm aalgorithm = algorithm::convolution_direct,
        prop_kind aprop_kind = prop_kind::forward,
        const padding_kind apadding_kind = padding_kind::zero) {
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
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
        &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
          "could not create a convolution forward primitive descriptor");
      reset(result);
      create_reorder_pds({src_desc, weights_desc});
    }
    descriptor(const tensor::descriptor &src_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &dst_desc,
        const tensor::dims strides,
        const tensor::dims dilates,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        const attr_t attr = attr_t(),
        algorithm aalgorithm = algorithm::convolution_direct,
        prop_kind aprop_kind = prop_kind::forward,
        const padding_kind apadding_kind = padding_kind::zero) {
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
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
        &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
        "could not create a convolution forward primitive descriptor");

      reset(result);
      create_reorder_pds({src_desc, weights_desc});
    }
  };

 public:
  using computation::expected_input_descriptor;
  using computation::expected_dst_descriptor;
  using computation::expected_weights_descriptor;

  template <typename T, typename ...Ts,
           typename = typename std::enable_if<
             std::is_same<T, tensor::descriptor>::value>::type>
  void init(const tensor::descriptor &src_desc,
      const tensor::descriptor &weights_desc,
      const tensor::descriptor &bias, const T &dst, Ts&&... args) {
    descriptor forward_descriptor(
        src_desc, weights_desc, bias, dst, std::forward<Ts>(args)...);

    computation::init(forward_descriptor, src_desc, weights_desc, bias);
  }

  template <typename T, typename ...Ts,
           typename  = typename std::enable_if<
             std::is_same<T, tensor::dims>::value>::type>
  void init(const tensor::descriptor &src_desc,
      const tensor::descriptor &weights_desc,
      const tensor::descriptor &dst, const T something,
      Ts&&... args) {
    descriptor forward_descriptor(src_desc, weights_desc, dst,
        something, std::forward<Ts>(args)...);

    computation::init(forward_descriptor, src_desc, weights_desc);
  }

  convolution_forward() = default;

  template <typename T, typename ...Ts>
  convolution_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& weights, const tensor& dst) {
    computation::execute(src, weights, dst);
  }

  void execute(const tensor& src, const tensor& weights, const tensor& bias,
      const tensor& dst) {
    computation::execute(src, weights, bias, dst);
  }

  template <class alloc, typename ...Ts>
  static void compute_impl(const tensor& src,
      const tensor& weights, const tensor& bias,
      const tensor::dims& dst_dims, tensor& dst, Ts&&... args) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        weights.get_dims(), bias.get_dims(), dst_dims, args...);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        weights.get_descriptor(), bias.get_descriptor(),
        tensor::descriptor {dst_dims, src.get_data_type()},
        std::forward<Ts>(args)...);

    // XXX: Performance evaluation
    // TODO: Custom allocator support
    auto src_in = src;
    auto weights_in = weights;
    if (src.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<alloc, convolution_forward>(
          comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }
    if (weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<alloc, convolution_forward>(
          comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    auto dst_desc = comp.expected_dst_descriptor();
    dst.reinit<alloc, convolution_forward>(std::move(dst_desc));
    comp.execute(src_in, weights_in, bias, dst);
  }

  template <class alloc, typename ...Ts>
  static void compute_impl(const tensor& src,
      const tensor& weights, const tensor::dims& dst_dims,
      tensor& dst, Ts&&... args) {
    tensor::descriptor result_desc(dst_dims, src.get_data_type());
    std::string key = utils::create_key(src.get_data_type(), src.get_dims(),
        weights.get_dims(), dst_dims, args...);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        weights.get_descriptor(), result_desc, std::forward<Ts>(args)...);

    // Performance evaluation
    auto src_in = src;
    auto weights_in = weights;

    // TODO: cut duplicated function call
    if (src.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<alloc, convolution_forward>(
          comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }
    if (weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<alloc, convolution_forward>(
          comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    auto dst_desc = comp.expected_dst_descriptor();
    dst.reinit<alloc, convolution_forward>(std::move(dst_desc));
    comp.execute(src_in, weights_in, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, const tensor& weights,
      const tensor::dims result_dims, tensor& dst, const tensor::dims strides,
      const tensor::dims dilates, const tensor::dims padding_l,
      const tensor::dims padding_r,
      const descriptor::attr_t attr = descriptor::attr_t(),
      algorithm aalogorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const padding_kind appading_kind = padding_kind::zero) {
    compute_impl<alloc>(src, weights, result_dims, dst, strides,
        dilates, padding_l, padding_r,
        attr, aalogorithm, aprop_kind, appading_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, const tensor& weights,
      const tensor& bias, const tensor::dims result_dims,
      tensor& dst, const tensor::dims strides,
      const tensor::dims dilates, const tensor::dims padding_l,
      const tensor::dims padding_r,
      const descriptor::attr_t attr = descriptor::attr_t(),
      algorithm aalogorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const padding_kind appading_kind = padding_kind::zero) {
    compute_impl<alloc>(src, weights, bias, result_dims, dst, strides,
        dilates, padding_l, padding_r,
        attr, aalogorithm, aprop_kind, appading_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, const tensor& weights,
      const tensor::dims result_dims, tensor& dst, const tensor::dims strides,
      const tensor::dims dilates, const tensor::dims padding_l,
      const tensor::dims padding_r, const int group,
      const descriptor::attr_t attr = descriptor::attr_t(),
      algorithm aalogorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const padding_kind appading_kind = padding_kind::zero) {

    auto weights_in = weights;
    if (group > 1) {
      auto gw_dims = weights.get_dims();
      gw_dims.insert(gw_dims.begin(), group);
      gw_dims[1] = gw_dims[1] / group;
      weights_in.init(
          tensor::descriptor {std::move(gw_dims), weights.get_data_type()},
          weights.get_data_handle());
    }

    if (dilates.empty() ||
        IDEEP_STD_EQUAL(dilates, 1) ||
        IDEEP_STD_EQUAL(dilates, 0)) {
      compute_impl<alloc>(src, weights_in, result_dims, dst, strides,
          padding_l, padding_r, attr, aalogorithm, aprop_kind, appading_kind);
    } else {
      compute_impl<alloc>(src, weights_in, result_dims, dst, strides,
          dilates, padding_l, padding_r,
          attr, aalogorithm, aprop_kind, appading_kind);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, const tensor& weights,
      const tensor& bias, const tensor::dims result_dims,
      tensor& dst, const tensor::dims strides,
      const tensor::dims dilates, const tensor::dims padding_l,
      const tensor::dims padding_r, const int group,
      const descriptor::attr_t attr = descriptor::attr_t(),
      algorithm aalogorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const padding_kind appading_kind = padding_kind::zero) {

    auto weights_in = weights;
    if (group > 1) {
      auto gw_dims = weights.get_dims();
      gw_dims.insert(gw_dims.begin(), group);
      gw_dims[1] = gw_dims[1] / group;
      weights_in.init(
          tensor::descriptor {std::move(gw_dims), weights.get_data_type()},
          weights.get_data_handle());
    }

    if (dilates.empty() ||
        IDEEP_STD_EQUAL(dilates, 1) ||
        IDEEP_STD_EQUAL(dilates, 0)) {
      compute_impl<alloc>(src, weights_in, bias, result_dims, dst, strides,
          padding_l, padding_r, attr, aalogorithm, aprop_kind, appading_kind);
    } else {
      compute_impl<alloc>(src, weights_in, bias, result_dims, dst, strides,
          dilates, padding_l, padding_r,
          attr, aalogorithm, aprop_kind, appading_kind);
    }
  }

  static tensor::descriptor expected_weights_descriptor(
      const tensor::dims weights_dims,
      tensor::data_type dtype = tensor::data_type::f32) {
    // Construct a dummy case
    auto it = weights_dims.end();
    auto o = *(it - 4), i = *(it - 3), h = *(it - 2), w = *(it - 1);
    auto ndims = weights_dims.size();
    auto g = ndims == 5 ? *(it - 5) : 1;
    tensor::dims x_dims = { 1, i * g, h, w};
    tensor::dims y_dims = { 1, o * g, 1, 1};
    tensor::descriptor x_desc(x_dims, dtype, format::nchw);
    tensor::descriptor y_desc(y_dims, dtype, format::nchw);
    tensor::descriptor weights_desc(weights_dims, dtype,
        ndims == 5 ? format::goihw : format::oihw);

    //  Because of:
    //        && implication(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
    //             || (jcp.stride_w == 1 && jcp.stride_h == 1))
    //  We guess conservatively

    if ( w > 7 ) {
      return weights_desc;
    }

    tensor::dims strides = {1, 1};
    tensor::dims dilates = {0, 0};
    tensor::dims padding_l = {0, 0};
    tensor::dims padding_r = {0, 0};

    convolution_forward comp(x_desc, weights_desc, y_desc,
        std::move(strides), std::move(dilates), std::move(padding_l),
        std::move(padding_r));

    return comp.expected_weights_descriptor();
  }
};

struct convolution_backward_data : public computation,
  public utils::computation_cache<convolution_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &grady_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &gradx_desc,
        const tensor::dims strides,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        const padding_kind apadding_kind = padding_kind::zero)
      : hint_(gradx_desc, weights_desc, grady_desc,
          strides, padding_l, padding_r)  {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_memory_desc_t diff_src_any = gradx_desc.format_any();
      mkldnn_memory_desc_t weights_any = weights_desc.format_any();
      mkldnn_memory_desc_t diff_dst_any = grady_desc.format_any();

      mkldnn_convolution_desc_t data;
      error::wrap_c_api(mkldnn_convolution_backward_data_desc_init(&data,
            convert_to_c(aalgorithm), &diff_src_any,
            &weights_any, &diff_dst_any,
            &strides[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward data descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(&result,
            &data, engine::cpu_engine().get(), hint_.get()),
      "could not create a convolution backward data primitive descriptor");
      reset(result);
      create_reorder_pds({grady_desc, weights_desc});
    }

    descriptor(const tensor::descriptor &grady_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &gradx_desc,
        const tensor::dims strides,
        const tensor::dims dilates,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        const padding_kind apadding_kind = padding_kind::zero)
      : hint_(gradx_desc, weights_desc, grady_desc,
          strides, dilates, padding_l, padding_r)  {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(dilates);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t diff_src_any = gradx_desc.format_any();
      mkldnn_memory_desc_t weights_any = weights_desc.format_any();
      mkldnn_memory_desc_t diff_dst_any = grady_desc.format_any();
      error::wrap_c_api(mkldnn_dilated_convolution_backward_data_desc_init(
            &data, convert_to_c(aalgorithm), &diff_src_any,
            &weights_any, &diff_dst_any, &strides[0], &dilates[0],
            &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward data descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(&result,
            &data, engine::cpu_engine().get(), hint_.get()),
      "could not create a convolution backward data primitive descriptor");
      reset(result);
      create_reorder_pds({grady_desc, weights_desc});
    }
  private:
    convolution_forward::descriptor hint_;
  };
public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template<typename ...Ts>
  void init(const tensor::descriptor &grady_desc,
      const tensor::descriptor &weights_desc,
      const tensor::descriptor &gradx_desc, Ts&&... args) {
    descriptor backward_data_descriptor(grady_desc, weights_desc,
        gradx_desc, std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor, grady_desc, weights_desc);
  }

  convolution_backward_data() = default;

  template <typename T, typename ...Ts>
  convolution_backward_data (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& grady, const tensor& weights,
      const tensor& gradx) {
    computation::execute(grady, weights, gradx);
  }

  template <class alloc, typename ...Ts>
  static void compute_impl(const tensor& grady, const tensor& weights,
      const tensor::dims& gradx_dims, tensor& gradx, Ts&&... args) {
    tensor::descriptor result_desc(gradx_dims, grady.get_data_type());
    auto key = utils::create_key(grady.get_data_type(), grady.get_dims(),
        weights.get_dims(), gradx_dims, args...);

    auto comp = fetch_or_create_m(key, grady.get_descriptor(),
        weights.get_descriptor(), result_desc, std::forward<Ts>(args)...);

    // XXX: Performance evaluation
    // TODO: Custom allocator support
    auto grady_in = grady;
    auto weights_in = weights;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<alloc, convolution_backward_data>(
          comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }
    if (weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<alloc, convolution_backward_data>(
          comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    gradx.reinit<alloc, convolution_backward_data>(
        comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& weights,
      const tensor::dims& gradx_dims, tensor& gradx, const tensor::dims strides,
      const tensor::dims dilates, const tensor::dims padding_l,
      const tensor::dims padding_r,
      algorithm aalgorithm = algorithm::convolution_direct,
      const padding_kind apadding_kind = padding_kind::zero) {
    compute_impl<alloc>(grady, weights, gradx_dims, gradx, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& weights,
      const tensor::dims& gradx_dims, tensor& gradx, const tensor::dims strides,
      const tensor::dims dilates, const tensor::dims padding_l,
      const tensor::dims padding_r, const int group,
      algorithm aalgorithm = algorithm::convolution_direct,
      const padding_kind apadding_kind = padding_kind::zero) {

    auto weights_in = weights;
    if (group > 1) {
      auto gw_dims = weights.get_dims();
      gw_dims.insert(gw_dims.begin(), group);
      gw_dims[1] = gw_dims[1] / group;
      weights_in.init(
          tensor::descriptor {std::move(gw_dims), weights.get_data_type()},
          weights.get_data_handle());
    }

    if (dilates.empty() ||
        IDEEP_STD_EQUAL(dilates, 1) ||
        IDEEP_STD_EQUAL(dilates, 0)) {
      compute_impl<alloc>(grady, weights_in, gradx_dims, gradx, strides,
          padding_l, padding_r, aalgorithm, apadding_kind);
    } else {
      compute_impl<alloc>(grady, weights_in, gradx_dims, gradx, strides,
          dilates, padding_l, padding_r, aalgorithm, apadding_kind);
    }
  }
};

struct convolution_backward_weights : public computation,
  public utils::computation_cache<convolution_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &grady_desc,
        const tensor::descriptor &gradw_desc,
        const tensor::descriptor &gradb_desc,
        const tensor::dims strides,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        const padding_kind apadding_kind = padding_kind::zero)
      : hint_(x_desc, gradw_desc, gradb_desc,
          grady_desc, strides, padding_l, padding_r) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t src_any = x_desc.format_any();
      mkldnn_memory_desc_t diff_weights_any = gradw_desc.format_any();
      mkldnn_memory_desc_t diff_bias_any = gradb_desc.format_any();
      mkldnn_memory_desc_t diff_dst_any = grady_desc.format_any();

      error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any,
            &diff_weights_any, &diff_bias_any,
            &diff_dst_any, &strides[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward weights primitive descriptor");
      reset(result);
      create_reorder_pds({x_desc, grady_desc});
    }
    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &grady_desc,
        const tensor::descriptor &gradw_desc,
        const tensor::dims strides,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        const padding_kind apadding_kind = padding_kind::zero)
      : hint_(x_desc, gradw_desc, grady_desc, strides, padding_l, padding_r) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t src_any = x_desc.format_any();
      mkldnn_memory_desc_t diff_weights_any = gradw_desc.format_any();
      mkldnn_memory_desc_t diff_dst_any = grady_desc.format_any();
      error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any,
            &diff_weights_any, nullptr, &diff_dst_any,
            &strides[0], &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward weights primitive descriptor");
      reset(result);
      create_reorder_pds({x_desc, grady_desc});
    }
    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &grady_desc,
        const tensor::descriptor &gradw_desc,
        const tensor::descriptor &gradb_desc,
        const tensor::dims strides,
        const tensor::dims dilates,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        const padding_kind apadding_kind = padding_kind::zero)
      : hint_(x_desc, gradw_desc, gradb_desc, grady_desc,
          strides, dilates, padding_l, padding_r) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(dilates);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t src_any = x_desc.format_any();
      mkldnn_memory_desc_t diff_weights_any = gradw_desc.format_any();
      mkldnn_memory_desc_t diff_bias_any = gradb_desc.format_any();
      mkldnn_memory_desc_t diff_dst_any = grady_desc.format_any();
      error::wrap_c_api(
          mkldnn_dilated_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any,
            &diff_weights_any, &diff_bias_any,
            &diff_dst_any, &strides[0], &dilates[0],
            &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward weights primitive descriptor");
      reset(result);
      create_reorder_pds({x_desc, grady_desc});
    }
    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &grady_desc,
        const tensor::descriptor &gradw_desc,
        const tensor::dims strides,
        const tensor::dims dilates,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm = algorithm::convolution_direct,
        const padding_kind apadding_kind = padding_kind::zero)
      :hint_(x_desc, gradw_desc, grady_desc,
          strides, dilates, padding_l, padding_r) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(dilates);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      mkldnn_convolution_desc_t data;
      mkldnn_memory_desc_t src_any = x_desc.format_any();
      mkldnn_memory_desc_t diff_weights_any = gradw_desc.format_any();
      mkldnn_memory_desc_t diff_dst_any = grady_desc.format_any();
      error::wrap_c_api(
          mkldnn_dilated_convolution_backward_weights_desc_init(
            &data, convert_to_c(aalgorithm), &src_any,
            &diff_weights_any, nullptr, &diff_dst_any,
            &strides[0], &dilates[0],  &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not create a convolution backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a convolution backward weights primitive descriptor");
      reset(result);
      create_reorder_pds({x_desc, grady_desc});
    }
  private:
    convolution_forward::descriptor hint_;
  };
public:
  using computation::expected_gradw_descriptor;
  using computation::expected_gradb_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &x_desc,
      const tensor::descriptor &grady_desc,
      const tensor::descriptor &gradw_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, grady_desc, gradw_desc,
        std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor, x_desc, grady_desc);
  }

  convolution_backward_weights() = default;

  template <typename T, typename ...Ts>
  convolution_backward_weights (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& grady, const tensor& gradw,
      const tensor& grad_bias) {
    computation::execute(src, grady, gradw, grad_bias);
  }

  void execute(const tensor& src, const tensor& grady, const tensor& gradw) {
    computation::execute(src, grady, gradw);
  }

  /*
   * This interface require MKL-DNN fixed beyoned
   * https://github.com/intel/mkl-dnn/commit/86f152b614c947b87633062a182c57775856a348
   */
  template <class alloc, typename ...Ts>
  static void compute_impl(const tensor& src, const tensor& grady,
      const tensor::dims& gradw_dims, tensor& gradw, tensor& gbias,
      Ts&&... args) {
    tensor::descriptor gradw_desc(gradw_dims, src.get_data_type());
    tensor::descriptor gradb_desc(
        tensor::dims {grady.get_dim(1)}, src.get_data_type());

    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        grady.get_dims(), gradw_dims, grady.get_dim(1), args...);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        grady.get_descriptor(), gradw_desc, gradb_desc,
        std::forward<Ts>(args)...);

    // XXX: Performance evaluation
    // TODO: Custom allocator support
    auto src_in = src;
    auto grady_in = grady;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<alloc, convolution_backward_weights>(
          comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<alloc, convolution_backward_weights>(
          comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<alloc, convolution_backward_weights>(
        comp.expected_gradw_descriptor());
    gbias.reinit<alloc, convolution_backward_weights>(
        comp.expected_gradb_descriptor());
    comp.execute(src_in, grady_in, gradw, gbias);
  }

  template <class alloc, typename ...Ts>
  static void compute_impl(const tensor& src, const tensor& grady,
      const tensor::dims& gradw_dims, tensor& gradw, Ts&&... args) {
    tensor::descriptor gradw_desc(gradw_dims, src.get_data_type());

    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        grady.get_dims(), gradw_dims, args...);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        grady.get_descriptor(), gradw_desc, std::forward<Ts>(args)...);

    // XXX: Performance evaluation
    // TODO: Custom allocator support
    auto src_in = src;
    auto grady_in = grady;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<alloc, convolution_backward_weights>(
          comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<alloc, convolution_backward_weights>(
          comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<alloc, convolution_backward_weights>(
        comp.expected_gradw_descriptor());
    comp.execute(src_in, grady_in, gradw);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& grady,
      const tensor::dims& gradw_dims, tensor& gradw,
      const tensor::dims strides, const tensor::dims dilates,
      const tensor::dims padding_l, const tensor::dims padding_r,
      algorithm aalgorithm = algorithm::convolution_direct,
      const padding_kind apadding_kind = padding_kind::zero) {
    compute_impl<alloc>(src, grady, gradw_dims, gradw, strides,
        dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src,
      const tensor& grady, const tensor::dims& gradw_dims, tensor& gradw,
      tensor& gradb, const tensor::dims strides, const tensor::dims dilates,
      const tensor::dims padding_l, const tensor::dims padding_r,
      algorithm aalgorithm = algorithm::convolution_direct,
      const padding_kind apadding_kind = padding_kind::zero) {
    compute_impl<alloc>(src, grady, gradw_dims, gradw, gradb,
        strides, dilates, padding_l, padding_r, aalgorithm, apadding_kind);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& grady,
      const tensor::dims& gradw_dims, tensor& gradw,
      const tensor::dims strides, const tensor::dims dilates,
      const tensor::dims padding_l, const tensor::dims padding_r,
      const int group, algorithm aalgorithm = algorithm::convolution_direct,
      const padding_kind apadding_kind = padding_kind::zero) {

    auto gw_dims_in = gradw_dims;
    if (group > 1) {
      gw_dims_in.insert(gw_dims_in.begin(), group);
      gw_dims_in[1] /= group;
    }

    if (dilates.empty() ||
        IDEEP_STD_EQUAL(dilates, 1) ||
        IDEEP_STD_EQUAL(dilates, 0)) {
      compute_impl<alloc>(src, grady, gw_dims_in, gradw, strides,
          padding_l, padding_r, aalgorithm, apadding_kind);
    } else {
      compute_impl<alloc>(src, grady, gw_dims_in, gradw, strides,
          dilates, padding_l, padding_r, aalgorithm, apadding_kind);
    }

    if (group > 1) {
      IDEEP_ENFORCE(group == gradw.get_dim(0),
          "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims[0] == group * gradw.get_dim(1),
          "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims.size() == gradw.ndims() - 1,
          "invalid ndim in grouped gradw");
      gradw.reshape(gradw_dims);
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src,
      const tensor& grady, const tensor::dims& gradw_dims, tensor& gradw,
      tensor& gradb, const tensor::dims strides, const tensor::dims dilates,
      const tensor::dims padding_l, const tensor::dims padding_r,
      const int group, algorithm aalgorithm = algorithm::convolution_direct,
      const padding_kind apadding_kind = padding_kind::zero) {

    auto gw_dims_in = gradw_dims;
    if (group > 1) {
      gw_dims_in.insert(gw_dims_in.begin(), group);
      gw_dims_in[1] /= group;
    }

    if (dilates.empty() ||
        IDEEP_STD_EQUAL(dilates, 1) ||
        IDEEP_STD_EQUAL(dilates, 0)) {
      compute_impl<alloc>(src, grady, gw_dims_in, gradw, gradb,
          strides, padding_l, padding_r, aalgorithm, apadding_kind);
    } else {
      compute_impl<alloc>(src, grady, gw_dims_in, gradw, gradb,
          strides, dilates, padding_l, padding_r, aalgorithm, apadding_kind);
    }

    if (group > 1) {
      IDEEP_ENFORCE(group == gradw.get_dim(0),
          "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims[0] == group * gradw.get_dim(1),
          "invalid dim 0 in grouped gradw");
      IDEEP_ENFORCE(gradw_dims.size() == gradw.ndims() - 1,
          "invalid ndim in grouped gradw");
      gradw.reshape(gradw_dims);
    }
  }
};

struct lrn_forward : public computation,
  public utils::computation_cache<lrn_forward> {
  struct descriptor : public descriptor_group {
    descriptor (const tensor::descriptor &x_desc,
        int local_size, float alpha, float beta, float k = 1.0,
        algorithm aalgorithm = algorithm::lrn_across_channels,
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_lrn_desc_t data;
      auto src_data = x_desc.get_mkldnn_memory_desc_t();
      error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data,
          mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
          src_data, local_size, alpha, beta, k),
          "could not create a lrn forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
              &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a lrn forward primitive descriptor");
      reset(result);
      // create_reorder_pds({x_desc});
    }
  };
public:
  using computation::expected_dst_descriptor;
  using computation::expected_workspace_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &x_desc, Ts&&... args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  lrn_forward() = default;

  template <typename T, typename ...Ts>
  lrn_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor &src, const tensor& dst, const tensor& workspace) {
    computation::execute(src, dst, workspace);
  }

  void execute(const tensor &src, tensor& dst) {
    if (dst.has_extra())
      computation::execute(src, dst, *dst.get_extra());
    else
      computation::execute(src, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, tensor& dst, int local_size,
      float alpha, float beta, float k = 1.0,
      algorithm aalgorithm = algorithm::lrn_across_channels,
      prop_kind aprop_kind = prop_kind::forward_training) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), local_size, alpha, beta, k,
        aalgorithm, aprop_kind);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        local_size, alpha, beta, k, aalgorithm, aprop_kind);

    bool with_workspace = aprop_kind == prop_kind::forward_training;

    if (dst != src) { // not inplace
      dst.reinit<alloc, lrn_forward>(comp.expected_dst_descriptor());
      if (with_workspace)
        dst.init_extra<alloc, lrn_forward>(
            comp.expected_workspace_descriptor());
    }

    comp.execute(src, dst);
  }
};

struct lrn_backward : public computation,
 public utils::computation_cache<lrn_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &gx_desc,
        int local_size, float alpha, float beta, float k = 1.0,
        algorithm aalgorithm = algorithm::lrn_across_channels)
      : hint_(x_desc, local_size, alpha, beta, k, aalgorithm) {
      mkldnn_lrn_desc_t data;
      error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
            convert_to_c(aalgorithm), gx_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(), local_size, alpha, beta, k),
          "could not create a lrn backward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(),
            hint_.get()),
          "could not create a backward lrn primitive descriptor");
      reset(result);
    }

  private:
    lrn_forward::descriptor hint_;
  };
public:
  using computation::expected_gradx_descriptor;

  template<typename ...Ts>
  void init(const tensor::descriptor &x_desc,
      const tensor::descriptor &grady_desc, Ts&&... args) {
    descriptor backward_data_descriptor(x_desc, grady_desc,
        std::forward<Ts>(args)...);
    computation::init(backward_data_descriptor, x_desc, grady_desc);
  }

  lrn_backward() = default;

  template<typename T, typename ...Ts>
  lrn_backward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& y,
      const tensor& gradx) {
    if (num_of_inputs() == 2)
      computation::execute(x, grady, gradx);
    else
      computation::execute(x, grady, *y.get_extra(), gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& x, const tensor& grady, const tensor& y,
      tensor& gradx, int local_size, float alpha, float beta, float k = 1.0,
      algorithm aalgorithm = algorithm::lrn_across_channels) {
    auto key = utils::create_key(x.get_data_type(), x.get_dims(),
        x.get_internal_format(), local_size, alpha, beta, k, aalgorithm);

    auto comp = fetch_or_create_m(key, x.get_descriptor(),
        grady.get_descriptor(), local_size, alpha, beta, k, aalgorithm);

    gradx.reinit<alloc, lrn_backward>(comp.expected_gradx_descriptor());
    comp.execute(x, grady, y, gradx);
  }
};

struct pooling_forward : public computation,
  public utils::computation_cache<pooling_forward> {
  struct descriptor : descriptor_group {
    descriptor() = default;
    descriptor(
        const tensor::descriptor &x_desc,
        const tensor::descriptor &y_desc,
        const tensor::dims strides,
        const tensor::dims kernel,
        const tensor::dims padding_l,
        const tensor::dims padding_r,
        algorithm aalgorithm,
        prop_kind aprop_kind = prop_kind::forward,
        const padding_kind apadding_kind = padding_kind::zero) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(kernel);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      auto src_data = x_desc.get_mkldnn_memory_desc_t();
      auto dst_data = y_desc.format_any();
      mkldnn_pooling_desc_t data;
      error::wrap_c_api(mkldnn_pooling_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind),
            convert_to_c(aalgorithm),
            src_data, &dst_data,
            &strides[0], &kernel[0],
            &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not init a forward pooling descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a forward pooling primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::expected_dst_descriptor;
  using computation::expected_workspace_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  pooling_forward() = default;

  template <typename T, typename ...Ts>
  pooling_forward(T arg, Ts &&...args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor &src, const tensor &dst, const tensor &workspace) {
    computation::execute(src, dst, workspace);
  }

  void execute(const tensor &src, tensor &dst) {
    if (dst.has_extra())
      computation::execute(src, dst, *dst.get_extra());
    else
      computation::execute(src, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src,
      const tensor::dims dst_dims, tensor& dst,
      const tensor::dims strides, const tensor::dims kernel,
      const tensor::dims padding_l, const tensor::dims padding_r,
      algorithm aalgorithm, prop_kind aprop_kind = prop_kind::forward,
      const padding_kind apadding_kind = padding_kind::zero) {
    tensor::descriptor dst_desc(dst_dims, src.get_data_type());
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), dst_dims, strides, kernel, padding_l,
        padding_r, aalgorithm, aprop_kind, apadding_kind);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        dst_desc, strides, kernel, padding_l, padding_r, aalgorithm,
        aprop_kind, apadding_kind);

    bool with_workspace = true
        && aprop_kind == prop_kind::forward_training
        && aalgorithm == mkldnn::pooling_max;

    if (dst != src) {
      dst.reinit<alloc, pooling_forward>(comp.expected_dst_descriptor());
      if (with_workspace)
        dst.init_extra<alloc, pooling_forward>(
            comp.expected_workspace_descriptor());
    }

    comp.execute(src, dst);
  }
};

struct pooling_backward : public computation,
  public utils::computation_cache<pooling_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &gradx_desc,
            const tensor::descriptor &grady_desc,
            const tensor::dims &strides,
            const tensor::dims &kernel,
            const tensor::dims &padding_l,
            const tensor::dims &padding_r,
            algorithm aalgorithm,
            const padding_kind apadding_kind = padding_kind::zero)
      : hint_([&]() {
              mkldnn::memory::validate_dims(strides);
              mkldnn::memory::validate_dims(kernel);
              mkldnn::memory::validate_dims(padding_l);
              mkldnn::memory::validate_dims(padding_r);
              auto gradx_data = gradx_desc.format_any();
              mkldnn_pooling_desc_t data;
              error::wrap_c_api(mkldnn_pooling_forward_desc_init(&data,
                    mkldnn::convert_to_c(prop_kind::forward),
                    convert_to_c(aalgorithm),
                    &gradx_data, grady_desc.get_mkldnn_memory_desc_t(),
                    &strides[0], &kernel[0],
                    &padding_l[0], &padding_r[0],
                    mkldnn::convert_to_c(apadding_kind)),
                  "could not init a forward pooling descriptor");
              mkldnn_primitive_desc_t result;
              error::wrap_c_api(mkldnn_primitive_desc_create(
                    &result, &data, engine::cpu_engine().get(), nullptr),
                  "could not create a forward pooling primitive descriptor");

              pooling_forward::descriptor hint;
              hint.reset(result);
              return hint;
            } ()) {
      mkldnn::memory::validate_dims(strides);
      mkldnn::memory::validate_dims(kernel);
      mkldnn::memory::validate_dims(padding_l);
      mkldnn::memory::validate_dims(padding_r);
      auto gradx_data = gradx_desc.format_any();
      mkldnn_pooling_desc_t data;
      error::wrap_c_api(mkldnn_pooling_backward_desc_init(&data,
            convert_to_c(aalgorithm),
            &gradx_data, grady_desc.get_mkldnn_memory_desc_t(),
            &strides[0], &kernel[0],
            &padding_l[0], &padding_r[0],
            mkldnn::convert_to_c(apadding_kind)),
          "could not init a backward pooling descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
                  &result, &data, engine::cpu_engine().get(),
                  hint_.get()),
              "could not create a backward pooling primitive descriptor");
      reset(result);
    }
  private:
    pooling_forward::descriptor hint_;
  };
public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &gradx_desc,
      const tensor::descriptor &grady_desc, Ts &&...args) {
    descriptor backward_descriptor(gradx_desc, grady_desc,
        std::forward<Ts>(args)...);
    computation::init(backward_descriptor, grady_desc, gradx_desc);
  }

  pooling_backward() = default;

  template <typename T, typename ...Ts>
  pooling_backward(T arg, Ts &&...args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor &grady, const tensor &y, const tensor &gradx) {
    if (num_of_inputs() == 1)
      computation::execute(grady, gradx);
    else
      computation::execute(grady, *y.get_extra(), gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &grady, const tensor &y, const tensor& x,
      tensor& gradx, const tensor::dims strides, const tensor::dims kernel,
      const tensor::dims padding_l, const tensor::dims padding_r,
      algorithm aalgorithm, padding_kind apadding_kind = padding_kind::zero) {
    auto grady_in = grady;
    // y is a scalar sometimes
    if (y.ndims() > 0 &&
        grady.get_descriptor() != y.get_descriptor()) {
      grady_in.init<alloc, pooling_backward>(y.get_descriptor());
      reorder::compute(grady, grady_in);
    }

    auto key = utils::create_key(grady_in.get_data_type(), grady_in.get_dims(),
        grady_in.get_internal_format(), x.get_dims(), strides, kernel, padding_l,
        padding_r, aalgorithm, apadding_kind);

    auto comp = fetch_or_create_m(key, x.get_descriptor(),
        grady_in.get_descriptor(), strides, kernel, padding_l, padding_r,
        aalgorithm, apadding_kind);

    gradx.reinit<alloc, pooling_backward>(comp.expected_gradx_descriptor());
    comp.execute(grady, y, gradx);
  }
};

struct eltwise_forward : public computation,
  public utils::computation_cache<eltwise_forward> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &x_desc,
        float alpha = 0.0, float beta = 0.0,
        algorithm alg_kind = algorithm::eltwise_relu,
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_eltwise_desc_t data;
      error::wrap_c_api(mkldnn_eltwise_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind),
            mkldnn::convert_to_c(alg_kind),
            x_desc.get_mkldnn_memory_desc_t(),
            alpha, beta),
              "could not create a eltwise forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
                &result, &data, engine::cpu_engine().get(), nullptr)
        , "could not create a eltwise forward primitive descriptor");
      reset(result);
    }
  };

public:
  using computation::computation;
  using computation::expected_dst_descriptor;

  template<typename ...Ts>
  void init(const tensor::descriptor &x_desc, Ts &&...args) {
    descriptor forward_descriptor(x_desc, std::forward<Ts>(args)...);
    computation::init(forward_descriptor, x_desc);
  }

  eltwise_forward() = default;

  template <typename T, typename ...Ts>
  eltwise_forward(T arg, Ts &&...args) {
    init(std::forward<T>(arg), std::forward<Ts>(args)...);
  }

  void execute(const tensor &x, const tensor &y) {
    computation::execute(x, y);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, tensor& dst,
      algorithm aalogorithm = algorithm::eltwise_relu,
      prop_kind aprop_kind = prop_kind::forward,
      float alpha = 0.0, float beta = 0.0) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), alpha, beta, aalogorithm, aprop_kind);

    auto comp = fetch_or_create_m(key, src.get_descriptor()
        , alpha, beta, aalogorithm, aprop_kind);

    if (dst != src)
      dst.reinit<alloc, eltwise_forward>(src.get_descriptor());
    comp.execute(src, dst);
  }
};

struct eltwise_backward : public computation,
  public utils::computation_cache<eltwise_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &grady_desc,
        const tensor::descriptor &x_desc,
        float alpha = 0.0, float beta = 0.0,
        algorithm alg_kind = algorithm::eltwise_relu)
      : hint_(x_desc, alg_kind) {
      mkldnn_eltwise_desc_t data;
      error::wrap_c_api(mkldnn_eltwise_backward_desc_init(&data,
            mkldnn::convert_to_c(alg_kind),
            grady_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(),
            static_cast<float>(alpha),
            static_cast<float>(beta)),
          "could not create a eltwise backward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), hint_.get()),
          "could not create a eltwise backward primitive descriptor");
      reset(result);
    }
  private:
    eltwise_forward::descriptor hint_;
  };
public:
  using computation::computation;
  using computation::expected_gradx_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &grady_desc,
      const tensor::descriptor &x_desc, Ts &&...args) {
    descriptor backward_descriptor(
        grady_desc, x_desc, std::forward<Ts>(args)...);
    computation::init(backward_descriptor, grady_desc, x_desc);
  }

  eltwise_backward() = default;

  template <typename T, typename ...Ts>
  eltwise_backward(T grady_desc, T src_desc, Ts &&...args) {
    init(std::forward<T>(grady_desc), std::forward<T>(src_desc),
        std::forward<Ts>(args)...);
  }

  void execute(const tensor &x, const tensor &grady, const tensor &gradx) {
    computation::execute(x, grady, gradx);
  }

  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  template<class alloc, typename ...Ts>
  static void compute_impl(const tensor &src, const tensor &grady,
      tensor& gradx, Ts &&...args) {
    // if grady is from outside, make it ours
    tensor grady_in = grady;
    if (grady.get_internal_format() != src.get_internal_format()) {
      grady_in.init<alloc, eltwise_backward>(src.get_descriptor());
      reorder::compute(grady, grady_in);
    }

    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), args...);

    auto comp = fetch_or_create_m(key, grady_in.get_descriptor(),
        src.get_descriptor(), std::forward<Ts>(args)...);

    gradx.reinit<alloc, eltwise_backward>(comp.expected_gradx_descriptor());
    comp.execute(src, grady_in, gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, const tensor &grady,
      tensor& gradx, algorithm aalogorithm = algorithm::eltwise_relu,
      float alpha = 0.0, float beta = 0.0) {
    compute_impl<alloc>(src, grady, gradx, alpha, beta, aalogorithm);
  }
};

struct sum : public computation,
  public utils::computation_cache<sum> {
  struct descriptor : public descriptor_group {
    descriptor(const std::vector<float> &scales,
        const std::vector<tensor::descriptor> &inputs) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_sum_primitive_desc_create(
              &result, nullptr,
              (int)c_api_inputs.size(),
              &scales[0], &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }

    descriptor(const std::vector<float> &scales,
        const std::vector<tensor::descriptor> &inputs,
        const tensor::descriptor output_desc) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_sum_primitive_desc_create(
              &result, output_desc.get_mkldnn_memory_desc_t(),
              (int)c_api_inputs.size(),
              &scales[0], &c_api_inputs[0]),
          "could not create a sum primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::execute;
  using computation::expected_dst_descriptor;

  void init(const std::vector<float> &scales,
      const std::vector<tensor::descriptor> &inputs) {
    descriptor forward_descriptor(scales, inputs);
    computation::init(forward_descriptor, inputs);
  }

  void init(const std::vector<float> &scales,
      const std::vector<tensor::descriptor> &inputs,
      const tensor::descriptor output) {
    descriptor forward_descriptor(scales, inputs, output);
    computation::init(forward_descriptor, inputs);
  }

  sum() = default;

  sum(const std::vector<float> &scales,
      const std::vector<tensor::descriptor> &inputs_desc,
      const tensor::descriptor output_desc) {
    init(scales, inputs_desc, output_desc);
  }

  sum(const std::vector<float> &scales,
      const std::vector<tensor::descriptor> &inputs_desc) {
    init(scales, inputs_desc);
  }

  void execute(const std::vector<tensor> &inputs, const tensor &output) {
    computation::execute(inputs, output);
  }

  template<class alloc = utils::allocator>
  static void compute(const std::vector<float> &scales,
      const std::vector<tensor> &inputs, tensor& output) {
    std::vector<tensor::descriptor> inputs_desc;
    for_each(inputs.begin(), inputs.end(), [&inputs_desc](tensor in) {
        inputs_desc.push_back(in.get_descriptor());
        });

    if (output.get_dims().size() == 0) {
      sum comp(scales, inputs_desc);
      output.reinit<alloc, sum>(comp.expected_dst_descriptor());
      comp.execute(inputs, output);
    } else {
      sum comp(scales, inputs_desc, output.get_descriptor());
      comp.execute(inputs, output);
    }
  }
};

struct concat : public computation,
  public utils::computation_cache<concat> {
  struct descriptor : public descriptor_group {
    descriptor(int concat_dimension,
        const std::vector<tensor::descriptor> &inputs) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_concat_primitive_desc_create(
              &result, nullptr,
              (int)c_api_inputs.size(),
              concat_dimension, &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }
    descriptor(int concat_dimension,
        const std::vector<tensor::descriptor> &inputs,
        const tensor::descriptor out_desc) {
      mkldnn_primitive_desc_t result;
      auto c_api_inputs = cpp_to_c(inputs);
      error::wrap_c_api(mkldnn_concat_primitive_desc_create(
              &result, out_desc.get_mkldnn_memory_desc_t(),
              (int)c_api_inputs.size(),
              concat_dimension, &c_api_inputs[0]),
          "could not create a concat primitive descriptor");
      reset(result);
    }
  };
public:
  using computation::execute;
  using computation::expected_dst_descriptor;

  void init(int concat_dimension,
      const std::vector<tensor::descriptor> &inputs) {
    descriptor forward_descriptor (concat_dimension, inputs);
    computation::init(forward_descriptor, inputs);
  }

  concat() = default;

  concat(int concat_dimension,
      const std::vector<tensor::descriptor> &inputs) {
    init(concat_dimension, inputs);
  }

  void execute(const std::vector<tensor> &inputs, const tensor &output) {
    computation::execute(inputs, output);
  }

  template<class alloc = utils::allocator>
  static void compute(std::vector<tensor> &inputs, int axis, tensor& dst) {
    std::vector<tensor::descriptor> tdesc;
    std::vector<tensor::data_type> inputs_dt;
    std::vector<tensor::dims> inputs_dims;
    std::vector<format> inputs_format;
    for (tensor elems : inputs) {
      tdesc.push_back(elems.get_descriptor());
      inputs_dt.push_back(elems.get_data_type());
      inputs_dims.push_back(elems.get_dims());
      inputs_format.push_back(elems.get_internal_format());
    }

    auto key = utils::create_key(inputs_dt, inputs_dims, inputs_format, axis);

    // FIXME
    // currently align all inputs format with first one
    for (int i = 1; i <tdesc.size(); i++) {
      if (inputs_format[i] != inputs_format[0]) {
        auto src_in = inputs[i];
        src_in.init<alloc, concat>(
            {inputs_dims[i], inputs_dt[i], inputs_format[0]});
        reorder::compute(inputs[i], src_in);
        inputs[i] = std::move(src_in);
        tdesc[i] = inputs[i].get_descriptor();
      }
    }

    auto comp = fetch_or_create_m(key, axis, tdesc);
    dst.reinit<alloc, concat>(comp.expected_dst_descriptor());
    comp.execute(inputs, dst);
  }

  static std::vector<int32_t> compute(
      std::vector<tensor> &inputs, int axis, bool add_axis, tensor& dst) {
    IDEEP_ENFORCE(axis < (inputs[0].ndims() + (add_axis ? 1 : 0)),
        "invalid axis in concat");
    for (int i = 0; i <inputs[0].ndims(); i++) {
      if (i == axis && !add_axis) continue;
      for (unsigned j = 1; j <inputs.size(); j++) {
        IDEEP_ENFORCE(inputs[j].get_dim(i) == inputs[0].get_dim(i),
          "invalid input dims in concat");
      }
    }

    int32_t dst_channels = 0;
    std::vector<int32_t> axis_info(inputs.size(), 0);
    for (unsigned k = 0; k <inputs.size(); k++) {
      axis_info[k] = add_axis ? 1 : inputs[k].get_dim(axis);
      dst_channels += axis_info[k];
    }

    tensor::dims dst_dims(inputs[0].get_dims());
    if (add_axis)
      dst_dims.insert(dst_dims.begin() + axis, dst_channels);
    else
      dst_dims[axis] = dst_channels;

    reorder reorder_;
    tensor::dims offset_dims(dst_dims.size(), 0);
    if (add_axis)
      dst.reinit({dst_dims, inputs[0].get_data_type()});
    else
      dst.reinit({dst_dims, inputs[0].get_data_type(),
          inputs[0].get_internal_format()});
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (add_axis) {
        tensor::dims in_dims(inputs[i].get_dims());
        in_dims.insert(in_dims.begin() + axis, 1);
        tensor::descriptor in_desc(inputs[i].get_descriptor().reshape(in_dims));
        auto view = dst.create_view(in_dims, offset_dims);
        reorder_.init(in_desc, view, dst.get_descriptor());
        reorder_({in_desc, inputs[i].get_data_handle()}, dst);
      } else {
        auto view = dst.create_view(inputs[i].get_dims(), offset_dims);
        reorder_.init(inputs[i].get_descriptor(), view, dst.get_descriptor());
        reorder_(inputs[i], dst);
      }
      offset_dims[axis] += axis_info[i];
    }

    return axis_info;
  }
};

struct softmax_forward : public computation {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &x_desc, int softmax_axis,
        prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_softmax_desc_t data;
      error::wrap_c_api(mkldnn_softmax_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind), x_desc.get_mkldnn_memory_desc_t(),
            softmax_axis),
          "could not create a softmax forward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
              &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a softmax forward primitive descriptor");
      reset(result);
    }
  };

public:
  using computation::expected_dst_descriptor;

  template<typename ...Ts>
  void init(const tensor::descriptor& src_desc,
      const tensor::descriptor& dst_desc, Ts&&... args) {
    descriptor softmax_descriptor(src_desc, std::forward<Ts>(args)...);
    computation::init(softmax_descriptor, src_desc, dst_desc);
  }

  void execute(const tensor& src, const tensor& dst) {
    computation::execute(src, dst);
  }
};

struct batch_norm_forward_base : public computation {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &src_desc, float epsilon,
        unsigned flags, prop_kind aprop_kind) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(
          mkldnn_batch_normalization_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(),
            epsilon, flags),
          "could not create a batch normalization forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
          &result, &data, engine::cpu_engine().get(), nullptr),
      "could not create a batch normalization forward primitive descriptor");
      reset(result);
    }
    descriptor(const tensor::descriptor &src_desc, float epsilon, attr_t attr,
        unsigned flags, prop_kind aprop_kind) {
      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(
          mkldnn_batch_normalization_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind), src_desc.get_mkldnn_memory_desc_t(),
            epsilon, flags),
          "could not create a batch normalization forward descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create_v2(
          &result, &data, attr.get(), engine::cpu_engine().get(), nullptr),
      "could not create a batch normalization forward primitive descriptor");
      reset(result);
    }
  };

public:
  using computation::expected_dst_descriptor;

  template<typename... Ts>
  void init(float epsilon, unsigned flags, prop_kind aprop_kind,
      const tensor::descriptor &src_desc, Ts&... rest) {
    descriptor batch_norm_forward(src_desc, epsilon, flags, aprop_kind);
    init(batch_norm_forward, src_desc, rest...);
  }

  /// Execute interface for (1, 0) (stats_is_src, use_scaleshift)
  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& dst) {
    computation::execute(src, mean, variance, dst);
  }

  /// Execute interface for (1, 1)
  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& weights, const tensor& dst) {
    computation::execute(src, mean, variance, weights, dst);
  }
};

struct batch_normalization_forward_inference : public batch_norm_forward_base,
  public utils::computation_cache<batch_normalization_forward_inference> {
public:
  using batch_norm_forward_base::execute;

  /// Execute interface for  (0, 0)
  void execute(const tensor& src, const tensor& dst) {
    computation::execute(src, dst);
  }

  /// Execute interface for  (0, 1)
  void execute(const tensor& src, const tensor& weights, const tensor& dst) {
    computation::execute(src, weights, dst);
  }

public:
  void init(const tensor::descriptor& src_desc, float epsilon,
      unsigned flag = batch_normalization_flag::use_global_stats |
      batch_normalization_flag::use_scale_shift) {
    descriptor batch_norm_forward(
        src_desc, epsilon, flag, prop_kind::forward_scoring);
    weights_.init(batch_norm_forward.expected_weights_descriptor());
    computation::init(batch_norm_forward);
  }

  batch_normalization_forward_inference () = default;

  template <typename T, typename ...Ts>
  batch_normalization_forward_inference (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  /// More functionality in this interface
  void execute(const tensor& src, const tensor& scale, const tensor& shift,
      const tensor& dst) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(),
        scale.get_data_handle(), scale.get_size());
    std::memcpy((char *)weights_.get_data_handle() + scale.get_size(),
        shift.get_data_handle(), shift.get_size());
    computation::execute(src, weights_, dst);
  }

  /// More functionality in this interface
  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& scale, const tensor& shift, const tensor& dst) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(),
        scale.get_data_handle(), scale.get_size());
    std::memcpy((char *)weights_.get_data_handle() + scale.get_size(),
        shift.get_data_handle(), shift.get_size());
    computation::execute(src, mean, variance, weights_, dst);
  }

  using computation::expected_dst_descriptor;

  // Inplace support?
  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& scale,
      const tensor& shift, tensor& dst, float epsilon) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), 3, epsilon);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        batch_normalization_flag::use_scale_shift, epsilon);

    if (dst != src)
      dst.reinit<alloc, batch_normalization_forward_inference>(
          comp.expected_dst_descriptor());
    comp.execute(src, scale, shift, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& mean,
      const tensor& variance, const tensor& scale,
      const tensor& shift, tensor& dst, float epsilon) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), 5, epsilon);

    auto comp = fetch_or_create_m(key, src.get_descriptor(), epsilon);

    if (dst != src)
      dst.reinit<alloc, batch_normalization_forward_inference>(
          comp.expected_dst_descriptor());
    comp.execute(src, mean, variance, scale, shift, dst);
  }
private:
  param weights_;
};

struct batch_normalization_forward_training : public batch_norm_forward_base,
  public utils::computation_cache<batch_normalization_forward_training> {
  float get_epsilon() const {
    const mkldnn_batch_normalization_desc_t *p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d),
        0, (void *)&p_desc),
      "could not query batch normalization descriptor");
    return p_desc->batch_norm_epsilon;
  }
public:
  using batch_norm_forward_base::execute;

  void init(const tensor::descriptor& src_desc, const tensor::descriptor& scale,
      const tensor::descriptor& shift, float momentum, float epsilon,
      unsigned flags = batch_normalization_flag::use_scale_shift) {
    assert(scale.ndims() == 1 && shift.ndims() == 1);
    descriptor batch_norm_forward(src_desc, epsilon, flags,
        prop_kind::forward_training);
    computation::init(batch_norm_forward, src_desc);

    // We borrown scale and bias for the shape of mean and variance
    weights_.init(batch_norm_forward.expected_weights_descriptor());
    sum_.init({momentum, 1.f - momentum}, {scale, shift});
  }

  batch_normalization_forward_training () = default;

  template <typename T, typename... Ts>
  batch_normalization_forward_training (T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  /// Execute interface for (0, 0)
  void execute(const tensor& src, const tensor& dst, const tensor& mean,
      const tensor& variance) {
    computation::execute(src, dst, mean, variance);
  }

  /// Execute interface for (0, 1)
  void execute(const tensor& src, const tensor& weights, const tensor& dst,
      const tensor& mean, const tensor& variance) {
    computation::execute(src, weights, dst, mean, variance);
  }

  void execute(const tensor& src, const tensor& scale, const tensor& shift,
      const tensor& dst, const tensor& mean, const tensor& variance) {
    // Small amount of buffer, car is good
    std::memcpy(weights_.get_data_handle(),
        scale.get_data_handle(), scale.get_size());
    std::memcpy((char *)weights_.get_data_handle() + scale.get_size(),
        shift.get_data_handle(), shift.get_size());
    computation::execute(src, weights_, dst, mean, variance);
  }

  void running_statistic(const tensor& mean, const tensor& variance,
      const tensor& running_mean, const tensor& running_var) {
    // TODO: provide accelerated version
    std::vector<tensor> inputs_for_mean {running_mean, mean};
    std::vector<tensor> inputs_for_var {running_var, variance};
    sum_.execute(inputs_for_mean, running_mean);
    sum_.execute(inputs_for_var, running_var);
  }

  // TODO: deprecates these two
  tensor::descriptor expected_mean_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 1);
  }

  tensor::descriptor expected_variance_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 2);
  }

  // TODO: this is good one
  tensor::descriptor expected_statistic_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 1);
  }

  using computation::expected_dst_descriptor;

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& scale, const tensor& shift,
      tensor& dst, tensor& mean, tensor& variance,
      float momentum, float epsilon) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), epsilon);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        scale.get_descriptor(), shift.get_descriptor(), momentum, epsilon);

    dst.reinit<alloc, batch_normalization_forward_training>(
        comp.expected_dst_descriptor());
    mean.reinit(comp.expected_statistic_descriptor());
    variance.reinit(comp.expected_statistic_descriptor());

    comp.execute(src, scale, shift, dst, mean, variance);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& scale,
      const tensor& shift, tensor& dst, tensor& mean,
      tensor& variance, tensor& running_mean,
      tensor& running_var, float momentum, float epsilon) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), epsilon);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        scale.get_descriptor(), shift.get_descriptor(), momentum, epsilon);

    // TODO: Substitue running statistics calculation with lighter version
    dst.reinit<alloc, batch_normalization_forward_training>(
        comp.expected_dst_descriptor());
    mean.reinit(comp.expected_statistic_descriptor());
    variance.reinit(comp.expected_statistic_descriptor());
    running_mean.reinit(comp.expected_statistic_descriptor());
    running_var.reinit(comp.expected_statistic_descriptor());

    comp.execute(src, scale, shift, dst, mean, variance);
    comp.running_statistic(mean, variance, running_mean, running_var);
  }

private:
  param weights_;
  sum sum_;
};

struct batch_normalization_backward : public computation,
  public utils::computation_cache<batch_normalization_backward> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &gradx_desc,
        const tensor::descriptor &x_desc,
        float epsilon, unsigned flags, prop_kind aprop_kind)
      : hint_(x_desc, epsilon, flags, prop_kind::forward_training) {

      mkldnn_batch_normalization_desc_t data;
      error::wrap_c_api(
          mkldnn_batch_normalization_backward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind),
            gradx_desc.get_mkldnn_memory_desc_t(),
            x_desc.get_mkldnn_memory_desc_t(),
            static_cast<float>(epsilon), flags),
          "could not create a batch normalization backward descriptor");

      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(
          &result, &data, engine::cpu_engine().get(),
          hint_.get()),
        "could not create a batch normalization backward primitive descriptor");
      reset(result);
    }
  private:
    batch_normalization_forward_training::descriptor hint_;
  };

  float get_epsilon() const {
    const mkldnn_batch_normalization_desc_t *p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d),
        0, (void *)&p_desc),
      "could not query batch normalization descriptor");
    return p_desc->batch_norm_epsilon;
  }

public:
  using computation::expected_gradx_descriptor;
  tensor::descriptor expected_grad_scale_descriptor() const {
    return expected_descriptor_of(query::src_pd, 2);
  }
  tensor::descriptor expected_grad_shift_descriptor() const {
    return expected_descriptor_of(query::src_pd, 1);
  }
  tensor::descriptor expected_statistic_descriptor() const {
    return expected_descriptor_of(query::src_pd, 1);
  }

  prop_kind get_prop_kind() const {
    const mkldnn_batch_normalization_desc_t *p_desc;
    error::wrap_c_api(mkldnn_primitive_desc_query(get_mkldnn_primitive_desc_t(),
        static_cast<mkldnn_query_t>(query::batch_normalization_d),
        0, (void *)&p_desc),
      "could not query batch normalization descriptor");
    return static_cast<prop_kind>(p_desc->prop_kind);
  }

  void init(const tensor::descriptor& gradx_desc,
      const tensor::descriptor& src_desc, float epsilon,
      unsigned flags = batch_normalization_flag::use_scale_shift,
      prop_kind aprop_kind=prop_kind::backward) {
    descriptor batch_norm_backward(gradx_desc, src_desc, epsilon,
        flags, aprop_kind);
    computation::init(batch_norm_backward);
    weights_.init(batch_norm_backward.expected_weights_descriptor());
  }

  batch_normalization_backward() = default;

  template <typename T, typename ...Ts>
  batch_normalization_backward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx,
      const tensor& gradw) {
    // We can sure that only scale is matter at this place
    std::memcpy(
        weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx, gradw);
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx,
      const tensor& gradw, const tensor& grad_shift) {
    // protect API integraty, should we use solid check instead of assert?
    assert(get_prop_kind() == prop_kind::backward);
    // We can sure that only scale is matter at this place
    // And single thread of memcpy should be fast enough
    std::memcpy(
        weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx, gradw);
    std::memcpy(grad_shift.get_data_handle(),
        (char *)gradw.get_data_handle() + grad_shift.get_size(),
        grad_shift.get_size());
  }

  void execute(const tensor& src, const tensor& mean, const tensor& variance,
      const tensor& grady, const tensor& scale, const tensor& gradx) {
    assert(get_prop_kind() == prop_kind::backward_data);
    std::memcpy(
        weights_.get_data_handle(), scale.get_data_handle(), scale.get_size());
    computation::execute(src, mean, variance, grady, weights_, gradx);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& mean,
      const tensor& variance, const tensor& grady, const tensor& scale,
      tensor& gradx, tensor& gradw, float epsilon) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), epsilon);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        src.get_descriptor(), epsilon);

    gradx.reinit<alloc, batch_normalization_backward>(
        comp.expected_gradx_descriptor());
    gradw.reinit(comp.expected_gradw_descriptor());
    comp.execute(
        src, mean, variance, grady, scale, gradx, gradw);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& mean,
      const tensor& variance, const tensor& grady, const tensor& scale,
      tensor& gradx, tensor& grad_scale, tensor& grad_shift, float epsilon) {
    auto key = utils::create_key(src.get_data_type(), src.get_dims(),
        src.get_internal_format(), epsilon);

    auto comp = fetch_or_create_m(key, src.get_descriptor(),
        src.get_descriptor(), epsilon);

    gradx.reinit<alloc, batch_normalization_backward>(
        comp.expected_gradx_descriptor());
    grad_scale.reinit(comp.expected_gradw_descriptor());
    grad_shift.reinit(mean.get_descriptor());
    comp.execute(
        src, mean, variance, grady, scale, gradx, grad_scale, grad_shift);
    grad_scale.set_descriptor(mean.get_descriptor());
  }
private:
  tensor weights_;
};

struct inner_product_forward: public computation,
  public utils::computation_cache<inner_product_forward> {
  struct descriptor: public descriptor_group {
    descriptor(const tensor::descriptor &src_desc,
            const tensor::descriptor &weights_desc,
            const tensor::descriptor &bias_desc,
            const tensor::descriptor &dst_desc,
            prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_inner_product_desc_t data;
      mkldnn_memory_desc_t src_data = src_desc.format_any();
      mkldnn_memory_desc_t weights_data = weights_desc.format_any();
      mkldnn_memory_desc_t bias_data = bias_desc.format_any();
      mkldnn_memory_desc_t dst_data = dst_desc.format_any();

      error::wrap_c_api(
          mkldnn_inner_product_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind), &src_data, &weights_data,
            &bias_data, &dst_data),
          "could not create a inner product forward descriptor");

      mkldnn_primitive_desc_t result;

      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a inner product forward primitive descriptor");
      reset(result);
      create_reorder_pds({src_desc, weights_desc});
    }

    descriptor(const tensor::descriptor &src_desc,
            const tensor::descriptor &weights_desc,
            const tensor::descriptor &dst_desc,
            prop_kind aprop_kind = prop_kind::forward) {
      mkldnn_inner_product_desc_t data;
      mkldnn_memory_desc_t src_data = src_desc.format_any();
      mkldnn_memory_desc_t weights_data = weights_desc.format_any();
      mkldnn_memory_desc_t dst_data = dst_desc.format_any();

      error::wrap_c_api(
          mkldnn_inner_product_forward_desc_init(&data,
            mkldnn::convert_to_c(aprop_kind), &src_data, &weights_data,
            nullptr, &dst_data),
          "could not create a inner product forward descriptor");

      mkldnn_primitive_desc_t result;

      error::wrap_c_api(mkldnn_primitive_desc_create(
            &result, &data, engine::cpu_engine().get(), nullptr),
          "could not create a inner product forward primitive descriptor");
      reset(result);
      create_reorder_pds({src_desc, weights_desc});
    }
  };
 public:
  using computation::execute;
  using computation::expected_dst_descriptor;
  using computation::expected_weights_descriptor;
  using computation::expected_src_descriptor;

  void init(const tensor::descriptor &src_desc,
      const tensor::descriptor &weights_desc,
      const tensor::descriptor &dst_desc) {
    descriptor forward_descriptor(src_desc, weights_desc, dst_desc);
    computation::init(forward_descriptor, src_desc, weights_desc);
  }

  void init(const tensor::descriptor &src_desc,
      const tensor::descriptor &weights_desc,
      const tensor::descriptor &bias_desc,
      const tensor::descriptor &dst_desc) {
    descriptor forward_descriptor(
        src_desc, weights_desc, bias_desc, dst_desc);
    computation::init(forward_descriptor, src_desc, weights_desc, bias_desc);
  }

  inner_product_forward() = default;

  template <typename T, typename ...Ts>
  inner_product_forward(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& src, const tensor& weights,
      const tensor& bias, tensor& dst) {
    tensor src_in = src;

    if (src.ndims() == 4 && weights.ndims() == 2) {
      //
      // tricky procedure, we change src back to public format
      // then we using another tensor with 2 dimension to describe the buffer
      //
      tensor::dims new_dims { src.get_dim(0),
        src.get_dim(1) * src.get_dim(2) * src.get_dim(3) };

      if (src.is_public_format())
        src_in.init({new_dims, src.get_data_type(), format::nc}, src.get_data_handle());
      else {
        src_in.init<alloc, inner_product_forward>(
            {src.get_dims(), src.get_data_type(), format::nchw});
        reorder::compute(src, src_in);
        src_in.set_descriptor({new_dims, src.get_data_type(), format::nc});
      }
    }

    tensor::dims dst_dims = {src_in.get_dim(0), weights.get_dim(0)};
    tensor::descriptor dst_desc(dst_dims, src_in.get_data_type());

    auto key = utils::create_key(src_in.get_data_type(), src_in.get_dims(),
        weights.get_dims(), bias.get_dims(), dst_dims);

    auto comp = fetch_or_create_m(key, src_in.get_descriptor(),
        weights.get_descriptor(), bias.get_descriptor(), dst_desc);

    auto weights_in = weights;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<alloc, inner_product_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }
    if (weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<alloc, inner_product_forward>(
          comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    dst.reinit<alloc, inner_product_forward>(
        comp.expected_dst_descriptor());
    comp.execute(src_in, weights_in, bias, dst);
  }

  template<class alloc = utils::allocator>
  static void compute(
      const tensor& src, const tensor& weights, tensor& dst) {
    tensor::dims dst_dims = {src.get_dim(0), weights.get_dim(0)};
    tensor::descriptor dst_desc(dst_dims, src.get_data_type());
    auto src_in = src;

    if (src.ndims() == 4 && weights.ndims() == 2) {
      //
      // tricky procedure, we change src back to public format
      // then we using another tensor with 2 dimension to describe the buffer
      //
      tensor::dims new_dims { src.get_dim(0),
        src.get_dim(1) * src.get_dim(2) * src.get_dim(3) };

      if (src.is_public_format())
        src_in.init({new_dims, src.get_data_type(), format::nc}, src.get_data_handle());
      else {
        src_in.init<alloc, inner_product_forward>(
            {src.get_dims(), src.get_data_type(), format::nchw});
        reorder::compute(src, src_in);
        src_in.set_descriptor({new_dims, src.get_data_type(), format::nc});
      }
    }

    auto key = utils::create_key(src_in.get_data_type(), src_in.get_dims(),
        weights.get_dims(), dst_dims);

    auto comp = fetch_or_create_m(key, src_in.get_descriptor(),
        weights.get_descriptor(), dst_desc);

    auto weights_in = weights;
    if (src_in.get_descriptor() != comp.expected_src_descriptor()) {
      src_in.init<alloc, inner_product_forward>(comp.expected_src_descriptor());
      reorder::compute(src, src_in);
    }
    if (weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<alloc, inner_product_forward>(
          comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    dst.reinit<alloc, inner_product_forward>(
        comp.expected_dst_descriptor());
    comp.execute(src_in, weights_in, dst);
  }
};

// TODO: parameter sequence adjust?
struct inner_product_backward_data: public computation,
  utils::computation_cache<inner_product_backward_data> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &gradx_desc,
        const tensor::descriptor &weights_desc,
        const tensor::descriptor &grady_desc)
      : hint_(gradx_desc, weights_desc, grady_desc) {
      auto diff_src_data = gradx_desc.format_any();
      auto weights_data = weights_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      mkldnn_inner_product_desc_t data;
      error::wrap_c_api(
          mkldnn_inner_product_backward_data_desc_init(&data,
            &diff_src_data, &weights_data,
            &diff_dst_data),
          "could not create a inner product backward data descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(&result,
              &data, engine::cpu_engine().get(), hint_.get()),
    "could not create a inner product backward data primitive descriptor");
      reset(result);
    }
  private:
    inner_product_forward::descriptor hint_;
  };
public:
  using computation::expected_gradx_descriptor;
  using computation::expected_grady_descriptor;
  using computation::expected_weights_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &gradx_desc,
      const tensor::descriptor &weights_desc,
      const tensor::descriptor &grady_desc) {
    descriptor backward_data_descriptor(gradx_desc, weights_desc, grady_desc);
    computation::init(backward_data_descriptor, grady_desc, weights_desc);
  }

  inner_product_backward_data() = default;

  template <typename T, typename ...Ts>
  inner_product_backward_data(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& grady, const tensor& weights, const tensor& gradx) {
    computation::execute(grady, weights, gradx);
  }

  template<class alloc = utils::allocator>
  static void compute( const tensor& grady, const tensor& weights,
      tensor::dims gradx_dims, tensor& gradx) {
    tensor::descriptor gradx_desc(gradx_dims, grady.get_data_type());

    auto key = utils::create_key(grady.get_data_type(), grady.get_dims(),
        weights.get_dims(), gradx_dims);

    auto comp = fetch_or_create_m(key, gradx_desc,
        weights.get_descriptor(), grady.get_descriptor());

    auto grady_in = grady;
    auto weights_in = weights;
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<alloc, inner_product_backward_data>(
          comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }
    if (weights.get_descriptor() != comp.expected_weights_descriptor()) {
      weights_in.init<alloc, inner_product_backward_data>(
          comp.expected_weights_descriptor());
      reorder::compute(weights, weights_in);
    }

    gradx.reinit<alloc, inner_product_backward_data>(
        comp.expected_gradx_descriptor());
    comp.execute(grady_in, weights_in, gradx);
  }
};

struct inner_product_backward_weights : public computation,
  public utils::computation_cache<inner_product_backward_weights> {
  struct descriptor : public descriptor_group {
    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &gradw_desc,
        const tensor::descriptor &gradb_desc,
        const tensor::descriptor &grady_desc)
      : hint_(x_desc, gradw_desc, gradb_desc, grady_desc) {
      mkldnn_inner_product_desc_t data;
      auto src_data = x_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      auto diff_weights_data = gradw_desc.format_any();
      auto diff_bias_data = gradb_desc.format_any();
      error::wrap_c_api(
          mkldnn_inner_product_backward_weights_desc_init(
            &data, &src_data, &diff_weights_data,
            &diff_bias_data, &diff_dst_data),
          "could not create a inner product backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(&result,
            &data, engine::cpu_engine().get(), hint_.get()),
    "cld not create a inner product backward weights primitive descriptor");
      reset(result);
    }

    descriptor(const tensor::descriptor &x_desc,
        const tensor::descriptor &gradw_desc,
        const tensor::descriptor &grady_desc)
    : hint_(x_desc, gradw_desc, grady_desc) {
      mkldnn_inner_product_desc_t data;
      auto src_data = x_desc.format_any();
      auto diff_dst_data = grady_desc.format_any();
      auto diff_weights_data = gradw_desc.format_any();
      error::wrap_c_api(
          mkldnn_inner_product_backward_weights_desc_init(
          &data, &src_data, &diff_weights_data,
          nullptr, &diff_dst_data),
          "could not create a inner product backward weights descriptor");
      mkldnn_primitive_desc_t result;
      error::wrap_c_api(mkldnn_primitive_desc_create(&result,
            &data, engine::cpu_engine().get(), hint_.get()),
    "cld not create a inner product backward weights primitive descriptor");
      reset(result);
    }
  private:
    inner_product_forward::descriptor hint_;
  };
public:
  using computation::expected_gradw_descriptor;
  using computation::expected_gradb_descriptor;

  template <typename ...Ts>
  void init(const tensor::descriptor &x_desc,
      const tensor::descriptor &grady_desc,
      const tensor::descriptor &gradw_desc, Ts&&... args) {
    descriptor backward_weights_descriptor(x_desc, grady_desc, gradw_desc,
        std::forward<Ts>(args)...);
    computation::init(backward_weights_descriptor, x_desc, grady_desc);
  }

  inner_product_backward_weights() = default;

  template <typename T, typename ...Ts>
  inner_product_backward_weights(T arg, Ts&&... args) {
    init(arg, std::forward<Ts>(args)...);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& gradw) {
    computation::execute(x, grady, gradw);
  }

  void execute(const tensor& x, const tensor& grady, const tensor& gradw
      , const tensor& gradb) {
    computation::execute(x, grady, gradw, gradb);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& x, const tensor& grady, tensor& gradw) {
    auto gradw_dims = x.get_dims();
    gradw_dims[0] = grady.get_dim(1);
    tensor::descriptor gradw_desc(gradw_dims, grady.get_data_type());

    auto key = utils::create_key(x.get_data_type(), x.get_dims(), gradw_dims,
        grady.get_dims());

    auto comp = fetch_or_create_m(key, x.get_descriptor(), gradw_desc,
        grady.get_descriptor());

    auto x_in = x;
    auto grady_in = grady;
    if (x.get_descriptor() != comp.expected_src_descriptor()) {
      x_in.init<alloc, inner_product_backward_weights>(comp.expected_src_descriptor());
      reorder::compute(x, x_in);
    }
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<alloc, inner_product_backward_weights>(
          comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<alloc, inner_product_backward_weights>(
        comp.expected_gradw_descriptor());
    comp.execute(x_in, grady_in, gradw);
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor& x, const tensor& grady, tensor& gradw,
      tensor& gradb) {
    auto gradw_dims = x.get_dims();
    gradw_dims[0] = grady.get_dim(1);

    tensor::dims gradb_dims = {grady.get_dim(1)};
    tensor::descriptor gradw_desc(gradw_dims, x.get_data_type());
    tensor::descriptor gradb_desc(gradb_dims, x.get_data_type());

    auto key = utils::create_key(x.get_data_type(), x.get_dims(), gradw_dims,
        gradb_dims, grady.get_dims());

    auto comp = fetch_or_create_m(key, x.get_descriptor(), gradw_desc, gradb_desc,
        grady.get_descriptor());

    auto x_in = x;
    auto grady_in = grady;
    if (x.get_descriptor() != comp.expected_src_descriptor()) {
      x_in.init<alloc, inner_product_backward_weights>(
          comp.expected_src_descriptor());
      reorder::compute(x, x_in);
    }
    if (grady.get_descriptor() != comp.expected_grady_descriptor()) {
      grady_in.init<alloc, inner_product_backward_weights>(
          comp.expected_grady_descriptor());
      reorder::compute(grady, grady_in);
    }

    gradw.reinit<alloc, inner_product_backward_weights>(
        comp.expected_gradw_descriptor());
    gradb.reinit(comp.expected_gradb_descriptor());
    comp.execute(x_in, grady_in, gradw, gradb);
  }
};

struct dropout_forward {
public:
  dropout_forward() = default;

  static void bernoulli_generate(const long n, const double p, int* r) {
    std::srand(std::time(0));
    const int seed = 17 + std::rand() % 4096;

    int nthr = omp_get_max_threads();

    # pragma omp parallel num_threads(nthr)
    {
      const int ithr = omp_get_thread_num();
      const long avg_amount = (n + nthr - 1) / nthr;
      const long my_offset = ithr * avg_amount;
      const long my_amount = std::min(my_offset + avg_amount, n) - my_offset;

      if (my_amount > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        vslSkipAheadStream(stream, my_offset);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream,
            my_amount, r + my_offset, p);
        vslDeleteStream(&stream);
      }
    }
  }

  template<class alloc, class T>
  static void compute_impl(const tensor &src, float ratio,
      tensor& dst, tensor& mask) {
    const auto scale = 1.0 / (1.0 - ratio);
    const auto size = src.get_nelems();
    mask.reinit<alloc, dropout_forward>(src.get_descriptor());
    dst.reinit<alloc, dropout_forward>(src.get_descriptor());

    std::unique_ptr<int[]> bernouli_nums(new int[size]);
    bernoulli_generate(size, 1.0 - ratio, bernouli_nums.get());

    const auto src_data = static_cast<T *>(src.get_data_handle());
    const auto mask_data = static_cast<T *>(mask.get_data_handle());
    const auto dst_data = static_cast<T *>(dst.get_data_handle());

    # pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      mask_data[i] = bernouli_nums[i] * scale;
      dst_data[i] = mask_data[i] * src_data[i];
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &src, float ratio,
      tensor& dst, tensor& mask) {
    switch(src.get_data_type()) {
    case tensor::data_type::f32:
      compute_impl<alloc, float>(src, ratio, dst, mask);
      break;
    case tensor::data_type::s32:
      compute_impl<alloc, int32_t>(src, ratio, dst, mask);
      break;
    case tensor::data_type::s16:
      compute_impl<alloc, int16_t>(src, ratio, dst, mask);
      break;
    case tensor::data_type::s8:
      compute_impl<alloc, int8_t>(src, ratio, dst, mask);
      break;
    case tensor::data_type::u8:
      compute_impl<alloc, uint8_t>(src, ratio, dst, mask);
      break;
    default:
      throw error(mkldnn_invalid_arguments, "Unsupported mkldnn data type!");
    }
  }
};

struct dropout_backward {
public:
  dropout_backward() = default;

  template<class alloc, class T>
  static void compute_impl(const tensor &mask, const tensor &gy, tensor& gx) {
    const auto size = mask.get_nelems();
    gx.reinit<alloc, dropout_backward>(gy.get_descriptor());

    const auto mask_data = static_cast<T *>(mask.get_data_handle());
    const auto gy_data = static_cast<T *>(gy.get_data_handle());
    const auto gx_data = static_cast<T *>(gx.get_data_handle());

    # pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      gx_data[i] = mask_data[i] * gy_data[i];
    }
  }

  template<class alloc = utils::allocator>
  static void compute(const tensor &mask, const tensor &gy, tensor& gx) {
    switch(gy.get_data_type()) {
    case tensor::data_type::f32:
      compute_impl<alloc, float>(mask, gy, gx);
      break;
    case tensor::data_type::s32:
      compute_impl<alloc, int32_t>(mask, gy, gx);
      break;
    case tensor::data_type::s16:
      compute_impl<alloc, int16_t>(mask, gy, gx);
      break;
    case tensor::data_type::s8:
      compute_impl<alloc, int8_t>(mask, gy, gx);
      break;
    case tensor::data_type::u8:
      compute_impl<alloc, uint8_t>(mask, gy, gx);
      break;
    default:
      throw error(mkldnn_invalid_arguments, "Unsupported mkldnn data type!");
    }
  }
};

struct eltwise_binary {
public:
  enum eltwise_binary_op {
    ELTWISE_ADD,
    ELTWISE_MUL,
    ELTWISE_DIV,
  };

  eltwise_binary() = default;

  template<class alloc = utils::allocator>
  static void compute(eltwise_binary_op op, tensor &inputA, tensor &inputB,
      tensor &outputC) {
    assert(inputA.ndims() >= inputB.ndims());
    assert(inputA.get_descriptor() == outputC.get_descriptor());
    if (inputA.get_dims() == inputB.get_dims()) {
      auto* inputB_data = inputB.get_data_handle();
      tensor scratch_tensor;
      if (inputA.get_internal_format() != inputB.get_internal_format()) {
        scratch_tensor.init<alloc, eltwise_binary>(inputA.get_descriptor());
        reorder::compute(inputB, scratch_tensor);
        inputB_data = scratch_tensor.get_data_handle();
      }
      switch (op) {
      case ELTWISE_ADD:
        utils::fast_math<utils::cpu_isa_t::avx2>::add<float>(
            static_cast<float*>(outputC.get_data_handle()),
            static_cast<float*>(inputA.get_data_handle()),
            static_cast<float*>(inputB_data),
            static_cast<unsigned>(inputA.get_nelems()));
        return;
      case ELTWISE_MUL:
      case ELTWISE_DIV:
      default:
        throw error(mkldnn_unimplemented, "Not implemented!");
      }
    } else {
      throw error(mkldnn_runtime_error, "Not implemented!");
    }
  }
};

struct sum_array {
public:
  typedef enum {
    NOERR = 0,
    UNSUPPORT_AXIS_COMMON_SUM,
    UNSUPPORT_AXIS_FAST_SUM,
    UNSUPPORT_DATA_TYPE,
  } err_num_t;

  sum_array() = default;

  template<typename data_t>
  static inline void sum_nChwXC_along_channel(data_t *src,
      tensor::descriptor src_desc, std::vector<int> axis, data_t *dst) {
    int mb = src_desc.get_dims()[0],
        ic = src_desc.get_dims()[1],
        ih = src_desc.get_dims()[2],
        iw = src_desc.get_dims()[3];
    const int cg = (int)src_desc.get_mkldnn_memory_desc_t()->format ==
        mkldnn_nChw16c ? 16 : 8;
    int cn = ic / cg;

    int blk_nthr = omp_get_max_threads(),
        blk_num = blk_nthr,
        blk_len = mb / blk_num,
        blk_len_ex = mb % blk_num;

    if (!blk_len)
      blk_nthr = mb;

    data_t *buf = reinterpret_cast<data_t *>(
        new char[ic * blk_nthr * sizeof(data_t)]);

    # pragma omp parallel num_threads(blk_nthr)
    {
      int ithr = omp_get_thread_num();
      int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
      int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
                   blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
      int bend = bstart + blen;

      data_t *loc_src = src + bstart * ic * ih * iw;
      if ((cg == 16) && (((unsigned long)buf & 0xf) == 0) &&
        (((unsigned long)loc_src & 0xf) == 0)) {
        for (int b = bstart; b < bend; b++) {
          data_t *loc_buf = buf + ithr * ic;
          for (int c = 0; c < cn; c++) {
            if (b == bstart)
              for (int o = 0; o < cg; o++)
                loc_buf[o] = 0;
            for (int hw = 0; hw < ih * iw; hw++) {
              __asm__(
                      "mov %0, %%rax\n"
                      "mov %1, %%rbx\n"
                      ".byte 0x62, 0xf1, 0x7c, 0x48, 0x10, 0x00\n" //vmovups (%%rax), %%zmm0
                      ".byte 0x62, 0xf1, 0x7c, 0x48, 0x58, 0x03\n" //vaddps (%%rbx), %%zmm0, %%zmm0
                      ".byte 0x62, 0xf1, 0x7c, 0x48, 0x11, 0x00\n" //vmovups %%zmm0, (%%rax)
                      :"+r"(loc_buf)
                      :"r"(loc_src)
                      :"rax", "rbx"
                      );
              loc_src += cg;
            }

            loc_buf += cg;
          }
        }
      } else if ((cg == 8) && (((unsigned long)buf & 0x7) == 0) &&
          (((unsigned long)loc_src & 0x7) == 0)) {
        for (int b = bstart; b < bend; b++) {
          data_t *loc_buf = buf + ithr * ic;
          for (int c = 0; c < cn; c++) {
            if (b == bstart)
              for (int o = 0; o < cg; o++)
                loc_buf[o] = 0;
            for (int hw = 0; hw < ih * iw; hw++) {
              __asm__(
                      "mov %0, %%rax\n"
                      "mov %1, %%rbx\n"
                      ".byte 0xc5, 0xfc, 0x10, 0x00\n" //vmovups (%%rax), %%ymm0
                      ".byte 0xc5, 0xfc, 0x58, 0x03\n" //vaddps (%%rbx), %%ymm0, %%ymm0
                      ".byte 0xc5, 0xfc, 0x11, 0x00\n" //vmovups %%ymm0, (%rax)
                      :"+r"(loc_buf)
                      :"r"(loc_src)
                      :"rax", "rbx"
                      );
              loc_src += cg;
            }

            loc_buf += cg;
          }
        }
      } else {
        for (int b = bstart; b < bend; b++) {
          data_t *loc_buf = buf + ithr * ic;
          for (int c = 0; c < cn; c++) {
            if (b == bstart)
              for (int o = 0; o < cg; o++)
                loc_buf[o] = 0;

            for (int hw = 0; hw < ih * iw; hw++) {
              for (int o = 0; o < cg; o++)
                loc_buf[o] += loc_src[o];
              loc_src += cg;
            }

            loc_buf += cg;
          }
        }
      }
    }

    // Allreduce
    int c_nthr = omp_get_max_threads(),
        c_num = c_nthr,
        c_len = ic / c_num,
        c_len_ex = ic % c_num;

    if (!c_len)
      c_nthr = ic;

    # pragma omp parallel num_threads(c_nthr)
    {
      int ithr = omp_get_thread_num();
      int clen = ithr < c_len_ex ? c_len + 1 : c_len;
      int cstart = ithr <= c_len_ex ? (c_len + 1) * ithr :
                   c_len_ex * (c_len + 1) + (ithr - c_len_ex) * c_len;
      int cend = cstart + clen;

      for (int c = cstart; c < cend; c++)
        dst[c] = 0;

      for (int i = 0; i < blk_nthr; i++) {
        data_t *loc_buf = buf + i * ic;
        for (int c = cstart; c < cend; c++)
          dst[c] += loc_buf[c];
      }
    }

    delete(reinterpret_cast<char *>(buf));
  }

  template<typename data_t>
  static inline tensor sum_fast_along_axis(tensor &src,
      std::vector<int> axis, err_num_t &err) {
    int axises = axis.size();
    std::vector<int> valid_axis_4dim = {0, 2, 3};

    err = NOERR;
    if (src.ndims() != 4 || axises != 3) {
      err = (err_num_t)-UNSUPPORT_AXIS_FAST_SUM;
      return tensor();
    }

    auto valid_axis = [](int axises,
                         std::vector<int> axis,
                         std::vector<int> valid_axis) -> bool {
      for (int i = 0; i < axises; i++)
        if (valid_axis[i] != axis[i])
          return false;
      return true;
    };

    switch ((int)src.get_internal_format()) {
    case mkldnn_nChw8c:
      if (!valid_axis(axises, axis, valid_axis_4dim))
        err = (err_num_t)-UNSUPPORT_AXIS_FAST_SUM;
      break;
    case mkldnn_nChw16c:
      if (!valid_axis(axises, axis, valid_axis_4dim))
        err = (err_num_t)-UNSUPPORT_AXIS_FAST_SUM;
      break;
    default:
      err = (err_num_t)-UNSUPPORT_AXIS_FAST_SUM;
      break;
    }

    if (err == (err_num_t)-UNSUPPORT_AXIS_FAST_SUM)
      return tensor();

    tensor dst;
    dst.init({{src.get_dims()[1]},
              src.get_data_type(),
              format::x});

    sum_nChwXC_along_channel((data_t *)src.get_data_handle(),
                             src.get_descriptor(), axis,
                             (data_t *)dst.get_data_handle());

    return dst;
  }

  template<typename data_t>
  static inline void sum_along_axis(data_t *src,
      tensor::descriptor src_desc, std::vector<int> axis, data_t *dst) {
    auto src_dims = src_desc.get_dims();
    auto src_ndims = src_desc.ndims();

    int tail = 1;
    for (int d = 1; d < src_ndims; d++)
      tail *= src_dims[d];

    bool along_mb = false;
    for (int a = 0; a < axis.size(); a++) {
      if (axis[a] == 0) {
        along_mb = true;
        break;
      }
    }

    int gbl_ws_size = 1;
    for (int d = 1; d < src_ndims; d++) {
      int a = 0;
      for (; a < axis.size(); a++)
        if (d == axis[a])
          break;

      if (a >= axis.size())
        gbl_ws_size *= src_dims[d];
    }

    int mb = src_dims[0];
    int blk_nthr = omp_get_max_threads(),
        blk_num = blk_nthr,
        blk_len = mb / blk_num,
        blk_len_ex = mb % blk_num;

    if (!blk_len)
      blk_nthr = mb;

    data_t *gbl_ws[blk_nthr];
    # pragma omp parallel num_threads(blk_nthr)
    {
      int ithr = omp_get_thread_num();
      int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
      int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
                   blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
      int bend = bstart + blen;

      data_t *loc_ws[blen];
      for (int b = bstart; b < bend; b++) {
        data_t *loc_src = src + b * tail;
        data_t *cur_src = loc_src;

        // Intialize for new blk
        std::vector<int> cur_dims;
        for (int d = 0; d < src_ndims; d++)
          cur_dims.push_back(src_dims[d]);

        std::vector<int> cur_axis;
        for (int a = 0; a < axis.size(); a++)
          if (axis[a] != 0)
            cur_axis.insert(cur_axis.begin(), axis[a]);

        // Sum along axis[a]
        for (int a = 0; a < cur_axis.size(); a++) {

          int cur_fore = 1;
          for (int d = 1; d < cur_axis[a]; d++)
            cur_fore *= cur_dims[d];

          int cur_tail = 1;
          for (int d = cur_axis[a] + 1; d < cur_dims.size(); d++)
            cur_tail *= cur_dims[d];

          int cur_ws_size = cur_fore * cur_tail;
          data_t *ws = reinterpret_cast<data_t *>(
              new char[cur_ws_size * sizeof(data_t)]);
          for (int o = 0; o < cur_ws_size; o++) ws[o] = 0;

          // kernel
          for (int base = 0, off = 0, w = 0; w < cur_ws_size;) {
            for (int t = 0; t < cur_dims[cur_axis[a]]; t++) {
              ws[w] += cur_src[off + t * cur_tail];
            }
            w++; if (0 == w % cur_tail) {
              off = base + cur_tail * cur_dims[cur_axis[a]];
              base = off;
            } else {
              off += 1;
            }
          }

          // adjust dims and cur_axis for sum in next axis
          cur_dims.erase(cur_dims.begin() + cur_axis[a]);
          for (int _a = a + 1; _a < cur_axis.size(); _a++) {
            if (cur_axis[_a] > cur_axis[a])
              cur_axis[_a] -= 1;
          }

          // refresh buffer
          if (cur_src != loc_src) delete(reinterpret_cast<char *>(cur_src));
          if (a == cur_axis.size() - 1) loc_ws[b - bstart] = ws;

          cur_src = ws;
        }
      }

      if (along_mb) {
        // local allreduce
        if (src_ndims == 2 && axis.size() == 1 && axis[0] == 0) {
          loc_ws[0] = reinterpret_cast<data_t *>(
              new char[tail * sizeof(data_t)]);
          for (int o = 0; o < tail; o++)
            loc_ws[0][o] = 0;
          for (int b = bstart; b < bend; b++) {
            data_t *loc_src = src + b * tail;
            for (int o = 0; o < tail; o++)
              loc_ws[0][o] += loc_src[o];
          }
        } else {
          for (int b = 1; b < blen; b++) {
            for (int o = 0; o < gbl_ws_size; o++)
              loc_ws[0][o] += loc_ws[b][o];
            delete(reinterpret_cast<char *>(loc_ws[b]));
          }
        }

        gbl_ws[ithr] = loc_ws[0];
      } else {
        // cpy to dst
        for (int b = bstart; b < bend; b++) {
          for (int o = 0; o < gbl_ws_size; o++)
            dst[b * gbl_ws_size + o] = loc_ws[b - bstart][o];
          delete(reinterpret_cast<char *>(loc_ws[b - bstart]));
        }
      }
    }

    if (along_mb) {
      // global allreduce
      int c_nthr = omp_get_max_threads(),
          c_num = c_nthr,
          c_len = gbl_ws_size / c_num,
          c_len_ex = gbl_ws_size % c_num;

      if (!c_len)
        c_nthr = gbl_ws_size;

      # pragma omp parallel num_threads(c_nthr)
      {
        int ithr = omp_get_thread_num();
        int clen = ithr < c_len_ex ? c_len + 1 : c_len;
        int cstart = ithr <= c_len_ex ? (c_len + 1) * ithr :
                     c_len_ex * (c_len + 1) + (ithr - c_len_ex) * c_len;
        int cend = cstart + clen;

        for (int c = cstart; c < cend; c++)
          dst[c] = 0;

        for (int i = 0; i < blk_nthr; i++) {
          data_t *loc_buf = gbl_ws[i];
          for (int c = cstart; c < cend; c++)
            dst[c] += loc_buf[c];
        }
      }

      for (int i = 0; i < blk_nthr; i++)
        delete(reinterpret_cast<char *>(gbl_ws[i]));
    }
  }

  template<typename data_t>
  static inline tensor sum_common_along_axis(tensor &src,
      std::vector<int> axis, err_num_t &err) {
    auto src_dims = src.get_dims();
    int dst_ndims = src.ndims() - axis.size();

    err = NOERR;
    // TODO: Support sum all
    if ((dst_ndims != 1 && dst_ndims != 2 && dst_ndims != 4) ||
        axis.size() == 0) {
      err = (err_num_t)-UNSUPPORT_AXIS_COMMON_SUM;
      return tensor();
    }

    tensor dst;
    dst.init({get_dst_dims(src.get_dims(), axis),
              src.get_data_type(),
              engine::default_format(dst_ndims)});

    sum_along_axis((data_t *)src.get_data_handle(),
                   src.get_descriptor(), axis,
                   (data_t *)dst.get_data_handle());

    return dst;
  }

  static tensor compute(tensor &src,
      std::vector<int> &axis, err_num_t &err) {
    if (optimized_format(src)) {
      switch(src.get_data_type()) {
      case tensor::data_type::f32:
        return sum_fast_along_axis<float>(src, axis, err);
      case tensor::data_type::s32:
        return sum_fast_along_axis<int32_t>(src, axis, err);
      case tensor::data_type::s16:
        return sum_fast_along_axis<int16_t>(src, axis, err);
      case tensor::data_type::s8:
        return sum_fast_along_axis<int8_t>(src, axis, err);
      case tensor::data_type::u8:
        return sum_fast_along_axis<uint8_t>(src, axis, err);
      default:
        break;
      }
    } else {
      switch(src.get_data_type()) {
      case tensor::data_type::f32:
        return sum_common_along_axis<float>(src, axis, err);
      case tensor::data_type::s32:
        return sum_common_along_axis<int32_t>(src, axis, err);
      case tensor::data_type::s16:
        return sum_common_along_axis<int16_t>(src, axis, err);
      case tensor::data_type::s8:
        return sum_common_along_axis<int8_t>(src, axis, err);
      case tensor::data_type::u8:
        return sum_common_along_axis<uint8_t>(src, axis, err);
      default:
        break;
      }
    }

    err = (err_num_t)-UNSUPPORT_DATA_TYPE;
    return tensor();
  }

private:
  static inline bool optimized_format(const tensor &t) {
    switch((int)t.get_internal_format()) {
    case mkldnn_nChw16c:
    case mkldnn_nChw8c:
    case mkldnn_OIhw8i8o:
    case mkldnn_OIhw16i16o:
    case mkldnn_OIhw8i16o2i:
    case mkldnn_OIhw8o16i2o:
    case mkldnn_OIhw8o8i:
    case mkldnn_OIhw16o16i:
    case mkldnn_Oihw8o:
    case mkldnn_Oihw16o:
        return true;
    default:
        return false;
    }
  }

  static inline tensor::dims get_dst_dims(tensor::dims src_dims,
      std::vector<int> axis) {
    tensor::dims dst_dims;
    for (unsigned d = 0; d < src_dims.size(); d++) {
      unsigned a = 0;
      for (; a < axis.size(); a++) {
        if (d == (unsigned)axis[a])
          break;
      }

      if (a >= axis.size())
        dst_dims.push_back(src_dims[d]);
    }

    return dst_dims;
  }
};

} // namespace mkldnn

#endif
