#ifndef IDEEP_PYTHON_BINDING_HPP
#define IDEEP_PYTHON_BINDING_HPP

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <sstream>

#include "oneapi/dnnl/dnnl_graph.hpp"

namespace ideep {

void bind_cpartition(pybind11::module& m) {
  pybind11::class_<dnnl::graph::compiled_partition> cp(m, "compiled_partition");

  cp.def(pybind11::init<>());
  cp.def("query_logical_tensor", &dnnl::graph::compiled_partition::query_logical_tensor);
  cp.def("get_inplace_ports", &dnnl::graph::compiled_partition::get_inplace_ports);
  cp.def("execute", &dnnl::graph::compiled_partition::execute);
}

const std::string engine_kind2str(dnnl::graph::engine::kind v) {
  if (v == dnnl::graph::engine::kind::any)
    return "any";
  if (v == dnnl::graph::engine::kind::cpu)
    return "cpu";
  if (v == dnnl::graph::engine::kind::gpu)
    return "gpu";
  return "unknown engine_kind";
}

auto eng2string = [](const dnnl::graph::engine& eng) {
  std::stringstream ss;
  ss << "engine(kind = " << engine_kind2str(eng.get_kind()) << ")";
  return ss.str();
};

void bind_engine(pybind11::module& m) {
  pybind11::class_<dnnl::graph::engine> eng(m, "engine");

  eng.def(pybind11::init<dnnl::graph::engine::kind, size_t>());
  eng.def("get_kind", &dnnl::graph::engine::get_kind);
  eng.def("get_count", &dnnl::graph::engine::get_count);
  eng.def("__repr__", eng2string);

  pybind11::enum_<dnnl::graph::engine::kind>(eng, "kind")
      .value("any", dnnl::graph::engine::kind::any)
      .value("cpu", dnnl::graph::engine::kind::cpu)
      .value("gpu", dnnl::graph::engine::kind::gpu)
      .export_values();
}

void bind_graph(pybind11::module& m) {
  pybind11::class_<dnnl::graph::graph> g(m, "graph");

  g.def(pybind11::init<dnnl::graph::engine::kind>());
  g.def(pybind11::init<dnnl::graph::engine::kind, dnnl::graph::fpmath_mode>());
  g.def("add_op", &dnnl::graph::graph::add_op);
  g.def("finalize", &dnnl::graph::graph::finalize);
  g.def("is_finalized", &dnnl::graph::graph::is_finalized);
  g.def("get_partitions", &dnnl::graph::graph::get_partitions);

  pybind11::enum_<dnnl::graph::fpmath_mode>(g, "fpmath_mode")
      .value("strict", dnnl::graph::fpmath_mode::strict)
      .value("bf16", dnnl::graph::fpmath_mode::bf16)
      .value("f16", dnnl::graph::fpmath_mode::f16)
      .value("tf32", dnnl::graph::fpmath_mode::tf32)
      .value("any", dnnl::graph::fpmath_mode::any)
      .export_values();
}

const std::string data_type2str(dnnl::graph::logical_tensor::data_type v) {
#define CASE(x)                        \
  case (dnnl::graph::logical_tensor::data_type::x): \
    return #x

  switch (v) {
    CASE(undef);
    CASE(f16);
    CASE(bf16);
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
    CASE(boolean);
    default:
      return "unknown data_type";
  }

#undef CASE
}

const std::string layout_type2str(dnnl::graph::logical_tensor::layout_type v) {
  if (v == dnnl::graph::logical_tensor::layout_type::undef)
    return "undef";
  if (v == dnnl::graph::logical_tensor::layout_type::any)
    return "any";
  if (v == dnnl::graph::logical_tensor::layout_type::strided)
    return "strided";
  if (v == dnnl::graph::logical_tensor::layout_type::opaque)
    return "opaque";
  return "unknown layout_type";
}

const std::string dims2string(const std::vector<int64_t>& dims) {
  std::stringstream ss;
  ss << "(";
  const char* delimer = "";
  for (const auto& d : dims) {
    ss << delimer << d;
    delimer = ", ";
  }
  ss << ")";
  return ss.str();
};

auto lt2string = [](const dnnl::graph::logical_tensor& lt) {
  std::stringstream ss;
  ss << "logical_tensor(id = " << lt.get_id()
     << ", dtype = " << data_type2str(lt.get_data_type())
     << ", layout = " << layout_type2str(lt.get_layout_type())
     << ", shape = " << dims2string(lt.get_dims());
  if (lt.get_layout_type() == dnnl::graph::logical_tensor::layout_type::opaque) {
    ss << ", layout_id = " << lt.get_layout_id();
  } else if (lt.get_layout_type() == dnnl::graph::logical_tensor::layout_type::strided) {
    ss << ", stride = " << dims2string(lt.get_strides());
  } else {
  }
  ss << ")";
  return ss.str();
};

void bind_logical_tensor(pybind11::module& m) {
  pybind11::class_<dnnl::graph::logical_tensor> lt(m, "logical_tensor");

  lt.def(pybind11::init<
         size_t,
         dnnl::graph::logical_tensor::data_type,
         dnnl::graph::logical_tensor::layout_type>());
  lt.def(pybind11::init<
         size_t,
         dnnl::graph::logical_tensor::data_type,
         int32_t,
         dnnl::graph::logical_tensor::layout_type,
         dnnl::graph::logical_tensor::property_type>());
  lt.def(pybind11::init<
         size_t,
         dnnl::graph::logical_tensor::data_type,
         dnnl::graph::logical_tensor::dims,
         dnnl::graph::logical_tensor::layout_type,
         dnnl::graph::logical_tensor::property_type>());
  lt.def(pybind11::init<
         size_t,
         dnnl::graph::logical_tensor::data_type,
         dnnl::graph::logical_tensor::dims,
         dnnl::graph::logical_tensor::dims,
         dnnl::graph::logical_tensor::property_type>());
  lt.def(pybind11::init<
         size_t,
         dnnl::graph::logical_tensor::data_type,
         dnnl::graph::logical_tensor::dims,
         size_t,
         dnnl::graph::logical_tensor::property_type>());
  lt.def("get_id", &dnnl::graph::logical_tensor::get_id);
  lt.def("get_data_type", &dnnl::graph::logical_tensor::get_data_type);
  lt.def("get_layout_type", &dnnl::graph::logical_tensor::get_layout_type);
  lt.def("get_property_type", &dnnl::graph::logical_tensor::get_property_type);
  lt.def("get_layout_id", &dnnl::graph::logical_tensor::get_layout_id);
  lt.def("get_mem_size", &dnnl::graph::logical_tensor::get_mem_size);
  lt.def("get_dims", &dnnl::graph::logical_tensor::get_dims);
  lt.def("get_strides", &dnnl::graph::logical_tensor::get_strides);
  lt.def("is_equal", &dnnl::graph::logical_tensor::is_equal);
  lt.def("__repr__", lt2string);

  pybind11::enum_<dnnl::graph::logical_tensor::data_type>(lt, "data_type")
      .value("undef", dnnl::graph::logical_tensor::data_type::undef)
      .value("f16", dnnl::graph::logical_tensor::data_type::f16)
      .value("bf16", dnnl::graph::logical_tensor::data_type::bf16)
      .value("f32", dnnl::graph::logical_tensor::data_type::f32)
      .value("s32", dnnl::graph::logical_tensor::data_type::s32)
      .value("s8", dnnl::graph::logical_tensor::data_type::s8)
      .value("u8", dnnl::graph::logical_tensor::data_type::u8)
      .value("boolean", dnnl::graph::logical_tensor::data_type::boolean)
      .export_values();

  pybind11::enum_<dnnl::graph::logical_tensor::layout_type>(lt, "layout_type")
      .value("undef", dnnl::graph::logical_tensor::layout_type::undef)
      .value("any", dnnl::graph::logical_tensor::layout_type::any)
      .value("strided", dnnl::graph::logical_tensor::layout_type::strided)
      .value("opaque", dnnl::graph::logical_tensor::layout_type::opaque)
      .export_values();

  pybind11::enum_<dnnl::graph::logical_tensor::property_type>(lt, "property_type")
      .value("undef", dnnl::graph::logical_tensor::property_type::undef)
      .value("variable", dnnl::graph::logical_tensor::property_type::variable)
      .value("constant", dnnl::graph::logical_tensor::property_type::constant)
      .export_values();
}

template <class T>
void set_op_attribute(dnnl::graph::op& aop, T x, dnnl::graph::op::attr attr) {
  if (pybind11::isinstance<pybind11::list>(x)) {
    if (pybind11::isinstance<pybind11::int_>(
            x.template cast<pybind11::list>()[0])) {
      std::vector<int64_t> int_attr = {};
      for (auto val : x.template cast<pybind11::list>()) {
        int_attr.push_back(val.template cast<int64_t>());
      }
      aop.set_attr<std::vector<int64_t>>(attr, int_attr);
    } else if (pybind11::isinstance<pybind11::float_>(
                   x.template cast<pybind11::list>()[0])) {
      std::vector<float> int_attr = {};
      for (auto val : x.template cast<pybind11::list>()) {
        int_attr.push_back(val.template cast<float>());
      }
      aop.set_attr<std::vector<float>>(attr, int_attr);
    } else {
      assert(!"unknown vector type");
    }
  } else if (pybind11::isinstance<pybind11::bool_>(x)) {
    aop.set_attr<bool>(attr, x.template cast<bool>());
  } else if (pybind11::isinstance<pybind11::int_>(x)) {
    aop.set_attr<int64_t>(attr, x.template cast<int64_t>());
  } else if (pybind11::isinstance<pybind11::float_>(x)) {
    aop.set_attr<float>(attr, x.template cast<float>());
  } else if (pybind11::isinstance<pybind11::str>(x)) {
    aop.set_attr<std::string>(attr, x.template cast<std::string>());
  } else {
    assert(!"unknown attribute type");
  }
}

void bind_op(pybind11::module& m) {
  pybind11::class_<dnnl::graph::op> opr(m, "op");

  opr.def(pybind11::init<size_t, dnnl::graph::op::kind, std::string>());
  opr.def(pybind11::init([](size_t id,
                            dnnl::graph::op::kind kind,
                            const std::vector<dnnl::graph::logical_tensor>& inputs,
                            const std::vector<dnnl::graph::logical_tensor>& outputs,
                            std::string name) {
    auto aop = dnnl::graph::op(id, kind, inputs, outputs, name);
    return aop;
  }));
  opr.def("set_attr", [](dnnl::graph::op& aop, dnnl::graph::op::attr key, pybind11::object val) {
    set_op_attribute(aop, val, key);
  });
  opr.def("add_input", &dnnl::graph::op::add_input);
  opr.def("add_inputs", &dnnl::graph::op::add_inputs);
  opr.def("add_output", &dnnl::graph::op::add_output);
  opr.def("add_outputs", &dnnl::graph::op::add_outputs);

  pybind11::enum_<dnnl::graph::op::kind>(opr, "kind")
      .value("Abs", dnnl::graph::op::kind::Abs)
      .value("AbsBackward", dnnl::graph::op::kind::AbsBackward)
      .value("Add", dnnl::graph::op::kind::Add)
      .value("AvgPool", dnnl::graph::op::kind::AvgPool)
      .value("AvgPoolBackward", dnnl::graph::op::kind::AvgPoolBackward)
      .value("BatchNormForwardTraining", dnnl::graph::op::kind::BatchNormForwardTraining)
      .value("BatchNormInference", dnnl::graph::op::kind::BatchNormInference)
      .value("BatchNormTrainingBackward", dnnl::graph::op::kind::BatchNormTrainingBackward)
      .value("BiasAdd", dnnl::graph::op::kind::BiasAdd)
      .value("BiasAddBackward", dnnl::graph::op::kind::BiasAddBackward)
      .value("Clamp", dnnl::graph::op::kind::Clamp)
      .value("ClampBackward", dnnl::graph::op::kind::ClampBackward)
      .value("Concat", dnnl::graph::op::kind::Concat)
      .value("Convolution", dnnl::graph::op::kind::Convolution)
      .value("ConvolutionBackwardData", dnnl::graph::op::kind::ConvolutionBackwardData)
      .value("ConvolutionBackwardWeights", dnnl::graph::op::kind::ConvolutionBackwardWeights)
      .value("ConvTranspose", dnnl::graph::op::kind::ConvTranspose)
      .value("ConvTransposeBackwardData", dnnl::graph::op::kind::ConvTransposeBackwardData)
      .value(
          "ConvTransposeBackwardWeights",
          dnnl::graph::op::kind::ConvTransposeBackwardWeights)
      .value("Dequantize", dnnl::graph::op::kind::Dequantize)
      .value("Divide", dnnl::graph::op::kind::Divide)
      .value("DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize)
      .value("DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize)
      .value("Elu", dnnl::graph::op::kind::Elu)
      .value("EluBackward", dnnl::graph::op::kind::EluBackward)
      .value("End", dnnl::graph::op::kind::End)
      .value("Exp", dnnl::graph::op::kind::Exp)
      .value("GELU", dnnl::graph::op::kind::GELU)
      .value("GELUBackward", dnnl::graph::op::kind::GELUBackward)
      .value("HardSigmoid", dnnl::graph::op::kind::HardSigmoid)
      .value("HardSigmoidBackward", dnnl::graph::op::kind::HardSigmoidBackward)
      .value("HardSwish", dnnl::graph::op::kind::HardSwish)
      .value("HardSwishBackward", dnnl::graph::op::kind::HardSwishBackward)
      .value("Interpolate", dnnl::graph::op::kind::Interpolate)
      .value("InterpolateBackward", dnnl::graph::op::kind::InterpolateBackward)
      .value("LayerNorm", dnnl::graph::op::kind::LayerNorm)
      .value("LayerNormBackward", dnnl::graph::op::kind::LayerNormBackward)
      .value("LeakyReLU", dnnl::graph::op::kind::LeakyReLU)
      .value("Log", dnnl::graph::op::kind::Log)
      .value("LogSoftmax", dnnl::graph::op::kind::LogSoftmax)
      .value("LogSoftmaxBackward", dnnl::graph::op::kind::LogSoftmaxBackward)
      .value("MatMul", dnnl::graph::op::kind::MatMul)
      .value("Maximum", dnnl::graph::op::kind::Maximum)
      .value("MaxPool", dnnl::graph::op::kind::MaxPool)
      .value("MaxPoolBackward", dnnl::graph::op::kind::MaxPoolBackward)
      .value("Minimum", dnnl::graph::op::kind::Minimum)
      .value("Mish", dnnl::graph::op::kind::Mish)
      .value("MishBackward", dnnl::graph::op::kind::MishBackward)
      .value("Multiply", dnnl::graph::op::kind::Multiply)
      .value("Pow", dnnl::graph::op::kind::Pow)
      .value("PReLU", dnnl::graph::op::kind::PReLU)
      .value("PReLUBackward", dnnl::graph::op::kind::PReLUBackward)
      .value("Quantize", dnnl::graph::op::kind::Quantize)
      .value("Reciprocal", dnnl::graph::op::kind::Reciprocal)
      .value("ReduceL1", dnnl::graph::op::kind::ReduceL1)
      .value("ReduceL2", dnnl::graph::op::kind::ReduceL2)
      .value("ReduceMax", dnnl::graph::op::kind::ReduceMax)
      .value("ReduceMean", dnnl::graph::op::kind::ReduceMean)
      .value("ReduceMin", dnnl::graph::op::kind::ReduceMin)
      .value("ReduceProd", dnnl::graph::op::kind::ReduceProd)
      .value("ReduceSum", dnnl::graph::op::kind::ReduceSum)
      .value("ReLU", dnnl::graph::op::kind::ReLU)
      .value("ReLUBackward", dnnl::graph::op::kind::ReLUBackward)
      .value("Reorder", dnnl::graph::op::kind::Reorder)
      .value("Round", dnnl::graph::op::kind::Round)
      .value("Select", dnnl::graph::op::kind::Select)
      .value("Sigmoid", dnnl::graph::op::kind::Sigmoid)
      .value("SigmoidBackward", dnnl::graph::op::kind::SigmoidBackward)
      .value("SoftMax", dnnl::graph::op::kind::SoftMax)
      .value("SoftMaxBackward", dnnl::graph::op::kind::SoftMaxBackward)
      .value("SoftPlus", dnnl::graph::op::kind::SoftPlus)
      .value("SoftPlusBackward", dnnl::graph::op::kind::SoftPlusBackward)
      .value("Sqrt", dnnl::graph::op::kind::Sqrt)
      .value("SqrtBackward", dnnl::graph::op::kind::SqrtBackward)
      .value("Square", dnnl::graph::op::kind::Square)
      .value("SquaredDifference", dnnl::graph::op::kind::SquaredDifference)
      .value("StaticReshape", dnnl::graph::op::kind::StaticReshape)
      .value("StaticTranspose", dnnl::graph::op::kind::StaticTranspose)
      .value("Subtract", dnnl::graph::op::kind::Subtract)
      .value("Tanh", dnnl::graph::op::kind::Tanh)
      .value("TanhBackward", dnnl::graph::op::kind::TanhBackward)
      .value("TypeCast", dnnl::graph::op::kind::TypeCast)
      .value("Wildcard", dnnl::graph::op::kind::Wildcard)
      .export_values();

  pybind11::enum_<dnnl::graph::op::attr>(opr, "attr")
      .value("undef", dnnl::graph::op::attr::undef)
      .value("alpha", dnnl::graph::op::attr::alpha)
      .value("beta", dnnl::graph::op::attr::beta)
      .value("epsilon", dnnl::graph::op::attr::epsilon)
      .value("max", dnnl::graph::op::attr::max)
      .value("min", dnnl::graph::op::attr::min)
      .value("momentum", dnnl::graph::op::attr::momentum)
      .value("scales", dnnl::graph::op::attr::scales)
      .value("axis", dnnl::graph::op::attr::axis)
      .value("begin_norm_axis", dnnl::graph::op::attr::begin_norm_axis)
      .value("groups", dnnl::graph::op::attr::groups)
      .value("axes", dnnl::graph::op::attr::axes)
      .value("dilations", dnnl::graph::op::attr::dilations)
      .value("dst_shape", dnnl::graph::op::attr::dst_shape)
      .value("kernel", dnnl::graph::op::attr::kernel)
      .value("order", dnnl::graph::op::attr::order)
      .value("output_padding", dnnl::graph::op::attr::output_padding)
      .value("pads_begin", dnnl::graph::op::attr::pads_begin)
      .value("pads_end", dnnl::graph::op::attr::pads_end)
      .value("shape", dnnl::graph::op::attr::shape)
      .value("sizes", dnnl::graph::op::attr::sizes)
      .value("src_shape", dnnl::graph::op::attr::src_shape)
      .value("strides", dnnl::graph::op::attr::strides)
      .value("weights_shape", dnnl::graph::op::attr::weights_shape)
      .value("zps", dnnl::graph::op::attr::zps)
      .value("exclude_pad", dnnl::graph::op::attr::exclude_pad)
      .value("keep_dims", dnnl::graph::op::attr::keep_dims)
      .value("keep_stats", dnnl::graph::op::attr::keep_stats)
      .value("per_channel_broadcast", dnnl::graph::op::attr::per_channel_broadcast)
      .value("special_zero", dnnl::graph::op::attr::special_zero)
      .value("transpose_a", dnnl::graph::op::attr::transpose_a)
      .value("transpose_b", dnnl::graph::op::attr::transpose_b)
      .value("use_affine", dnnl::graph::op::attr::use_affine)
      .value("use_dst", dnnl::graph::op::attr::use_dst)
      .value("auto_broadcast", dnnl::graph::op::attr::auto_broadcast)
      .value("auto_pad", dnnl::graph::op::attr::auto_pad)
      .value(
          "coordinate_transformation_mode",
          dnnl::graph::op::attr::coordinate_transformation_mode)
      .value("data_format", dnnl::graph::op::attr::data_format)
      .value("mode", dnnl::graph::op::attr::mode)
      .value("qtype", dnnl::graph::op::attr::qtype)
      .value("rounding_type", dnnl::graph::op::attr::rounding_type)
      .value("weights_format", dnnl::graph::op::attr::weights_format)
      .export_values();
}

void bind_partition(pybind11::module& m) {
  pybind11::class_<dnnl::graph::partition> p(m, "partition");

  p.def(pybind11::init<>());
  p.def(pybind11::init(
      [](const dnnl::graph::op& op, dnnl::graph::engine::kind ekind) { return dnnl::graph::partition(op, ekind); }));
  p.def("get_ops_num", &dnnl::graph::partition::get_ops_num);
  p.def("get_ops", &dnnl::graph::partition::get_ops);
  p.def("get_id", &dnnl::graph::partition::get_id);
  p.def("is_supported", &dnnl::graph::partition::is_supported);
  p.def("get_input_ports", &dnnl::graph::partition::get_input_ports);
  p.def("get_output_ports", &dnnl::graph::partition::get_output_ports);
  p.def("get_engine_kind", &dnnl::graph::partition::get_engine_kind);
  p.def("compile", &dnnl::graph::partition::compile);

  pybind11::enum_<dnnl::graph::partition::policy>(p, "policy")
      .value("fusion", dnnl::graph::partition::policy::fusion)
      .value("debug", dnnl::graph::partition::policy::debug)
      .export_values();
}

void bind_stream(pybind11::module& m) {
  pybind11::class_<dnnl::graph::stream> strm(m, "stream");

  strm.def(pybind11::init<dnnl::graph::engine&>());
  strm.def("get_engine", &dnnl::graph::stream::get_engine);
  strm.def("wait", &dnnl::graph::stream::wait);
}

static size_t size_of(dnnl::graph::logical_tensor::data_type dtype) {
  switch (dtype) {
    case dnnl::graph::logical_tensor::data_type::f32:
    case dnnl::graph::logical_tensor::data_type::s32:
      return 4U;
    case dnnl::graph::logical_tensor::data_type::s8:
    case dnnl::graph::logical_tensor::data_type::u8:
      return 1U;
    case dnnl::graph::logical_tensor::data_type::f16:
    case dnnl::graph::logical_tensor::data_type::bf16:
      return 2U;
    case dnnl::graph::logical_tensor::data_type::boolean:
      return sizeof(bool);
    default:
      return 0;
  }
}

static std::string format_string(dnnl::graph::logical_tensor::data_type dtype) {
  switch (dtype) {
    case dnnl::graph::logical_tensor::data_type::f32:
    case dnnl::graph::logical_tensor::data_type::f16:
    case dnnl::graph::logical_tensor::data_type::bf16:
      return pybind11::format_descriptor<float>::format();
      break;
    case dnnl::graph::logical_tensor::data_type::u8:
      return pybind11::format_descriptor<uint8_t>::format();
      break;
    case dnnl::graph::logical_tensor::data_type::s8:
      return pybind11::format_descriptor<int8_t>::format();
      break;
    case dnnl::graph::logical_tensor::data_type::boolean:
      return pybind11::format_descriptor<bool>::format();
      break;
    case dnnl::graph::logical_tensor::data_type::s32:
      return pybind11::format_descriptor<int32_t>::format();
      break;
    default:
      throw std::runtime_error("unknown data type");
  }
}

pybind11::buffer_info to_buffer_info(dnnl::graph::tensor& t, dnnl::graph::logical_tensor& lt) {
  auto strides = lt.get_strides();
  auto shapes = lt.get_dims();
  auto dtype = lt.get_data_type();
  std::transform(
      strides.begin(), strides.end(), strides.begin(), [&](int64_t i) {
        return i * size_of(dtype);
      });
  return pybind11::buffer_info(
      t.get_data_handle(), /* Pointer to buffer */
      size_of(dtype), /* Size of one scalar */
      format_string(dtype), /* Python struct-style format descriptor */
      shapes.size(), /* Number of dimensions */
      shapes, /* Buffer dimensions */
      strides);
}

dnnl::graph::logical_tensor::data_type convert_from_array_dtype(const pybind11::array& a) {
  auto tgt_dtype = a.dtype();
  if (tgt_dtype.is(pybind11::dtype::of<float>())) {
    return dnnl::graph::logical_tensor::data_type::f32;
  } else if (tgt_dtype.is(pybind11::dtype::of<int8_t>())) {
    return dnnl::graph::logical_tensor::data_type::s8;
  } else if (tgt_dtype.is(pybind11::dtype::of<uint8_t>())) {
    return dnnl::graph::logical_tensor::data_type::u8;
  } else if (tgt_dtype.is(pybind11::dtype::of<int32_t>())) {
    return dnnl::graph::logical_tensor::data_type::s32;
  } else if (tgt_dtype.is(pybind11::dtype::of<bool>())) {
    return dnnl::graph::logical_tensor::data_type::boolean;
  } else {
    // not support fp16 and bf16 yet
    assert(!"unsupported data type.");
  }
  return dnnl::graph::logical_tensor::data_type::undef;
}

void bind_tensor(pybind11::module& m) {
  pybind11::class_<dnnl::graph::tensor> t(m, "tensor", pybind11::buffer_protocol());

  t.def(pybind11::init([](dnnl::graph::logical_tensor& lt, dnnl::graph::engine& eng, pybind11::buffer b) {
    auto bufinfo = b.request();
    return dnnl::graph::tensor(lt, eng, bufinfo.ptr);
  }));
  t.def(pybind11::init([](dnnl::graph::logical_tensor& lt, dnnl::graph::engine& eng) {
    return dnnl::graph::tensor(lt, eng, nullptr);
  }));
  t.def(pybind11::init([](dnnl::graph::logical_tensor& lt, dnnl::graph::engine& eng, int64_t data_ptr) {
    return dnnl::graph::tensor(lt, eng, reinterpret_cast<void*>(data_ptr));
  }));
  t.def(
      "set_data_handle",
      [](dnnl::graph::tensor& self, int64_t data_ptr) {
        self.set_data_handle(reinterpret_cast<void*>(data_ptr));
      },
      pybind11::arg("data_ptr"));
  t.def("get_data_handle", [](dnnl::graph::tensor& self) {
    return reinterpret_cast<int64_t>(self.get_data_handle());
  });
  t.def("get_engine", &dnnl::graph::tensor::get_engine);
  t.def("from_numpy", [](pybind11::array& b, const dnnl::graph::engine& eng) {
    // create a logical tensor with id `0`
    dnnl::graph::logical_tensor::dims shape{b.shape(), b.shape() + b.ndim()};
    dnnl::graph::logical_tensor::dims strides{b.strides(), b.strides() + b.ndim()};
    dnnl::graph::logical_tensor lt{0, convert_from_array_dtype(b), shape, strides};
    // get mutable pointer from array
    return dnnl::graph::tensor{lt, eng, b.mutable_data()};
  });
  t.def(
      "to_numpy",
      [](dnnl::graph::tensor& self, dnnl::graph::logical_tensor& lt) {
        auto bufinfo = to_buffer_info(self, lt);
        // pass any valid pybind object to `base` to make sure there is
        // no memory copy between returned array and self tensor, see
        // details at https://github.com/pybind/pybind11/issues/323
        return pybind11::array(
            pybind11::dtype(bufinfo),
            bufinfo.shape,
            bufinfo.strides,
            bufinfo.ptr,
            /* base = */ pybind11::str{});
      },
      pybind11::arg("lt"));
}

void bind_status(pybind11::module& m) {
  pybind11::enum_<dnnl::graph::status>(m, "status")
      .value("success", dnnl::graph::status::success)
      .value("out_of_memory", dnnl::graph::status::out_of_memory)
      .value("invalid_arguments", dnnl::graph::status::invalid_arguments)
      .value("unimplemented", dnnl::graph::status::unimplemented)
      .value("last_impl_reached", dnnl::graph::status::last_impl_reached)
      .value("runtime_error", dnnl::graph::status::runtime_error)
      .value("not_required", dnnl::graph::status::not_required)
      .value("invalid_graph", dnnl::graph::status::invalid_graph)
      .value("invalid_graph_op", dnnl::graph::status::invalid_graph_op)
      .value("invalid_shape", dnnl::graph::status::invalid_shape)
      .value("invalid_data_type", dnnl::graph::status::invalid_data_type)
      .export_values();
}

void bind_compiled_partition_cache_capacity(pybind11::module& m) {
  m.def(
      "set_compiled_partition_cache_capacity",
      &dnnl::graph::set_compiled_partition_cache_capacity);
  m.def(
      "get_compiled_partition_cache_capacity",
      &dnnl::graph::get_compiled_partition_cache_capacity);
}

void bind_constant_tensor_cache(pybind11::module& m) {
  m.def("set_constant_tensor_cache", &dnnl::graph::set_constant_tensor_cache);
  m.def("get_constant_tensor_cache", &dnnl::graph::get_constant_tensor_cache);
}

void initOnednnGraphPythonBindings(pybind11::module& m) {
  m.doc() = R"pbdoc(
        oneDNN Graph API Python binding
        -------------------------------
        .. currentmodule:: onednn_graph_python_binding
        .. autosummary::
           :toctree: _generate
    )pbdoc";

  bind_status(m);
  bind_graph(m);
  bind_logical_tensor(m);
  bind_engine(m);
  bind_op(m);
  bind_tensor(m);
  bind_partition(m);
  bind_cpartition(m);
  bind_stream(m);
  bind_compiled_partition_cache_capacity(m);
  bind_constant_tensor_cache(m);
}

} // namespace ideep

#endif
