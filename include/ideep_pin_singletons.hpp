#ifndef _IDEEP_PIN_SINGLETONS_HPP_
#define _IDEEP_PIN_SINGLETONS_HPP_

#include "ideep.hpp"
#include "mkldnn_compat.hpp"

namespace ideep {

engine& engine::cpu_engine() {
  static engine cpu_engine(kind::cpu, 0);
  return cpu_engine;
}

engine& engine::gpu_engine() {
  static engine gpu_engine(kind::gpu, 0);
  return gpu_engine;
}

struct RegisterEngineAllocator {
  RegisterEngineAllocator(
      engine& eng,
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    // change runtime flag start with "MKLDNN_" to "DNNL_"
    EnvSetter env_setter;
    eng.set_allocator(malloc, free);
  }
};

} // namespace ideep

#endif
