#ifndef _MKLDNN_COMPAT_HPP_
#define _MKLDNN_COMPAT_HPP_

#include "ideep.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace ideep {
struct EnvSetter {
  // oneDNN will only accept runtime flags which start with "DNNL_/ONEDNN_" from
  // version v2.5. If user setting runtime flags start with MKLDNN_, we need to
  // keep it works for a while before we finally deprecated it. This is a
  // compatibility layer for runtime flags start with MKLDNN_
  EnvSetter() {
    for (auto name : mkldnn_runtime_flags) {
      query_and_set_env(name.c_str());
    }
  }

  void query_and_set_env(std::string name) {
    std::string value;
    if (getenv_user(name, value)) {
      std::string dnnl_name = "DNNL_";
      dnnl_name += std::string(name);
#ifdef _WIN32
      SetEnvironmentVariable(dnnl_name.c_str(), value.c_str());
#else
      setenv(dnnl_name.c_str(), value.c_str(), 1);
#endif
    }
  }

  bool getenv_user(std::string name, std::string& value) {
    std::string name_str = "MKLDNN_" + std::string(name);
    size_t value_length = 0;
    const char* p = getenv(name_str.c_str());
    value_length = p == nullptr ? 0 : strlen(p);
    if (value_length > 0) {
      value += std::string(p);
      return true;
    }
    return false;
  }

  // current runtime flags in mkldnn
  const std::vector<std::string> mkldnn_runtime_flags = {
      "VERBOSE",
      "ITT_TASK_LEVEL",
      "PRIMITIVE_CACHE_CAPACITY",
      "SC_STACK_SIZE",
      "SC_SOFT_STACK_LIMIT",
      "JIT_PROFILE",
      "VERBOSE_TIMESTAMP",
      "DEFAULT_FPMATH_MODE",
      "MAX_CPU_ISA",
      "CPU_ISA_HINTS"};
};

} // namespace ideep

#endif
