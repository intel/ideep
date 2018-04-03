#ifndef _IDEEP_PIN_SINGLETONS_HPP_
#define _IDEEP_PIN_SINGLETONS_HPP_

#include "ideep.hpp"

namespace ideep {
/// Put these in only one library
engine &engine::cpu_engine() {
  static engine cpu_engine;
  return cpu_engine;
}

}

#endif
