#ifndef _IDEEP_PIN_SINGLETONS_HPP_
#define _IDEEP_PIN_SINGLETONS_HPP_

#include <ideep.hpp>

/// Put these in only one library
ideep::engine &ideep::engine::cpu_engine() {
  static engine cpu_engine;
  return cpu_engine;
}

#endif
