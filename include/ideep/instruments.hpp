#ifndef _ITTNOTIFY_HPP_
#define _ITTNOTIFY_HPP_
#include <ittnotify.h>

namespace ideep {
namespace instruments {
class ittnofity {
public:
  static void start() {
    __itt_pause();
  }

  static void resume() {
  }

  static void dettach() {
  }
};
}
}

#endif
