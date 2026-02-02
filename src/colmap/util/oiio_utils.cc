#include "colmap/util/oiio_utils.h"

#include <cstdlib>
#include <mutex>

namespace colmap {

static std::once_flag oiio_setup_flag;

void EnsureOpenImageIOInitialized() {
  std::call_once(oiio_setup_flag, []() {
    OIIO::attribute("threads", 1);
    OIIO::attribute("exr_threads", 1);

#if OIIO_VERSION >= OIIO_MAKE_VERSION(2, 5, 3)
    std::atexit([]() { OIIO::shutdown(); });
#endif
  });
}

}  // namespace colmap
