#pragma once

#include <OpenImageIO/imageio.h>

namespace colmap {

// Declaration of the thread-safe, one-time initialization function.
void InitializeOpenImageIO();

}  // namespace colmap
