// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <OpenImageIO/imageio.h>

namespace colmap {

// Declaration of the thread-safe, one-time initialization function.
void EnsureOpenImageIOInitialized();

}  // namespace colmap
