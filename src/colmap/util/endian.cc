// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/endian.h"

#include "colmap/util/logging.h"

#include <boost/predef/other/endian.h>

namespace colmap {

bool IsLittleEndian() {
#if BOOST_ENDIAN_LITTLE_BYTE
  return true;
#elif BOOST_ENDIAN_LITTLE_WORD
  // We do not support such exotic architectures.
  LOG(FATAL) << "Unsupported byte ordering";
#else
  return false;
#endif
}

bool IsBigEndian() {
#if BOOST_ENDIAN_BIG_BYTE
  return true;
#elif BOOST_ENDIAN_BIG_WORD
  // We do not support such exotic architectures.
  LOG(FATAL) << "Unsupported byte ordering";
#else
  return false;
#endif
}

}  // namespace colmap
