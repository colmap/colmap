#pragma once

#ifdef _MSC_VER
// Caspar uses 'uint' which is not defined in MSVC.
typedef unsigned int uint;

#ifdef __cplusplus
// solver.cc uses std::to_string but misses the <string> header.
#include <string>
#endif
#endif