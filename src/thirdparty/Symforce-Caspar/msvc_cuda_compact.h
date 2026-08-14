#pragma once

#ifdef _MSC_VER
// Caspar CUDA sources use 'uint', which is not defined by MSVC.
typedef unsigned int uint;
#endif
