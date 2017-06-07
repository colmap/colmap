#ifndef COLMAP_SRC_BASE_SERIALIZATION_H_
#define COLMAP_SRC_BASE_SERIALIZATION_H_

#include <string>

#include "base/reconstruction.h"

namespace colmap {

// These are the main entry functions for serialization and the interface you
// should be using.
// If the filename provided is invalid the functions may throw or terminate.
template <typename Type>
void WriteToBinaryFile(const std::string& path, const Type& data);

template <typename Type>
void ReadFromBinaryFile(const std::string& path, Type* data);
}  // namespace colmap

#include "base/serialization_impl.h"

#endif  // COLMAP_SRC_BASE_SERIALIZATION_H_
