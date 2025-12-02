#pragma once

#include "glomap/controllers/global_mapper.h"

namespace glomap {

// Use default values for most of the settings from database
int RunGlobalMapper(int argc, char** argv);

// Use default values for most of the settings from colmap reconstruction
int RunGlobalMapperResume(int argc, char** argv);

}  // namespace glomap
