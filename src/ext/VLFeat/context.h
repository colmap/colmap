// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_EXT_VLFEAT_CONTEXT_H_
#define COLMAP_EXT_VLFEAT_CONTEXT_H_

extern "C" {
#include "generic.h"
}

// This class must be constructed before calling any VLFeat functions.
// It should be in the same translation unit as the VLFeat code, e.g.:
//
//    namespace {
//      VLContextManager vl_context_manager;
//    }
//
//    void Foo() {
//      vl_sift_new(...);
//    }
//
class VLContextManager {
 public:
  VLContextManager() { vl_constructor(); }
  ~VLContextManager() { vl_destructor(); }
};

#endif  // COLMAP_EXT_VLFEAT_CONTEXT_H_
