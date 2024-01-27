#pragma once

#include "pycolmap/pipeline/extract_features.h"
#include "pycolmap/pipeline/images.h"
#include "pycolmap/pipeline/match_features.h"
#include "pycolmap/pipeline/meshing.h"
#include "pycolmap/pipeline/mvs.h"
#include "pycolmap/pipeline/sfm.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindPipeline(py::module& m) {
  BindImages(m);
  BindExtractFeatures(m);
  BindMatchFeatures(m);
  BindSfM(m);
  BindMVS(m);
  BindMeshing(m);
}
