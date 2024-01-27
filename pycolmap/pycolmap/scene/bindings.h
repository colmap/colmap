#pragma once

#include "pycolmap/scene/camera.h"
#include "pycolmap/scene/correspondence_graph.h"
#include "pycolmap/scene/database.h"
#include "pycolmap/scene/image.h"
#include "pycolmap/scene/point2D.h"
#include "pycolmap/scene/point3D.h"
#include "pycolmap/scene/reconstruction.h"
#include "pycolmap/scene/track.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindScene(py::module& m) {
  BindImage(m);
  BindCamera(m);
  BindPoint2D(m);
  BindTrack(m);
  BindPoint3D(m);
  BindCorrespondenceGraph(m);
  BindReconstruction(m);
  BindDatabase(m);
}
