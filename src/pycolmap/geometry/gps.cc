#include "colmap/geometry/gps.h"

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindGPS(py::module& m) {
  py::enum_<GPSTransform::Ellipsoid> PyGPSTransfromEllipsoid(
      m, "GPSTransfromEllipsoid");
  PyGPSTransfromEllipsoid.value("GRS80", GPSTransform::Ellipsoid::GRS80)
      .value("WGS84", GPSTransform::Ellipsoid::WGS84);
  AddStringToEnumConstructor(PyGPSTransfromEllipsoid);

  py::class_ext_<GPSTransform> PyGPSTransform(m, "GPSTransform");
  PyGPSTransform
      .def(py::init<const GPSTransform::Ellipsoid>(),
           "ellipsoid"_a = GPSTransform::Ellipsoid::GRS80)
      .def("ellipsoid_to_ecef", &GPSTransform::EllipsoidToECEF, "lat_lon_alt"_a)
      .def("ecef_to_ellipsoid", &GPSTransform::ECEFToEllipsoid, "xyz_in_ecef"_a)
      .def("ellipsoid_to_enu",
           &GPSTransform::EllipsoidToENU,
           "lat_lon_alt"_a,
           "ref_lat"_a,
           "ref_lon"_a)
      .def("ecef_to_enu",
           &GPSTransform::ECEFToENU,
           "xyz_in_ecef"_a,
           "ref_lat"_a,
           "ref_lon"_a)
      .def("enu_to_ellipsoid",
           &GPSTransform::ENUToEllipsoid,
           "xyz_in_enu"_a,
           "ref_lat"_a,
           "ref_lon"_a,
           "ref_alt"_a)
      .def("enu_to_ecef",
           &GPSTransform::ENUToECEF,
           "xyz_in_enu"_a,
           "ref_lat"_a,
           "ref_lon"_a,
           "ref_alt"_a)
      .def("ellipsoid_to_utm", &GPSTransform::EllipsoidToUTM, "lat_lon_alt"_a)
      .def("utm_to_ellipsoid",
           &GPSTransform::UTMToEllipsoid,
           "xyz_in_utm"_a,
           "zone"_a,
           "is_north"_a);
}
