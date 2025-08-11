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

  py::enum_<GPSTransform::CartesianFrame> PyGPSTransfromCartesianFrame(
      m, "GPSTransfromCartesianFrame");
  PyGPSTransfromCartesianFrame.value("ECEF", GPSTransform::CartesianFrame::ECEF)
      .value("ENU", GPSTransform::CartesianFrame::ENU)
      .value("UTM", GPSTransform::CartesianFrame::UTM);
  AddStringToEnumConstructor(PyGPSTransfromCartesianFrame);

  py::class_ext_<GPSTransform> PyGPSTransform(m, "GPSTransform");
  PyGPSTransform
      .def(py::init<const GPSTransform::Ellipsoid>(),
           "ellipsoid"_a = GPSTransform::Ellipsoid::GRS80)

      .def("ellipsoid_to_ecef",
           py::overload_cast<const Eigen::Vector3d&>(
               &GPSTransform::EllipsoidToECEF, py::const_),
           "lat_lon_alt"_a)
      .def("ellipsoid_to_ecef",
           py::overload_cast<const std::vector<Eigen::Vector3d>&>(
               &GPSTransform::EllipsoidToECEF, py::const_),
           "lat_lon_alt"_a)

      .def("ecef_to_ellipsoid",
           py::overload_cast<const Eigen::Vector3d&>(
               &GPSTransform::ECEFToEllipsoid, py::const_),
           "xyz_in_ecef"_a)
      .def("ecef_to_ellipsoid",
           py::overload_cast<const std::vector<Eigen::Vector3d>&>(
               &GPSTransform::ECEFToEllipsoid, py::const_),
           "xyz_in_ecef"_a)

      .def("ellipsoid_to_enu",
           py::overload_cast<const Eigen::Vector3d&, double, double>(
               &GPSTransform::EllipsoidToENU, py::const_),
           "lat_lon_alt"_a,
           "ref_lat"_a,
           "ref_lon"_a)
      .def("ellipsoid_to_enu",
           py::overload_cast<const std::vector<Eigen::Vector3d>&,
                             double,
                             double>(&GPSTransform::EllipsoidToENU, py::const_),
           "lat_lon_alt"_a,
           "ref_lat"_a,
           "ref_lon"_a)

      .def("ecef_to_enu",
           py::overload_cast<const Eigen::Vector3d&, double, double>(
               &GPSTransform::ECEFToENU, py::const_),
           "xyz_in_ecef"_a,
           "ref_lat"_a,
           "ref_lon"_a)
      .def("ecef_to_enu",
           py::overload_cast<const std::vector<Eigen::Vector3d>&,
                             double,
                             double>(&GPSTransform::ECEFToENU, py::const_),
           "xyz_in_ecef"_a,
           "ref_lat"_a,
           "ref_lon"_a)

      .def("enu_to_ellipsoid",
           py::overload_cast<const Eigen::Vector3d&, double, double, double>(
               &GPSTransform::ENUToEllipsoid, py::const_),
           "xyz_in_enu"_a,
           "ref_lat"_a,
           "ref_lon"_a,
           "ref_alt"_a)
      .def("enu_to_ellipsoid",
           py::overload_cast<const std::vector<Eigen::Vector3d>&,
                             double,
                             double,
                             double>(&GPSTransform::ENUToEllipsoid, py::const_),
           "xyz_in_enu"_a,
           "ref_lat"_a,
           "ref_lon"_a,
           "ref_alt"_a)

      .def("enu_to_ecef",
           py::overload_cast<const Eigen::Vector3d&, double, double, double>(
               &GPSTransform::ENUToECEF, py::const_),
           "xyz_in_enu"_a,
           "ref_lat"_a,
           "ref_lon"_a,
           "ref_alt"_a)
      .def("enu_to_ecef",
           py::overload_cast<const std::vector<Eigen::Vector3d>&,
                             double,
                             double,
                             double>(&GPSTransform::ENUToECEF, py::const_),
           "xyz_in_enu"_a,
           "ref_lat"_a,
           "ref_lon"_a,
           "ref_alt"_a)

      .def("ellipsoid_to_utm",
           py::overload_cast<const Eigen::Vector3d&>(
               &GPSTransform::EllipsoidToUTM, py::const_),
           "lat_lon_alt"_a)
      .def("ellipsoid_to_utm",
           py::overload_cast<const std::vector<Eigen::Vector3d>&>(
               &GPSTransform::EllipsoidToUTM, py::const_),
           "lat_lon_alt"_a)

      .def("utm_to_ellipsoid",
           py::overload_cast<const Eigen::Vector3d&, int>(
               &GPSTransform::UTMToEllipsoid, py::const_),
           "xyz_in_utm"_a,
           "zone"_a)
      .def("utm_to_ellipsoid",
           py::overload_cast<const std::vector<Eigen::Vector3d>&, int>(
               &GPSTransform::UTMToEllipsoid, py::const_),
           "xyz_in_utm"_a,
           "zone"_a);
}
