#include "colmap/estimators/imu_preintegration.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindImuPreintegration(py::module& m) {
  auto PyImuIntegrationMethod =
      py::enum_<ImuIntegrationMethod>(m, "ImuIntegrationMethod")
          .value("MIDPOINT", ImuIntegrationMethod::MIDPOINT)
          .value("RK4", ImuIntegrationMethod::RK4);
  AddStringToEnumConstructor(PyImuIntegrationMethod);

  using ImuPreintegrationOptions = ImuPreintegrationOptions;
  py::classh<ImuPreintegrationOptions> PyImuPreintegrationOptions(
      m, "ImuPreintegrationOptions");
  PyImuPreintegrationOptions.def(py::init<>())
      .def_readwrite("method", &ImuPreintegrationOptions::method)
      .def_readwrite("integration_noise_density",
                     &ImuPreintegrationOptions::integration_noise_density)
      .def_readwrite("reintegrate_vel_norm_thres",
                     &ImuPreintegrationOptions::reintegrate_vel_norm_thres)
      .def_readwrite("reintegrate_angle_norm_thres",
                     &ImuPreintegrationOptions::reintegrate_angle_norm_thres);

  py::classh<PreintegratedImuData> PyPreintegratedImuData(
      m, "PreintegratedImuData");
  PyPreintegratedImuData.def(py::init<>())
      .def_readwrite("delta_t", &PreintegratedImuData::delta_t)
      .def_readwrite("delta_R", &PreintegratedImuData::delta_R)
      .def_readwrite("delta_p", &PreintegratedImuData::delta_p)
      .def_readwrite("delta_v", &PreintegratedImuData::delta_v)
      .def_readwrite("dR_dbg", &PreintegratedImuData::dR_dbg)
      .def_readwrite("dp_dbg", &PreintegratedImuData::dp_dbg)
      .def_readwrite("dv_dbg", &PreintegratedImuData::dv_dbg)
      .def_readwrite("dp_dba", &PreintegratedImuData::dp_dba)
      .def_readwrite("dv_dba", &PreintegratedImuData::dv_dba)
      .def_readwrite("biases", &PreintegratedImuData::biases)
      .def_readwrite("covariance", &PreintegratedImuData::covariance)
      .def_readwrite("sqrt_information",
                     &PreintegratedImuData::sqrt_information)
      .def_readwrite("gravity_magnitude",
                     &PreintegratedImuData::gravity_magnitude)
      .def("finalize", &PreintegratedImuData::Finalize);

  py::classh<ImuPreintegrator> PyImuPreintegrator(m, "ImuPreintegrator");
  PyImuPreintegrator
      .def(py::init<const ImuPreintegrationOptions&,
                    const ImuCalibration&,
                    timestamp_t,
                    timestamp_t>(),
           "options"_a,
           "calib"_a,
           "t_start"_a,
           "t_end"_a)
      .def("reset", &ImuPreintegrator::Reset)
      .def("has_started", &ImuPreintegrator::HasStarted)
      .def("set_linearization_biases",
           &ImuPreintegrator::SetLinearizationBiases,
           "biases"_a)
      .def("feed_imu",
           py::overload_cast<const ImuMeasurement&>(&ImuPreintegrator::FeedImu),
           "measurement"_a)
      .def("feed_imu",
           py::overload_cast<const std::vector<ImuMeasurement>&>(
               &ImuPreintegrator::FeedImu),
           "measurements"_a)
      .def("extract", &ImuPreintegrator::Extract)
      .def(
          "update",
          [](ImuPreintegrator& self, PreintegratedImuData& data) {
            self.Update(&data);
          },
          "data"_a,
          "Extract into an existing data object (in-place update).")
      .def("should_reintegrate",
           &ImuPreintegrator::ShouldReintegrate,
           "biases"_a)
      .def("reintegrate", py::overload_cast<>(&ImuPreintegrator::Reintegrate))
      .def("reintegrate",
           py::overload_cast<const Eigen::Vector6d&>(
               &ImuPreintegrator::Reintegrate),
           "biases"_a)
      .def_property_readonly("measurements", &ImuPreintegrator::Measurements);
}
