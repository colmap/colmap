#include "colmap/estimators/imu_preintegration.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindImuPreintegration(py::module& m) {
  using ImuPreITGOpt = ImuPreintegrationOptions;
  py::class_<ImuPreITGOpt> PyImuPreintegrationOptions(
      m, "ImuPreintegrationOptions");
  PyImuPreintegrationOptions.def(py::init<>())
      .def_readwrite("use_integration_noise",
                     &ImuPreITGOpt::use_integration_noise)
      .def_readwrite("integration_noise_density",
                     &ImuPreITGOpt::integration_noise_density)
      .def_readwrite("reintegrate_vel_norm_thres",
                     &ImuPreITGOpt::reintegrate_vel_norm_thres)
      .def_readwrite("reintegrate_angle_norm_thres",
                     &ImuPreITGOpt::reintegrate_angle_norm_thres);

  using PIM = PreintegratedImuMeasurement;
  py::class_<PIM> PyPreintegratedImuMeasurement(m,
                                                "PreintegratedImuMeasurement");
  PyPreintegratedImuMeasurement
      .def(py::init<const ImuPreintegrationOptions&,
                    const ImuCalibration&,
                    const double,
                    const double>(),
           "options"_a,
           "calib"_a,
           "t_start"_a,
           "t_end"_a)
      .def("reset", &PIM::Reset)
      .def("has_started", &PIM::HasStarted)
      .def("set_acc_rect_mat", &PIM::SetAccRectMat, "mat"_a)
      .def("set_gyro_rect_mat", &PIM::SetGyroRectMat, "mat"_a)
      .def("set_biases", &PIM::SetBiases, "biases"_a)
      .def("add_measurement", &PIM::AddMeasurement, "measurement"_a)
      .def("add_measurements", &PIM::AddMeasurements, "measurements"_a)
      .def("finish", &PIM::Finish)
      .def("has_finished", &PIM::HasFinished)
      .def("check_reintegrate", &PIM::CheckReintegrate, "biases"_a)
      .def("reintegrate", py::overload_cast<>(&PIM::Reintegrate))
      .def("reintegrate",
           py::overload_cast<const Eigen::Vector6d&>(&PIM::Reintegrate),
           "biases"_a)
      .def_property_readonly("delta_t", &PIM::DeltaT)
      .def_property_readonly("delta_R", &PIM::DeltaR)
      .def_property_readonly("delta_p", &PIM::DeltaP)
      .def_property_readonly("delta_v", &PIM::DeltaV)
      .def_property_readonly("dR_dbg", &PIM::dR_dbg)
      .def_property_readonly("dp_dba", &PIM::dp_dba)
      .def_property_readonly("dp_dbg", &PIM::dp_dbg)
      .def_property_readonly("dv_dba", &PIM::dv_dba)
      .def_property_readonly("dv_dbg", &PIM::dv_dbg)
      .def_property_readonly("biases", &PIM::Biases)
      .def_property_readonly("covariance", &PIM::Covariance)
      .def_property_readonly("sqrt_information", &PIM::SqrtInformation)
      .def_property_readonly("gravity_magnitude", &PIM::GravityMagnitude)
      .def_property_readonly("measurements", &PIM::Measurements);

  m.def("PreintegratedImuMeasurementCost",
        &PreintegratedImuMeasurementCostFunction::Create,
        "preintegrated_imu_measurement"_a);
}
