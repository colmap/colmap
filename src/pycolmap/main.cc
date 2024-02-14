#include "colmap/math/random.h"
#include "colmap/util/version.h"

#include "pycolmap/estimators/bindings.h"
#include "pycolmap/feature/sift.h"
#include "pycolmap/geometry/bindings.h"
#include "pycolmap/helpers.h"
#include "pycolmap/optim/bindings.h"
#include "pycolmap/pipeline/bindings.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/bindings.h"
#include "pycolmap/sfm/bindings.h"
#include "pycolmap/utils.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace colmap;

struct Logging {
  // TODO: Replace with google::LogSeverity in glog >= v0.7.0
  enum class LogSeverity {
    GLOG_INFO = google::GLOG_INFO,
    GLOG_WARNING = google::GLOG_WARNING,
    GLOG_ERROR = google::GLOG_ERROR,
    GLOG_FATAL = google::GLOG_FATAL,
  };
};  // dummy class

std::pair<std::string, int> GetPythonCallFrame() {
  const auto frame = py::module_::import("sys").attr("_getframe")(0);
  const std::string file = py::str(frame.attr("f_code").attr("co_filename"));
  const std::string function = py::str(frame.attr("f_code").attr("co_name"));
  const int line = py::int_(frame.attr("f_lineno"));
  return std::make_pair(file + ":" + function, line);
}

void BindLogging(py::module& m) {
  py::class_<Logging> PyLogging(m, "logging", py::module_local());
  PyLogging.def_readwrite_static("minloglevel", &FLAGS_minloglevel)
      .def_readwrite_static("stderrthreshold", &FLAGS_stderrthreshold)
      .def_readwrite_static("log_dir", &FLAGS_log_dir)
      .def_readwrite_static("logtostderr", &FLAGS_logtostderr)
      .def_readwrite_static("alsologtostderr", &FLAGS_alsologtostderr)
      .def_readwrite_static("verbose_level", &FLAGS_v)
      .def_static(
          "set_log_destination",
          [](const Logging::LogSeverity severity, const std::string& path) {
            google::SetLogDestination(
                static_cast<google::LogSeverity>(severity), path.c_str());
          })
      .def_static(
          "verbose",
          [](const int level, const std::string& msg) {
            if (VLOG_IS_ON(level)) {
              const auto frame = GetPythonCallFrame();
              google::LogMessage(frame.first.c_str(), frame.second).stream()
                  << msg;
            }
          })
      .def_static(
          "info",
          [](const std::string& msg) {
            const auto frame = GetPythonCallFrame();
            google::LogMessage(frame.first.c_str(), frame.second).stream()
                << msg;
          })
      .def_static("warning",
                  [](const std::string& msg) {
                    const auto frame = GetPythonCallFrame();
                    google::LogMessage(
                        frame.first.c_str(), frame.second, google::GLOG_WARNING)
                            .stream()
                        << msg;
                  })
      .def_static("error",
                  [](const std::string& msg) {
                    const auto frame = GetPythonCallFrame();
                    google::LogMessage(
                        frame.first.c_str(), frame.second, google::GLOG_ERROR)
                            .stream()
                        << msg;
                  })
      .def_static("fatal", [](const std::string& msg) {
        const auto frame = GetPythonCallFrame();
        google::LogMessageFatal(frame.first.c_str(), frame.second).stream()
            << msg;
      });
  py::enum_<Logging::LogSeverity>(PyLogging, "Level")
      .value("INFO", Logging::LogSeverity::GLOG_INFO)
      .value("WARNING", Logging::LogSeverity::GLOG_WARNING)
      .value("ERROR", Logging::LogSeverity::GLOG_ERROR)
      .value("FATAL", Logging::LogSeverity::GLOG_FATAL)
      .export_values();

#if defined(GLOG_VERSION_MAJOR) && \
    (GLOG_VERSION_MAJOR > 0 || GLOG_VERSION_MINOR >= 6)
  if (!google::IsGoogleLoggingInitialized())
#endif
  {
    google::InitGoogleLogging("");
    google::InstallFailureSignalHandler();
  }
  FLAGS_alsologtostderr = true;
}

PYBIND11_MODULE(pycolmap, m) {
  m.doc() = "COLMAP plugin";
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#else
  m.attr("__version__") = py::str("dev");
#endif
  m.attr("has_cuda") = IsGPU(Device::AUTO);
  m.attr("COLMAP_version") = py::str(GetVersionInfo());
  m.attr("COLMAP_build") = py::str(GetBuildInfo());

  auto PyDevice = py::enum_<Device>(m, "Device")
                      .value("auto", Device::AUTO)
                      .value("cpu", Device::CPU)
                      .value("cuda", Device::CUDA);
  AddStringToEnumConstructor(PyDevice);

  BindLogging(m);
  BindGeometry(m);
  BindOptim(m);
  BindScene(m);
  BindEstimators(m);
  BindSfMObjects(m);
  BindSift(m);
  BindPipeline(m);

  m.def("set_random_seed",
        &SetPRNGSeed,
        "Initialize the PRNG with the given seed.");

  py::add_ostream_redirect(m, "ostream");
}
