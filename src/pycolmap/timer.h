#pragma once

#include "colmap/util/timer.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindTimer(py::module& m) {
  py::class_<colmap::Timer>(m, "Timer")
      .def(py::init<>())
      .def("start", &colmap::Timer::Start)
      .def("restart", &colmap::Timer::Restart)
      .def("pause", &colmap::Timer::Pause)
      .def("resume", &colmap::Timer::Resume)
      .def("reset", &colmap::Timer::Reset)
      .def("elapsed_micro_seconds", &colmap::Timer::ElapsedMicroSeconds)
      .def("elapsed_seconds", &colmap::Timer::ElapsedSeconds)
      .def("elapsed_minutes", &colmap::Timer::ElapsedMinutes)
      .def("elapsed_hours", &colmap::Timer::ElapsedHours)
      .def("print_seconds", &colmap::Timer::PrintSeconds)
      .def("print_minutes", &colmap::Timer::PrintMinutes)
      .def("print_hours", &colmap::Timer::PrintHours);
}
