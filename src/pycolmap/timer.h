#pragma once

#include "colmap/util/timer.h"

#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

void BindTimer(py::module& m) {
  py::class_<Timer>(m, "Timer")
      .def(py::init<>())
      .def("start", &Timer::Start)
      .def("restart", &Timer::Restart)
      .def("pause", &Timer::Pause)
      .def("resume", &Timer::Resume)
      .def("reset", &Timer::Reset)
      .def("elapsed_micro_seconds", &Timer::ElapsedMicroSeconds)
      .def("elapsed_seconds", &Timer::ElapsedSeconds)
      .def("elapsed_minutes", &Timer::ElapsedMinutes)
      .def("elapsed_hours", &Timer::ElapsedHours)
      .def("print_seconds", &Timer::PrintSeconds)
      .def("print_minutes", &Timer::PrintMinutes)
      .def("print_hours", &Timer::PrintHours);
}
