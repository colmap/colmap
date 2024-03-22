#pragma once

#include "colmap/util/timer.h"

#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

void BindTimer(py::module& m) {
  py::class_<Timer>(m, "Timer")
    .def(py::init<>())
    .def("Start", &Timer::Start)
    .def("Restart", &Timer::Restart)
    .def("Pause", &Timer::Pause)
    .def("Resume", &Timer::Resume)
    .def("Reset", &Timer::Reset)
    .def("ElapsedMicroSeconds", &Timer::ElapsedMicroSeconds)
    .def("ElapsedSeconds", &Timer::ElapsedSeconds)
    .def("ElapsedMinutes", &Timer::ElapsedMinutes)
    .def("ElapsedHours", &Timer::ElapsedHours)
    .def("PrintSeconds", &Timer::PrintSeconds)
    .def("PrintMinutes", &Timer::PrintMinutes)
    .def("PrintHours", &Timer::PrintHours);
}
