/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

/**
 * Helper functions for working with python arrays.
 *
 * Caspar supports host arrays using the __array_interface__ attribute:
 * https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface
 * and device arrays using the __cuda_array_interface__ attribute:
 * https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html
 */

#pragma once

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace caspar {

size_t GetNumRows(const py::object& obj);
size_t GetNumCols(const py::object& obj);

void AssertHostMemory(const py::object& obj);
void AssertDeviceMemory(const py::object& obj);

void AssertNumRowsEquals(const py::object& obj, size_t n);
void AssertNumColsEquals(const py::object& obj, size_t n);

float* AsFloatPtr(const py::object& obj);
double* AsDoublePtr(const py::object& obj);
int* AsIntPtr(const py::object& obj);
uint* AsUintPtr(const py::object& obj);
uint8_t* AsCharPtr(const py::object& obj);
uint2* AsUint2Ptr(const py::object& obj);

}  // namespace caspar
