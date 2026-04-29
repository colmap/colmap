/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "pybind_array_tools.h"

#include <algorithm>

namespace py = pybind11;

namespace caspar {

py::object GetInterface(const py::object& obj) {
  if (py::hasattr(obj, "__cuda_array_interface__")) {
    return obj.attr("__cuda_array_interface__");
  } else if (py::hasattr(obj, "__array_interface__")) {
    return obj.attr("__array_interface__");
  } else {
    const std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error(
        "Object does not have __cuda_array_interface__ or __array_interface__\n" + info);
  }
}

size_t GetNumRows(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  py::tuple shape = cuda_array_interface["shape"].cast<py::tuple>();
  return shape[0].cast<size_t>();
}

size_t GetNumCols(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  py::tuple shape = cuda_array_interface["shape"].cast<py::tuple>();
  return shape[1].cast<size_t>();
}

void AssertHostMemory(const py::object& obj) {
  if (!py::hasattr(obj, "__array_interface__")) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Object does not have expose __array_interface__\n" + info);
  }
}

void AssertDeviceMemory(const py::object& obj) {
  if (!py::hasattr(obj, "__cuda_array_interface__")) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Object does not have expose __cuda_array_interface__\n" + info);
  }
}

void AssertNumRowsEquals(const py::object& obj, size_t n) {
  if (GetNumRows(obj) != n) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected array with " + std::to_string(n) + " rows, got " +
                             std::to_string(GetNumRows(obj)) + " rows\n" + info);
  }
}

void AssertNumColsEquals(const py::object& obj, size_t n) {
  if (GetNumCols(obj) != n) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected array with " + std::to_string(n) + " cols, got " +
                             std::to_string(GetNumRows(obj)) + " cols\n" + info);
  }
}

void AssertContiguous(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  if (!cuda_array_interface["strides"].is_none()) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("This function only supports contiguous arrays\n" + info);
  }
}

void Assert1D(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  if (py::len(cuda_array_interface["shape"]) != 1) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected 1D array, got " +
                             std::to_string(py::len(cuda_array_interface["shape"])) + "D array\n" +
                             info);
  }
}

void Assert2D(const py::object& obj) {
  AssertContiguous(obj);
  py::object cuda_array_interface = GetInterface(obj);
  if (py::len(cuda_array_interface["shape"]) != 2) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected 2D array, got " +
                             std::to_string(py::len(cuda_array_interface["shape"])) + "D array\n" +
                             info);
  }
}

void Assert2DNxk(const py::object& obj, size_t k) {
  py::object cuda_array_interface = GetInterface(obj);
  Assert2D(obj);
  py::tuple shape = cuda_array_interface["shape"].cast<py::tuple>();
  if (shape[1].cast<int>() != 2) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error(
        "Expected 2D array with second dimension of size " + std::to_string(k) + ", got " +
        py::str(cuda_array_interface["shape"]).cast<std::string>() + "\n" + info);
  }
}

void AssertType(const py::object& obj, const std::vector<std::string>& types) {
  py::object cuda_array_interface = GetInterface(obj);
  std::string obj_type = cuda_array_interface["typestr"].cast<std::string>();
  if (std::find(types.begin(), types.end(), obj_type) == types.end()) {
    std::string types_str = "";
    for (const auto& type : types) {
      if (!types_str.empty()) {
        types_str += " or ";
      }
      types_str += type;
    }
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected array of type " + types_str + ", got " + obj_type + "\n" +
                             info);
  }
}

void AssertFloat(const py::object& obj) {
  AssertType(obj, {"<f4"});
}

void AssertDouble(const py::object& obj) {
  AssertType(obj, {"<f8"});
}

void AssertUint(const py::object& obj) {
  AssertType(obj, {"<u4", "<i4"});
}

void AssertChar(const py::object& obj) {
  AssertType(obj, {"|u1", "|i1"});
}

void AssertFloatVec(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  AssertFloat(obj);
  AssertContiguous(obj);
}

void AssertDoubleVec(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  AssertDouble(obj);
  AssertContiguous(obj);
}

void AssertUintVec(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  AssertUint(obj);
  AssertContiguous(obj);
}

void AssertCharVec(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  AssertChar(obj);
  AssertContiguous(obj);
}

void AssertUint2Vec(const py::object& obj) {
  py::object cuda_array_interface = GetInterface(obj);
  AssertUint(obj);
  Assert2DNxk(obj, 2);
}

float* AsFloatPtr(const py::object& obj) {
  AssertFloatVec(obj);
  py::tuple data = GetInterface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<float*>(data[0].cast<size_t>());
}

double* AsDoublePtr(const py::object& obj) {
  AssertDoubleVec(obj);
  py::tuple data = GetInterface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<double*>(data[0].cast<size_t>());
}

int* AsIntPtr(const py::object& obj) {
  AssertUintVec(obj);
  py::tuple data = GetInterface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<int*>(data[0].cast<size_t>());
}

uint* AsUintPtr(const py::object& obj) {
  AssertUintVec(obj);
  py::tuple data = GetInterface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<uint*>(data[0].cast<size_t>());
}

uint8_t* AsCharPtr(const py::object& obj) {
  AssertCharVec(obj);
  py::tuple data = GetInterface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<uint8_t*>(data[0].cast<size_t>());
}

uint2* AsUint2Ptr(const py::object& obj) {
  AssertUint2Vec(obj);
  py::tuple data = GetInterface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<uint2*>(data[0].cast<size_t>());
}

}  // namespace caspar
