/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "pybind_array_tools.h"

#include <algorithm>

namespace py = pybind11;

namespace caspar {

py::object get_interface(const py::object& obj) {
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
  py::object cuda_array_interface = get_interface(obj);
  py::tuple shape = cuda_array_interface["shape"].cast<py::tuple>();
  return shape[0].cast<size_t>();
}

size_t GetNumCols(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
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

void assert_contiguous(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  if (!cuda_array_interface["strides"].is_none()) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("This function only supports contiguous arrays\n" + info);
  }
}

void assert_1d(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  if (py::len(cuda_array_interface["shape"]) != 1) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected 1D array, got " +
                             std::to_string(py::len(cuda_array_interface["shape"])) + "D array\n" +
                             info);
  }
}

void assert_2d(const py::object& obj) {
  assert_contiguous(obj);
  py::object cuda_array_interface = get_interface(obj);
  if (py::len(cuda_array_interface["shape"]) != 2) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error("Expected 2D array, got " +
                             std::to_string(py::len(cuda_array_interface["shape"])) + "D array\n" +
                             info);
  }
}

void assert_2d_nxk(const py::object& obj, size_t k) {
  py::object cuda_array_interface = get_interface(obj);
  assert_2d(obj);
  py::tuple shape = cuda_array_interface["shape"].cast<py::tuple>();
  if (shape[1].cast<int>() != 2) {
    std::string info = py::repr(obj).cast<std::string>();
    throw std::runtime_error(
        "Expected 2D array with second dimension of size " + std::to_string(k) + ", got " +
        py::str(cuda_array_interface["shape"]).cast<std::string>() + "\n" + info);
  }
}

void assert_type(const py::object& obj, const std::vector<std::string>& types) {
  py::object cuda_array_interface = get_interface(obj);
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

void assert_float(const py::object& obj) {
  assert_type(obj, {"<f4"});
}

void assert_double(const py::object& obj) {
  assert_type(obj, {"<f8"});
}

void assert_uint(const py::object& obj) {
  assert_type(obj, {"<u4", "<i4"});
}

void assert_char(const py::object& obj) {
  assert_type(obj, {"|u1", "|i1"});
}

void assert_floatvec(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  assert_float(obj);
  assert_contiguous(obj);
}

void assert_doublevec(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  assert_double(obj);
  assert_contiguous(obj);
}

void assert_uintvec(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  assert_uint(obj);
  assert_contiguous(obj);
}

void assert_charvec(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  assert_char(obj);
  assert_contiguous(obj);
}

void assert_uint2vec(const py::object& obj) {
  py::object cuda_array_interface = get_interface(obj);
  assert_uint(obj);
  assert_2d_nxk(obj, 2);
}

float* AsFloatPtr(const py::object& obj) {
  assert_floatvec(obj);
  py::tuple data = get_interface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<float*>(data[0].cast<size_t>());
}

double* AsDoublePtr(const py::object& obj) {
  assert_doublevec(obj);
  py::tuple data = get_interface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<double*>(data[0].cast<size_t>());
}

int* AsIntPtr(const py::object& obj) {
  assert_uintvec(obj);
  py::tuple data = get_interface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<int*>(data[0].cast<size_t>());
}

uint* AsUintPtr(const py::object& obj) {
  assert_uintvec(obj);
  py::tuple data = get_interface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<uint*>(data[0].cast<size_t>());
}

uint8_t* AsCharPtr(const py::object& obj) {
  assert_charvec(obj);
  py::tuple data = get_interface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<uint8_t*>(data[0].cast<size_t>());
}

uint2* AsUint2Ptr(const py::object& obj) {
  assert_uint2vec(obj);
  py::tuple data = get_interface(obj)["data"].cast<py::tuple>();
  return reinterpret_cast<uint2*>(data[0].cast<size_t>());
}

}  // namespace caspar
