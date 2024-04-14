#pragma once

#include "colmap/util/types.h"

#include <memory>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;

namespace PYBIND11_NAMESPACE {
namespace detail {

// Bind COLMAP's backport implementation of std::span. This copies the content
// into a list. We could instead create a view with an Eigen::Map but the cast
// should be explicit and cannot be automatic - likely not worth the added
// logic.
template <typename Type>
struct type_caster<span<Type>> : list_caster<span<Type>, Type> {};

// Autocast os.PathLike to std::string
// Adapted from pybind11/stl/filesystem.h
template <>
struct type_caster<std::string> {
 public:
  PYBIND11_TYPE_CASTER(std::string, const_name(PYBIND11_STRING_NAME));

  bool load(handle src, bool) {
    PyObject* buf = PyOS_FSPath(src.ptr());
    if (!buf) {
      PyErr_Clear();
      return false;
    }
    PyObject* native = nullptr;
    if (PyUnicode_FSConverter(buf, &native) != 0) {
      if (auto* c_str = PyBytes_AsString(native)) {
        value = c_str;
      }
    }
    Py_XDECREF(native);
    Py_DECREF(buf);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    return true;
  }

  static handle cast(const std::string& s, return_value_policy rvp, handle h) {
    return string_caster<std::string>::cast(s, rvp, h);
  }
};

// Autocast from numpy.ndarray to std::vector<Eigen::Vector>
template <typename Scalar, int Size>
struct type_caster<std::vector<Eigen::Matrix<Scalar, Size, 1>>> {
 public:
  using MatrixType =
      typename Eigen::Matrix<Scalar, Eigen::Dynamic, Size, Eigen::RowMajor>;
  using VectorType = typename Eigen::Matrix<Scalar, Size, 1>;
  using props = EigenProps<MatrixType>;
  PYBIND11_TYPE_CASTER(std::vector<VectorType>, props::descriptor);

  bool load(handle src, bool) {
    const auto buf = array::ensure(src);
    if (!buf) {
      return false;
    }
    const buffer_info info = buf.request();
    if (info.ndim != 2 || info.shape[1] != Size) {
      return false;
    }
    const size_t num_elements = info.shape[0];
    value.resize(num_elements);
    const auto& mat = src.cast<Eigen::Ref<const MatrixType>>();
    Eigen::Map<MatrixType>(
        reinterpret_cast<Scalar*>(value.data()), num_elements, Size) = mat;
    return true;
  }

  static handle cast(const std::vector<VectorType>& vec,
                     return_value_policy /* policy */,
                     handle h) {
    Eigen::Map<const MatrixType> mat(
        reinterpret_cast<const Scalar*>(vec.data()), vec.size(), Size);
    return type_caster<Eigen::Map<const MatrixType>>::cast(
        mat, return_value_policy::copy, h);
  }
};

}  // namespace detail

template <typename type_, typename... options>
class class_ext_ : public class_<type_, options...> {
 public:
  using Parent = class_<type_, options...>;
  using Parent::class_;  // inherit constructors
  using type = type_;

  template <typename C, typename D, typename... Extra>
  class_ext_& def_readwrite(const char* name, D C::*pm, const Extra&... extra) {
    static_assert(
        std::is_same<C, type>::value || std::is_base_of<C, type>::value,
        "def_readwrite() requires a class member (or base class member)");
    cpp_function fget([pm](type&c) -> D& { return c.*pm; }, is_method(*this)),
        fset([pm](type&c, const D&value) { c.*pm = value; }, is_method(*this));
    this->def_property(
        name, fget, fset, return_value_policy::reference_internal, extra...);
    return *this;
  }

  template <typename... Args>
  class_ext_& def(Args&&... args) {
    Parent::def(std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  class_ext_& def_property(Args&&... args) {
    Parent::def_property(std::forward<Args>(args)...);
    return *this;
  }
};

}  // namespace PYBIND11_NAMESPACE
