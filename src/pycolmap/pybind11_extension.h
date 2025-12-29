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

namespace PYBIND11_NAMESPACE {
namespace detail {

// Bind COLMAP's backport implementation of std::span. This copies the content
// into a list. We could instead create a view with an Eigen::Map but the cast
// should be explicit and cannot be automatic - likely not worth the added
// logic.
template <typename Type>
struct type_caster<colmap::span<Type>> : list_caster<colmap::span<Type>, Type> {
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
class classh_ext : public classh<type_, options...> {
 public:
  using Parent = classh<type_, options...>;
  using Parent::Parent;  // inherit constructors
  using type = type_;

  template <typename C, typename D, typename... Extra>
  classh_ext& def_readwrite(const char* name,
                            D C::* pm,
                            const Extra&... extra) {
    static_assert(
        std::is_same<C, type>::value || std::is_base_of<C, type>::value,
        "def_readwrite() requires a class member (or base class member)");
    cpp_function fget([pm](type& c) -> D& { return c.*pm; }, is_method(*this)),
        fset([pm](type& c, const D& value) { c.*pm = value; },
             is_method(*this));
    this->def_property(
        name, fget, fset, return_value_policy::reference_internal, extra...);
    return *this;
  }

  template <typename... Args>
  classh_ext& def(Args&&... args) {
    Parent::def(std::forward<Args>(args)...);
    return *this;
  }

  template <typename... Args>
  classh_ext& def_property(Args&&... args) {
    Parent::def_property(std::forward<Args>(args)...);
    return *this;
  }
};

}  // namespace PYBIND11_NAMESPACE
