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

// Fix long-standing bug https://github.com/pybind/pybind11/issues/4529
// TODO(sarlinpe): remove when https://github.com/pybind/pybind11/pull/4972
// appears in the next release of pybind11.
template <typename Map,
          typename holder_type = std::unique_ptr<Map>,
          typename... Args>
class_<Map, holder_type> bind_map_fix(handle scope,
                                      const std::string& name,
                                      Args&&... args) {
  using KeyType = typename Map::key_type;
  using MappedType = typename Map::mapped_type;
  using StrippedKeyType = detail::remove_cvref_t<KeyType>;
  using StrippedMappedType = detail::remove_cvref_t<MappedType>;
  using KeysView = detail::keys_view<StrippedKeyType>;
  using ValuesView = detail::values_view<StrippedMappedType>;
  using ItemsView = detail::items_view<StrippedKeyType, StrippedMappedType>;
  using Class_ = class_<Map, holder_type>;

  // If either type is a non-module-local bound type then make the map binding
  // non-local as well; otherwise (e.g. both types are either module-local or
  // converting) the map will be module-local.
  auto* tinfo = detail::get_type_info(typeid(MappedType));
  bool local = !tinfo || tinfo->module_local;
  if (local) {
    tinfo = detail::get_type_info(typeid(KeyType));
    local = !tinfo || tinfo->module_local;
  }

  Class_ cl(scope,
            name.c_str(),
            pybind11::module_local(local),
            std::forward<Args>(args)...);
  std::string key_type_name(detail::type_info_description(typeid(KeyType)));
  std::string mapped_type_name(
      detail::type_info_description(typeid(MappedType)));

  // Wrap KeysView[KeyType] if it wasn't already wrapped
  if (!detail::get_type_info(typeid(KeysView))) {
    class_<KeysView> keys_view(scope,
                               ("KeysView[" + key_type_name + "]").c_str(),
                               pybind11::module_local(local));
    keys_view.def("__len__", &KeysView::len);
    keys_view.def("__iter__",
                  &KeysView::iter,
                  keep_alive<0, 1>() /* Essential: keep view alive while
                                        iterator exists */
    );
    keys_view.def(
        "__contains__",
        static_cast<bool (KeysView::*)(const KeyType&)>(&KeysView::contains));
    // Fallback for when the object is not of the key type
    keys_view.def(
        "__contains__",
        static_cast<bool (KeysView::*)(const object&)>(&KeysView::contains));
  }
  // Similarly for ValuesView:
  if (!detail::get_type_info(typeid(ValuesView))) {
    class_<ValuesView> values_view(
        scope,
        ("ValuesView[" + mapped_type_name + "]").c_str(),
        pybind11::module_local(local));
    values_view.def("__len__", &ValuesView::len);
    values_view.def("__iter__",
                    &ValuesView::iter,
                    keep_alive<0, 1>() /* Essential: keep view alive while
                                          iterator exists */
    );
  }
  // Similarly for ItemsView:
  if (!detail::get_type_info(typeid(ItemsView))) {
    class_<ItemsView> items_view(scope,
                                 ("ItemsView[" + key_type_name + ", ")
                                     .append(mapped_type_name + "]")
                                     .c_str(),
                                 pybind11::module_local(local));
    items_view.def("__len__", &ItemsView::len);
    items_view.def("__iter__",
                   &ItemsView::iter,
                   keep_alive<0, 1>() /* Essential: keep view alive while
                                         iterator exists */
    );
  }

  cl.def(init<>());

  // Register stream insertion operator (if possible)
  detail::map_if_insertion_operator<Map, Class_>(cl, name);

  cl.def(
      "__bool__",
      [](const Map& m) -> bool { return !m.empty(); },
      "Check whether the map is nonempty");

  cl.def(
      "__iter__",
      [](Map& m) { return make_key_iterator(m.begin(), m.end()); },
      keep_alive<0, 1>() /* Essential: keep map alive while iterator exists */
  );

  cl.def(
      "keys",
      [](Map& m) {
        return std::unique_ptr<KeysView>(
            new detail::KeysViewImpl<Map, KeysView>(m));
      },
      keep_alive<0, 1>() /* Essential: keep map alive while view exists */
  );

  cl.def(
      "values",
      [](Map& m) {
        return std::unique_ptr<ValuesView>(
            new detail::ValuesViewImpl<Map, ValuesView>(m));
      },
      keep_alive<0, 1>() /* Essential: keep map alive while view exists */
  );

  cl.def(
      "items",
      [](Map& m) {
        return std::unique_ptr<ItemsView>(
            new detail::ItemsViewImpl<Map, ItemsView>(m));
      },
      keep_alive<0, 1>() /* Essential: keep map alive while view exists */
  );

  cl.def(
      "__getitem__",
      [](Map& m, const KeyType& k) -> MappedType& {
        auto it = m.find(k);
        if (it == m.end()) {
          throw key_error();
        }
        return it->second;
      },
      return_value_policy::reference_internal  // ref + keepalive
  );

  cl.def("__contains__", [](Map& m, const KeyType& k) -> bool {
    auto it = m.find(k);
    if (it == m.end()) {
      return false;
    }
    return true;
  });
  // Fallback for when the object is not of the key type
  cl.def("__contains__", [](Map&, const object&) -> bool { return false; });

  // Assignment provided only if the type is copyable
  detail::map_assignment<Map, Class_>(cl);

  cl.def("__delitem__", [](Map& m, const KeyType& k) {
    auto it = m.find(k);
    if (it == m.end()) {
      throw key_error();
    }
    m.erase(it);
  });

  // Always use a lambda in case of `using` declaration
  cl.def("__len__", [](const Map& m) { return m.size(); });

  return cl;
}
}  // namespace PYBIND11_NAMESPACE
