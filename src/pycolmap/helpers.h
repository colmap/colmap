#pragma once

#include "colmap/util/cancellation.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"
#include "colmap/util/threading.h"

#include "pycolmap/feature/opaque_types.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

const Eigen::IOFormat vec_fmt(Eigen::StreamPrecision,
                              Eigen::DontAlignCols,
                              ", ",
                              ", ");

template <typename T>
T PyStringToEnum(const py::enum_<T>& enm, const std::string& value) {
  const auto values = enm.attr("__members__").template cast<py::dict>();
  const auto str_val = py::str(value);
  if (!values.contains(str_val)) {
    LOG(FATAL_THROW) << "Invalid string value " << value << " for enum "
                     << enm.attr("__name__").template cast<std::string>();
  }
  return T(values[str_val].template cast<T>());
}

template <typename T>
void AddStringToEnumConstructor(py::enum_<T>& enm) {
  enm.def(py::init([enm](const std::string& value) {
            return PyStringToEnum(enm, py::str(value));  // str constructor
          }),
          py::arg("name"));
  enm.attr("__repr__") = enm.attr("__str__");
  py::implicitly_convertible<std::string, T>();
}

inline void UpdateFromDict(py::object& self, const py::dict& dict) {
  for (const auto& it : dict) {
    if (!py::isinstance<py::str>(it.first)) {
      LOG(FATAL_THROW) << "Dictionary key is not a string: "
                       << py::str(it.first);
    }
    const py::str name = py::reinterpret_borrow<py::str>(it.first);
    const py::handle& value = it.second;
    const auto attr = self.attr(name);
    try {
      if (py::hasattr(attr, "mergedict") && py::isinstance<py::dict>(value)) {
        attr.attr("mergedict").attr("__call__")(value);
      } else {
        self.attr(name) = value;
      }
    } catch (const py::error_already_set& ex) {
      if (ex.matches(PyExc_TypeError)) {
        // If fail we try bases of the class
        const py::list bases =
            attr.attr("__class__").attr("__bases__").cast<py::list>();
        bool success_on_base = false;
        for (auto& base : bases) {
          try {
            self.attr(name) = base(value);
            success_on_base = true;
            break;
          } catch (const py::error_already_set&) {
            continue;  // We anyway throw afterwards
          }
        }
        if (success_on_base) {
          continue;
        }
        std::ostringstream ss;
        ss << self.attr("__class__")
                  .attr("__name__")
                  .template cast<std::string>()
           << "." << name.template cast<std::string>() << ": Could not convert "
           << py::type::of(value.cast<py::object>())
                  .attr("__name__")
                  .template cast<std::string>()
           << ": " << py::str(value).template cast<std::string>() << " to '"
           << py::type::of(attr).attr("__name__").template cast<std::string>()
           << "'.";
        // We write the err message to give info even if exceptions
        // is catched outside, e.g. in function overload resolve
        LOG(ERROR) << "Internal TypeError: " << ss.str();
        throw(py::type_error(std::string("Failed to merge dict into class: ") +
                             "Could not assign " +
                             name.template cast<std::string>()));
      } else if (ex.matches(PyExc_AttributeError) &&
                 py::str(ex.value()).cast<std::string>() ==
                     std::string("can't set attribute")) {
        std::ostringstream ss;
        ss << self.attr("__class__")
                  .attr("__name__")
                  .template cast<std::string>()
           << "." << name.template cast<std::string>() << " defined readonly.";
        throw py::attribute_error(ss.str());
      } else if (ex.matches(PyExc_AttributeError)) {
        LOG(ERROR) << "Internal AttributeError: "
                   << py::str(ex.value()).cast<std::string>();
        throw;
      } else {
        LOG(ERROR) << "Internal Error: "
                   << py::str(ex.value()).cast<std::string>();
        throw;
      }
    }
  }
}

inline bool AttributeIsFunction(const std::string& name,
                                const py::object& value) {
  return (name.find("__") == 0 || name.rfind("__") != std::string::npos ||
          py::hasattr(value, "__func__") || py::hasattr(value, "__call__"));
}

inline std::vector<std::string> ListObjectAttributes(const py::object& pyself) {
  std::vector<std::string> attributes;
  for (const auto& handle : pyself.attr("__dir__")()) {
    const py::str attribute = py::reinterpret_borrow<py::str>(handle);
    const auto value = pyself.attr(attribute);
    if (AttributeIsFunction(attribute, value)) {
      continue;
    }
    attributes.push_back(attribute);
  }
  return attributes;
}

template <typename T, typename... options>
py::dict ConvertToDict(const T& self,
                       std::vector<std::string> attributes,
                       const bool recursive) {
  const py::object pyself = py::cast(self);
  if (attributes.empty()) {
    attributes = ListObjectAttributes(pyself);
  }
  py::dict dict;
  for (const auto& attr : attributes) {
    const auto value = pyself.attr(attr.c_str());
    if (recursive && py::hasattr(value, "todict")) {
      dict[attr.c_str()] =
          value.attr("todict").attr("__call__")().template cast<py::dict>();
    } else {
      dict[attr.c_str()] = value;
    }
  }
  return dict;
}

template <typename T, typename... options>
std::string CreateSummary(const T& self, bool write_type) {
  std::ostringstream ss;
  auto pyself = py::cast(self);
  const std::string prefix = "    ";
  bool after_subsummary = false;
  ss << pyself.attr("__class__").attr("__name__").template cast<std::string>()
     << ":";
  for (auto& handle : pyself.attr("__dir__")()) {
    const py::str name = py::reinterpret_borrow<py::str>(handle);
    py::object attribute;
    try {
      attribute = pyself.attr(name);
    } catch (const std::exception&) {
      // Some properties are not valid for some uninitialized objects.
      continue;
    }
    if (AttributeIsFunction(name, attribute)) {
      continue;
    }
    ss << "\n";
    if (!after_subsummary) {
      ss << prefix;
    }
    ss << name.template cast<std::string>();
    if (py::hasattr(attribute, "summary")) {
      std::string summ = attribute.attr("summary")
                             .attr("__call__")(write_type)
                             .template cast<std::string>();
      static const std::regex newline_regex("\n");
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      summ = std::regex_replace(summ, newline_regex, "\n" + prefix);
      ss << ": " << summ;
    } else {
      if (write_type) {
        const std::string type_str =
            py::str(py::type::of(attribute).attr("__name__"));
        ss << ": " << type_str;
        after_subsummary = true;
      }
      std::string value = py::str(attribute);
      if (value.length() > 80 && py::hasattr(attribute, "__len__")) {
        const int length = attribute.attr("__len__")().template cast<int>();
        value = colmap::StringPrintf(
            "%c ... %d elements ... %c", value.front(), length, value.back());
      }
      ss << " = " << value;
      after_subsummary = false;
    }
  }
  return ss.str();
}

template <typename T>
std::string CreateRepresentationFromAttributes(const T& self) {
  std::ostringstream ss;
  auto pyself = py::cast(self);
  ss << pyself.attr("__class__").attr("__name__").template cast<std::string>()
     << "(";
  bool is_first = true;
  for (auto& handle : pyself.attr("__dir__")()) {
    const py::str name = py::reinterpret_borrow<py::str>(handle);
    py::object attribute;
    try {
      attribute = pyself.attr(name);
    } catch (const std::exception&) {
      // Some properties are not valid for some uninitialized objects.
      continue;
    }
    if (AttributeIsFunction(name, attribute)) {
      continue;
    }
    if (!is_first) {
      ss << ", ";
    }
    is_first = false;
    ss << name.template cast<std::string>() << "=";
    if (py::isinstance<py::str>(attribute)) {
      ss << "'" << py::str(attribute) << "'";
    } else {
      ss << py::str(attribute);
    }
  }
  ss << ")";
  return ss.str();
}

template <typename T, typename = void>
struct IsOstreamable : std::false_type {};

template <typename T>
struct IsOstreamable<
    T,
    std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};

template <typename T>
std::string CreateRepresentation(const T& self) {
  if constexpr (IsOstreamable<T>::value) {
    std::ostringstream ss;
    ss << self;
    return ss.str();
  } else {
    return CreateRepresentationFromAttributes<T>(self);
  }
}

template <typename T, typename... options>
void AddDefaultsToDocstrings(py::classh<T, options...> cls) {
  auto obj = cls();
  for (auto& handle : obj.attr("__dir__")()) {
    const std::string attribute = py::str(handle);
    py::object member;
    try {
      member = obj.attr(attribute.c_str());
    } catch (const std::exception&) {
      // Some properties are not valid for some uninitialized objects.
      continue;
    }
    auto prop = cls.attr(attribute.c_str());
    if (AttributeIsFunction(attribute, member)) {
      continue;
    }
    const auto type_name = py::type::of(member).attr("__name__");
    const std::string doc =
        colmap::StringPrintf("%s (%s, default: %s)",
                             py::str(prop.doc()).cast<std::string>().c_str(),
                             type_name.template cast<std::string>().c_str(),
                             py::str(member).cast<std::string>().c_str());
    prop.doc() = py::str(doc);
  }
}

template <typename T, typename = void>
struct has_equality_operator : std::false_type {};

template <typename T>
struct has_equality_operator<
    T,
    std::void_t<decltype(std::declval<T&>() == std::declval<T&>())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_less_than_operator : std::false_type {};

template <typename T>
struct has_less_than_operator<
    T,
    std::void_t<decltype(std::declval<T&>() < std::declval<T&>())>>
    : std::true_type {};

template <typename T, typename = void>
struct is_hashable : std::false_type {};

template <typename T>
struct is_hashable<T, std::void_t<decltype(std::hash<T>{}(std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename... options>
void MakeDataclass(py::classh<T, options...> cls,
                   const std::vector<std::string>& attributes = {}) {
  AddDefaultsToDocstrings(cls);
  if (!py::hasattr(cls, "summary")) {
    cls.def("summary", &CreateSummary<T>, py::arg("write_type") = false);
  }
  if (!cls.attr("__dict__").contains("__repr__")) {
    cls.def("__repr__", &CreateRepresentation<T>);
  }
  cls.def("mergedict", &UpdateFromDict, py::arg("kwargs"));
  cls.def(
      "todict",
      [attributes](const T& self, const bool recursive) {
        return ConvertToDict(self, attributes, recursive);
      },
      py::arg("recursive") = true);

  if constexpr (std::is_copy_constructible_v<T>) {
    cls.def(py::init([cls](const py::dict& dict) {
      py::object self = cls();
      self.attr("mergedict").attr("__call__")(dict);
      return self.cast<T>();
    }));
    cls.def(py::init([cls](const py::kwargs& kwargs) {
      py::dict dict = kwargs.cast<py::dict>();
      return cls(dict).template cast<T>();
    }));
    py::implicitly_convertible<py::dict, T>();
    py::implicitly_convertible<py::kwargs, T>();

    if (!cls.attr("__dict__").contains("__copy__")) {
      cls.def("__copy__", [](const T& self) { return T(self); });
    }
    if (!cls.attr("__dict__").contains("__deepcopy__")) {
      cls.def("__deepcopy__",
              [](const T& self, const py::dict&) { return T(self); });
    }

    cls.def(py::pickle(
        [attributes](const T& self) {
          return ConvertToDict(self, attributes, /*recursive=*/false);
        },
        [cls](const py::dict& dict) {
          py::object self = cls();
          self.attr("mergedict").attr("__call__")(dict);
          return self.cast<T>();
        }));
  }

  if constexpr (has_equality_operator<T>::value) {
    cls.def(py::self == py::self);
    if constexpr (is_hashable<T>::value) {
      cls.def("__hash__", [](const T& self) { return std::hash<T>()(self); });
    } else {
      cls.attr("__hash__") = py::none();
    }
  } else {
    if constexpr (std::is_copy_constructible_v<T>) {
      cls.def("__eq__", [attributes](const T& self, const py::object& other) {
        if (!py::isinstance<T>(other)) {
          return false;
        }
        py::dict self_dict = ConvertToDict(self, attributes, true);
        py::dict other_dict = ConvertToDict(other.cast<T>(), attributes, true);
        return self_dict.equal(other_dict);
      });
    }
    cls.attr("__hash__") = py::none();
  }

  if constexpr (has_less_than_operator<T>::value) {
    cls.def(py::self < py::self);
  }
}

// Catch python keyboard interrupts

/*
// single
if (PyInterrupt().Raised()) {
    // stop the execution and raise an exception
    throw py::error_already_set();
}

// loop
PyInterrupt py_interrupt = PyInterrupt(2.0)
for (...) {
    if (py_interrupt.Raised()) {
        // stop the execution and raise an exception
        throw py::error_already_set();
    }
    // Do your workload here
}


*/
struct PyInterrupt {
  using clock = std::chrono::steady_clock;
  using sec = std::chrono::duration<double>;
  explicit PyInterrupt(double gap = -1.0) : start(clock::now()), gap_(gap) {}

  inline bool Raised();

 private:
  std::mutex mutex_;
  bool found = false;
  colmap::Timer timer_;
  clock::time_point start;
  double gap_;
};

inline bool PyInterrupt::Raised() {
  const sec duration = clock::now() - start;
  if (!found && duration.count() > gap_) {
    std::lock_guard<std::mutex> lock(mutex_);
    py::gil_scoped_acquire acq;
    found = (PyErr_CheckSignals() != 0);
    start = clock::now();
  }
  return found;
}

[[noreturn]] inline void ThrowPythonError() {
  py::gil_scoped_acquire acquire;
  throw py::error_already_set();
}

[[noreturn]] inline void ThrowCancelled() {
  py::gil_scoped_acquire acquire;
  PyErr_SetString(PyExc_InterruptedError, "Operation cancelled");
  throw py::error_already_set();
}

class PyInterruptChecker {
 public:
  explicit PyInterruptChecker(
      std::shared_ptr<colmap::CancellationToken> cancellation_token = nullptr,
      const double gap = 1.0)
      : py_interrupt_(gap),
        cancellation_token_(std::move(cancellation_token)),
        calling_thread_id_(std::this_thread::get_id()) {}

  // The callback may run on worker threads, but Python signals may only be
  // checked from the thread that created this object.
  std::function<bool()> Callback() {
    return [this]() {
      if (std::this_thread::get_id() == calling_thread_id_ &&
          py_interrupt_.Raised()) {
        python_interrupt_raised_.store(true, std::memory_order_relaxed);
      }
      return python_interrupt_raised_.load(std::memory_order_relaxed) ||
             (cancellation_token_ && cancellation_token_->IsCancelled());
    };
  }

  void CheckAndThrow() const {
    if (python_interrupt_raised_.load(std::memory_order_relaxed)) {
      ThrowPythonError();
    }
    if (cancellation_token_ && cancellation_token_->IsCancelled()) {
      ThrowCancelled();
    }
  }

 private:
  PyInterrupt py_interrupt_;
  std::shared_ptr<colmap::CancellationToken> cancellation_token_;
  std::thread::id calling_thread_id_;
  std::atomic<bool> python_interrupt_raised_{false};
};

// Instead of thread.Wait() call this to allow interrupts through python
inline void PyWait(colmap::Thread* thread,
                   const std::shared_ptr<colmap::CancellationToken>&
                       cancellation_token = nullptr,
                   double gap = 1.0) {
  const PyInterrupt::sec poll_interval(gap);
  PyInterrupt py_interrupt(gap);
  while (thread->IsRunning()) {
    if (cancellation_token && cancellation_token->IsCancelled()) {
      thread->Stop();
      thread->Wait();
      ThrowCancelled();
    }
    if (py_interrupt.Raised()) {
      LOG(ERROR) << "Stopping thread...";
      thread->Stop();
      thread->Wait();
      ThrowPythonError();
    }
    std::this_thread::sleep_for(poll_interval);
  }
  // after finishing join the thread to avoid abort
  thread->Wait();
  if (cancellation_token && cancellation_token->IsCancelled()) {
    ThrowCancelled();
  }
}

// Test if pyceres is available
inline bool IsPyceresAvailable() {
  try {
    py::module::import("pyceres");
  } catch (const py::error_already_set&) {
    return false;
  }
  return true;
}

template <typename Parent>
inline void DefDeprecation(
    Parent& parent,
    std::string old_name,
    std::string new_name,
    std::optional<std::string> custom_warning = std::nullopt) {
  const std::string doc =
      colmap::StringPrintf("Deprecated, use ``%s`` instead.", new_name.c_str());
  parent.def(
      old_name.c_str(),
      [parent,
       old_name,
       new_name = std::move(new_name),
       custom_warning = std::move(custom_warning)](const py::args& args,
                                                   const py::kwargs& kwargs) {
        if (custom_warning) {
          PyErr_WarnEx(PyExc_DeprecationWarning, custom_warning->c_str(), 1);
        } else {
          std::ostringstream warning;
          warning << old_name << "() is deprecated, use " << new_name
                  << "() instead.";
          PyErr_WarnEx(PyExc_DeprecationWarning, warning.str().c_str(), 1);
        }
        return parent.attr(new_name.c_str())(*args, **kwargs);
      },
      doc.c_str());
}
