#pragma once

#include "colmap/util/string.h"
#include "colmap/util/threading.h"

#include "pycolmap/log_exceptions.h"

#include <exception>
#include <regex>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

const Eigen::IOFormat vec_fmt(Eigen::StreamPrecision,
                              Eigen::DontAlignCols,
                              ", ",
                              ", ");

template <typename T>
T pyStringToEnum(const py::enum_<T>& enm, const std::string& value) {
  const auto values = enm.attr("__members__").template cast<py::dict>();
  const auto str_val = py::str(value);
  if (values.contains(str_val)) {
    return T(values[str_val].template cast<T>());
  }
  const std::string msg =
      "Invalid string value " + value + " for enum " +
      std::string(enm.attr("__name__").template cast<std::string>());
  THROW_EXCEPTION(std::out_of_range, msg.c_str());
  return T();
}

template <typename T>
void AddStringToEnumConstructor(py::enum_<T>& enm) {
  enm.def(py::init([enm](const std::string& value) {
    return pyStringToEnum(enm, py::str(value));  // str constructor
  }));
  py::implicitly_convertible<std::string, T>();
}

void UpdateFromDict(py::object& self, const py::dict& dict) {
  for (const auto& it : dict) {
    if (!py::isinstance<py::str>(it.first)) {
      const std::string msg = "Dictionary key is not a string: " +
                              py::str(it.first).template cast<std::string>();
      THROW_EXCEPTION(std::invalid_argument, msg.c_str());
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
        std::stringstream ss;
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
        std::stringstream ss;
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

bool AttributeIsFunction(const std::string& name, const py::object& value) {
  return (name.find("__") == 0 || name.rfind("__") != std::string::npos ||
          py::hasattr(value, "__func__") || py::hasattr(value, "__call__"));
}

std::vector<std::string> ListObjectAttributes(const py::object& pyself) {
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
  std::stringstream ss;
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
    } catch (const std::exception& e) {
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
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      summ = std::regex_replace(summ, std::regex("\n"), "\n" + prefix);
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
        value = StringPrintf(
            "%c ... %d elements ... %c", value.front(), length, value.back());
      }
      ss << " = " << value;
      after_subsummary = false;
    }
  }
  return ss.str();
}

template <typename T, typename... options>
void AddDefaultsToDocstrings(py::class_<T, options...> cls) {
  auto obj = cls();
  for (auto& handle : obj.attr("__dir__")()) {
    const std::string attribute = py::str(handle);
    py::object member;
    try {
      member = obj.attr(attribute.c_str());
    } catch (const std::exception& e) {
      // Some properties are not valid for some uninitialized objects.
      continue;
    }
    auto prop = cls.attr(attribute.c_str());
    if (AttributeIsFunction(attribute, member)) {
      continue;
    }
    const auto type_name = py::type::of(member).attr("__name__");
    const std::string doc =
        StringPrintf("%s (%s, default: %s)",
                     py::str(prop.doc()).cast<std::string>().c_str(),
                     type_name.template cast<std::string>().c_str(),
                     py::str(member).cast<std::string>().c_str());
    prop.doc() = py::str(doc);
  }
}

template <typename T, typename... options>
void MakeDataclass(py::class_<T, options...> cls,
                   const std::vector<std::string>& attributes = {}) {
  AddDefaultsToDocstrings(cls);
  if (!py::hasattr(cls, "summary")) {
    cls.def("summary", &CreateSummary<T>, "write_type"_a = false);
  }
  cls.def("mergedict", &UpdateFromDict);
  cls.def(
      "todict",
      [attributes](const T& self, const bool recursive) {
        return ConvertToDict(self, attributes, recursive);
      },
      "recursive"_a = true);

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

  cls.def("__copy__", [](const T& self) { return T(self); });
  cls.def("__deepcopy__",
          [](const T& self, const py::dict&) { return T(self); });

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
  explicit PyInterrupt(double gap = -1.0);

  inline bool Raised();

 private:
  std::mutex mutex_;
  bool found = false;
  Timer timer_;
  clock::time_point start;
  double gap_;
};

PyInterrupt::PyInterrupt(double gap) : gap_(gap), start(clock::now()) {}

bool PyInterrupt::Raised() {
  const sec duration = clock::now() - start;
  if (!found && duration.count() > gap_) {
    std::lock_guard<std::mutex> lock(mutex_);
    py::gil_scoped_acquire acq;
    found = (PyErr_CheckSignals() != 0);
    start = clock::now();
  }
  return found;
}

// Instead of thread.Wait() call this to allow interrupts through python
void PyWait(Thread* thread, double gap = 2.0) {
  PyInterrupt py_interrupt(gap);
  while (thread->IsRunning()) {
    if (py_interrupt.Raised()) {
      LOG(ERROR) << "Stopping thread...";
      thread->Stop();
      thread->Wait();
      throw py::error_already_set();
    }
  }
  // after finishing join the thread to avoid abort
  thread->Wait();
}
