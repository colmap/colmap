#include "caspar_mappings_pybinding.h"
#include "pybind_array_tools.h"
#include "shared_indices_pybinding.h"
#include "solver_params_pybinding.h"
#include "solver_pybinding.h"
#include "sort_indices_pybinding.h"

namespace {

using namespace caspar;

}  // namespace

PYBIND11_MODULE(caspar_lib, module) {
  module.doc() =
      "Module containing bindings for a generated caspar library. See the "
      "generated pyi file for more details.";
  module.def("shared_indices", &caspar::shared_indices_pybinding);

  caspar::add_casmappings_pybindings(module);
  caspar::add_solver_pybinding(module);
  caspar::add_solver_params_pybinding(module);
}