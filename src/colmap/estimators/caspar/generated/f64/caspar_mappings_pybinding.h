#pragma once

#include "caspar_mappings.h"
#include "pybind_array_tools.h"

namespace caspar {

void add_casmappings_pybindings(pybind11::module_ module) {
  module.def(
      "ConstPinholeCalib_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        ConstPinholeCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                            AsDoublePtr(cas_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "ConstPinholeCalib_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        ConstPinholeCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                            AsDoublePtr(stacked_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "ConstPixel_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 2) {
          throw std::runtime_error("The stacked data must have 2 columns.");
        }
        if (GetNumRows(cas_data) != 2) {
          throw std::runtime_error("The caspar data must have 2 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        ConstPixel_stacked_to_caspar(AsDoublePtr(stacked_data),
                                     AsDoublePtr(cas_data),
                                     cas_stride,
                                     0,
                                     num_objects);
      });
  module.def(
      "ConstPixel_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 2) {
          throw std::runtime_error("The stacked data must have 2 columns.");
        }
        if (GetNumRows(cas_data) != 2) {
          throw std::runtime_error("The caspar data must have 2 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        ConstPixel_caspar_to_stacked(AsDoublePtr(cas_data),
                                     AsDoublePtr(stacked_data),
                                     cas_stride,
                                     0,
                                     num_objects);
      });
  module.def(
      "ConstPoint_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 3) {
          throw std::runtime_error("The stacked data must have 3 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        ConstPoint_stacked_to_caspar(AsDoublePtr(stacked_data),
                                     AsDoublePtr(cas_data),
                                     cas_stride,
                                     0,
                                     num_objects);
      });
  module.def(
      "ConstPoint_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 3) {
          throw std::runtime_error("The stacked data must have 3 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        ConstPoint_caspar_to_stacked(AsDoublePtr(cas_data),
                                     AsDoublePtr(stacked_data),
                                     cas_stride,
                                     0,
                                     num_objects);
      });
  module.def(
      "ConstPose_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 7) {
          throw std::runtime_error("The stacked data must have 7 columns.");
        }
        if (GetNumRows(cas_data) != 8) {
          throw std::runtime_error("The caspar data must have 8 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        ConstPose_stacked_to_caspar(AsDoublePtr(stacked_data),
                                    AsDoublePtr(cas_data),
                                    cas_stride,
                                    0,
                                    num_objects);
      });
  module.def(
      "ConstPose_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 7) {
          throw std::runtime_error("The stacked data must have 7 columns.");
        }
        if (GetNumRows(cas_data) != 8) {
          throw std::runtime_error("The caspar data must have 8 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        ConstPose_caspar_to_stacked(AsDoublePtr(cas_data),
                                    AsDoublePtr(stacked_data),
                                    cas_stride,
                                    0,
                                    num_objects);
      });
  module.def(
      "ConstSimpleRadialCalib_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        ConstSimpleRadialCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                 AsDoublePtr(cas_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "ConstSimpleRadialCalib_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        ConstSimpleRadialCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                                 AsDoublePtr(stacked_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "PinholeCalib_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        PinholeCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                       AsDoublePtr(cas_data),
                                       cas_stride,
                                       0,
                                       num_objects);
      });
  module.def(
      "PinholeCalib_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        PinholeCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                       AsDoublePtr(stacked_data),
                                       cas_stride,
                                       0,
                                       num_objects);
      });
  module.def(
      "Point_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 3) {
          throw std::runtime_error("The stacked data must have 3 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        Point_stacked_to_caspar(AsDoublePtr(stacked_data),
                                AsDoublePtr(cas_data),
                                cas_stride,
                                0,
                                num_objects);
      });
  module.def(
      "Point_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 3) {
          throw std::runtime_error("The stacked data must have 3 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        Point_caspar_to_stacked(AsDoublePtr(cas_data),
                                AsDoublePtr(stacked_data),
                                cas_stride,
                                0,
                                num_objects);
      });
  module.def(
      "Pose_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 7) {
          throw std::runtime_error("The stacked data must have 7 columns.");
        }
        if (GetNumRows(cas_data) != 8) {
          throw std::runtime_error("The caspar data must have 8 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        Pose_stacked_to_caspar(AsDoublePtr(stacked_data),
                               AsDoublePtr(cas_data),
                               cas_stride,
                               0,
                               num_objects);
      });
  module.def(
      "Pose_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 7) {
          throw std::runtime_error("The stacked data must have 7 columns.");
        }
        if (GetNumRows(cas_data) != 8) {
          throw std::runtime_error("The caspar data must have 8 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        Pose_caspar_to_stacked(AsDoublePtr(cas_data),
                               AsDoublePtr(stacked_data),
                               cas_stride,
                               0,
                               num_objects);
      });
  module.def(
      "SimpleRadialCalib_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        SimpleRadialCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                            AsDoublePtr(cas_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "SimpleRadialCalib_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 4) {
          throw std::runtime_error("The stacked data must have 4 columns.");
        }
        if (GetNumRows(cas_data) != 4) {
          throw std::runtime_error("The caspar data must have 4 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        SimpleRadialCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                            AsDoublePtr(stacked_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
}

}  // namespace caspar