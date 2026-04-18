#pragma once

#include "caspar_mappings.h"
#include "pybind_array_tools.h"

namespace caspar {

void add_casmappings_pybindings(pybind11::module_ module) {
  module.def(
      "ConstPinholeExtraCalib_stacked_to_caspar",
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
        ConstPinholeExtraCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                 AsDoublePtr(cas_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "ConstPinholeExtraCalib_caspar_to_stacked",
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

        ConstPinholeExtraCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                                 AsDoublePtr(stacked_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "ConstPinholeFocal_stacked_to_caspar",
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
        ConstPinholeFocal_stacked_to_caspar(AsDoublePtr(stacked_data),
                                            AsDoublePtr(cas_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "ConstPinholeFocal_caspar_to_stacked",
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

        ConstPinholeFocal_caspar_to_stacked(AsDoublePtr(cas_data),
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
      "ConstSimpleRadialExtraCalib_stacked_to_caspar",
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
        ConstSimpleRadialExtraCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                      AsDoublePtr(cas_data),
                                                      cas_stride,
                                                      0,
                                                      num_objects);
      });
  module.def(
      "ConstSimpleRadialExtraCalib_caspar_to_stacked",
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

        ConstSimpleRadialExtraCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                                      AsDoublePtr(stacked_data),
                                                      cas_stride,
                                                      0,
                                                      num_objects);
      });
  module.def(
      "ConstSimpleRadialFocal_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 1) {
          throw std::runtime_error("The stacked data must have 1 columns.");
        }
        if (GetNumRows(cas_data) != 1) {
          throw std::runtime_error("The caspar data must have 1 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        ConstSimpleRadialFocal_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                 AsDoublePtr(cas_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "ConstSimpleRadialFocal_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 1) {
          throw std::runtime_error("The stacked data must have 1 columns.");
        }
        if (GetNumRows(cas_data) != 1) {
          throw std::runtime_error("The caspar data must have 1 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        ConstSimpleRadialFocal_caspar_to_stacked(AsDoublePtr(cas_data),
                                                 AsDoublePtr(stacked_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "PinholeExtraCalib_stacked_to_caspar",
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
        PinholeExtraCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                            AsDoublePtr(cas_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "PinholeExtraCalib_caspar_to_stacked",
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

        PinholeExtraCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                            AsDoublePtr(stacked_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "PinholeFocal_stacked_to_caspar",
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
        PinholeFocal_stacked_to_caspar(AsDoublePtr(stacked_data),
                                       AsDoublePtr(cas_data),
                                       cas_stride,
                                       0,
                                       num_objects);
      });
  module.def(
      "PinholeFocal_caspar_to_stacked",
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

        PinholeFocal_caspar_to_stacked(AsDoublePtr(cas_data),
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
      "SimpleRadialExtraCalib_stacked_to_caspar",
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
        SimpleRadialExtraCalib_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                 AsDoublePtr(cas_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "SimpleRadialExtraCalib_caspar_to_stacked",
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

        SimpleRadialExtraCalib_caspar_to_stacked(AsDoublePtr(cas_data),
                                                 AsDoublePtr(stacked_data),
                                                 cas_stride,
                                                 0,
                                                 num_objects);
      });
  module.def(
      "SimpleRadialFocal_stacked_to_caspar",
      [](pybind11::object stacked_data, pybind11::object cas_data) {
        if (GetNumCols(stacked_data) != 1) {
          throw std::runtime_error("The stacked data must have 1 columns.");
        }
        if (GetNumRows(cas_data) != 1) {
          throw std::runtime_error("The caspar data must have 1 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }
        SimpleRadialFocal_stacked_to_caspar(AsDoublePtr(stacked_data),
                                            AsDoublePtr(cas_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
  module.def(
      "SimpleRadialFocal_caspar_to_stacked",
      [](pybind11::object cas_data, pybind11::object stacked_data) {
        if (GetNumCols(stacked_data) != 1) {
          throw std::runtime_error("The stacked data must have 1 columns.");
        }
        if (GetNumRows(cas_data) != 1) {
          throw std::runtime_error("The caspar data must have 1 rows.");
        }
        int num_objects = GetNumRows(stacked_data);
        int cas_stride = GetNumCols(cas_data);
        if (cas_stride < num_objects) {
          throw std::runtime_error(
              "The caspar data must have at least as many columns as "
              "stacked_data has rows.");
        }

        SimpleRadialFocal_caspar_to_stacked(AsDoublePtr(cas_data),
                                            AsDoublePtr(stacked_data),
                                            cas_stride,
                                            0,
                                            num_objects);
      });
}

}  // namespace caspar