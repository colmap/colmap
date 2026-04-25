#pragma once

#include "caspar_mappings.h"
#include "pybind_array_tools.h"

namespace caspar {

void add_casmappings_pybindings(pybind11::module_ module) {
  module.def(
      "ConstPinholeFocalAndExtra_stacked_to_caspar",
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
        ConstPinholeFocalAndExtra_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                    AsDoublePtr(cas_data),
                                                    cas_stride,
                                                    0,
                                                    num_objects);
      });
  module.def(
      "ConstPinholeFocalAndExtra_caspar_to_stacked",
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

        ConstPinholeFocalAndExtra_caspar_to_stacked(AsDoublePtr(cas_data),
                                                    AsDoublePtr(stacked_data),
                                                    cas_stride,
                                                    0,
                                                    num_objects);
      });
  module.def(
      "ConstPinholePose_stacked_to_caspar",
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
        ConstPinholePose_stacked_to_caspar(AsDoublePtr(stacked_data),
                                           AsDoublePtr(cas_data),
                                           cas_stride,
                                           0,
                                           num_objects);
      });
  module.def(
      "ConstPinholePose_caspar_to_stacked",
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

        ConstPinholePose_caspar_to_stacked(AsDoublePtr(cas_data),
                                           AsDoublePtr(stacked_data),
                                           cas_stride,
                                           0,
                                           num_objects);
      });
  module.def(
      "ConstPinholePrincipalPoint_stacked_to_caspar",
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
        ConstPinholePrincipalPoint_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                     AsDoublePtr(cas_data),
                                                     cas_stride,
                                                     0,
                                                     num_objects);
      });
  module.def(
      "ConstPinholePrincipalPoint_caspar_to_stacked",
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

        ConstPinholePrincipalPoint_caspar_to_stacked(AsDoublePtr(cas_data),
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
      "ConstSimpleRadialFocalAndExtra_stacked_to_caspar",
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
        ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
            AsDoublePtr(stacked_data),
            AsDoublePtr(cas_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "ConstSimpleRadialFocalAndExtra_caspar_to_stacked",
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

        ConstSimpleRadialFocalAndExtra_caspar_to_stacked(
            AsDoublePtr(cas_data),
            AsDoublePtr(stacked_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "ConstSimpleRadialPose_stacked_to_caspar",
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
        ConstSimpleRadialPose_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                AsDoublePtr(cas_data),
                                                cas_stride,
                                                0,
                                                num_objects);
      });
  module.def(
      "ConstSimpleRadialPose_caspar_to_stacked",
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

        ConstSimpleRadialPose_caspar_to_stacked(AsDoublePtr(cas_data),
                                                AsDoublePtr(stacked_data),
                                                cas_stride,
                                                0,
                                                num_objects);
      });
  module.def(
      "ConstSimpleRadialPrincipalPoint_stacked_to_caspar",
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
        ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
            AsDoublePtr(stacked_data),
            AsDoublePtr(cas_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "ConstSimpleRadialPrincipalPoint_caspar_to_stacked",
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

        ConstSimpleRadialPrincipalPoint_caspar_to_stacked(
            AsDoublePtr(cas_data),
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
      "PinholeFocalAndExtra_stacked_to_caspar",
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
        PinholeFocalAndExtra_stacked_to_caspar(AsDoublePtr(stacked_data),
                                               AsDoublePtr(cas_data),
                                               cas_stride,
                                               0,
                                               num_objects);
      });
  module.def(
      "PinholeFocalAndExtra_caspar_to_stacked",
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

        PinholeFocalAndExtra_caspar_to_stacked(AsDoublePtr(cas_data),
                                               AsDoublePtr(stacked_data),
                                               cas_stride,
                                               0,
                                               num_objects);
      });
  module.def(
      "PinholePose_stacked_to_caspar",
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
        PinholePose_stacked_to_caspar(AsDoublePtr(stacked_data),
                                      AsDoublePtr(cas_data),
                                      cas_stride,
                                      0,
                                      num_objects);
      });
  module.def(
      "PinholePose_caspar_to_stacked",
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

        PinholePose_caspar_to_stacked(AsDoublePtr(cas_data),
                                      AsDoublePtr(stacked_data),
                                      cas_stride,
                                      0,
                                      num_objects);
      });
  module.def(
      "PinholePrincipalPoint_stacked_to_caspar",
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
        PinholePrincipalPoint_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                AsDoublePtr(cas_data),
                                                cas_stride,
                                                0,
                                                num_objects);
      });
  module.def(
      "PinholePrincipalPoint_caspar_to_stacked",
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

        PinholePrincipalPoint_caspar_to_stacked(AsDoublePtr(cas_data),
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
  module.def(
      "SimpleRadialFocalAndExtra_stacked_to_caspar",
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
        SimpleRadialFocalAndExtra_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                    AsDoublePtr(cas_data),
                                                    cas_stride,
                                                    0,
                                                    num_objects);
      });
  module.def(
      "SimpleRadialFocalAndExtra_caspar_to_stacked",
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

        SimpleRadialFocalAndExtra_caspar_to_stacked(AsDoublePtr(cas_data),
                                                    AsDoublePtr(stacked_data),
                                                    cas_stride,
                                                    0,
                                                    num_objects);
      });
  module.def(
      "SimpleRadialPose_stacked_to_caspar",
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
        SimpleRadialPose_stacked_to_caspar(AsDoublePtr(stacked_data),
                                           AsDoublePtr(cas_data),
                                           cas_stride,
                                           0,
                                           num_objects);
      });
  module.def(
      "SimpleRadialPose_caspar_to_stacked",
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

        SimpleRadialPose_caspar_to_stacked(AsDoublePtr(cas_data),
                                           AsDoublePtr(stacked_data),
                                           cas_stride,
                                           0,
                                           num_objects);
      });
  module.def(
      "SimpleRadialPrincipalPoint_stacked_to_caspar",
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
        SimpleRadialPrincipalPoint_stacked_to_caspar(AsDoublePtr(stacked_data),
                                                     AsDoublePtr(cas_data),
                                                     cas_stride,
                                                     0,
                                                     num_objects);
      });
  module.def(
      "SimpleRadialPrincipalPoint_caspar_to_stacked",
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

        SimpleRadialPrincipalPoint_caspar_to_stacked(AsDoublePtr(cas_data),
                                                     AsDoublePtr(stacked_data),
                                                     cas_stride,
                                                     0,
                                                     num_objects);
      });
}

}  // namespace caspar