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
        ConstPinholeFocalAndExtra_stacked_to_caspar(AsFloatPtr(stacked_data),
                                                    AsFloatPtr(cas_data),
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

        ConstPinholeFocalAndExtra_caspar_to_stacked(AsFloatPtr(cas_data),
                                                    AsFloatPtr(stacked_data),
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
        ConstPinholePrincipalPoint_stacked_to_caspar(AsFloatPtr(stacked_data),
                                                     AsFloatPtr(cas_data),
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

        ConstPinholePrincipalPoint_caspar_to_stacked(AsFloatPtr(cas_data),
                                                     AsFloatPtr(stacked_data),
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
        ConstPixel_stacked_to_caspar(AsFloatPtr(stacked_data),
                                     AsFloatPtr(cas_data),
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

        ConstPixel_caspar_to_stacked(AsFloatPtr(cas_data),
                                     AsFloatPtr(stacked_data),
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
        ConstPoint_stacked_to_caspar(AsFloatPtr(stacked_data),
                                     AsFloatPtr(cas_data),
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

        ConstPoint_caspar_to_stacked(AsFloatPtr(cas_data),
                                     AsFloatPtr(stacked_data),
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
        ConstPose_stacked_to_caspar(AsFloatPtr(stacked_data),
                                    AsFloatPtr(cas_data),
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

        ConstPose_caspar_to_stacked(AsFloatPtr(cas_data),
                                    AsFloatPtr(stacked_data),
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
            AsFloatPtr(stacked_data),
            AsFloatPtr(cas_data),
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
            AsFloatPtr(cas_data),
            AsFloatPtr(stacked_data),
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
            AsFloatPtr(stacked_data),
            AsFloatPtr(cas_data),
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
            AsFloatPtr(cas_data),
            AsFloatPtr(stacked_data),
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
        PinholeFocalAndExtra_stacked_to_caspar(AsFloatPtr(stacked_data),
                                               AsFloatPtr(cas_data),
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

        PinholeFocalAndExtra_caspar_to_stacked(AsFloatPtr(cas_data),
                                               AsFloatPtr(stacked_data),
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
        PinholePrincipalPoint_stacked_to_caspar(AsFloatPtr(stacked_data),
                                                AsFloatPtr(cas_data),
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

        PinholePrincipalPoint_caspar_to_stacked(AsFloatPtr(cas_data),
                                                AsFloatPtr(stacked_data),
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
        Point_stacked_to_caspar(AsFloatPtr(stacked_data),
                                AsFloatPtr(cas_data),
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

        Point_caspar_to_stacked(AsFloatPtr(cas_data),
                                AsFloatPtr(stacked_data),
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
        Pose_stacked_to_caspar(AsFloatPtr(stacked_data),
                               AsFloatPtr(cas_data),
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

        Pose_caspar_to_stacked(AsFloatPtr(cas_data),
                               AsFloatPtr(stacked_data),
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
        SimpleRadialFocalAndExtra_stacked_to_caspar(AsFloatPtr(stacked_data),
                                                    AsFloatPtr(cas_data),
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

        SimpleRadialFocalAndExtra_caspar_to_stacked(AsFloatPtr(cas_data),
                                                    AsFloatPtr(stacked_data),
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
        SimpleRadialPrincipalPoint_stacked_to_caspar(AsFloatPtr(stacked_data),
                                                     AsFloatPtr(cas_data),
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

        SimpleRadialPrincipalPoint_caspar_to_stacked(AsFloatPtr(cas_data),
                                                     AsFloatPtr(stacked_data),
                                                     cas_stride,
                                                     0,
                                                     num_objects);
      });
}

}  // namespace caspar