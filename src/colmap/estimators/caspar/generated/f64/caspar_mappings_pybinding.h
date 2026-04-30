#pragma once

#include "caspar_mappings.h"
#include "pybind_array_tools.h"

namespace caspar {

void add_casmappings_pybindings(pybind11::module_ module) {
  module.def(
      "const_pinhole_focal_stacked_to_caspar",
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
        ConstPinholeFocalStackedToCaspar(AsDoublePtr(stacked_data),
                                         AsDoublePtr(cas_data),
                                         cas_stride,
                                         0,
                                         num_objects);
      });
  module.def(
      "const_pinhole_focal_caspar_to_stacked",
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

        ConstPinholeFocalCasparToStacked(AsDoublePtr(cas_data),
                                         AsDoublePtr(stacked_data),
                                         cas_stride,
                                         0,
                                         num_objects);
      });
  module.def(
      "const_pinhole_pose_stacked_to_caspar",
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
        ConstPinholePoseStackedToCaspar(AsDoublePtr(stacked_data),
                                        AsDoublePtr(cas_data),
                                        cas_stride,
                                        0,
                                        num_objects);
      });
  module.def(
      "const_pinhole_pose_caspar_to_stacked",
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

        ConstPinholePoseCasparToStacked(AsDoublePtr(cas_data),
                                        AsDoublePtr(stacked_data),
                                        cas_stride,
                                        0,
                                        num_objects);
      });
  module.def(
      "const_pinhole_principal_point_stacked_to_caspar",
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
        ConstPinholePrincipalPointStackedToCaspar(AsDoublePtr(stacked_data),
                                                  AsDoublePtr(cas_data),
                                                  cas_stride,
                                                  0,
                                                  num_objects);
      });
  module.def(
      "const_pinhole_principal_point_caspar_to_stacked",
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

        ConstPinholePrincipalPointCasparToStacked(AsDoublePtr(cas_data),
                                                  AsDoublePtr(stacked_data),
                                                  cas_stride,
                                                  0,
                                                  num_objects);
      });
  module.def(
      "const_pixel_stacked_to_caspar",
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
        ConstPixelStackedToCaspar(AsDoublePtr(stacked_data),
                                  AsDoublePtr(cas_data),
                                  cas_stride,
                                  0,
                                  num_objects);
      });
  module.def(
      "const_pixel_caspar_to_stacked",
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

        ConstPixelCasparToStacked(AsDoublePtr(cas_data),
                                  AsDoublePtr(stacked_data),
                                  cas_stride,
                                  0,
                                  num_objects);
      });
  module.def(
      "const_point_stacked_to_caspar",
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
        ConstPointStackedToCaspar(AsDoublePtr(stacked_data),
                                  AsDoublePtr(cas_data),
                                  cas_stride,
                                  0,
                                  num_objects);
      });
  module.def(
      "const_point_caspar_to_stacked",
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

        ConstPointCasparToStacked(AsDoublePtr(cas_data),
                                  AsDoublePtr(stacked_data),
                                  cas_stride,
                                  0,
                                  num_objects);
      });
  module.def(
      "const_simple_radial_focal_and_distortion_stacked_to_caspar",
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
        ConstSimpleRadialFocalAndDistortionStackedToCaspar(
            AsDoublePtr(stacked_data),
            AsDoublePtr(cas_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "const_simple_radial_focal_and_distortion_caspar_to_stacked",
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

        ConstSimpleRadialFocalAndDistortionCasparToStacked(
            AsDoublePtr(cas_data),
            AsDoublePtr(stacked_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "const_simple_radial_pose_stacked_to_caspar",
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
        ConstSimpleRadialPoseStackedToCaspar(AsDoublePtr(stacked_data),
                                             AsDoublePtr(cas_data),
                                             cas_stride,
                                             0,
                                             num_objects);
      });
  module.def(
      "const_simple_radial_pose_caspar_to_stacked",
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

        ConstSimpleRadialPoseCasparToStacked(AsDoublePtr(cas_data),
                                             AsDoublePtr(stacked_data),
                                             cas_stride,
                                             0,
                                             num_objects);
      });
  module.def(
      "const_simple_radial_principal_point_stacked_to_caspar",
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
        ConstSimpleRadialPrincipalPointStackedToCaspar(
            AsDoublePtr(stacked_data),
            AsDoublePtr(cas_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "const_simple_radial_principal_point_caspar_to_stacked",
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

        ConstSimpleRadialPrincipalPointCasparToStacked(
            AsDoublePtr(cas_data),
            AsDoublePtr(stacked_data),
            cas_stride,
            0,
            num_objects);
      });
  module.def(
      "pinhole_calib_stacked_to_caspar",
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
        PinholeCalibStackedToCaspar(AsDoublePtr(stacked_data),
                                    AsDoublePtr(cas_data),
                                    cas_stride,
                                    0,
                                    num_objects);
      });
  module.def(
      "pinhole_calib_caspar_to_stacked",
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

        PinholeCalibCasparToStacked(AsDoublePtr(cas_data),
                                    AsDoublePtr(stacked_data),
                                    cas_stride,
                                    0,
                                    num_objects);
      });
  module.def(
      "pinhole_focal_stacked_to_caspar",
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
        PinholeFocalStackedToCaspar(AsDoublePtr(stacked_data),
                                    AsDoublePtr(cas_data),
                                    cas_stride,
                                    0,
                                    num_objects);
      });
  module.def(
      "pinhole_focal_caspar_to_stacked",
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

        PinholeFocalCasparToStacked(AsDoublePtr(cas_data),
                                    AsDoublePtr(stacked_data),
                                    cas_stride,
                                    0,
                                    num_objects);
      });
  module.def(
      "pinhole_pose_stacked_to_caspar",
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
        PinholePoseStackedToCaspar(AsDoublePtr(stacked_data),
                                   AsDoublePtr(cas_data),
                                   cas_stride,
                                   0,
                                   num_objects);
      });
  module.def(
      "pinhole_pose_caspar_to_stacked",
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

        PinholePoseCasparToStacked(AsDoublePtr(cas_data),
                                   AsDoublePtr(stacked_data),
                                   cas_stride,
                                   0,
                                   num_objects);
      });
  module.def(
      "pinhole_principal_point_stacked_to_caspar",
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
        PinholePrincipalPointStackedToCaspar(AsDoublePtr(stacked_data),
                                             AsDoublePtr(cas_data),
                                             cas_stride,
                                             0,
                                             num_objects);
      });
  module.def(
      "pinhole_principal_point_caspar_to_stacked",
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

        PinholePrincipalPointCasparToStacked(AsDoublePtr(cas_data),
                                             AsDoublePtr(stacked_data),
                                             cas_stride,
                                             0,
                                             num_objects);
      });
  module.def(
      "point_stacked_to_caspar",
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
        PointStackedToCaspar(AsDoublePtr(stacked_data),
                             AsDoublePtr(cas_data),
                             cas_stride,
                             0,
                             num_objects);
      });
  module.def(
      "point_caspar_to_stacked",
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

        PointCasparToStacked(AsDoublePtr(cas_data),
                             AsDoublePtr(stacked_data),
                             cas_stride,
                             0,
                             num_objects);
      });
  module.def(
      "simple_radial_calib_stacked_to_caspar",
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
        SimpleRadialCalibStackedToCaspar(AsDoublePtr(stacked_data),
                                         AsDoublePtr(cas_data),
                                         cas_stride,
                                         0,
                                         num_objects);
      });
  module.def(
      "simple_radial_calib_caspar_to_stacked",
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

        SimpleRadialCalibCasparToStacked(AsDoublePtr(cas_data),
                                         AsDoublePtr(stacked_data),
                                         cas_stride,
                                         0,
                                         num_objects);
      });
  module.def(
      "simple_radial_focal_and_distortion_stacked_to_caspar",
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
        SimpleRadialFocalAndDistortionStackedToCaspar(AsDoublePtr(stacked_data),
                                                      AsDoublePtr(cas_data),
                                                      cas_stride,
                                                      0,
                                                      num_objects);
      });
  module.def(
      "simple_radial_focal_and_distortion_caspar_to_stacked",
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

        SimpleRadialFocalAndDistortionCasparToStacked(AsDoublePtr(cas_data),
                                                      AsDoublePtr(stacked_data),
                                                      cas_stride,
                                                      0,
                                                      num_objects);
      });
  module.def(
      "simple_radial_pose_stacked_to_caspar",
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
        SimpleRadialPoseStackedToCaspar(AsDoublePtr(stacked_data),
                                        AsDoublePtr(cas_data),
                                        cas_stride,
                                        0,
                                        num_objects);
      });
  module.def(
      "simple_radial_pose_caspar_to_stacked",
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

        SimpleRadialPoseCasparToStacked(AsDoublePtr(cas_data),
                                        AsDoublePtr(stacked_data),
                                        cas_stride,
                                        0,
                                        num_objects);
      });
  module.def(
      "simple_radial_principal_point_stacked_to_caspar",
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
        SimpleRadialPrincipalPointStackedToCaspar(AsDoublePtr(stacked_data),
                                                  AsDoublePtr(cas_data),
                                                  cas_stride,
                                                  0,
                                                  num_objects);
      });
  module.def(
      "simple_radial_principal_point_caspar_to_stacked",
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

        SimpleRadialPrincipalPointCasparToStacked(AsDoublePtr(cas_data),
                                                  AsDoublePtr(stacked_data),
                                                  cas_stride,
                                                  0,
                                                  num_objects);
      });
}

}  // namespace caspar