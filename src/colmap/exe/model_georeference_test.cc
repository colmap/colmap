// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/exe/model_georeference.h"

#include "colmap/geometry/gps.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// The LichtFeld Studio import contract was established by exporting a scene,
// opening it in LichtFeld Studio, and confirming it displays upright. That
// check cannot run in CI, so these tests assert every element of the contract
// at the unit level instead: a change to the matrices or the basis mapping is
// then a release blocker rather than a regression nobody sees until the scene
// is already sideways in the viewer.

TEST(ModelGeoreference, LichtfeldColmapParsesAndRetainsExactSpelling) {
  // Contract item 1: `LICHTFELD_COLMAP` parses successfully and retains the
  // exact spelling; near-miss spellings and case variants are rejected.
  OutputCoordinateFrame frame;
  ASSERT_TRUE(OutputCoordinateFrameFromString("LICHTFELD_COLMAP", &frame));
  EXPECT_EQ(frame, OutputCoordinateFrame::LICHTFELD_COLMAP);

  EXPECT_FALSE(OutputCoordinateFrameFromString("lichtfeld_colmap", &frame));
  EXPECT_FALSE(OutputCoordinateFrameFromString("LichtfeldColmap", &frame));
  EXPECT_FALSE(OutputCoordinateFrameFromString("LICHTFELD-COLMAP", &frame));
  EXPECT_FALSE(OutputCoordinateFrameFromString("", &frame));

  ASSERT_TRUE(OutputCoordinateFrameFromString("ENU_Z_UP", &frame));
  EXPECT_EQ(frame, OutputCoordinateFrame::ENU_Z_UP);
}

TEST(ModelGeoreference, LichtfeldColmapIsProperRotation) {
  // Contract item 2: GeometryFromEnu(LICHTFELD_COLMAP) is a proper rotation
  // (determinant +1, not a reflection) with unit scale and zero translation.
  const Sim3d geometry_from_enu =
      GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP);
  EXPECT_NEAR(geometry_from_enu.scale(), 1.0, 1e-12);
  EXPECT_LT(geometry_from_enu.translation().norm(), 1e-12);
  const Eigen::Matrix3d rotation =
      geometry_from_enu.rotation().toRotationMatrix();
  EXPECT_NEAR(rotation.determinant(), 1.0, 1e-12);
  // Orthonormal: R^T R = I.
  EXPECT_LT(
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm(),
      1e-12);
}

TEST(ModelGeoreference, LichtfeldColmapBasisMapping) {
  // Contract item 3: maps ENU East to raw +X, North to raw +Z, and Up to raw
  // -Y.
  const Eigen::Matrix3d rotation =
      GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP)
          .rotation()
          .toRotationMatrix();
  const Eigen::Vector3d east(1.0, 0.0, 0.0);
  const Eigen::Vector3d north(0.0, 1.0, 0.0);
  const Eigen::Vector3d up(0.0, 0.0, 1.0);
  EXPECT_LT((rotation * east - Eigen::Vector3d(1.0, 0.0, 0.0)).norm(), 1e-12);
  EXPECT_LT((rotation * north - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1e-12);
  EXPECT_LT((rotation * up - Eigen::Vector3d(0.0, -1.0, 0.0)).norm(), 1e-12);
}

TEST(ModelGeoreference, LichtfeldVisualizerBoundaryMapsUpToDisplayedY) {
  // Contract item 4: applying LichtFeld's own data-to-visualizer boundary
  // transform on top of the raw written geometry maps ENU Up to displayed
  // +Y -- i.e. the full chain (visualizer_from_geometry * geometry_from_enu)
  // sends ENU-up to the visualizer's +Y, matching the empirically-verified
  // upright display.
  const Sim3d geometry_from_enu =
      GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP);
  const Sim3d visualizer_from_geometry = LichtfeldVisualizerFromColmapData();
  const Eigen::Vector3d enu_up(0.0, 0.0, 1.0);
  const Eigen::Vector3d displayed_up = visualizer_from_geometry.rotation() *
                                       (geometry_from_enu.rotation() * enu_up);
  EXPECT_LT((displayed_up - Eigen::Vector3d(0.0, 1.0, 0.0)).norm(), 1e-12);

  // The boundary transform itself is a pure axis-sign-flip rotation
  // (determinant +1), not applied by this exporter but recorded in the
  // report's consumer_profile for downstream composition.
  const Eigen::Matrix3d boundary_rotation =
      visualizer_from_geometry.rotation().toRotationMatrix();
  EXPECT_NEAR(boundary_rotation.determinant(), 1.0, 1e-12);
}

TEST(ModelGeoreference, TransformAppliesToPointsCentersAndOrientations) {
  // Contract item 5: a model export transforms points, camera centers, and
  // camera orientations -- not only points.
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 3;
  options.num_points3D = 10;
  SynthesizeDataset(options, &reconstruction);

  const image_t image_id = reconstruction.RegImageIds().front();
  const Eigen::Vector3d original_center =
      reconstruction.Image(image_id).ProjectionCenter();
  const Eigen::Quaterniond original_rotation =
      reconstruction.Image(image_id).CamFromWorld().rotation();
  const point3D_t point_id = reconstruction.Points3D().begin()->first;
  const Eigen::Vector3d original_point = reconstruction.Point3D(point_id).xyz;

  const Sim3d geometry_from_enu =
      GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP);
  reconstruction.Transform(geometry_from_enu);

  // Points transform by the Sim3d directly.
  EXPECT_LT((reconstruction.Point3D(point_id).xyz -
             geometry_from_enu * original_point)
                .norm(),
            1e-9);

  // Camera centers transform by the Sim3d directly.
  EXPECT_LT((reconstruction.Image(image_id).ProjectionCenter() -
             geometry_from_enu * original_center)
                .norm(),
            1e-9);

  // Camera orientations are not left untouched: applying the transform
  // changed the orientation, and applying its exact inverse recovers the
  // original orientation -- proving orientation was transformed
  // consistently with points/centers, not merely passed through.
  const Eigen::Quaterniond transformed_rotation =
      reconstruction.Image(image_id).CamFromWorld().rotation();
  EXPECT_GT(original_rotation.angularDistance(transformed_rotation), 1e-6);
  reconstruction.Transform(Inverse(geometry_from_enu));
  EXPECT_LT(original_rotation.angularDistance(
                reconstruction.Image(image_id).CamFromWorld().rotation()),
            1e-9);
}

TEST(ModelGeoreference, GeometryEnuRoundTripIsIdentityForEveryOutputFrame) {
  // Contract item 6: enu_from_geometry * geometry_from_enu is identity, for
  // every supported output frame (not just LICHTFELD_COLMAP).
  const Eigen::Vector3d probe(1.0, 2.0, 3.0);
  for (const OutputCoordinateFrame frame :
       {OutputCoordinateFrame::ENU_Z_UP,
        OutputCoordinateFrame::LICHTFELD_COLMAP}) {
    const Sim3d geometry_from_enu = GeometryFromEnu(frame);
    const Sim3d enu_from_geometry = Inverse(geometry_from_enu);
    const Eigen::Vector3d round_trip =
        enu_from_geometry * (geometry_from_enu * probe);
    EXPECT_LT((round_trip - probe).norm(), 1e-12);
  }
}

TEST(ModelGeoreference, EcefGeometryRoundTripIsIdentity) {
  // Contract item 7: ecef_from_geometry * geometry_from_ecef is identity,
  // composed through a real WGS84 ECEF-from-ENU rotation exactly as the
  // georeference report does (not a synthetic placeholder rotation).
  const double origin_lat = 45.5;
  const double origin_lon = -73.6;
  const double origin_alt = 120.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const Eigen::Matrix3d ecef_from_enu_rotation =
      GPSTransform::ECEFFromENU(origin_lat, origin_lon);
  const Eigen::Vector3d origin_ecef = gps_transform.EllipsoidToECEF(
      {Eigen::Vector3d(origin_lat, origin_lon, origin_alt)})[0];
  const Sim3d ecef_from_enu(
      1.0, Eigen::Quaterniond(ecef_from_enu_rotation), origin_ecef);

  const Sim3d geometry_from_enu =
      GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP);
  const Sim3d enu_from_geometry = Inverse(geometry_from_enu);
  const Sim3d ecef_from_geometry = ecef_from_enu * enu_from_geometry;
  const Sim3d geometry_from_ecef = Inverse(ecef_from_geometry);

  const Eigen::Vector3d probe(1.0, 2.0, 3.0);
  const Eigen::Vector3d round_trip =
      ecef_from_geometry * (geometry_from_ecef * probe);
  EXPECT_LT((round_trip - probe).norm(), 1e-6);
}

TEST(ModelGeoreference, EcefTransformSurvivesDeletionOnlyCrop) {
  // For a point in the serialized geometry frame,
  // point_ecef = ecef_from_geometry * point_geometry. If an editor only
  // deletes points/Gaussians, every surviving coordinate is unchanged and
  // the same ecef_from_geometry remains exactly valid for a tiny remaining
  // segment -- no cameras or deleted geometry are required to apply it.
  const double origin_lat = 45.5;
  const double origin_lon = -73.6;
  const double origin_alt = 120.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const Eigen::Matrix3d ecef_from_enu_rotation =
      GPSTransform::ECEFFromENU(origin_lat, origin_lon);
  const Eigen::Vector3d origin_ecef = gps_transform.EllipsoidToECEF(
      {Eigen::Vector3d(origin_lat, origin_lon, origin_alt)})[0];
  const Sim3d ecef_from_enu(
      1.0, Eigen::Quaterniond(ecef_from_enu_rotation), origin_ecef);
  const Sim3d geometry_from_enu =
      GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP);
  const Sim3d ecef_from_geometry = ecef_from_enu * Inverse(geometry_from_enu);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 3;
  options.num_points3D = 10;
  SynthesizeDataset(options, &reconstruction);
  reconstruction.Transform(geometry_from_enu);

  std::vector<point3D_t> point_ids;
  for (const auto& [point_id, _] : reconstruction.Points3D()) {
    point_ids.push_back(point_id);
  }
  ASSERT_GE(point_ids.size(), 4u);

  // Record the ECEF position of two surviving points before the crop.
  const point3D_t surviving_id_1 = point_ids[0];
  const point3D_t surviving_id_2 = point_ids[1];
  const Eigen::Vector3d ecef_before_1 =
      ecef_from_geometry * reconstruction.Point3D(surviving_id_1).xyz;
  const Eigen::Vector3d ecef_before_2 =
      ecef_from_geometry * reconstruction.Point3D(surviving_id_2).xyz;

  // Crop: delete every other point (a deletion-only edit; no camera is
  // touched and no image is deregistered).
  for (size_t i = 2; i < point_ids.size(); ++i) {
    reconstruction.DeletePoint3D(point_ids[i]);
  }
  ASSERT_EQ(reconstruction.NumPoints3D(), 2u);

  // The exact same transform, recomputed with no knowledge of the crop
  // (same origin + output frame only), still recovers the correct ECEF
  // position for both surviving points.
  const Sim3d ecef_from_geometry_recomputed =
      ecef_from_enu *
      Inverse(GeometryFromEnu(OutputCoordinateFrame::LICHTFELD_COLMAP));
  EXPECT_LT((ecef_from_geometry_recomputed *
                 reconstruction.Point3D(surviving_id_1).xyz -
             ecef_before_1)
                .norm(),
            1e-9);
  EXPECT_LT((ecef_from_geometry_recomputed *
                 reconstruction.Point3D(surviving_id_2).xyz -
             ecef_before_2)
                .norm(),
            1e-9);
}

// Contract items 8 (frame_contract.geometry_frame remains LICHTFELD_COLMAP)
// and 9 (raw up_axis remains -Y, consumer display up-axis remains Y) reduce
// to the basis-mapping assertions above (items 3 and 4) plus the report's
// literal frame-name/up-axis strings, which are covered end-to-end -- against
// the actual serialized reconstruction and the actual JSON report written by
// a real model_aligner report run -- by
// ModelAligner.OutputCoordinateFrameLichtfeldColmap in model_test.cc.

}  // namespace
}  // namespace colmap
