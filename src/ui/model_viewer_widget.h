// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_UI_MODEL_VIEWER_WIDGET_H_
#define COLMAP_SRC_UI_MODEL_VIEWER_WIDGET_H_

#include <QtCore>
#include <QtOpenGL>

#include <QOpenGLFunctions_3_2_Core>

#include "base/database.h"
#include "base/reconstruction.h"
#include "ui/colormaps.h"
#include "ui/image_viewer_widget.h"
#include "ui/line_painter.h"
#include "ui/movie_grabber_widget.h"
#include "ui/point_painter.h"
#include "ui/point_viewer_widget.h"
#include "ui/render_options.h"
#include "ui/triangle_painter.h"
#include "util/option_manager.h"

namespace colmap {

class ModelViewerWidget : public QOpenGLWidget,
                          protected QOpenGLFunctions_3_2_Core {
 public:
  const float kInitNearPlane = 1.0f;
  const float kMinNearPlane = 1e-3f;
  const float kMaxNearPlane = 1e5f;
  const float kNearPlaneScaleSpeed = 0.02f;
  const float kFarPlane = 1e5f;
  const float kInitFocusDistance = 100.0f;
  const float kMinFocusDistance = 1e-5f;
  const float kMaxFocusDistance = 1e8f;
  const float kFieldOfView = 25.0f;
  const float kFocusSpeed = 2.0f;
  const float kInitPointSize = 1.0f;
  const float kMinPointSize = 0.5f;
  const float kMaxPointSize = 100.0f;
  const float kPointScaleSpeed = 0.1f;
  const float kInitImageSize = 0.2f;
  const float kMinImageSize = 1e-6f;
  const float kMaxImageSize = 1e3f;
  const float kImageScaleSpeed = 0.1f;
  const int kDoubleClickInterval = 250;

  ModelViewerWidget(QWidget* parent, OptionManager* options);

  void ReloadReconstruction();
  void ClearReconstruction();

  int GetProjectionType() const;

  void SetPointColormap(PointColormapBase* colormap);

  void UpdateMovieGrabber();

  void EnableCoordinateGrid();
  void DisableCoordinateGrid();

  void ChangeFocusDistance(const float delta);
  void ChangeNearPlane(const float delta);
  void ChangePointSize(const float delta);
  void ChangeCameraSize(const float delta);

  void RotateView(const float x, const float y, const float prev_x,
                  const float prev_y);
  void TranslateView(const float x, const float y, const float prev_x,
                     const float prev_y);

  void ResetView();

  QMatrix4x4 ModelViewMatrix() const;
  void SetModelViewMatrix(const QMatrix4x4& matrix);

  void SelectObject(const int x, const int y);
  void SelectMoviewGrabberView(const size_t view_idx);

  QImage GrabImage();
  void GrabMovie();

  void ShowPointInfo(const point3D_t point3D_id);
  void ShowImageInfo(const image_t image_id);

  float PointSize() const;
  float ImageSize() const;
  void SetPointSize(const float point_size);
  void SetImageSize(const float image_size);

  void SetBackgroundColor(const float r, const float g, const float b);

  // Copy of current scene data that is displayed
  Reconstruction* reconstruction;
  EIGEN_STL_UMAP(camera_t, Camera) cameras;
  EIGEN_STL_UMAP(image_t, Image) images;
  EIGEN_STL_UMAP(point3D_t, Point3D) points3D;
  std::vector<image_t> reg_image_ids;

  QLabel* statusbar_status_label;

 protected:
  void initializeGL() override;
  void resizeGL(int width, int height) override;
  void paintGL() override;

 private:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;

  void SetupPainters();
  void SetupView();

  void Upload();
  void UploadCoordinateGridData();
  void UploadPointData(const bool selection_mode = false);
  void UploadPointConnectionData();
  void UploadImageData(const bool selection_mode = false);
  void UploadImageConnectionData();
  void UploadMovieGrabberData();

  void ComposeProjectionMatrix();

  float ZoomScale() const;
  float AspectRatio() const;
  float OrthographicWindowExtent() const;

  Eigen::Vector3f PositionToArcballVector(const float x, const float y) const;

  OptionManager* options_;

  QMatrix4x4 model_view_matrix_;
  QMatrix4x4 projection_matrix_;

  LinePainter coordinate_axes_painter_;
  LinePainter coordinate_grid_painter_;

  PointPainter point_painter_;
  LinePainter point_connection_painter_;

  LinePainter image_line_painter_;
  TrianglePainter image_triangle_painter_;
  LinePainter image_connection_painter_;

  LinePainter movie_grabber_path_painter_;
  LinePainter movie_grabber_line_painter_;
  TrianglePainter movie_grabber_triangle_painter_;

  PointViewerWidget* point_viewer_widget_;
  DatabaseImageViewerWidget* image_viewer_widget_;
  MovieGrabberWidget* movie_grabber_widget_;

  std::unique_ptr<PointColormapBase> point_colormap_;

  bool mouse_is_pressed_;
  QTimer mouse_press_timer_;
  QPoint prev_mouse_pos_;

  float focus_distance_;

  std::vector<std::pair<size_t, char>> selection_buffer_;
  image_t selected_image_id_;
  point3D_t selected_point3D_id_;
  size_t selected_movie_grabber_view_;

  bool coordinate_grid_enabled_;

  // Size of points (dynamic): does not require re-uploading of points.
  float point_size_;
  // Size of image models (not dynamic): requires re-uploading of image models.
  float image_size_;
  // Near clipping plane.
  float near_plane_;

  float bg_color_[3];
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_MODEL_VIEWER_WIDGET_H_
