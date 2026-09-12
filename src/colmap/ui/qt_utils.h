// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_alignment.h"

#include <QtCore>
#include <QtGui>
#include <QtOpenGL>

#include <Eigen/Core>

namespace colmap {

// Loads a monochrome SVG icon from a Qt resource path and tints it to match
// the current application palette, so that icons remain legible under both
// light and dark themes. The icons under :/media are flat silhouettes without
// an explicit fill color (which defaults to black per the SVG spec), so a
// solid tint to the palette foreground color is exact. The tint is baked in
// when the icon is created, so a theme change at runtime is only reflected
// once icons are recreated.
QIcon ThemedIcon(const QString& resource_path);

Eigen::Matrix4f QMatrixToEigen(const QMatrix4x4& matrix);

QMatrix4x4 EigenToQMatrix(const Eigen::Matrix4f& matrix);

QImage BitmapToQImageRGB(const Bitmap& bitmap);

void DrawKeypoints(QPixmap* image,
                   const FeatureKeypoints& points,
                   const QColor& color = Qt::red);

QPixmap ShowImagesSideBySide(const QPixmap& image1, const QPixmap& image2);

QPixmap DrawMatches(const QPixmap& image1,
                    const QPixmap& image2,
                    const FeatureKeypoints& points1,
                    const FeatureKeypoints& points2,
                    const FeatureMatches& matches,
                    const QColor& keypoints_color = Qt::red);

}  // namespace colmap
