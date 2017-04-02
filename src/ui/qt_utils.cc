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

#include "ui/qt_utils.h"

#include "base/camera_models.h"
#include "util/misc.h"

namespace colmap {

Eigen::Matrix4f QMatrixToEigen(const QMatrix4x4& matrix) {
  Eigen::Matrix4f eigen;
  for (size_t r = 0; r < 4; ++r) {
    for (size_t c = 0; c < 4; ++c) {
      eigen(r, c) = matrix(r, c);
    }
  }
  return eigen;
}

QMatrix4x4 EigenToQMatrix(const Eigen::Matrix4f& matrix) {
  QMatrix4x4 qt;
  for (size_t r = 0; r < 4; ++r) {
    for (size_t c = 0; c < 4; ++c) {
      qt(r, c) = matrix(r, c);
    }
  }
  return qt;
}

QImage BitmapToQImageRGB(const Bitmap& bitmap) {
  QImage image(bitmap.Width(), bitmap.Height(), QImage::Format_RGB32);
  for (int y = 0; y < image.height(); ++y) {
    QRgb* image_line = (QRgb*)image.scanLine(y);
    for (int x = 0; x < image.width(); ++x) {
      BitmapColor<uint8_t> color;
      if (bitmap.GetPixel(x, y, &color)) {
        image_line[x] = qRgba(color.r, color.g, color.b, 255);
      }
    }
  }
  return image;
}

QPixmap ShowImagesSideBySide(const QPixmap& image1, const QPixmap& image2) {
  QPixmap image = QPixmap(QSize(image1.width() + image2.width(),
                                std::max(image1.height(), image2.height())));

  image.fill(Qt::black);

  QPainter painter(&image);
  painter.drawImage(0, 0, image1.toImage());
  painter.drawImage(image1.width(), 0, image2.toImage());

  return image;
}

void DrawKeypoints(QPixmap* pixmap, const FeatureKeypoints& points,
                   const QColor& color) {
  if (pixmap->isNull()) {
    return;
  }

  const int pen_width = std::max(pixmap->width(), pixmap->height()) / 2048 + 1;
  const int radius = 3 * pen_width + (3 * pen_width) % 2;
  const float radius2 = radius / 2.0f;

  QPainter painter(pixmap);
  painter.setRenderHint(QPainter::Antialiasing);

  QPen pen;
  pen.setWidth(pen_width);
  pen.setColor(color);
  painter.setPen(pen);

  for (const auto& point : points) {
    painter.drawEllipse(point.x - radius2, point.y - radius2, radius, radius);
  }
}

QPixmap DrawMatches(const QPixmap& image1, const QPixmap& image2,
                    const FeatureKeypoints& points1,
                    const FeatureKeypoints& points2,
                    const FeatureMatches& matches,
                    const QColor& keypoints_color) {
  QPixmap image = ShowImagesSideBySide(image1, image2);

  QPainter painter(&image);
  painter.setRenderHint(QPainter::Antialiasing);

  // Draw keypoints

  const int pen_width = std::max(image.width(), image.height()) / 2048 + 1;
  const int radius = 3 * pen_width + (3 * pen_width) % 2;
  const float radius2 = radius / 2.0f;

  QPen pen;
  pen.setWidth(pen_width);
  pen.setColor(keypoints_color);
  painter.setPen(pen);

  for (const auto& point : points1) {
    painter.drawEllipse(point.x - radius2, point.y - radius2, radius, radius);
  }
  for (const auto& point : points2) {
    painter.drawEllipse(image1.width() + point.x - radius2, point.y - radius2,
                        radius, radius);
  }

  // Draw matches

  pen.setWidth(std::max(pen_width / 2, 1));

  for (const auto& match : matches) {
    const point2D_t idx1 = match.point2D_idx1;
    const point2D_t idx2 = match.point2D_idx2;
    pen.setColor(QColor(0, 255, 0));
    painter.setPen(pen);
    painter.drawLine(QPoint(points1[idx1].x, points1[idx1].y),
                     QPoint(image1.width() + points2[idx2].x, points2[idx2].y));
  }

  return image;
}

}  // namespace colmap
