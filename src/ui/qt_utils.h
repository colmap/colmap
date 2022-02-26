// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_UI_QT_UTILS_H_
#define COLMAP_SRC_UI_QT_UTILS_H_

#include <Eigen/Core>

#include <QtCore>
#include <QtOpenGL>

#include "feature/types.h"
#include "util/bitmap.h"
#include "util/types.h"

namespace colmap {

Eigen::Matrix4f QMatrixToEigen(const QMatrix4x4& matrix);

QMatrix4x4 EigenToQMatrix(const Eigen::Matrix4f& matrix);

QImage BitmapToQImageRGB(const Bitmap& bitmap);

void DrawKeypoints(QPixmap* image, const FeatureKeypoints& points,
                   const QColor& color = Qt::red);

QPixmap ShowImagesSideBySide(const QPixmap& image1, const QPixmap& image2);

QPixmap DrawMatches(const QPixmap& image1, const QPixmap& image2,
                    const FeatureKeypoints& points1,
                    const FeatureKeypoints& points2,
                    const FeatureMatches& matches,
                    const QColor& keypoints_color = Qt::red);

}  // namespace colmap

#endif  // COLMAP_SRC_UI_QT_UTILS_H_
