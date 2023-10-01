// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/ui/match_matrix_widget.h"

namespace colmap {

MatchMatrixWidget::MatchMatrixWidget(QWidget* parent, OptionManager* options)
    : ImageViewerWidget(parent), options_(options) {
  setWindowTitle("Match matrix");
}

void MatchMatrixWidget::Show() {
  Database database(*options_->database_path);

  if (database.NumImages() == 0) {
    return;
  }

  // Sort the images according to their name.
  std::vector<Image> images = database.ReadAllImages();
  std::sort(images.begin(),
            images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  // Allocate the match matrix image.
  Bitmap match_matrix;
  match_matrix.Allocate(images.size(), images.size(), true);
  match_matrix.Fill(BitmapColor<uint8_t>(255));

  // Map image identifiers to match matrix locations.
  std::unordered_map<image_t, size_t> image_id_to_idx;
  for (size_t idx = 0; idx < images.size(); ++idx) {
    image_id_to_idx.emplace(images[idx].ImageId(), idx);
  }

  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::vector<int> num_inliers;
  database.ReadTwoViewGeometryNumInliers(&image_pairs, &num_inliers);

  // Fill the match matrix.
  if (!num_inliers.empty()) {
    const double max_value =
        std::log1p(*std::max_element(num_inliers.begin(), num_inliers.end()));
    for (size_t i = 0; i < image_pairs.size(); ++i) {
      const double value = std::log1p(num_inliers[i]) / max_value;
      const size_t idx1 = image_id_to_idx.at(image_pairs[i].first);
      const size_t idx2 = image_id_to_idx.at(image_pairs[i].second);
      const BitmapColor<float> color(255 * JetColormap::Red(value),
                                     255 * JetColormap::Green(value),
                                     255 * JetColormap::Blue(value));
      match_matrix.SetPixel(idx1, idx2, color.Cast<uint8_t>());
      match_matrix.SetPixel(idx2, idx1, color.Cast<uint8_t>());
    }
  }

  ShowBitmap(match_matrix);
}

}  // namespace colmap
