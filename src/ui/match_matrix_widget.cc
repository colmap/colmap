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

#include "ui/match_matrix_widget.h"

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
  std::sort(images.begin(), images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  // Allocate the match matrix image.
  Bitmap match_matrix;
  match_matrix.Allocate(images.size(), images.size(), true);
  match_matrix.Fill(BitmapColor<uint8_t>(255, 255, 255));

  // Map image identifiers to match matrix locations.
  std::unordered_map<image_t, size_t> image_id_to_idx;
  for (size_t idx = 0; idx < images.size(); ++idx) {
    image_id_to_idx.emplace(images[idx].ImageId(), idx);
  }

  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::vector<int> num_inliers;
  database.ReadInlierMatchesGraph(&image_pairs, &num_inliers);

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

  ShowBitmap(match_matrix, true);
}

}  // namespace colmap
