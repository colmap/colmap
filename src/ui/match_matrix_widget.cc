// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "ui/colormaps.h"

namespace colmap {

const double MatchMatrixWidget::kZoomFactor = 1.33;

MatchMatrixWidget::MatchMatrixWidget(QWidget* parent, OptionManager* options)
    : QWidget(parent), options_(options), current_scale_(1.0) {
  setWindowFlags(Qt::Window);
  resize(parent->width() - 20, parent->height() - 20);
  setWindowTitle("Match matrix");

  QGridLayout* grid = new QGridLayout(this);
  grid->setContentsMargins(5, 5, 5, 5);

  image_label_ = new QLabel(this);
  image_scroll_area_ = new QScrollArea(this);
  image_scroll_area_->setWidget(image_label_);

  grid->addWidget(image_scroll_area_, 0, 0);

  QHBoxLayout* button_layout = new QHBoxLayout();

  QFont font;
  font.setPointSize(10);

  QPushButton* zoom_in_button = new QPushButton(tr("+"), this);
  zoom_in_button->setFont(font);
  zoom_in_button->setFixedWidth(50);
  button_layout->addWidget(zoom_in_button);
  connect(zoom_in_button, &QPushButton::released, this,
          &MatchMatrixWidget::ZoomIn);

  QPushButton* zoom_out_button = new QPushButton(tr("-"), this);
  zoom_out_button->setFont(font);
  zoom_out_button->setFixedWidth(50);
  button_layout->addWidget(zoom_out_button);
  connect(zoom_out_button, &QPushButton::released, this,
          &MatchMatrixWidget::ZoomOut);

  grid->addLayout(button_layout, 1, 0, Qt::AlignRight);
}

void MatchMatrixWidget::Update() {
  Database database;
  database.Open(*options_->database_path);

  // Sort the images according to their name.
  std::vector<Image> images = database.ReadAllImages();
  std::sort(images.begin(), images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  // Allocate the match matrix image.
  QImage match_matrix(images.size(), images.size(), QImage::Format_RGB32);
  match_matrix.fill(Qt::white);

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
    const double max_value = std::log(
        1.0 + *std::max_element(num_inliers.begin(), num_inliers.end()));
    for (size_t i = 0; i < image_pairs.size(); ++i) {
      const double value = std::log(1.0 + num_inliers[i]) / max_value;
      const size_t idx1 = image_id_to_idx.at(image_pairs[i].first);
      const size_t idx2 = image_id_to_idx.at(image_pairs[i].second);
      const QColor color(255 * JetColormap::Red(value),
                         255 * JetColormap::Green(value),
                         255 * JetColormap::Blue(value));
      match_matrix.setPixel(idx1, idx2, color.rgba());
      match_matrix.setPixel(idx2, idx1, color.rgba());
    }
  }

  // Remember the original image for zoom in/out.
  image_ = QPixmap::fromImage(match_matrix);

  current_scale_ = 1.0;
  const double scale =
      (image_scroll_area_->height() - 5) / static_cast<double>(image_.height());
  ScaleImage(scale);
}

void MatchMatrixWidget::closeEvent(QCloseEvent* event) {
  image_ = QPixmap();
  image_label_->clear();
}

void MatchMatrixWidget::ScaleImage(const double scale) {
  current_scale_ *= scale;

  const Qt::TransformationMode transform_mode =
      current_scale_ > 1.0 ? Qt::FastTransformation : Qt::SmoothTransformation;

  image_label_->setPixmap(image_.scaledToWidth(
      static_cast<int>(current_scale_ * image_.width()), transform_mode));

  image_label_->adjustSize();
}

void MatchMatrixWidget::ZoomIn() { ScaleImage(kZoomFactor); }

void MatchMatrixWidget::ZoomOut() { ScaleImage(1.0 / kZoomFactor); }

}  // namespace colmap
