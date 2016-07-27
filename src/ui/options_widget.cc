// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "ui/options_widget.h"

#include <cmath>

namespace colmap {

OptionsWidget::OptionsWidget(QWidget* parent) : QWidget(parent) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);

  QFont font;
  font.setPointSize(10);
  setFont(font);

  grid_layout_ = new QGridLayout(this);
  grid_layout_->setVerticalSpacing(3);
  grid_layout_->setAlignment(Qt::AlignTop);
  setLayout(grid_layout_);
}

void OptionsWidget::showEvent(QShowEvent* event) { ReadOptions(); }

void OptionsWidget::closeEvent(QCloseEvent* event) { WriteOptions(); }

void OptionsWidget::hideEvent(QHideEvent* event) { WriteOptions(); }

QSpinBox* OptionsWidget::AddOptionInt(int* option,
                                      const std::string& label_text,
                                      const int min, const int max) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QSpinBox* spinbox = new QSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  grid_layout_->addWidget(spinbox, grid_layout_->rowCount() - 1, 1);

  options_int_.emplace_back(spinbox, option);

  return spinbox;
}

QDoubleSpinBox* OptionsWidget::AddOptionDouble(
    double* option, const std::string& label_text, const double min,
    const double max, const double step, const int decimals) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QDoubleSpinBox* spinbox = new QDoubleSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setSingleStep(step);
  spinbox->setDecimals(decimals);
  grid_layout_->addWidget(spinbox, grid_layout_->rowCount() - 1, 1);

  options_double_.emplace_back(spinbox, option);

  return spinbox;
}

QDoubleSpinBox* OptionsWidget::AddOptionDoubleLog(
    double* option, const std::string& label_text, const double min,
    const double max, const double step, const int decimals) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QDoubleSpinBox* spinbox = new QDoubleSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setSingleStep(step);
  spinbox->setDecimals(decimals);
  grid_layout_->addWidget(spinbox, grid_layout_->rowCount() - 1, 1);

  options_double_log_.emplace_back(spinbox, option);

  return spinbox;
}

QCheckBox* OptionsWidget::AddOptionBool(bool* option,
                                        const std::string& label_text) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QCheckBox* checkbox = new QCheckBox(this);
  grid_layout_->addWidget(checkbox, grid_layout_->rowCount() - 1, 1);

  options_bool_.emplace_back(checkbox, option);

  return checkbox;
}

QLineEdit* OptionsWidget::AddOptionText(std::string* option,
                                        const std::string& label_text) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QLineEdit* line_edit = new QLineEdit(this);
  grid_layout_->addWidget(line_edit, grid_layout_->rowCount() - 1, 1);

  options_text_.emplace_back(line_edit, option);

  return line_edit;
}

QLineEdit* OptionsWidget::AddOptionFilePath(std::string* option,
                                            const std::string& label_text) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QLineEdit* line_edit = new QLineEdit(this);
  grid_layout_->addWidget(line_edit, grid_layout_->rowCount() - 1, 1);

  auto SelectPathFunc = [this, line_edit]() {
    line_edit->setText(QFileDialog::getOpenFileName(this, tr("Select path")));
  };

  QPushButton* select_button = new QPushButton(tr("Select path"), this);
  select_button->setFont(font());
  connect(select_button, &QPushButton::released, this, SelectPathFunc);
  grid_layout_->addWidget(select_button, grid_layout_->rowCount(), 1);

  options_path_.emplace_back(line_edit, option);

  return line_edit;
}

QLineEdit* OptionsWidget::AddOptionDirPath(std::string* option,
                                           const std::string& label_text) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QLineEdit* line_edit = new QLineEdit(this);
  grid_layout_->addWidget(line_edit, grid_layout_->rowCount() - 1, 1);

  auto SelectPathFunc = [this, line_edit]() {
    line_edit->setText(
        QFileDialog::getExistingDirectory(this, tr("Select path")));
  };

  QPushButton* select_button = new QPushButton(tr("Select path"), this);
  select_button->setFont(font());
  connect(select_button, &QPushButton::released, this, SelectPathFunc);
  grid_layout_->addWidget(select_button, grid_layout_->rowCount(), 1);

  options_path_.emplace_back(line_edit, option);

  return line_edit;
}

void OptionsWidget::AddSpacer() {
  QLabel* label = new QLabel("", this);
  label->setFont(font());
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0, 2, 1);
}

void OptionsWidget::AddSection(const std::string& label_text) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setContentsMargins(0, 0, 0, 5);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0, 1, 2,
                          Qt::AlignHCenter);
}

void OptionsWidget::ReadOptions() {
  for (auto& option : options_int_) {
    option.first->setValue(*option.second);
  }

  for (auto& option : options_double_) {
    option.first->setValue(*option.second);
  }

  for (auto& option : options_double_log_) {
    option.first->setValue(std::log10(*option.second));
  }

  for (auto& option : options_bool_) {
    option.first->setChecked(*option.second);
  }

  for (auto& option : options_text_) {
    option.first->setText(QString::fromStdString(*option.second));
  }

  for (auto& option : options_path_) {
    option.first->setText(QString::fromStdString(*option.second));
  }
}

void OptionsWidget::WriteOptions() {
  for (auto& option : options_int_) {
    *option.second = option.first->value();
  }

  for (auto& option : options_double_) {
    *option.second = option.first->value();
  }

  for (auto& option : options_double_log_) {
    *option.second = std::pow(10, option.first->value());
  }

  for (auto& option : options_bool_) {
    *option.second = option.first->isChecked();
  }

  for (auto& option : options_text_) {
    *option.second = option.first->text().toUtf8().constData();
  }

  for (auto& option : options_path_) {
    *option.second = option.first->text().toUtf8().constData();
  }
}

}  // namespace colmap
