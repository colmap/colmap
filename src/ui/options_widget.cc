// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#include "ui/options_widget.h"

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

void OptionsWidget::AddOptionRow(const std::string& label_text, QWidget* widget,
                                 void* option) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  widget->setFont(font());
  grid_layout_->addWidget(widget, grid_layout_->rowCount() - 1, 1);

  option_rows_.emplace(option, std::make_pair(label, widget));
  widget_rows_.emplace(widget, std::make_pair(label, widget));
}

void OptionsWidget::AddWidgetRow(const std::string& label_text,
                                 QWidget* widget) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  widget->setFont(font());
  grid_layout_->addWidget(widget, grid_layout_->rowCount() - 1, 1);

  widget_rows_.emplace(widget, std::make_pair(label, widget));
}

void OptionsWidget::AddLayoutRow(const std::string& label_text,
                                 QLayout* layout) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(label, grid_layout_->rowCount(), 0);

  QWidget* layout_widget = new QWidget(this);
  layout_widget->setLayout(layout);
  layout->setContentsMargins(0, 0, 0, 0);

  grid_layout_->addWidget(layout_widget, grid_layout_->rowCount() - 1, 1);

  layout_rows_.emplace(layout, std::make_pair(label, layout_widget));
}

QSpinBox* OptionsWidget::AddOptionInt(int* option,
                                      const std::string& label_text,
                                      const int min, const int max) {
  QSpinBox* spinbox = new QSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);

  AddOptionRow(label_text, spinbox, option);

  options_int_.emplace_back(spinbox, option);

  return spinbox;
}

QDoubleSpinBox* OptionsWidget::AddOptionDouble(
    double* option, const std::string& label_text, const double min,
    const double max, const double step, const int decimals) {
  QDoubleSpinBox* spinbox = new QDoubleSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setSingleStep(step);
  spinbox->setDecimals(decimals);

  AddOptionRow(label_text, spinbox, option);

  options_double_.emplace_back(spinbox, option);

  return spinbox;
}

QDoubleSpinBox* OptionsWidget::AddOptionDoubleLog(
    double* option, const std::string& label_text, const double min,
    const double max, const double step, const int decimals) {
  QDoubleSpinBox* spinbox = new QDoubleSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setSingleStep(step);
  spinbox->setDecimals(decimals);

  AddOptionRow(label_text, spinbox, option);

  options_double_log_.emplace_back(spinbox, option);

  return spinbox;
}

QCheckBox* OptionsWidget::AddOptionBool(bool* option,
                                        const std::string& label_text) {
  QCheckBox* checkbox = new QCheckBox(this);

  AddOptionRow(label_text, checkbox, option);

  options_bool_.emplace_back(checkbox, option);

  return checkbox;
}

QLineEdit* OptionsWidget::AddOptionText(std::string* option,
                                        const std::string& label_text) {
  QLineEdit* line_edit = new QLineEdit(this);

  AddOptionRow(label_text, line_edit, option);

  options_text_.emplace_back(line_edit, option);

  return line_edit;
}

QLineEdit* OptionsWidget::AddOptionFilePath(std::string* option,
                                            const std::string& label_text) {
  QLineEdit* line_edit = new QLineEdit(this);

  AddOptionRow(label_text, line_edit, option);

  auto SelectPathFunc = [this, line_edit]() {
    line_edit->setText(QFileDialog::getOpenFileName(this, tr("Select file")));
  };

  QPushButton* select_button = new QPushButton(tr("Select file"), this);
  select_button->setFont(font());
  connect(select_button, &QPushButton::released, this, SelectPathFunc);
  grid_layout_->addWidget(select_button, grid_layout_->rowCount(), 1);

  options_path_.emplace_back(line_edit, option);

  return line_edit;
}

QLineEdit* OptionsWidget::AddOptionDirPath(std::string* option,
                                           const std::string& label_text) {
  QLineEdit* line_edit = new QLineEdit(this);

  AddOptionRow(label_text, line_edit, option);

  auto SelectPathFunc = [this, line_edit]() {
    line_edit->setText(
        QFileDialog::getExistingDirectory(this, tr("Select folder")));
  };

  QPushButton* select_button = new QPushButton(tr("Select folder"), this);
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

void OptionsWidget::AddSection(const std::string& title) {
  QLabel* label = new QLabel(tr(title.c_str()), this);
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

void OptionsWidget::showEvent(QShowEvent* event) { ReadOptions(); }

void OptionsWidget::closeEvent(QCloseEvent* event) { WriteOptions(); }

void OptionsWidget::hideEvent(QHideEvent* event) { WriteOptions(); }

void OptionsWidget::ShowOption(void* option) {
  auto& option_row = option_rows_.at(option);
  option_row.first->show();
  option_row.second->show();
}

void OptionsWidget::HideOption(void* option) {
  auto& option_row = option_rows_.at(option);
  option_row.first->hide();
  option_row.second->hide();
}

void OptionsWidget::ShowWidget(QWidget* widget) {
  auto& widget_row = widget_rows_.at(widget);
  widget_row.first->show();
  widget_row.second->show();
}

void OptionsWidget::HideWidget(QWidget* widget) {
  auto& widget_row = widget_rows_.at(widget);
  widget_row.first->hide();
  widget_row.second->hide();
}

void OptionsWidget::ShowLayout(QLayout* layout) {
  auto& layout_row = layout_rows_.at(layout);
  layout_row.first->show();
  layout_row.second->show();
}

void OptionsWidget::HideLayout(QLayout* layout) {
  auto& layout_row = layout_rows_.at(layout);
  layout_row.first->hide();
  layout_row.second->hide();
}

}  // namespace colmap
