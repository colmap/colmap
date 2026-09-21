// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/options_widget.h"

#include <limits>

namespace colmap {
namespace {

// Maps the stored "unlimited" sentinel (INT_MAX) to the displayed value -1.
int UnlimitedOptionToSpinbox(int option, int max) {
  return option >= max ? -1 : option;
}

// Maps the displayed value -1 back to the stored "unlimited" sentinel
// (INT_MAX).
int UnlimitedSpinboxToOption(int value) {
  return value < 0 ? std::numeric_limits<int>::max() : value;
}

}  // namespace

OptionsWidget::OptionsWidget(QWidget* parent) : QWidget(parent) {
  QFont font;
  font.setPointSize(10);
  setFont(font);

  grid_layout_ = new QGridLayout(this);
  grid_layout_->setVerticalSpacing(3);
  grid_layout_->setAlignment(Qt::AlignTop);
  setLayout(grid_layout_);
}

QLabel* OptionsWidget::CreateRowLabel(const std::string& label_text) {
  QLabel* label = new QLabel(tr(label_text.c_str()), this);
  label->setFont(font());
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  return label;
}

void OptionsWidget::AddOptionRow(const std::string& label_text,
                                 QWidget* widget,
                                 void* option) {
  QLabel* label = CreateRowLabel(label_text);
  const int row = grid_layout_->rowCount();
  grid_layout_->addWidget(label, row, 0);

  widget->setFont(font());
  grid_layout_->addWidget(widget, row, 1);

  option_rows_.emplace(option, std::make_pair(label, widget));
  widget_rows_.emplace(widget, std::make_pair(label, widget));
}

void OptionsWidget::AddWidgetRow(const std::string& label_text,
                                 QWidget* widget) {
  QLabel* label = CreateRowLabel(label_text);
  const int row = grid_layout_->rowCount();
  grid_layout_->addWidget(label, row, 0);

  widget->setFont(font());
  grid_layout_->addWidget(widget, row, 1);

  widget_rows_.emplace(widget, std::make_pair(label, widget));
}

void OptionsWidget::AddLayoutRow(const std::string& label_text,
                                 QLayout* layout) {
  QLabel* label = CreateRowLabel(label_text);
  const int row = grid_layout_->rowCount();
  grid_layout_->addWidget(label, row, 0);

  QWidget* layout_widget = new QWidget(this);
  layout_widget->setLayout(layout);
  layout->setContentsMargins(0, 0, 0, 0);

  grid_layout_->addWidget(layout_widget, row, 1);

  layout_rows_.emplace(layout, std::make_pair(label, layout_widget));
}

QSpinBox* OptionsWidget::AddOptionInt(int* option,
                                      const std::string& label_text,
                                      const int min,
                                      const int max) {
  QSpinBox* spinbox = new QSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setValue(*option);

  AddOptionRow(label_text, spinbox, option);

  options_int_.emplace_back(spinbox, option);

  return spinbox;
}

QSpinBox* OptionsWidget::AddOptionIntUnlimited(int* option,
                                               const std::string& label_text,
                                               const int max) {
  QSpinBox* spinbox = new QSpinBox(this);
  spinbox->setMinimum(-1);
  spinbox->setMaximum(max);
  spinbox->setSpecialValueText(tr("unlimited"));
  spinbox->setValue(UnlimitedOptionToSpinbox(*option, max));

  AddOptionRow(label_text, spinbox, option);

  options_int_unlimited_.emplace_back(spinbox, option);

  return spinbox;
}

QDoubleSpinBox* OptionsWidget::AddOptionDouble(double* option,
                                               const std::string& label_text,
                                               const double min,
                                               const double max,
                                               const double step,
                                               const int decimals) {
  QDoubleSpinBox* spinbox = new QDoubleSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setSingleStep(step);
  spinbox->setDecimals(decimals);
  spinbox->setValue(*option);

  AddOptionRow(label_text, spinbox, option);

  options_double_.emplace_back(spinbox, option);

  return spinbox;
}

QDoubleSpinBox* OptionsWidget::AddOptionDoubleLog(double* option,
                                                  const std::string& label_text,
                                                  const double min,
                                                  const double max,
                                                  const double step,
                                                  const int decimals) {
  QDoubleSpinBox* spinbox = new QDoubleSpinBox(this);
  spinbox->setMinimum(min);
  spinbox->setMaximum(max);
  spinbox->setSingleStep(step);
  spinbox->setDecimals(decimals);
  spinbox->setValue(*option);

  AddOptionRow(label_text, spinbox, option);

  options_double_log_.emplace_back(spinbox, option);

  return spinbox;
}

QCheckBox* OptionsWidget::AddOptionBool(bool* option,
                                        const std::string& label_text) {
  QCheckBox* checkbox = new QCheckBox(this);
  checkbox->setChecked(*option);

  AddOptionRow(label_text, checkbox, option);

  options_bool_.emplace_back(checkbox, option);

  return checkbox;
}

QLineEdit* OptionsWidget::AddOptionText(std::string* option,
                                        const std::string& label_text) {
  QLineEdit* line_edit = new QLineEdit(this);
  line_edit->setText(QString::fromStdString(*option));

  AddOptionRow(label_text, line_edit, option);

  options_text_.emplace_back(line_edit, option);

  return line_edit;
}

QLineEdit* OptionsWidget::AddOptionPath(std::filesystem::path* option,
                                        const std::string& label_text,
                                        bool directory) {
  QLineEdit* line_edit = new QLineEdit(this);
  line_edit->setText(QString::fromStdString(option->string()));

  AddOptionRow(label_text, line_edit, option);

  auto SelectPathFunc = [this, line_edit, directory]() {
    line_edit->setText(
        directory ? QFileDialog::getExistingDirectory(this, tr("Select folder"))
                  : QFileDialog::getOpenFileName(this, tr("Select file")));
  };

  QPushButton* select_button = new QPushButton(
      directory ? tr("Select folder") : tr("Select file"), this);
  select_button->setFont(font());
  connect(select_button, &QPushButton::released, this, SelectPathFunc);
  grid_layout_->addWidget(select_button, grid_layout_->rowCount(), 1);

  options_path_.emplace_back(line_edit, option);

  return line_edit;
}

QLineEdit* OptionsWidget::AddOptionFilePath(std::filesystem::path* option,
                                            const std::string& label_text) {
  return AddOptionPath(option, label_text, /*directory=*/false);
}

QLineEdit* OptionsWidget::AddOptionDirPath(std::filesystem::path* option,
                                           const std::string& label_text) {
  return AddOptionPath(option, label_text, /*directory=*/true);
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
  grid_layout_->addWidget(
      label, grid_layout_->rowCount(), 0, 1, 2, Qt::AlignHCenter);
}

void OptionsWidget::ReadOptions() {
  for (auto& option : options_int_) {
    option.first->setValue(*option.second);
  }

  for (auto& option : options_int_unlimited_) {
    option.first->setValue(
        UnlimitedOptionToSpinbox(*option.second, option.first->maximum()));
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
    option.first->setText(QString::fromStdString(option.second->string()));
  }
}

void OptionsWidget::WriteOptions() {
  for (auto& option : options_int_) {
    *option.second = option.first->value();
  }

  for (auto& option : options_int_unlimited_) {
    *option.second = UnlimitedSpinboxToOption(option.first->value());
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
  SetRowVisible(option_rows_, option, /*visible=*/true);
}

void OptionsWidget::HideOption(void* option) {
  SetRowVisible(option_rows_, option, /*visible=*/false);
}

void OptionsWidget::ShowWidget(QWidget* widget) {
  SetRowVisible(widget_rows_, widget, /*visible=*/true);
}

void OptionsWidget::HideWidget(QWidget* widget) {
  SetRowVisible(widget_rows_, widget, /*visible=*/false);
}

void OptionsWidget::ShowLayout(QLayout* layout) {
  SetRowVisible(layout_rows_, layout, /*visible=*/true);
}

void OptionsWidget::HideLayout(QLayout* layout) {
  SetRowVisible(layout_rows_, layout, /*visible=*/false);
}

}  // namespace colmap
