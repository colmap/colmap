// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <QtWidgets>
#include <algorithm>
#include <functional>

namespace colmap {

// Keep real cell widgets for visible rows and retain the focused row until
// focus moves elsewhere. The table retains its ordinary items and selection
// model; the factory recreates controls from current item data.
class TableWidgetButtonRows : public QObject {
 public:
  using Factory = std::function<QWidget*(int row, int column)>;

  TableWidgetButtonRows(QTableWidget* table,
                        int first_column,
                        int end_column,
                        Factory factory)
      : QObject(table),
        table_(table),
        first_column_(first_column),
        end_column_(end_column),
        factory_(std::move(factory)) {
    table_->viewport()->installEventFilter(this);
    connect(
        table_->verticalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
          ScheduleUpdate();
        });
    connect(
        table_->verticalHeader(), &QHeaderView::sectionResized, this, [this]() {
          ScheduleUpdate();
        });
    connect(qApp, &QApplication::focusChanged, this, [this]() {
      if (focused_row_ >= 0) ScheduleUpdate();
    });
  }

  // Call after replacing item data, including replacement at the same size.
  void Refresh() {
    for (int row = begin_; row < end_ && row < table_->rowCount(); ++row) {
      RemoveRow(row);
    }
    RemoveRow(focused_row_);
    focused_row_ = -1;
    begin_ = end_ = 0;
    Update();
  }

 private:
  bool eventFilter(QObject* object, QEvent* event) override {
    // Scroll bursts need only one update, but newly exposed controls must be
    // installed before the viewport paints rather than appearing a frame late.
    if (object == table_->viewport() && event->type() == QEvent::Paint &&
        update_pending_) {
      Update();
    }
    if (object == table_->viewport() &&
        (event->type() == QEvent::Resize || event->type() == QEvent::Show)) {
      ScheduleUpdate();
    }
    return QObject::eventFilter(object, event);
  }

  bool RowHasFocus(int row) const {
    QWidget* focus = QApplication::focusWidget();
    if (focus == nullptr) return false;
    for (int col = first_column_; col < end_column_; ++col) {
      QWidget* widget = table_->cellWidget(row, col);
      if (widget != nullptr &&
          (widget == focus || widget->isAncestorOf(focus))) {
        return true;
      }
    }
    return false;
  }

  void RemoveRow(int row) {
    if (row < 0 || row >= table_->rowCount()) return;
    for (int col = first_column_; col < end_column_; ++col) {
      table_->removeCellWidget(row, col);
    }
  }

  void ScheduleUpdate() {
    if (update_pending_) return;
    update_pending_ = true;
    QTimer::singleShot(0, this, [this]() {
      update_pending_ = false;
      Update();
    });
  }

  void Update() {
    if (updating_) return;
    QScopedValueRollback<bool> updating(updating_, true);
    int first = 0;
    int last = 0;
    if (table_->rowCount() > 0) {
      const int top = table_->rowAt(0);
      first = std::max(0, top - 1);
      // Before the first layout, create one row rather than treating an
      // unknown viewport as permission to instantiate the entire table.
      if (top < 0) {
        last = 1;
      } else {
        const int bottom =
            table_->rowAt(std::max(0, table_->viewport()->height() - 1));
        last = bottom < 0 ? table_->rowCount()
                          : std::min(table_->rowCount(), bottom + 2);
      }
    }
    if (focused_row_ >= 0) {
      if (focused_row_ >= first && focused_row_ < last) {
        focused_row_ = -1;
      } else if (focused_row_ >= table_->rowCount() ||
                 !RowHasFocus(focused_row_)) {
        RemoveRow(focused_row_);
        focused_row_ = -1;
      }
    }
    for (int row = begin_; row < end_ && row < table_->rowCount(); ++row) {
      if (row >= first && row < last) continue;
      if (RowHasFocus(row)) {
        focused_row_ = row;
      } else {
        RemoveRow(row);
      }
    }
    for (int row = first; row < last; ++row) {
      if (table_->isRowHidden(row)) continue;
      for (int col = first_column_; col < end_column_; ++col) {
        if (table_->cellWidget(row, col) == nullptr) {
          if (QWidget* widget = factory_(row, col)) {
            table_->setCellWidget(row, col, widget);
          }
        }
      }
    }
    begin_ = first;
    end_ = last;
  }

  QTableWidget* table_;
  int first_column_;
  int end_column_;
  Factory factory_;
  int begin_ = 0;
  int end_ = 0;
  int focused_row_ = -1;
  bool update_pending_ = false;
  bool updating_ = false;
};

}  // namespace colmap
