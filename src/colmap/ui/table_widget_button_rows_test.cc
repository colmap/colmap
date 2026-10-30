// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/table_widget_button_rows.h"

#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace {

class TableWidgetButtonRowsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    static int argc = 1;
    static char program[] = "table_widget_button_rows_test";
    static char* argv[] = {program, nullptr};
    application_ = std::make_unique<QApplication>(argc, argv);
  }

  static void TearDownTestSuite() { application_.reset(); }

  void DrainEvents() {
    QApplication::processEvents();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
  }

  int CountCellWidgets(QTableWidget& table) {
    int count = 0;
    for (int row = 0; row < table.rowCount(); ++row) {
      for (int col = 1; col < 4; ++col) {
        count += table.cellWidget(row, col) != nullptr;
      }
    }
    return count;
  }

  inline static std::unique_ptr<QApplication> application_;
};

TEST_F(TableWidgetButtonRowsTest, BoundsWidgetsAndPreservesScrolledActions) {
  QTableWidget table(7000, 5);
  table.resize(700, 400);
  table.verticalHeader()->setDefaultSectionSize(25);
  int selected_row = -1;
  int selected_column = -1;
  TableWidgetButtonRows rows(
      &table, 1, 4, [&](int row, int column) -> QWidget* {
        auto* button = new QPushButton(QString::number(row));
        QObject::connect(button, &QPushButton::released, [&, row, column]() {
          selected_row = row;
          selected_column = column;
        });
        return button;
      });
  rows.Refresh();
  table.show();
  DrainEvents();
  const int maximum_rows = table.viewport()->height() / 25 + 4;
  EXPECT_GT(CountCellWidgets(table), 0);
  EXPECT_LE(CountCellWidgets(table), 3 * maximum_rows);
  QPointer<QWidget> old_button = table.cellWidget(0, 1);
  // The focused row is intentionally retained; this case checks retirement
  // of an ordinary, unfocused row instead.
  table.setFocus();
  ASSERT_NE(QApplication::focusWidget(), old_button);

  table.verticalScrollBar()->setValue(table.verticalScrollBar()->maximum());
  DrainEvents();
  EXPECT_TRUE(old_button.isNull());
  EXPECT_EQ(table.cellWidget(0, 1), nullptr);
  EXPECT_LE(CountCellWidgets(table), 3 * maximum_rows);
  auto* button = qobject_cast<QPushButton*>(table.cellWidget(6999, 3));
  ASSERT_NE(button, nullptr);
  button->click();
  EXPECT_EQ(selected_row, 6999);
  EXPECT_EQ(selected_column, 3);
}

TEST_F(TableWidgetButtonRowsTest,
       RefreshReadsReplacementDataAndHandlesEmptyRows) {
  QTableWidget table(3, 5);
  table.resize(700, 400);
  for (int row = 0; row < 3; ++row) {
    table.setItem(row, 0, new QTableWidgetItem("old"));
  }
  TableWidgetButtonRows rows(&table, 1, 4, [&](int row, int) -> QWidget* {
    const auto* item = table.item(row, 0);
    return item ? new QPushButton(item->text()) : nullptr;
  });
  rows.Refresh();
  table.show();
  DrainEvents();
  for (int row = 0; row < 3; ++row) table.item(row, 0)->setText("new");
  rows.Refresh();
  DrainEvents();
  EXPECT_EQ(qobject_cast<QPushButton*>(table.cellWidget(0, 1))->text(), "new");
  table.clearContents();
  table.setRowCount(0);
  rows.Refresh();
  DrainEvents();
  EXPECT_EQ(CountCellWidgets(table), 0);
  table.setRowCount(2);
  table.setItem(0, 0, new QTableWidgetItem("reopened"));
  rows.Refresh();
  DrainEvents();
  ASSERT_NE(table.cellWidget(0, 1), nullptr);
  EXPECT_EQ(table.cellWidget(1, 1), nullptr);
}

TEST_F(TableWidgetButtonRowsTest, ResizeAndFastScrollStayBounded) {
  QTableWidget table(7000, 5);
  table.resize(700, 200);
  table.verticalHeader()->setDefaultSectionSize(25);
  TableWidgetButtonRows rows(
      &table, 1, 4, [](int, int) { return new QPushButton("Image"); });
  table.show();
  rows.Refresh();
  DrainEvents();
  for (int value = 0; value <= table.verticalScrollBar()->maximum();
       value += 50) {
    table.verticalScrollBar()->setValue(value);
  }
  table.resize(700, 800);
  DrainEvents();
  EXPECT_LE(CountCellWidgets(table), 3 * (table.viewport()->height() / 25 + 4));
  EXPECT_LT(table.findChildren<QPushButton*>().size(), 120);
}

TEST_F(TableWidgetButtonRowsTest, PendingUpdateIsCancelledWithTable) {
  auto table = std::make_unique<QTableWidget>(7000, 5);
  new TableWidgetButtonRows(
      table.get(), 1, 4, [](int, int) { return new QPushButton("Image"); });
  table->show();
  table->resize(800, 500);
  table.reset();
  DrainEvents();
}

TEST_F(TableWidgetButtonRowsTest, RowShrinkExposesControlsWithoutScrolling) {
  QTableWidget table(7000, 5);
  table.resize(700, 400);
  table.verticalHeader()->setDefaultSectionSize(25);
  TableWidgetButtonRows rows(
      &table, 1, 4, [](int, int) { return new QPushButton("Image"); });
  table.show();
  rows.Refresh();
  DrainEvents();
  table.verticalScrollBar()->setValue(100);
  DrainEvents();
  const int first = table.rowAt(0);
  ASSERT_GE(first, 0);
  table.setRowHeight(first, 300);
  DrainEvents();
  table.setRowHeight(first, 25);
  DrainEvents();
  const int last = table.rowAt(table.viewport()->height() - 1);
  ASSERT_GT(last, first + 2);
  EXPECT_NE(table.cellWidget(last, 2), nullptr);
}

TEST_F(TableWidgetButtonRowsTest, HiddenRowsDoNotReceiveControls) {
  QTableWidget table(7000, 5);
  table.resize(700, 400);
  for (int row = 1; row < 7000; ++row) table.setRowHidden(row, true);
  int created = 0;
  TableWidgetButtonRows rows(&table, 1, 4, [&](int, int) {
    ++created;
    return new QPushButton("Image");
  });
  table.show();
  rows.Refresh();
  DrainEvents();
  EXPECT_EQ(CountCellWidgets(table), 3);
  EXPECT_LE(created, 6);
}

TEST_F(TableWidgetButtonRowsTest, RetainsFocusUntilItMovesToAnotherRow) {
  QTableWidget table(7000, 5);
  table.resize(1200, 800);
  table.verticalHeader()->setDefaultSectionSize(25);
  TableWidgetButtonRows rows(&table, 1, 4, [](int row, int) {
    return new QPushButton(QString::number(row));
  });
  table.show();
  table.activateWindow();
  rows.Refresh();
  DrainEvents();
  QPointer<QWidget> focused = table.cellWidget(0, 1);
  ASSERT_NE(focused, nullptr);
  focused->setFocus();
  DrainEvents();
  ASSERT_EQ(QApplication::focusWidget(), focused);
  const int end_scroll = table.verticalScrollBar()->maximum();
  table.verticalScrollBar()->setValue(end_scroll);
  DrainEvents();
  EXPECT_EQ(QApplication::focusWidget(), focused);
  EXPECT_EQ(table.verticalScrollBar()->value(), end_scroll);
  EXPECT_EQ(table.cellWidget(0, 1), focused);
  EXPECT_LE(CountCellWidgets(table), 3 * (table.viewport()->height() / 25 + 5));
  ASSERT_NE(table.cellWidget(6999, 3), nullptr);
  table.cellWidget(6999, 3)->setFocus();
  DrainEvents();
  EXPECT_TRUE(focused.isNull());
  EXPECT_EQ(QApplication::focusWidget(), table.cellWidget(6999, 3));
}

TEST_F(TableWidgetButtonRowsTest, CoalescesScrollBurstsAndPreservesSelection) {
  QTableWidget table(7000, 5);
  table.resize(1200, 800);
  table.verticalHeader()->setDefaultSectionSize(25);
  table.setSelectionBehavior(QAbstractItemView::SelectRows);
  table.setSelectionMode(QAbstractItemView::SingleSelection);
  for (int row = 0; row < 7000; ++row)
    table.setItem(row, 0, new QTableWidgetItem(QString::number(row)));
  int created = 0;
  TableWidgetButtonRows rows(&table, 1, 4, [&](int row, int) {
    ++created;
    return new QPushButton(QString::number(row));
  });
  table.show();
  rows.Refresh();
  DrainEvents();
  const int before = created;
  table.setCurrentCell(300, 0);
  for (int i = 0; i < 40; ++i)
    table.verticalScrollBar()->setValue((i * 173) %
                                        table.verticalScrollBar()->maximum());
  DrainEvents();
  EXPECT_EQ(table.currentRow(), 300);
  EXPECT_EQ(table.selectionModel()->selectedRows().size(), 1);
  const int first = table.rowAt(0);
  ASSERT_GE(first, 0);
  ASSERT_NE(table.cellWidget(first, 1), nullptr);
  EXPECT_EQ(qobject_cast<QPushButton*>(table.cellWidget(first, 1))->text(),
            QString::number(first));
  EXPECT_LE(created - before, 3 * (table.viewport()->height() / 25 + 4));
}

TEST_F(TableWidgetButtonRowsTest, InstallsExposedControlsBeforePainting) {
  QTableWidget table(7000, 5);
  table.resize(1200, 800);
  table.verticalHeader()->setDefaultSectionSize(25);
  TableWidgetButtonRows rows(&table, 1, 4, [](int row, int) {
    return new QPushButton(QString::number(row));
  });
  table.show();
  rows.Refresh();
  DrainEvents();
  table.verticalScrollBar()->setValue(table.verticalScrollBar()->maximum());
  // Render immediately, without draining the timer queued by scrolling.
  QImage image(table.viewport()->size(), QImage::Format_ARGB32);
  table.viewport()->render(&image);
  ASSERT_NE(table.cellWidget(6999, 1), nullptr);
  EXPECT_EQ(qobject_cast<QPushButton*>(table.cellWidget(6999, 1))->text(),
            "6999");
}

}  // namespace
}  // namespace colmap
