/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/trips_panel.h"

TripsPanel::TripsPanel(QWidget* parent) : QFrame(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);

  // main content
  main_layout->addSpacing(9);
  center_layout = new QStackedLayout();

  driveStatsWidget = new DriveStats;
  driveStatsWidget->setStyleSheet(R"(
    QLabel[type="title"] { font-size: 19px; font-weight: 185; }
    QLabel[type="number"] { font-size: 29px; font-weight: 185; }
    QLabel[type="unit"] { font-size: 19px; font-weight: 111; color: #A0A0A0; }
  )");
  center_layout->addWidget(driveStatsWidget);

  main_layout->addLayout(center_layout, 1);

  setStyleSheet(R"(
    * {
      color: white;
    }
    TripsPanel > QLabel {
      font-size: 20px;
    }
  )");
}
