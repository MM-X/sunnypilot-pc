/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/offroad_home.h"

#include <QStackedWidget>

#include "selfdrive/ui/sunnypilot/qt/widgets/drive_stats.h"

OffroadHomeSP::OffroadHomeSP(QWidget *parent) : OffroadHome(parent) {
  QStackedWidget *left_widget = new QStackedWidget(this);
  DriveStats *driveStatsWidget = new DriveStats(this);
  driveStatsWidget->setStyleSheet(R"(
    QLabel[type="title"] { font-size: 19px; font-weight: 185; }
    QLabel[type="number"] { font-size: 29px; font-weight: 185; }
    QLabel[type="unit"] { font-size: 19px; font-weight: 111; color: #A0A0A0; }
  )");
  left_widget->addWidget(driveStatsWidget);
  left_widget->setStyleSheet("border-radius: 4px;");

  home_layout->insertWidget(0, left_widget);
}
