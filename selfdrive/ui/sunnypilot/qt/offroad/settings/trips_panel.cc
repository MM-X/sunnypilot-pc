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
#ifdef RK_BUILD
  main_layout->addSpacing(9);
#else
  main_layout->addSpacing(25);
#endif
  center_layout = new QStackedLayout();

  driveStatsWidget = new DriveStats;
  driveStatsWidget->setStyleSheet(
#ifdef RK_BUILD
R"UI0(    QLabel[type="title"] { font-size: 19px; font-weight: 185; }
    QLabel[type="number"] { font-size: 29px; font-weight: 185; }
    QLabel[type="unit"] { font-size: 19px; font-weight: 111; color: #A0A0A0; })UI0"
#else
R"UI1(    QLabel[type="title"] { font-size: 51px; font-weight: 500; }
    QLabel[type="number"] { font-size: 78px; font-weight: 500; }
    QLabel[type="unit"] { font-size: 51px; font-weight: 300; color: #A0A0A0; })UI1"
#endif
R"UI2(  )UI2");
  center_layout->addWidget(driveStatsWidget);

  main_layout->addLayout(center_layout, 1);

  setStyleSheet(R"UI3(
    * {
      color: white;
    }
    TripsPanel > QLabel {)UI3"
#ifdef RK_BUILD
R"UI4(      font-size: 20px;)UI4"
#else
R"UI5(      font-size: 55px;)UI5"
#endif
R"UI6(    }
  )UI6");
}
