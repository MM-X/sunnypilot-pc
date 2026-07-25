/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/widgets/drive_stats.h"

#include <QDebug>
#include <QGridLayout>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"

static QLabel* newLabel(const QString& text, const QString &type) {
  QLabel* label = new QLabel(text);
  label->setProperty("type", type);
  return label;
}

DriveStats::DriveStats(QWidget* parent) : QFrame(parent) {
  metric_ = Params().getBool("IsMetric");

  QVBoxLayout* main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setContentsMargins(18, 18, 18, 22);
#else
  main_layout->setContentsMargins(50, 50, 50, 60);
#endif

  auto add_stats_layouts = [=](const QString &title, StatsLabels& labels) {
    QGridLayout* grid_layout = new QGridLayout;
#ifdef RK_BUILD
    grid_layout->setVerticalSpacing(4);
    grid_layout->setContentsMargins(0, 14, 0, 4);
#else
    grid_layout->setVerticalSpacing(10);
    grid_layout->setContentsMargins(0, 10, 0, 10);
#endif

    int row = 0;
    grid_layout->addWidget(newLabel(title, "title"), row++, 0, 1, 3);
    grid_layout->addItem(new QSpacerItem(0,
#ifdef RK_BUILD
      11
#else
      30
#endif
    ), row++, 0, 1, 1);

    grid_layout->addWidget(labels.routes = newLabel("0", "number"), row, 0, Qt::AlignLeft);
    grid_layout->addWidget(labels.distance = newLabel("0", "number"), row, 1, Qt::AlignLeft);
    grid_layout->addWidget(labels.hours = newLabel("0", "number"), row, 2, Qt::AlignLeft);

    grid_layout->addWidget(newLabel((tr("Drives")), "unit"), row + 1, 0, Qt::AlignLeft);
    grid_layout->addWidget(labels.distance_unit = newLabel(getDistanceUnit(), "unit"), row + 1, 1, Qt::AlignLeft);
    grid_layout->addWidget(newLabel(tr("Hours"), "unit"), row + 1, 2, Qt::AlignLeft);

    main_layout->addLayout(grid_layout);
  };

  add_stats_layouts(tr("ALL TIME"), all_);
  main_layout->addStretch();
  add_stats_layouts(tr("PAST WEEK"), week_);

  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1.1/devices/" + *dongleId + "/stats";
    RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_DriveStats", 30);
    QObject::connect(repeater, &RequestRepeater::requestDone, this, &DriveStats::parseResponse);
  }

  setStyleSheet(R"UI0(
    DriveStats {
      background-color: #333333;)UI0"
#ifdef RK_BUILD
R"UI1(      border-radius: 4px;)UI1"
#else
R"UI2(      border-radius: 10px;)UI2"
#endif
R"UI3(    }
)UI3"
#ifdef RK_BUILD
R"UI4(    QLabel[type="title"] { font-size: 51px; font-weight: 185; }
    QLabel[type="number"] { font-size: 78px; font-weight: 185; }
    QLabel[type="unit"] { font-size: 51px; font-weight: 111; color: #A0A0A0; })UI4"
#else
R"UI5(    QLabel[type="title"] { font-size: 51px; font-weight: 500; }
    QLabel[type="number"] { font-size: 78px; font-weight: 500; }
    QLabel[type="unit"] { font-size: 51px; font-weight: 300; color: #A0A0A0; })UI5"
#endif
R"UI6(  )UI6");
}

void DriveStats::updateStats() {
  auto update = [=](const QJsonObject& obj, StatsLabels& labels) {
    labels.routes->setText(QString::number((int)obj["routes"].toDouble()));
    labels.distance->setText(QString::number(int(obj["distance"].toDouble() * (metric_ ? MILE_TO_KM : 1))));
    labels.distance_unit->setText(getDistanceUnit());
    labels.hours->setText(QString::number((int)(obj["minutes"].toDouble() / 60)));
  };

  QJsonObject json = stats_.object();
  update(json["all"].toObject(), all_);
  update(json["week"].toObject(), week_);
}

void DriveStats::parseResponse(const QString& response, bool success) {
  if (!success) return;

  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    return;
  }
  stats_ = doc;
  updateStats();
}

void DriveStats::showEvent(QShowEvent* event) {
  bool metric = Params().getBool("IsMetric");
  if (metric_ != metric) {
    metric_ = metric;
    updateStats();
  }
}
