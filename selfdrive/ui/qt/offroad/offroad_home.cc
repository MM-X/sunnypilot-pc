/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/qt/offroad/offroad_home.h"

#include "selfdrive/ui/qt/offroad/experimental_mode.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/prime.h"

// OffroadHome: the offroad home page

OffroadHome::OffroadHome(QWidget* parent) : QFrame(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setContentsMargins(15, 15, 15, 15);
#else
  main_layout->setContentsMargins(40, 40, 40, 40);
#endif

  // top header
  header_layout = new QHBoxLayout();
  header_layout->setContentsMargins(0, 0, 0, 0);
#ifdef RK_BUILD
  header_layout->setSpacing(6);
#else
  header_layout->setSpacing(16);
#endif

  update_notif = new QPushButton(tr("UPDATE"));
  update_notif->setVisible(false);
  update_notif->setStyleSheet("background-color: #364DEF;");
  QObject::connect(update_notif, &QPushButton::clicked, [=]() { center_layout->setCurrentIndex(1); });
  header_layout->addWidget(update_notif, 0, Qt::AlignHCenter | Qt::AlignLeft);

  alert_notif = new QPushButton();
  alert_notif->setVisible(false);
  alert_notif->setStyleSheet("background-color: #E22C2C;");
  QObject::connect(alert_notif, &QPushButton::clicked, [=] { center_layout->setCurrentIndex(2); });
  header_layout->addWidget(alert_notif, 0, Qt::AlignHCenter | Qt::AlignLeft);

  version = new ElidedLabel();
  header_layout->addWidget(version, 0, Qt::AlignHCenter | Qt::AlignRight);

  main_layout->addLayout(header_layout);

  // main content
#ifdef RK_BUILD
  main_layout->addSpacing(9);
#else
  main_layout->addSpacing(25);
#endif
  center_layout = new QStackedLayout();

  QWidget *home_widget = new QWidget(this);
  {
    home_layout = new QHBoxLayout(home_widget);
    home_layout->setContentsMargins(0, 0, 0, 0);
#ifdef RK_BUILD
    home_layout->setSpacing(11);
#else
    home_layout->setSpacing(30);
#endif

#ifndef SUNNYPILOT
    // left: PrimeAdWidget
    QStackedWidget *left_widget = new QStackedWidget(this);
    QVBoxLayout *left_prime_layout = new QVBoxLayout();
    left_prime_layout->setContentsMargins(0, 0, 0, 0);
    QWidget *prime_user = new PrimeUserWidget();
    prime_user->setStyleSheet(
#ifdef RK_BUILD
R"UI0(    border-radius: 4px;)UI0"
#else
R"UI1(    border-radius: 10px;)UI1"
#endif
R"UI2(    background-color: #333333;
    )UI2");
    left_prime_layout->addWidget(prime_user);
    left_prime_layout->addStretch();
    left_widget->addWidget(new LayoutWidget(left_prime_layout));
    left_widget->addWidget(new PrimeAdWidget);
#ifdef RK_BUILD
    left_widget->setStyleSheet("border-radius: 4px;");
#else
    left_widget->setStyleSheet("border-radius: 10px;");
#endif

    connect(uiState()->prime_state, &PrimeState::changed, [left_widget]() {
      left_widget->setCurrentIndex(uiState()->prime_state->isSubscribed() ? 0 : 1);
    });

    home_layout->addWidget(left_widget, 1);
#endif

    // right: ExperimentalModeButton, SetupWidget
    QWidget* right_widget = new QWidget(this);
    QVBoxLayout* right_column = new QVBoxLayout(right_widget);
    right_column->setContentsMargins(0, 0, 0, 0);
#ifdef RK_BUILD
    right_widget->setFixedWidth(277);
    right_column->setSpacing(11);
#else
    right_widget->setFixedWidth(750);
    right_column->setSpacing(30);
#endif

    ExperimentalModeButton *experimental_mode = new ExperimentalModeButton(this);
    QObject::connect(experimental_mode, &ExperimentalModeButton::openSettings, this, &OffroadHome::openSettings);
    right_column->addWidget(experimental_mode, 1);

    SetupWidget *setup_widget = new SetupWidget;
    QObject::connect(setup_widget, &SetupWidget::openSettings, this, &OffroadHome::openSettings);
    right_column->addWidget(setup_widget, 1);

    home_layout->addWidget(right_widget, 1);
  }
  center_layout->addWidget(home_widget);

  // add update & alerts widgets
  update_widget = new UpdateAlert();
  QObject::connect(update_widget, &UpdateAlert::dismiss, [=]() { center_layout->setCurrentIndex(0); });
  center_layout->addWidget(update_widget);
  alerts_widget = new OffroadAlert();
  QObject::connect(alerts_widget, &OffroadAlert::dismiss, [=]() { center_layout->setCurrentIndex(0); });
  center_layout->addWidget(alerts_widget);

  main_layout->addLayout(center_layout, 1);

  // set up refresh timer
  timer = new QTimer(this);
  timer->callOnTimeout(this, &OffroadHome::refresh);

  setStyleSheet(R"UI3(
    * {
      color: white;
    }
    OffroadHome {
      background-color: black;
    }
    OffroadHome > QPushButton {)UI3"
#ifdef RK_BUILD
R"UI4(      padding: 6px 12px;
      border-radius: 2px;
      font-size: 14px;
      font-weight: 184;)UI4"
#else
R"UI5(      padding: 15px 30px;
      border-radius: 5px;
      font-size: 40px;
      font-weight: 500;)UI5"
#endif
R"UI6(    }
    OffroadHome > QLabel {)UI6"
#ifdef RK_BUILD
R"UI7(      font-size: 20px;)UI7"
#else
R"UI8(      font-size: 55px;)UI8"
#endif
R"UI9(    }
  )UI9");
}

void OffroadHome::showEvent(QShowEvent *event) {
  refresh();
  timer->start(10 * 1000);
}

void OffroadHome::hideEvent(QHideEvent *event) {
  timer->stop();
}

void OffroadHome::refresh() {
  version->setText(getBrand() + " " +  QString::fromStdString(params.get("UpdaterCurrentDescription")));

  bool updateAvailable = update_widget->refresh();
  int alerts = alerts_widget->refresh();

  // pop-up new notification
  int idx = center_layout->currentIndex();
  if (!updateAvailable && !alerts) {
    idx = 0;
  } else if (updateAvailable && (!update_notif->isVisible() || (!alerts && idx == 2))) {
    idx = 1;
  } else if (alerts && (!alert_notif->isVisible() || (!updateAvailable && idx == 1))) {
    idx = 2;
  }
  center_layout->setCurrentIndex(idx);

  update_notif->setVisible(updateAvailable);
  alert_notif->setVisible(alerts);
  if (alerts) {
    alert_notif->setText(QString::number(alerts) + (alerts > 1 ? tr(" ALERTS") : tr(" ALERT")));
  }
}
