/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

#include <QPainter>
#include <QStyleOption>

QFrame *horizontal_line(QWidget *parent) {
  QFrame *line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet(
#ifdef RK_BUILD
R"UI0(    border-width: 1px;)UI0"
#else
R"UI1(    border-width: 2px;)UI1"
#endif
R"UI2(    border-bottom-style: solid;
    border-color: gray;
  )UI2");
#ifdef RK_BUILD
  line->setFixedHeight(4);
#else
  line->setFixedHeight(10);
#endif
  return line;
}

QFrame *vertical_space(int height, QWidget *parent) {
  QFrame *v_space = new QFrame(parent);
  v_space->setFrameShape(QFrame::StyledPanel);
  v_space->setFixedHeight(height);
  return v_space;
}

// AbstractControlSP
std::vector<AbstractControlSP*> AbstractControlSP::advanced_controls_;
AbstractControlSP::~AbstractControlSP() { UnregisterAdvancedControl(this); }

void AbstractControlSP::RegisterAdvancedControl(AbstractControlSP *ctrl) { advanced_controls_.push_back(ctrl); }

void AbstractControlSP::UnregisterAdvancedControl(AbstractControlSP *ctrl) {
  advanced_controls_.erase(std::remove(advanced_controls_.begin(), advanced_controls_.end(), ctrl), advanced_controls_.end());
}

void AbstractControlSP::UpdateAllAdvancedControls() {
  bool visibility = Params().getBool("ShowAdvancedControls");
  advanced_controls_.erase(std::remove(advanced_controls_.begin(), advanced_controls_.end(), nullptr), advanced_controls_.end());
  for (auto *ctrl : advanced_controls_) ctrl->setVisible(visibility);
}

AbstractControlSP::AbstractControlSP(const QString &title, const QString &desc, const QString &icon, QWidget *parent, bool advancedControl)
    : AbstractControl(title, desc, icon, parent), isAdvancedControl(advancedControl) {
  if (isAdvancedControl) RegisterAdvancedControl(this);

  main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);

  hlayout = new QHBoxLayout;
  hlayout->setMargin(0);
#ifdef RK_BUILD
  hlayout->setSpacing(7);
#else
  hlayout->setSpacing(20);
#endif

  // title
  title_label = new QPushButton(title);
#ifdef RK_BUILD
  title_label->setFixedHeight(44);
  title_label->setStyleSheet("font-size: 18px; font-weight: 166; text-align: left; border: none;");
#else
  title_label->setFixedHeight(120);
  title_label->setStyleSheet("font-size: 50px; font-weight: 450; text-align: left; border: none;");
#endif
  hlayout->addWidget(title_label, 1);

  // value next to control button
  value = new ElidedLabelSP();
  value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  value->setStyleSheet("color: #aaaaaa");
  hlayout->addWidget(value);

  main_layout->addLayout(hlayout);

  // description
  description = new QLabel(desc);
#ifdef RK_BUILD
  description->setContentsMargins(15, 7, 15, 7);
  description->setStyleSheet("font-size: 15px; color: grey");
#else
  description->setContentsMargins(40, 20, 40, 20);
  description->setStyleSheet("font-size: 40px; color: grey");
#endif
  description->setWordWrap(true);
  description->setVisible(false);
  main_layout->addWidget(description);

  connect(title_label, &QPushButton::clicked, [=]() {
    if (!description->isVisible()) {
      emit showDescriptionEvent();
    }

    if (!description->text().isEmpty()) {
      description->setVisible(!description->isVisible());
    }
  });

  main_layout->addStretch();
}

void AbstractControlSP::hideEvent(QHideEvent *e) {
  if (description != nullptr) {
    description->hide();
  }
}

AbstractControlSP_SELECTOR::AbstractControlSP_SELECTOR(const QString &title, const QString &desc, const QString &icon, QWidget *parent, bool advancedControl)
    : AbstractControlSP(title, desc, icon, parent, advancedControl) {

  if (title_label != nullptr) {
    delete title_label;
    title_label = nullptr;
  }

  if (description != nullptr) {
    delete description;
    description = nullptr;
  }

  if (value != nullptr) {
    ReplaceWidget(value, new QWidget());
    value = nullptr;
  }

  QLayoutItem *item;
  while ((item = main_layout->takeAt(0)) != nullptr) {
    if (item->widget()) {
      delete item->widget();
    }
    delete item;
  }

  main_layout->setMargin(0);

  hlayout = new QHBoxLayout;
  hlayout->setMargin(0);
  hlayout->setSpacing(0);

  // title
  if (!title.isEmpty()) {
    title_label = new QPushButton(title);
    title_label->setFixedHeight(
#ifdef RK_BUILD
      44
#else
      120
#endif
    );
    title_label->setStyleSheet(
#ifdef RK_BUILD
      "font-size: 18px; font-weight: 166; text-align: left; border: none; padding: 0 0 0 0"
#else
      "font-size: 50px; font-weight: 450; text-align: left; border: none; padding: 0 0 0 0"
#endif
    );
    main_layout->addWidget(title_label, 1);

    connect(title_label, &QPushButton::clicked, [=]() {
      if (!description->isVisible()) {
        emit showDescriptionEvent();
      }

      if (!description->text().isEmpty()) {
        bool isVisible = !description->isVisible();
        description->setVisible(isVisible);

        if (isVisible && spacingItem) {
          main_layout->removeItem(spacingItem);
        } else if (!isVisible && spacingItem != nullptr && main_layout->indexOf(spacingItem) == -1) {
          main_layout->insertItem(main_layout->indexOf(description), spacingItem);
        }
      }
    });
  } else {
#ifdef RK_BUILD
    main_layout->addSpacing(7);
#else
    main_layout->addSpacing(20);
#endif
  }

  main_layout->addLayout(hlayout);
  if (!desc.isEmpty() && spacingItem != nullptr && main_layout->indexOf(spacingItem) == -1) {
    main_layout->insertItem(main_layout->count(), spacingItem);
  }

  // description
  description = new QLabel(desc);
#ifdef RK_BUILD
  description->setContentsMargins(15, 7, 15, 7);
  description->setStyleSheet("font-size: 15px; color: grey");
#else
  description->setContentsMargins(40, 20, 40, 20);
  description->setStyleSheet("font-size: 40px; color: grey");
#endif
  description->setWordWrap(true);
  description->setVisible(false);
  main_layout->addWidget(description);

  main_layout->addStretch();
}

void AbstractControlSP_SELECTOR::hideEvent(QHideEvent *e) {
  if (description != nullptr) {
    description->hide();
  }

  if (spacingItem != nullptr && main_layout->indexOf(spacingItem) == -1) {
    main_layout->insertItem(main_layout->indexOf(description), spacingItem);
  }
}

// controls

ButtonControlSP::ButtonControlSP(const QString &title, const QString &text, const QString &desc, QWidget *parent, bool advancedControl)
    : AbstractControlSP(title, desc, "", parent, advancedControl) {

  btn.setText(text);
  btn.setStyleSheet(R"UI3(
    QPushButton {
      padding: 0;)UI3"
#ifdef RK_BUILD
R"UI4(      border-radius: 18px;
      font-size: 13px;
      font-weight: 185;)UI4"
#else
R"UI5(      border-radius: 50px;
      font-size: 35px;
      font-weight: 500;)UI5"
#endif
R"UI6(      color: #E4E4E4;
      background-color: #393939;
    }
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      color: #33E4E4E4;
    }
  )UI6");
#ifdef RK_BUILD
  btn.setFixedSize(92, 37);
#else
  btn.setFixedSize(250, 100);
#endif
  QObject::connect(&btn, &QPushButton::clicked, this, &ButtonControlSP::clicked);
  hlayout->addWidget(&btn);
}

// ElidedLabelSP

ElidedLabelSP::ElidedLabelSP(QWidget *parent) : ElidedLabelSP({}, parent) {
}

ElidedLabelSP::ElidedLabelSP(const QString &text, QWidget *parent) : QLabel(text.trimmed(), parent) {
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  setMinimumWidth(1);
}

void ElidedLabelSP::resizeEvent(QResizeEvent *event) {
  QLabel::resizeEvent(event);
  lastText_ = elidedText_ = "";
}

void ElidedLabelSP::paintEvent(QPaintEvent *event) {
  const QString curText = text();
  if (curText != lastText_) {
    elidedText_ = fontMetrics().elidedText(curText, Qt::ElideRight, contentsRect().width());
    lastText_ = curText;
  }

  QPainter painter(this);
  drawFrame(&painter);
  QStyleOption opt;
  opt.initFrom(this);
  style()->drawItemText(&painter, contentsRect(), alignment(), opt.palette, isEnabled(), elidedText_, foregroundRole());
}

// ParamControlSP

ParamControlSP::ParamControlSP(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent, bool advancedControl)
    : ToggleControlSP(title, desc, icon, false, parent, advancedControl){

  key = param.toStdString();
  QObject::connect(this, &ParamControlSP::toggleFlipped, this, &ParamControlSP::toggleClicked);

  hlayout->removeWidget(&toggle);
  hlayout->insertWidget(0, &toggle);

  hlayout->removeWidget(this->icon_label);
  hlayout->insertWidget(1, this->icon_label);
}

void ParamControlSP::toggleClicked(bool state) {
  auto do_confirm = [this]() {
    QString content("<body><h2 style=\"text-align: center;\">" + title_label->text() + "</h2><br>"
#ifdef RK_BUILD
                    "<p style=\"text-align: center; margin: 0 47px; font-size: 18px;\">" + getDescription() + "</p></body>");
#else
                    "<p style=\"text-align: center; margin: 0 128px; font-size: 50px;\">" + getDescription() + "</p></body>");
#endif
    return ConfirmationDialog(content, tr("Enable"), tr("Cancel"), true, this).exec();
  };

  bool confirmed = store_confirm && params.getBool(key + "Confirmed");
  if (!confirm || confirmed || !state || do_confirm()) {
    if (store_confirm && state) params.putBool(key + "Confirmed", true);
    params.putBool(key, state);
    setIcon(state);
  } else {
    toggle.togglePosition();
  }
}
