#include "selfdrive/ui/qt/offroad/onboarding.h"

#include <string>

#include <QLabel>
#include <QPainter>
#include <QTransform>
#include <QVBoxLayout>

#include "common/util.h"
#include "common/params.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/input.h"

TrainingGuide::TrainingGuide(QWidget *parent) : QFrame(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
}

void TrainingGuide::mouseReleaseEvent(QMouseEvent *e) {
  if (click_timer.elapsed() < 250) {
    return;
  }
  click_timer.restart();

  auto contains = [this](QRect r, const QPoint &pt) {
    if (image.size() != image_raw_size) {
      QTransform transform;
      transform.translate((width()- image.width()) / 2.0, (height()- image.height()) / 2.0);
      transform.scale(image.width() / (float)image_raw_size.width(), image.height() / (float)image_raw_size.height());
      r= transform.mapRect(r);
    }
    return r.contains(pt);
  };

  if (contains(boundingRect[currentIndex], e->pos())) {
    if (currentIndex == 9) {
#ifdef RK_BUILD
      const QRect yes = QRect(262, 298, 197, 60);
#else
      const QRect yes = QRect(707, 804, 531, 164);
#endif
      Params().putBool("RecordFront", contains(yes, e->pos()));
    }
    currentIndex += 1;
  } else if (currentIndex == (boundingRect.size() - 2) && contains(boundingRect.last(), e->pos())) {
    currentIndex = 0;
  }

  if (currentIndex >= (boundingRect.size() - 1)) {
    emit completedTraining();
  } else {
    update();
  }
}

void TrainingGuide::showEvent(QShowEvent *event) {
  currentIndex = 0;
  click_timer.start();
}

QImage TrainingGuide::loadImage(int id) {
  QImage img(img_path + QString("step%1.png").arg(id));
  image_raw_size = img.size();
  if (image_raw_size != rect().size()) {
    img = img.scaled(width(), height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
  }
  return img;
}

void TrainingGuide::paintEvent(QPaintEvent *event) {
  QPainter painter(this);

  QRect bg(0, 0, painter.device()->width(), painter.device()->height());
  painter.fillRect(bg, QColor("#000000"));

  image = loadImage(currentIndex);
  QRect rect(image.rect());
  rect.moveCenter(bg.center());
  painter.drawImage(rect.topLeft(), image);

  // progress bar
  if (currentIndex > 0 && currentIndex < (boundingRect.size() - 2)) {
#ifdef RK_BUILD
    const int h = 7;
#else
    const int h = 20;
#endif
    const int w = (currentIndex / (float)(boundingRect.size() - 2)) * width();
    painter.fillRect(QRect(0, height() - h, w, h), QColor("#465BEA"));
  }
}

void TermsPage::showEvent(QShowEvent *event) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setContentsMargins(17, 13, 17, 17);
#else
  main_layout->setContentsMargins(45, 35, 45, 45);
#endif
  main_layout->setSpacing(0);

  QVBoxLayout *vlayout = new QVBoxLayout();
#ifdef RK_BUILD
  vlayout->setContentsMargins(61, 61, 61, 0);
#else
  vlayout->setContentsMargins(165, 165, 165, 0);
#endif
  main_layout->addLayout(vlayout);

  QLabel *title = new QLabel(tr("Welcome to sunnypilot"));
#ifdef RK_BUILD
  title->setStyleSheet("font-size: 34px; font-weight: 186;");
#else
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
#endif
  vlayout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

#ifdef RK_BUILD
  vlayout->addSpacing(33);
#else
  vlayout->addSpacing(90);
#endif
  QLabel *desc = new QLabel(tr("You must accept the Terms and Conditions to use sunnypilot. Read the latest terms at <span style='color: #465BEA;'>https://comma.ai/terms</span> before continuing."));
  desc->setWordWrap(true);
#ifdef RK_BUILD
  desc->setStyleSheet("font-size: 30px; font-weight: 112;");
#else
  desc->setStyleSheet("font-size: 80px; font-weight: 300;");
#endif
  vlayout->addWidget(desc, 0);

  vlayout->addStretch();

  QHBoxLayout* buttons = new QHBoxLayout;
  buttons->setMargin(0);
#ifdef RK_BUILD
  buttons->setSpacing(17);
#else
  buttons->setSpacing(45);
#endif
  main_layout->addLayout(buttons);

  QPushButton *decline_btn = new QPushButton(tr("Decline"));
  buttons->addWidget(decline_btn);
  QObject::connect(decline_btn, &QPushButton::clicked, this, &TermsPage::declinedTerms);

  accept_btn = new QPushButton(tr("Agree"));
  accept_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #465BEA;
    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )");
  buttons->addWidget(accept_btn);
  QObject::connect(accept_btn, &QPushButton::clicked, this, &TermsPage::acceptedTerms);
}

void DeclinePage::showEvent(QShowEvent *event) {
  if (layout()) {
    return;
  }

  QVBoxLayout *main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setMargin(17);
  main_layout->setSpacing(15);
#else
  main_layout->setMargin(45);
  main_layout->setSpacing(40);
#endif

  QLabel *text = new QLabel(this);
  text->setText(tr("You must accept the Terms and Conditions in order to use sunnypilot."));
#ifdef RK_BUILD
  text->setStyleSheet(R"(font-size: 30px; font-weight: 112; margin: 74px;)");
#else
  text->setStyleSheet(R"(font-size: 80px; font-weight: 300; margin: 200px;)");
#endif
  text->setWordWrap(true);
  main_layout->addWidget(text, 0, Qt::AlignCenter);

  QHBoxLayout* buttons = new QHBoxLayout;
#ifdef RK_BUILD
  buttons->setSpacing(17);
#else
  buttons->setSpacing(45);
#endif
  main_layout->addLayout(buttons);

  QPushButton *back_btn = new QPushButton(tr("Back"));
  buttons->addWidget(back_btn);

  QObject::connect(back_btn, &QPushButton::clicked, this, &DeclinePage::getBack);

  QPushButton *uninstall_btn = new QPushButton(tr("Decline, uninstall %1").arg(getBrand()));
  uninstall_btn->setStyleSheet("background-color: #B73D3D");
  buttons->addWidget(uninstall_btn);
  QObject::connect(uninstall_btn, &QPushButton::clicked, [=]() {
    Params().putBool("DoUninstall", true);
  });
}

void OnboardingWindow::updateActiveScreen() {
  if (!accepted_terms) {
    setCurrentIndex(0);
  } else if (!training_done) {
    setCurrentIndex(1);
  } else {
    emit onboardingDone();
  }
}

OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  std::string current_terms_version = params.get("TermsVersion");
  std::string current_training_version = params.get("TrainingVersion");
  accepted_terms = params.get("HasAcceptedTerms") == current_terms_version;
  training_done = params.get("CompletedTrainingVersion") == current_training_version;

  TermsPage* terms = new TermsPage(this);
  addWidget(terms);
  connect(terms, &TermsPage::acceptedTerms, [=]() {
    params.put("HasAcceptedTerms", current_terms_version);
    accepted_terms = true;
    updateActiveScreen();
  });
  connect(terms, &TermsPage::declinedTerms, [=]() { setCurrentIndex(2); });

  TrainingGuide* tr = new TrainingGuide(this);
  addWidget(tr);
  connect(tr, &TrainingGuide::completedTraining, [=]() {
    training_done = true;
    params.put("CompletedTrainingVersion", current_training_version);
    updateActiveScreen();
  });

  DeclinePage* declinePage = new DeclinePage(this);
  addWidget(declinePage);
  connect(declinePage, &DeclinePage::getBack, [=]() { updateActiveScreen(); });

  setStyleSheet(R"UI0(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {)UI0"
#ifdef RK_BUILD
R"UI1(      height: 60px;
      font-size: 20px;
      font-weight: 148;
      border-radius: 4px;)UI1"
#else
R"UI2(      height: 160px;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;)UI2"
#endif
R"UI3(      background-color: #4F4F4F;
    }
  )UI3");
  updateActiveScreen();
}
