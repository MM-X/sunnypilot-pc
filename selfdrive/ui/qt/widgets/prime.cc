#include "selfdrive/ui/qt/widgets/prime.h"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include <QrCode.hpp>

#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/wifi.h"

using qrcodegen::QrCode;

PairingQRWidget::PairingQRWidget(QWidget* parent) : QWidget(parent) {
  timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &PairingQRWidget::refresh);
}

void PairingQRWidget::showEvent(QShowEvent *event) {
  refresh();
  timer->start(5 * 60 * 1000);
  device()->setOffroadBrightness(100);
}

void PairingQRWidget::hideEvent(QHideEvent *event) {
  timer->stop();
  device()->setOffroadBrightness(BACKLIGHT_OFFROAD);
}

void PairingQRWidget::refresh() {
  QString pairToken = CommaApi::create_jwt({{"pair", true}});
  QString qrString = "https://connect.comma.ai/?pair=" + pairToken;
  this->updateQrCode(qrString);
  update();
}

void PairingQRWidget::updateQrCode(const QString &text) {
  QrCode qr = QrCode::encodeText(text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  QImage im(sz, sz, QImage::Format_RGB32);

  QRgb black = qRgb(0, 0, 0);
  QRgb white = qRgb(255, 255, 255);
  for (int y = 0; y < sz; y++) {
    for (int x = 0; x < sz; x++) {
      im.setPixel(x, y, qr.getModule(x, y) ? black : white);
    }
  }

  // Integer division to prevent anti-aliasing
  int final_sz = ((width() / sz) - 1) * sz;
  img = QPixmap::fromImage(im.scaled(final_sz, final_sz, Qt::KeepAspectRatio), Qt::MonoOnly);
}

void PairingQRWidget::paintEvent(QPaintEvent *e) {
  QPainter p(this);
  p.fillRect(rect(), Qt::white);

  QSize s = (size() - img.size()) / 2;
  p.drawPixmap(s.width(), s.height(), img);
}


PairingPopup::PairingPopup(QWidget *parent) : DialogBase(parent) {
  QHBoxLayout *hlayout = new QHBoxLayout(this);
  hlayout->setContentsMargins(0, 0, 0, 0);
  hlayout->setSpacing(0);

  setStyleSheet("PairingPopup { background-color: #E0E0E0; }");

  // text
  QVBoxLayout *vlayout = new QVBoxLayout();
#ifdef RK_BUILD
  vlayout->setContentsMargins(31, 26, 18, 26);
  vlayout->setSpacing(18);
#else
  vlayout->setContentsMargins(85, 70, 50, 70);
  vlayout->setSpacing(50);
#endif
  hlayout->addLayout(vlayout, 1);
  {
    QPushButton *close = new QPushButton(QIcon(":/icons/close.svg"), "", this);
#ifdef RK_BUILD
    close->setIconSize(QSize(30, 30));
#else
    close->setIconSize(QSize(80, 80));
#endif
    close->setStyleSheet("border: none;");
    vlayout->addWidget(close, 0, Qt::AlignLeft);
    QObject::connect(close, &QPushButton::clicked, this, &QDialog::reject);

#ifdef RK_BUILD
    vlayout->addSpacing(11);
#else
    vlayout->addSpacing(30);
#endif

    QLabel *title = new QLabel(tr("Pair your device to your comma account"), this);
#ifdef RK_BUILD
    title->setStyleSheet("font-size: 28px; color: black;");
#else
    title->setStyleSheet("font-size: 75px; color: black;");
#endif
    title->setWordWrap(true);
    vlayout->addWidget(title);

    QLabel *instructions = new QLabel(QString(
#ifdef RK_BUILD
R"UI0(      <ol type='1' style='margin-left: 6px;'>
        <li style='margin-bottom: 18px;'>%1</li>
        <li style='margin-bottom: 18px;'>%2</li>
        <li style='margin-bottom: 18px;'>%3</li>)UI0"
#else
R"UI1(      <ol type='1' style='margin-left: 15px;'>
        <li style='margin-bottom: 50px;'>%1</li>
        <li style='margin-bottom: 50px;'>%2</li>
        <li style='margin-bottom: 50px;'>%3</li>)UI1"
#endif
R"UI2(      </ol>
    )UI2").arg(tr("Go to https://connect.comma.ai on your phone"))
    .arg(tr("Click \"add new device\" and scan the QR code on the right"))
    .arg(tr("Bookmark connect.comma.ai to your home screen to use it like an app")), this);

#ifdef RK_BUILD
    instructions->setStyleSheet("font-size: 17px; font-weight: bold; color: black;");
#else
    instructions->setStyleSheet("font-size: 47px; font-weight: bold; color: black;");
#endif
    instructions->setWordWrap(true);
    vlayout->addWidget(instructions);

    vlayout->addStretch();
  }

  // QR code
  PairingQRWidget *qr = new PairingQRWidget(this);
  hlayout->addWidget(qr, 1);
}

int PairingPopup::exec() {
  if (!util::system_time_valid()) {
    ConfirmationDialog::alert(tr("Please connect to Wi-Fi to complete initial pairing"), parentWidget());
    return QDialog::Rejected;
  }
  return DialogBase::exec();
}


PrimeUserWidget::PrimeUserWidget(QWidget *parent) : QFrame(parent) {
  setObjectName("primeWidget");
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
#ifdef RK_BUILD
  mainLayout->setContentsMargins(21, 15, 21, 15);
  mainLayout->setSpacing(7);
#else
  mainLayout->setContentsMargins(56, 40, 56, 40);
  mainLayout->setSpacing(20);
#endif

  QLabel *subscribed = new QLabel(tr("✓ SUBSCRIBED"));
#ifdef RK_BUILD
  subscribed->setStyleSheet("font-size: 15px; font-weight: bold; color: #86FF4E;");
#else
  subscribed->setStyleSheet("font-size: 41px; font-weight: bold; color: #86FF4E;");
#endif
  mainLayout->addWidget(subscribed);

  QLabel *commaPrime = new QLabel(tr("comma prime"));
#ifdef RK_BUILD
  commaPrime->setStyleSheet("font-size: 28x; font-weight: bold;");
#else
  commaPrime->setStyleSheet("font-size: 75px; font-weight: bold;");
#endif
  mainLayout->addWidget(commaPrime);
}


PrimeAdWidget::PrimeAdWidget(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setContentsMargins(30, 33, 30, 22);
#else
  main_layout->setContentsMargins(80, 90, 80, 60);
#endif
  main_layout->setSpacing(0);

  QLabel *upgrade = new QLabel(tr("Upgrade Now"));
#ifdef RK_BUILD
  upgrade->setStyleSheet("font-size: 28px; font-weight: bold;");
#else
  upgrade->setStyleSheet("font-size: 75px; font-weight: bold;");
#endif
  main_layout->addWidget(upgrade, 0, Qt::AlignTop);
#ifdef RK_BUILD
  main_layout->addSpacing(18);
#else
  main_layout->addSpacing(50);
#endif

  QLabel *description = new QLabel(tr("Become a comma prime member at connect.comma.ai"));
#ifdef RK_BUILD
  description->setStyleSheet("font-size: 21px; font-weight: light; color: white;");
#else
  description->setStyleSheet("font-size: 56px; font-weight: light; color: white;");
#endif
  description->setWordWrap(true);
  main_layout->addWidget(description, 0, Qt::AlignTop);

  main_layout->addStretch();

  QLabel *features = new QLabel(tr("PRIME FEATURES:"));
#ifdef RK_BUILD
  features->setStyleSheet("font-size: 15px; font-weight: bold; color: #E5E5E5;");
#else
  features->setStyleSheet("font-size: 41px; font-weight: bold; color: #E5E5E5;");
#endif
  main_layout->addWidget(features, 0, Qt::AlignBottom);
#ifdef RK_BUILD
  main_layout->addSpacing(11);
#else
  main_layout->addSpacing(30);
#endif

  QVector<QString> bullets = {tr("Remote access"), tr("24/7 LTE connectivity"), tr("1 year of drive storage"), tr("Remote snapshots")};
  for (auto &b : bullets) {
    const QString check = "<b><font color='#465BEA'>✓</font></b> ";
    QLabel *l = new QLabel(check + b);
    l->setAlignment(Qt::AlignLeft);
#ifdef RK_BUILD
    l->setStyleSheet("font-size: 18px; margin-bottom: 5px;");
#else
    l->setStyleSheet("font-size: 50px; margin-bottom: 15px;");
#endif
    main_layout->addWidget(l, 0, Qt::AlignBottom);
  }

  setStyleSheet(R"UI3(
    PrimeAdWidget {)UI3"
#ifdef RK_BUILD
R"UI4(      border-radius: 4px;)UI4"
#else
R"UI5(      border-radius: 10px;)UI5"
#endif
R"UI6(      background-color: #333333;
    }
  )UI6");
}


SetupWidget::SetupWidget(QWidget* parent) : QFrame(parent) {
  mainLayout = new QStackedWidget;

  // Unpaired, registration prompt layout

  QFrame* finishRegistration = new QFrame;
  finishRegistration->setObjectName("primeWidget");
  QVBoxLayout* finishRegistrationLayout = new QVBoxLayout(finishRegistration);
#ifdef RK_BUILD
  finishRegistrationLayout->setSpacing(14);
  finishRegistrationLayout->setContentsMargins(24, 18, 24, 18);
#else
  finishRegistrationLayout->setSpacing(38);
  finishRegistrationLayout->setContentsMargins(64, 48, 64, 48);
#endif

  QLabel* registrationTitle = new QLabel(tr("Finish Setup"));
#ifdef RK_BUILD
  registrationTitle->setStyleSheet("font-size: 28px; font-weight: bold;");
#else
  registrationTitle->setStyleSheet("font-size: 75px; font-weight: bold;");
#endif
  finishRegistrationLayout->addWidget(registrationTitle);

  QLabel* registrationDescription = new QLabel(tr("Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer."));
  registrationDescription->setWordWrap(true);
#ifdef RK_BUILD
  registrationDescription->setStyleSheet("font-size: 18px; font-weight: light;");
#else
  registrationDescription->setStyleSheet("font-size: 50px; font-weight: light;");
#endif
  finishRegistrationLayout->addWidget(registrationDescription);

  finishRegistrationLayout->addStretch();

  QPushButton* pair = new QPushButton(tr("Pair device"));
  pair->setStyleSheet(R"UI7(
    QPushButton {)UI7"
#ifdef RK_BUILD
R"UI8(      font-size: 20px;
      font-weight: 185;
      border-radius: 4px;)UI8"
#else
R"UI9(      font-size: 55px;
      font-weight: 500;
      border-radius: 10px;)UI9"
#endif
R"UI10(      background-color: #465BEA;)UI10"
#ifdef RK_BUILD
R"UI11(      padding: 24px;)UI11"
#else
R"UI12(      padding: 64px;)UI12"
#endif
R"UI13(    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )UI13");
  finishRegistrationLayout->addWidget(pair);

  popup = new PairingPopup(this);
  QObject::connect(pair, &QPushButton::clicked, popup, &PairingPopup::exec);

  mainLayout->addWidget(finishRegistration);

  // build stacked layout
  QVBoxLayout *outer_layout = new QVBoxLayout(this);
  outer_layout->setContentsMargins(0, 0, 0, 0);
  outer_layout->addWidget(mainLayout);

  QWidget *content = new QWidget;
  content_layout = new QVBoxLayout(content);
  content_layout->setContentsMargins(0, 0, 0, 0);
#ifdef RK_BUILD
  content_layout->setSpacing(11);
#else
  content_layout->setSpacing(30);
#endif

  WiFiPromptWidget *wifi_prompt = new WiFiPromptWidget;
  QObject::connect(wifi_prompt, &WiFiPromptWidget::openSettings, this, &SetupWidget::openSettings);
  content_layout->addWidget(wifi_prompt);
  content_layout->addStretch();

  mainLayout->addWidget(content);

  mainLayout->setCurrentIndex(1);

  setStyleSheet(R"UI14(
    #primeWidget {)UI14"
#ifdef RK_BUILD
R"UI15(      border-radius: 4px;)UI15"
#else
R"UI16(      border-radius: 10px;)UI16"
#endif
R"UI17(      background-color: #333333;
    }
  )UI17");

  // Retain size while hidden
  QSizePolicy sp_retain = sizePolicy();
  sp_retain.setRetainSizeWhenHidden(true);
  setSizePolicy(sp_retain);

  QObject::connect(uiState()->prime_state, &PrimeState::changed, [this](PrimeState::Type type) {
    if (type == PrimeState::PRIME_TYPE_UNPAIRED) {
      mainLayout->setCurrentIndex(0);  // Display "Pair your device" widget
    } else {
      popup->reject();
      mainLayout->setCurrentIndex(1);  // Display Wi-Fi prompt widget
    }
  });
}
