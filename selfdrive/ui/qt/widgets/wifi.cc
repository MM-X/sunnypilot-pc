#include "selfdrive/ui/qt/widgets/wifi.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>

WiFiPromptWidget::WiFiPromptWidget(QWidget *parent) : QFrame(parent) {
  // Setup Firehose Mode
  QVBoxLayout *main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setContentsMargins(21, 15, 21, 15);
  main_layout->setSpacing(16);

#else
  main_layout->setContentsMargins(56, 40, 56, 40);
  main_layout->setSpacing(42);  
  
#endif
  QLabel *title = new QLabel(tr("<span style='font-family: \"Noto Color Emoji\";'>🔥</span> Firehose Mode <span style='font-family: Noto Color Emoji;'>🔥</span>"));
#ifdef RK_BUILD
  title->setStyleSheet("font-size: 24px; font-weight: 185;");
#else
  title->setStyleSheet("font-size: 64px; font-weight: 500;");
#endif
  main_layout->addWidget(title);

  QLabel *desc = new QLabel(tr("Maximize your training data uploads to improve openpilot's driving models."));
#ifdef RK_BUILD
  desc->setStyleSheet("font-size: 15px; font-weight: 148;");
#else
  desc->setStyleSheet("font-size: 40px; font-weight: 400;");
#endif
  desc->setWordWrap(true);
  main_layout->addWidget(desc);

  QPushButton *settings_btn = new QPushButton(tr("Open"));
  connect(settings_btn, &QPushButton::clicked, [=]() { emit openSettings(1, "FirehosePanel"); });
  settings_btn->setStyleSheet(R"UI0(
    QPushButton {)UI0"
#ifdef RK_BUILD
R"UI1(      font-size: 18px;
      font-weight: 185;
      border-radius: 4px;)UI1"
#else
R"UI2(      font-size: 48px;
      font-weight: 500;
      border-radius: 10px;)UI2"
#endif
R"UI3(      background-color: #465BEA;)UI3"
#ifdef RK_BUILD
R"UI4(      padding: 12px;)UI4"
#else
R"UI5(      padding: 32px;)UI5"
#endif
R"UI6(    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )UI6");
  main_layout->addWidget(settings_btn);

  setStyleSheet(R"UI7(
    WiFiPromptWidget {
      background-color: #333333;)UI7"
#ifdef RK_BUILD
R"UI8(      border-radius: 4px;)UI8"
#else
R"UI9(      border-radius: 10px;)UI9"
#endif
R"UI10(    }
  )UI10");
}