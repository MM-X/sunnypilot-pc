#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWidget>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

int main(int argc, char *argv[]) {
  initApp(argc, argv);
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  QGridLayout *main_layout = new QGridLayout(&window);
  main_layout->setMargin(18);

  QLabel *label = new QLabel(argv[1]);
  label->setWordWrap(true);
  label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  ScrollView *scroll = new ScrollView(label);
  scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  main_layout->addWidget(scroll, 0, 0, Qt::AlignTop);

  // Scroll to the bottom
  QObject::connect(scroll->verticalScrollBar(), &QAbstractSlider::rangeChanged, [=]() {
    scroll->verticalScrollBar()->setValue(scroll->verticalScrollBar()->maximum());
  });

  QPushButton *btn = new QPushButton();
#ifdef __aarch64__
  btn->setText(QObject::tr("Reboot"));
  QObject::connect(btn, &QPushButton::clicked, [=]() {
    Hardware::reboot();
  });
#else
  btn->setText(QObject::tr("Exit"));
  QObject::connect(btn, &QPushButton::clicked, &a, &QApplication::quit);
#endif
  main_layout->addWidget(btn, 0, 0, Qt::AlignRight | Qt::AlignBottom);

  window.setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      background-color: black;
      font-size: 24px;
    }
    QPushButton {
      padding: 18px;
      padding-right: 38px;
      padding-left: 38px;
      border: 1px solid white;
      border-radius: 8px;
      margin-right: 16px;
    }
  )");

  return a.exec();
}
