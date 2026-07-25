#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QMouseEvent>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>

#include "common/params.h"
#include "selfdrive/ui/qt/qt_window.h"

class TrainingGuide : public QFrame {
  Q_OBJECT

public:
  explicit TrainingGuide(QWidget *parent = 0);

private:
  void showEvent(QShowEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent* e) override;
  QImage loadImage(int id);

  QImage image;
  QSize image_raw_size;
  int currentIndex = 0;

  // Bounding boxes for each training guide step
#ifdef RK_BUILD
  const QRect continueBtn = {681, 0, 118, 400};
#else
  const QRect continueBtn = {1840, 0, 320, 1080};
#endif
  QVector<QRect> boundingRect {
#ifdef RK_BUILD
    QRect(41, 298, 229, 61),
#else
    QRect(112, 804, 618, 164),
#endif
    continueBtn,
    continueBtn,
#ifdef RK_BUILD
    QRect(608, 207, 78, 116),
    QRect(616, 195, 68, 40),
#else
    QRect(1641, 558, 210, 313),
    QRect(1662, 528, 184, 108),
#endif
    continueBtn,
#ifdef RK_BUILD
    QRect(672, 230, 78, 63),
    QRect(500, 0, 184, 280),
    QRect(570, 143, 173, 88),
    QRect(41, 298, 417, 61),
    QRect(592, 74, 117, 123),
#else
    QRect(1814, 621, 211, 170),
    QRect(1350, 0, 497, 755),
    QRect(1540, 386, 468, 238),
    QRect(112, 804, 1126, 164),
    QRect(1598, 199, 316, 333),
#endif
    continueBtn,
#ifdef RK_BUILD
    QRect(505, 33, 295, 367),
#else
    QRect(1364, 90, 796, 990),
#endif
    continueBtn,
#ifdef RK_BUILD
    QRect(590, 42, 118, 316),
    QRect(511, 189, 145, 90),
#else
    QRect(1593, 114, 318, 853),
    QRect(1379, 511, 391, 243),
#endif
    continueBtn,
    continueBtn,
#ifdef RK_BUILD
    QRect(233, 298, 232, 61),
    QRect(40, 298, 157, 61),
#else
    QRect(630, 804, 626, 164),
    QRect(108, 804, 426, 164),
#endif
  };

  const QString img_path = "../assets/training/";
  QElapsedTimer click_timer;

signals:
  void completedTraining();
};


class TermsPage : public QFrame {
  Q_OBJECT

public:
  explicit TermsPage(QWidget *parent = 0) : QFrame(parent) {}

private:
  void showEvent(QShowEvent *event) override;

protected:
  QPushButton *accept_btn;

signals:
  void acceptedTerms();
  void declinedTerms();
};

class DeclinePage : public QFrame {
  Q_OBJECT

public:
  explicit DeclinePage(QWidget *parent = 0) : QFrame(parent) {}

private:
  void showEvent(QShowEvent *event) override;

signals:
  void getBack();
};

class OnboardingWindow : public QStackedWidget {
  Q_OBJECT

public:
  explicit OnboardingWindow(QWidget *parent = 0);
  inline void showTrainingGuide() { setCurrentIndex(1); }
  virtual inline bool completed() const { return accepted_terms && training_done; }

protected:
  virtual void updateActiveScreen();

  Params params;
  bool accepted_terms = false, training_done = false;

signals:
  void onboardingDone();
};
