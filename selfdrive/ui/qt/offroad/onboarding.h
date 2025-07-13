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
  const QRect continueBtn = {681, 0, 118, 400};
  QVector<QRect> boundingRect {
    QRect(41, 298, 229, 61),
    continueBtn,
    continueBtn,
    QRect(608, 207, 78, 116),
    QRect(616, 195, 68, 40),
    continueBtn,
    QRect(672, 230, 78, 63),
    QRect(500, 0, 184, 280),
    QRect(570, 143, 173, 88),
    QRect(41, 298, 417, 61),
    QRect(592, 74, 117, 123),
    continueBtn,
    QRect(505, 33, 295, 367),
    continueBtn,
    QRect(590, 42, 118, 316),
    QRect(511, 189, 145, 90),
    continueBtn,
    continueBtn,
    QRect(233, 298, 232, 61),
    QRect(40, 298, 157, 61),
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
