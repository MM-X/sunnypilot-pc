#include "selfdrive/ui/qt/widgets/scrollview.h"

#include <QScrollBar>
#include <QScroller>

// TODO: disable horizontal scrolling and resize

ScrollView::ScrollView(QWidget *w, QWidget *parent) : QScrollArea(parent) {
  setWidget(w);
  setWidgetResizable(true);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setStyleSheet("background-color: transparent; border:none");

  QString style = R"UI0(
    QScrollBar:vertical {
      border: none;
      background: transparent;)UI0"
#ifdef RK_BUILD
R"UI1(      width: 4px;)UI1"
#else
R"UI2(      width: 10px;)UI2"
#endif
R"UI3(      margin: 0;
    }
    QScrollBar::handle:vertical {
      min-height: 0px;)UI3"
#ifdef RK_BUILD
R"UI4(      border-radius: 2px;)UI4"
#else
R"UI5(      border-radius: 5px;)UI5"
#endif
R"UI6(      background-color: white;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
      height: 0px;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
      background: none;
    }
  )UI6";
  verticalScrollBar()->setStyleSheet(style);
  horizontalScrollBar()->setStyleSheet(style);

  QScroller *scroller = QScroller::scroller(this->viewport());
  QScrollerProperties sp = scroller->scrollerProperties();

  sp.setScrollMetric(QScrollerProperties::VerticalOvershootPolicy, QVariant::fromValue<QScrollerProperties::OvershootPolicy>(QScrollerProperties::OvershootAlwaysOff));
  sp.setScrollMetric(QScrollerProperties::HorizontalOvershootPolicy, QVariant::fromValue<QScrollerProperties::OvershootPolicy>(QScrollerProperties::OvershootAlwaysOff));
  sp.setScrollMetric(QScrollerProperties::MousePressEventDelay, 0.01);
  scroller->grabGesture(this->viewport(), QScroller::LeftMouseButtonGesture);
  scroller->setScrollerProperties(sp);
}

void ScrollView::hideEvent(QHideEvent *e) {
  verticalScrollBar()->setValue(0);
}
