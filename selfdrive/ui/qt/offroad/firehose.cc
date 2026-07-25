#include "selfdrive/ui/qt/offroad/firehose.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QScrollArea>
#include <QStackedLayout>
#include <QProgressBar>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>

#ifdef SUNNYPILOT
#define UIState UIStateSP
#endif

FirehosePanel::FirehosePanel(SettingsWindow *parent) : QWidget((QWidget*)parent) {
  layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  layout->setContentsMargins(15, 15, 15, 15);
  layout->setSpacing(7);
#else
  layout->setContentsMargins(40, 40, 40, 40);
  layout->setSpacing(20);
#endif

  // header
  QLabel *title = new QLabel(tr("Firehose Mode"));
#ifdef RK_BUILD
  title->setStyleSheet("font-size: 37px; font-weight: 185; font-family: 'Noto Color Emoji';");
#else
  title->setStyleSheet("font-size: 100px; font-weight: 500; font-family: 'Noto Color Emoji';");
#endif
  layout->addWidget(title, 0, Qt::AlignCenter);

  // Create a container for the content
  QFrame *content = new QFrame();
#ifdef RK_BUILD
  content->setStyleSheet("background-color: #292929; border-radius: 6px; padding: 8px;");
#else
  content->setStyleSheet("background-color: #292929; border-radius: 15px; padding: 20px;");
#endif
  QVBoxLayout *content_layout = new QVBoxLayout(content);
#ifdef RK_BUILD
  content_layout->setSpacing(7);
#else
  content_layout->setSpacing(20);
#endif

  // Top description
  QLabel *description = new QLabel(tr("sunnypilot learns to drive by watching humans, like you, drive.\n\nFirehose Mode allows you to maximize your training data uploads to improve openpilot's driving models. More data means bigger models, which means better Experimental Mode."));
#ifdef RK_BUILD
  description->setStyleSheet("font-size: 16px; padding-bottom: 8px;");
#else
  description->setStyleSheet("font-size: 45px; padding-bottom: 20px;");
#endif
  description->setWordWrap(true);
  content_layout->addWidget(description);

  // Add a separator
  QFrame *line = new QFrame();
  line->setFrameShape(QFrame::HLine);
  line->setFrameShadow(QFrame::Sunken);
#ifdef RK_BUILD
  line->setStyleSheet("background-color: #444444; margin-top: 2px; margin-bottom: 2px;");
#else
  line->setStyleSheet("background-color: #444444; margin-top: 5px; margin-bottom: 5px;");
#endif
  content_layout->addWidget(line);

  toggle_label = new QLabel(tr("Firehose Mode: ACTIVE"));
#ifdef RK_BUILD
  toggle_label->setStyleSheet("font-size: 22px; font-weight: bold; color: white;");
#else
  toggle_label->setStyleSheet("font-size: 60px; font-weight: bold; color: white;");
#endif
  content_layout->addWidget(toggle_label);

  // Add contribution label
  contribution_label = new QLabel();
#ifdef RK_BUILD
  contribution_label->setStyleSheet("font-size: 20px; margin-top: 4x; margin-bottom: 4px;");
#else
  contribution_label->setStyleSheet("font-size: 52px; margin-top: 10px; margin-bottom: 10px;");
#endif
  contribution_label->setWordWrap(true);
  contribution_label->hide();
  content_layout->addWidget(contribution_label);

  // Add a separator before detailed instructions
  QFrame *line2 = new QFrame();
  line2->setFrameShape(QFrame::HLine);
  line2->setFrameShadow(QFrame::Sunken);
#ifdef RK_BUILD
  line2->setStyleSheet("background-color: #444444; margin-top: 4px; margin-bottom: 4px;");
#else
  line2->setStyleSheet("background-color: #444444; margin-top: 10px; margin-bottom: 10px;");
#endif
  content_layout->addWidget(line2);

  // Detailed instructions at the bottom
  detailed_instructions = new QLabel(tr(
    "For maximum effectiveness, bring your device inside and connect to a good USB-C adapter and Wi-Fi weekly.<br>"
    "<br>"
    "Firehose Mode can also work while you're driving if connected to a hotspot or unlimited SIM card.<br>"
    "<br><br>"
    "<b>Frequently Asked Questions</b><br><br>"
    "<i>Does it matter how or where I drive?</i> Nope, just drive as you normally would.<br><br>"
    "<i>Do all of my segments get pulled in Firehose Mode?</i> No, we selectively pull a subset of your segments.<br><br>"
    "<i>What's a good USB-C adapter?</i> Any fast phone or laptop charger should be fine.<br><br>"
    "<i>Does it matter which software I run?</i> Yes, only upstream sunnypilot (and particular forks) are able to be used for training."
  ));
#ifdef RK_BUILD
  detailed_instructions->setStyleSheet("font-size: 14px; color: #E4E4E4;");
#else
  detailed_instructions->setStyleSheet("font-size: 40px; color: #E4E4E4;");
#endif
  detailed_instructions->setWordWrap(true);
  content_layout->addWidget(detailed_instructions);

  layout->addWidget(content, 1);

  // Set up the API request for firehose stats
  const QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  firehose_stats = new RequestRepeater(this, CommaApi::BASE_URL + "/v1/devices/" + dongle_id + "/firehose_stats",
                                       "ApiCache_FirehoseStats", 30, true);
  QObject::connect(firehose_stats, &RequestRepeater::requestDone, [=](const QString &response, bool success) {
    if (success) {
      QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
      QJsonObject json = doc.object();
      int count = json["firehose"].toInt();
      contribution_label->setText(tr("<b>%n segment(s)</b> of your driving is in the training dataset so far.", "", count));
      contribution_label->show();
    }
  });

  QObject::connect(uiState(), &UIState::uiUpdate, this, &FirehosePanel::refresh);
}

void FirehosePanel::refresh() {
  auto deviceState = (*uiState()->sm)["deviceState"].getDeviceState();
  auto networkType = deviceState.getNetworkType();
  bool networkMetered = deviceState.getNetworkMetered();

  bool is_active = !networkMetered && (networkType != cereal::DeviceState::NetworkType::NONE);
  if (is_active) {
    toggle_label->setText(tr("ACTIVE"));
#ifdef RK_BUILD
    toggle_label->setStyleSheet("font-size: 22px; font-weight: bold; color: #2ecc71;");
#else
    toggle_label->setStyleSheet("font-size: 60px; font-weight: bold; color: #2ecc71;");
#endif
  } else {
#ifdef RK_BUILD
    toggle_label->setText(tr("<span stylesheet='font-size: 22px; font-weight: bold; color: #e74c3c;'>INACTIVE</span>: connect to an unmetered network"));
    toggle_label->setStyleSheet("font-size: 22px;");
#else
    toggle_label->setText(tr("<span stylesheet='font-size: 60px; font-weight: bold; color: #e74c3c;'>INACTIVE</span>: connect to an unmetered network"));
    toggle_label->setStyleSheet("font-size: 60px;");
#endif
  }
}
