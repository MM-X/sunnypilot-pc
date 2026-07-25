#include "selfdrive/ui/qt/network/networking.h"

#include <algorithm>

#include <QHBoxLayout>
#include <QScrollBar>
#include <QStyle>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"
#else
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"
#endif

#ifdef RK_BUILD
static const int ICON_WIDTH = 18;
#else
static const int ICON_WIDTH = 49;
#endif

// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QFrame(parent) {
  main_layout = new QStackedLayout(this);

  wifi = new WifiManager(this);
  connect(wifi, &WifiManager::refreshSignal, this, &Networking::refresh);
  connect(wifi, &WifiManager::wrongPassword, this, &Networking::wrongPassword);

  wifiScreen = new QWidget(this);
  QVBoxLayout* vlayout = new QVBoxLayout(wifiScreen);
#ifdef RK_BUILD
  vlayout->setContentsMargins(7, 7, 7, 7);
#else
  vlayout->setContentsMargins(20, 20, 20, 20);
#endif
  if (show_advanced) {
    QPushButton* advancedSettings = new QPushButton(tr("Advanced"));
    advancedSettings->setObjectName("advanced_btn");
#ifdef RK_BUILD
    advancedSettings->setStyleSheet("margin-right: 11px;");
    advancedSettings->setFixedSize(148, 37);
#else
    advancedSettings->setStyleSheet("margin-right: 30px;");
    advancedSettings->setFixedSize(400, 100);
#endif
    connect(advancedSettings, &QPushButton::clicked, [=]() { main_layout->setCurrentWidget(an); });
#ifdef RK_BUILD
    vlayout->addSpacing(4);
#else
    vlayout->addSpacing(10);
#endif
    vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
#ifdef RK_BUILD
    vlayout->addSpacing(4);
#else
    vlayout->addSpacing(10);
#endif
  }

  wifiWidget = new WifiUI(this, wifi);
  wifiWidget->setObjectName("wifiWidget");
  connect(wifiWidget, &WifiUI::connectToNetwork, this, &Networking::connectToNetwork);

  ScrollView *wifiScroller = new ScrollView(wifiWidget, this);
  wifiScroller->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  vlayout->addWidget(wifiScroller, 1);
  main_layout->addWidget(wifiScreen);

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=]() { main_layout->setCurrentWidget(wifiScreen); });
  connect(an, &AdvancedNetworking::requestWifiScreen, [=]() { main_layout->setCurrentWidget(wifiScreen); });
  main_layout->addWidget(an);

  QPalette pal = palette();
  pal.setColor(QPalette::Window, QColor(0x29, 0x29, 0x29));
  setAutoFillBackground(true);
  setPalette(pal);

  setStyleSheet(R"UI0(
    #wifiWidget > QPushButton, #back_btn, #advanced_btn {)UI0"
#ifdef RK_BUILD
R"UI1(      font-size: 18px;)UI1"
#else
R"UI2(      font-size: 50px;)UI2"
#endif
R"UI3(      margin: 0px;)UI3"
#ifdef RK_BUILD
R"UI4(      padding: 6px;)UI4"
#else
R"UI5(      padding: 15px;)UI5"
#endif
R"UI6(      border-width: 0;)UI6"
#ifdef RK_BUILD
R"UI7(      border-radius: 12px;)UI7"
#else
R"UI8(      border-radius: 30px;)UI8"
#endif
R"UI9(      color: #dddddd;
      background-color: #393939;
    }
    #back_btn:pressed, #advanced_btn:pressed {
      background-color:  #4a4a4a;
    }
  )UI9");
  main_layout->setCurrentWidget(wifiScreen);
}

void Networking::setPrimeType(PrimeState::Type type) {
  an->setGsmVisible(type == PrimeState::PRIME_TYPE_NONE || type == PrimeState::PRIME_TYPE_UNKNOWN || \
                    type == PrimeState::PRIME_TYPE_PURPLE ||  type == PrimeState::PRIME_TYPE_LITE);
  wifi->ipv4_forward = (type == PrimeState::PRIME_TYPE_NONE || type == PrimeState::PRIME_TYPE_LITE);
}

void Networking::refresh() {
  wifiWidget->refresh();
  an->refresh();
}

void Networking::connectToNetwork(const Network n) {
  if (wifi->isKnownConnection(n.ssid)) {
    wifi->activateWifiConnection(n.ssid);
  } else if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n, false);
  } else if (n.security_type == SecurityType::WPA) {
    QString pass = InputDialog::getText(tr("Enter password"), this, tr("for \"%1\"").arg(QString::fromUtf8(n.ssid)), true, 8);
    if (!pass.isEmpty()) {
      wifi->connect(n, false, pass);
    }
  }
}

void Networking::wrongPassword(const QString &ssid) {
  if (wifi->seenNetworks.contains(ssid)) {
    const Network &n = wifi->seenNetworks.value(ssid);
    QString pass = InputDialog::getText(tr("Wrong password"), this, tr("for \"%1\"").arg(QString::fromUtf8(n.ssid)), true, 8);
    if (!pass.isEmpty()) {
      wifi->connect(n, false, pass);
    }
  }
}

void Networking::showEvent(QShowEvent *event) {
  wifi->start();
}

void Networking::hideEvent(QHideEvent *event) {
  main_layout->setCurrentWidget(wifiScreen);
  wifi->stop();
}

// AdvancedNetworking functions

AdvancedNetworking::AdvancedNetworking(QWidget* parent, WifiManager* wifi): QWidget(parent), wifi(wifi) {

  QVBoxLayout* main_layout = new QVBoxLayout(this);
#ifdef RK_BUILD
  main_layout->setMargin(15);
  main_layout->setSpacing(7);
#else
  main_layout->setMargin(40);
  main_layout->setSpacing(20);
#endif

  // Back button
  QPushButton* back = new QPushButton(tr("Back"));
  back->setObjectName("back_btn");
#ifdef RK_BUILD
  back->setFixedSize(148, 37);
#else
  back->setFixedSize(400, 100);
#endif
  connect(back, &QPushButton::clicked, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  ListWidget *list = new ListWidget(this);
  // Enable tethering layout
  tetheringToggle = new ToggleControl(tr("Enable Tethering"), "", "", wifi->isTetheringEnabled());
  list->addItem(tetheringToggle);
  QObject::connect(tetheringToggle, &ToggleControl::toggleFlipped, this, &AdvancedNetworking::toggleTethering);

  // Change tethering password
  ButtonControl *editPasswordButton = new ButtonControl(tr("Tethering Password"), tr("EDIT"));
  connect(editPasswordButton, &ButtonControl::clicked, [=]() {
    QString pass = InputDialog::getText(tr("Enter new tethering password"), this, "", true, 8, wifi->getTetheringPassword());
    if (!pass.isEmpty()) {
      wifi->changeTetheringPassword(pass);
    }
  });
  list->addItem(editPasswordButton);

  // IP address
  ipLabel = new LabelControl(tr("IP Address"), wifi->ipv4_address);
  list->addItem(ipLabel);

  // Roaming toggle
  const bool roamingEnabled = params.getBool("GsmRoaming");
  roamingToggle = new ToggleControl(tr("Enable Roaming"), "", "", roamingEnabled);
  QObject::connect(roamingToggle, &ToggleControl::toggleFlipped, [=](bool state) {
    params.putBool("GsmRoaming", state);
    wifi->updateGsmSettings(state, QString::fromStdString(params.get("GsmApn")), params.getBool("GsmMetered"));
  });
  list->addItem(roamingToggle);

  // APN settings
  editApnButton = new ButtonControl(tr("APN Setting"), tr("EDIT"));
  connect(editApnButton, &ButtonControl::clicked, [=]() {
    const QString cur_apn = QString::fromStdString(params.get("GsmApn"));
    QString apn = InputDialog::getText(tr("Enter APN"), this, tr("leave blank for automatic configuration"), false, -1, cur_apn).trimmed();

    if (apn.isEmpty()) {
      params.remove("GsmApn");
    } else {
      params.put("GsmApn", apn.toStdString());
    }
    wifi->updateGsmSettings(params.getBool("GsmRoaming"), apn, params.getBool("GsmMetered"));
  });
  list->addItem(editApnButton);

  // Cellular metered toggle (prime lite or none)
  const bool metered = params.getBool("GsmMetered");
  cellularMeteredToggle = new ToggleControl(tr("Cellular Metered"), tr("Prevent large data uploads when on a metered cellular connection"), "", metered);
  QObject::connect(cellularMeteredToggle, &SshToggle::toggleFlipped, [=](bool state) {
    params.putBool("GsmMetered", state);
    wifi->updateGsmSettings(params.getBool("GsmRoaming"), QString::fromStdString(params.get("GsmApn")), state);
  });
  list->addItem(cellularMeteredToggle);

  // Wi-Fi metered toggle
  std::vector<QString> metered_button_texts{tr("default"), tr("metered"), tr("unmetered")};
  wifiMeteredToggle = new MultiButtonControl(tr("Wi-Fi Network Metered"), tr("Prevent large data uploads when on a metered Wi-Fi connection"), "", metered_button_texts);
  QObject::connect(wifiMeteredToggle, &MultiButtonControl::buttonClicked, [=](int id) {
    wifiMeteredToggle->setEnabled(false);
    MeteredType metered = MeteredType::UNKNOWN;
    if (id == NM_METERED_YES) {
      metered = MeteredType::YES;
    } else if (id == NM_METERED_NO) {
      metered = MeteredType::NO;
    }
    auto pending_call = wifi->setCurrentNetworkMetered(metered);
    if (pending_call) {
      QDBusPendingCallWatcher *watcher = new QDBusPendingCallWatcher(*pending_call);
      QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, [=]() {
        refresh();
        watcher->deleteLater();
      });
    }
  });
  list->addItem(wifiMeteredToggle);

  // Hidden Network
  hiddenNetworkButton = new ButtonControl(tr("Hidden Network"), tr("CONNECT"));
  connect(hiddenNetworkButton, &ButtonControl::clicked, [=]() {
    QString ssid = InputDialog::getText(tr("Enter SSID"), this, "", false, 1);
    if (!ssid.isEmpty()) {
      QString pass = InputDialog::getText(tr("Enter password"), this, tr("for \"%1\"").arg(ssid), true, -1);
      Network hidden_network;
      hidden_network.ssid = ssid.toUtf8();
      if (!pass.isEmpty()) {
        hidden_network.security_type = SecurityType::WPA;
        wifi->connect(hidden_network, true, pass);
      } else {
        wifi->connect(hidden_network, true);
      }
      emit requestWifiScreen();
    }
  });
  list->addItem(hiddenNetworkButton);

  // Set initial config
  wifi->updateGsmSettings(roamingEnabled, QString::fromStdString(params.get("GsmApn")), metered);

  main_layout->addWidget(new ScrollView(list, this));
  main_layout->addStretch(1);
}

void AdvancedNetworking::setGsmVisible(bool visible) {
  roamingToggle->setVisible(visible);
  editApnButton->setVisible(visible);
  cellularMeteredToggle->setVisible(visible);
}

void AdvancedNetworking::refresh() {
  ipLabel->setText(wifi->ipv4_address);
  tetheringToggle->setEnabled(true);

  if (wifi->isTetheringEnabled() || wifi->ipv4_address == "") {
    wifiMeteredToggle->setEnabled(false);
    wifiMeteredToggle->setCheckedButton(0);
  } else if (wifi->ipv4_address != "") {
    MeteredType metered = wifi->currentNetworkMetered();
    wifiMeteredToggle->setEnabled(true);
    wifiMeteredToggle->setCheckedButton(static_cast<int>(metered));
  }

  update();
}

void AdvancedNetworking::toggleTethering(bool enabled) {
  wifi->setTetheringEnabled(enabled);
  tetheringToggle->setEnabled(false);
  if (enabled) {
    wifiMeteredToggle->setEnabled(false);
    wifiMeteredToggle->setCheckedButton(0);
  }
}

// WifiUI functions

WifiUI::WifiUI(QWidget *parent, WifiManager* wifi) : QWidget(parent), wifi(wifi) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  // load imgs
  for (const auto &s : {"low", "medium", "high", "full"}) {
    QPixmap pix(ASSET_PATH + "/icons/wifi_strength_" + s + ".svg");
#ifdef RK_BUILD
    strengths.push_back(pix.scaledToHeight(25, Qt::SmoothTransformation));
#else
    strengths.push_back(pix.scaledToHeight(68, Qt::SmoothTransformation));
#endif
  }
  lock = QPixmap(ASSET_PATH + "icons/lock_closed.svg").scaledToWidth(ICON_WIDTH, Qt::SmoothTransformation);
  checkmark = QPixmap(ASSET_PATH + "icons/checkmark.svg").scaledToWidth(ICON_WIDTH, Qt::SmoothTransformation);
  circled_slash = QPixmap(ASSET_PATH + "icons/circled_slash.svg").scaledToWidth(ICON_WIDTH, Qt::SmoothTransformation);

  scanningLabel = new QLabel(tr("Scanning for networks..."));
#ifdef RK_BUILD
  scanningLabel->setStyleSheet("font-size: 24px;");
#else
  scanningLabel->setStyleSheet("font-size: 65px;");
#endif
  main_layout->addWidget(scanningLabel, 0, Qt::AlignCenter);

  wifi_list_widget = new ListWidget(this);
  wifi_list_widget->setVisible(false);
  main_layout->addWidget(wifi_list_widget);

  setStyleSheet(R"UI10(
    QScrollBar::handle:vertical {
      min-height: 0px;)UI10"
#ifdef RK_BUILD
R"UI11(      border-radius: 2px;)UI11"
#else
R"UI12(      border-radius: 4px;)UI12"
#endif
R"UI13(      background-color: #8A8A8A;
    }
    #forgetBtn {)UI13"
#ifdef RK_BUILD
R"UI14(      font-size: 12px;
      font-weight: 222;)UI14"
#else
R"UI15(      font-size: 32px;
      font-weight: 600;)UI15"
#endif
R"UI16(      color: #292929;
      background-color: #BDBDBD;
      border-width: 1px solid #828282;)UI16"
#ifdef RK_BUILD
R"UI17(      border-radius: 1px;
      padding: 14px;
      padding-bottom: 6px;
      padding-top: 6px;)UI17"
#else
R"UI18(      border-radius: 5px;
      padding: 40px;
      padding-bottom: 16px;
      padding-top: 16px;)UI18"
#endif
R"UI19(    }
    #forgetBtn:pressed {
      background-color: #828282;
    }
    #connecting {)UI19"
#ifdef RK_BUILD
R"UI20(      font-size: 12px;
      font-weight: 222;)UI20"
#else
R"UI21(      font-size: 32px;
      font-weight: 600;)UI21"
#endif
R"UI22(      color: white;
      border-radius: 0;)UI22"
#ifdef RK_BUILD
R"UI23(      padding: 10px;
      padding-left: 16px;
      padding-right: 16px;)UI23"
#else
R"UI24(      padding: 27px;
      padding-left: 43px;
      padding-right: 43px;)UI24"
#endif
R"UI25(      background-color: black;
    }
    #ssidLabel {
      text-align: left;
      border: none;)UI25"
#ifdef RK_BUILD
R"UI26(      padding-top: 18px;
      padding-bottom: 18px;)UI26"
#else
R"UI27(      padding-top: 50px;
      padding-bottom: 50px;)UI27"
#endif
R"UI28(    }
    #ssidLabel:disabled {
      color: #696969;
    }
  )UI28");
}

void WifiUI::refresh() {
  bool is_empty = wifi->seenNetworks.isEmpty();
  scanningLabel->setVisible(is_empty);
  wifi_list_widget->setVisible(!is_empty);
  if (is_empty) return;

  setUpdatesEnabled(false);

  const bool is_tethering_enabled = wifi->isTetheringEnabled();
  QList<Network> sortedNetworks = wifi->seenNetworks.values();
  std::sort(sortedNetworks.begin(), sortedNetworks.end(), compare_by_strength);

  int n = 0;
  for (Network &network : sortedNetworks) {
    QPixmap status_icon;
    if (network.connected == ConnectedType::CONNECTED) {
      status_icon = checkmark;
    } else if (network.security_type == SecurityType::UNSUPPORTED) {
      status_icon = circled_slash;
    } else if (network.security_type == SecurityType::WPA) {
      status_icon = lock;
    }
    bool show_forget_btn = wifi->isKnownConnection(network.ssid) && !is_tethering_enabled;
    QPixmap strength = strengths[strengthLevel(network.strength)];

    auto item = getItem(n++);
    item->setItem(network, status_icon, show_forget_btn, strength);
    item->setVisible(true);
  }
  for (; n < wifi_items.size(); ++n) wifi_items[n]->setVisible(false);

  setUpdatesEnabled(true);
}

WifiItem *WifiUI::getItem(int n) {
  auto item = n < wifi_items.size() ? wifi_items[n] : wifi_items.emplace_back(new WifiItem(tr("CONNECTING..."), tr("FORGET")));
  if (!item->parentWidget()) {
    QObject::connect(item, &WifiItem::connectToNetwork, this, &WifiUI::connectToNetwork);
    QObject::connect(item, &WifiItem::forgotNetwork, [this](const Network n) {
      if (ConfirmationDialog::confirm(tr("Forget Wi-Fi Network \"%1\"?").arg(QString::fromUtf8(n.ssid)), tr("Forget"), this))
        wifi->forgetConnection(n.ssid);
    });
    wifi_list_widget->addItem(item);
  }
  return item;
}

// WifiItem

WifiItem::WifiItem(const QString &connecting_text, const QString &forget_text, QWidget *parent) : QWidget(parent) {
  QHBoxLayout *hlayout = new QHBoxLayout(this);
#ifdef RK_BUILD
  hlayout->setContentsMargins(16, 0, 27, 0);
  hlayout->setSpacing(18);
#else
  hlayout->setContentsMargins(44, 0, 73, 0);
  hlayout->setSpacing(50);
#endif

  hlayout->addWidget(ssidLabel = new ElidedLabel());
  ssidLabel->setObjectName("ssidLabel");
  ssidLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  hlayout->addWidget(connecting = new QPushButton(connecting_text), 0, Qt::AlignRight);
  connecting->setObjectName("connecting");
  hlayout->addWidget(forgetBtn = new QPushButton(forget_text), 0, Qt::AlignRight);
  forgetBtn->setObjectName("forgetBtn");
  hlayout->addWidget(iconLabel = new QLabel(), 0, Qt::AlignRight);
  hlayout->addWidget(strengthLabel = new QLabel(), 0, Qt::AlignRight);

  iconLabel->setFixedWidth(ICON_WIDTH);
  QObject::connect(forgetBtn, &QPushButton::clicked, [this]() { emit forgotNetwork(network); });
  QObject::connect(ssidLabel, &ElidedLabel::clicked, [this]() {
    if (network.connected == ConnectedType::DISCONNECTED) emit connectToNetwork(network);
  });
}

void WifiItem::setItem(const Network &n, const QPixmap &status_icon, bool show_forget_btn, const QPixmap &strength_icon) {
  network = n;

  ssidLabel->setText(n.ssid);
  ssidLabel->setEnabled(n.security_type != SecurityType::UNSUPPORTED);
#ifdef RK_BUILD
  ssidLabel->setFont(InterFont(20, network.connected == ConnectedType::DISCONNECTED ? QFont::Normal : QFont::Bold));
#else
  ssidLabel->setFont(InterFont(55, network.connected == ConnectedType::DISCONNECTED ? QFont::Normal : QFont::Bold));
#endif

  connecting->setVisible(n.connected == ConnectedType::CONNECTING);
  forgetBtn->setVisible(show_forget_btn);

  iconLabel->setPixmap(status_icon);
  strengthLabel->setPixmap(strength_icon);
}
