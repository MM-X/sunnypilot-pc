/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/max_time_offroad.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/brightness.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

enum class DeviceSleepModeStatus {
  DEFAULT,
  OFFROAD,
};

class DevicePanelSP : public DevicePanel {
  Q_OBJECT

public:
  explicit DevicePanelSP(SettingsWindowSP *parent = 0);
  void showEvent(QShowEvent *event) override;
  void setOffroadMode();
  void updateState();
  void resetSettings();

private:
  std::map<QString, PushButtonSP*> buttons;
  PushButtonSP *offroadBtn;
  MaxTimeOffroad *maxTimeOffroad;
  ButtonParamControlSP *toggleDeviceBootMode;
  Brightness *brightness;
  OptionControlSP *interactivityTimeout;

  const QString alwaysOffroadStyle = R"UI0(
    PushButtonSP {)UI0"
#ifdef RK_BUILD
R"UI1(      border-radius: 7px;
      font-size: 18px;
      font-weight: 166;
      height: 55px;
      padding: 0 9px 0 9px;)UI1"
#else
R"UI2(      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;)UI2"
#endif
R"UI3(      color: #FFFFFF;
      background-color: #393939;
    }
    PushButtonSP:pressed {
      background-color: #4A4A4A;
    }
  )UI3";

  const QString autoOffroadStyle = R"UI4(
    PushButtonSP {)UI4"
#ifdef RK_BUILD
R"UI5(      border-radius: 7px;
      font-size: 18px;
      font-weight: 166;
      height: 55px;
      padding: 0 9px 0 9px;)UI5"
#else
R"UI6(      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;)UI6"
#endif
R"UI7(      color: #FFFFFF;
      background-color: #E22C2C;
    }
    PushButtonSP:pressed {
      background-color: #FF2424;
    }
  )UI7";

  const QString rebootButtonStyle = R"UI8(
    PushButtonSP {)UI8"
#ifdef RK_BUILD
R"UI9(      border-radius: 7px;
      font-size: 18px;
      font-weight: 166;
      height: 55px;
      padding: 0 9px 0 9px;)UI9"
#else
R"UI10(      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;)UI10"
#endif
R"UI11(      color: #FFFFFF;
      background-color: #393939;
    }
    PushButtonSP:pressed {
      background-color: #4A4A4A;
    }
  )UI11";

  const QString powerOffButtonStyle = R"UI12(
    PushButtonSP {)UI12"
#ifdef RK_BUILD
R"UI13(      border-radius: 7px;
      font-size: 18px;
      font-weight: 166;
      height: 55px;
      padding: 0 9px 0 9px;)UI13"
#else
R"UI14(      border-radius: 20px;
      font-size: 50px;
      font-weight: 450;
      height: 150px;
      padding: 0 25px 0 25px;)UI14"
#endif
R"UI15(      color: #FFFFFF;
      background-color: #E22C2C;
    }
    PushButtonSP:pressed {
      background-color: #FF2424;
    }
  )UI15";

  static QString deviceSleepModeDescription(DeviceSleepModeStatus status = DeviceSleepModeStatus::DEFAULT) {
    QString def_str = tr("⁍ Default: Device will boot/wake-up normally & will be ready to engage.");
    QString offrd_str = tr("⁍ Offroad: Device will be in Always Offroad mode after boot/wake-up.");

    if (status == DeviceSleepModeStatus::DEFAULT) {
      def_str = "<font color='white'><b>" + def_str + "</b></font>";
    } else if (status == DeviceSleepModeStatus::OFFROAD) {
      offrd_str = "<font color='white'><b>" + offrd_str + "</b></font>";
    }

    return QString("%1<br><br>%2<br>%3")
             .arg(tr("Controls state of the device after boot/sleep."))
             .arg(def_str)
             .arg(offrd_str);
  }
};
