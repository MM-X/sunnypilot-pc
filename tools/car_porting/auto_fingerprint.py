#!/usr/bin/env python3

import argparse
from collections import defaultdict
from opendbc.car.debug.format_fingerprints import format_brand_fw_versions

from opendbc.car.fingerprints import MIGRATION
from opendbc.car.fw_versions import MODEL_TO_BRAND, match_fw_to_car
from openpilot.tools.lib.logreader import LogReader, ReadMode
from openpilot.tools.lib.route import Route

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto fingerprint from a route")
  parser.add_argument("route", help="The route name to use")
  parser.add_argument("data_dir", help="local directory with routes")
  parser.add_argument("segments", help="第几段数据", default=None, nargs="?")
  parser.add_argument("platform", help="The platform, or leave empty to auto-determine using fuzzy", default=None, nargs="?")
  args = parser.parse_args()
  r = Route(args.route, data_dir=args.data_dir)
  lr = LogReader(r.log_paths()[int(args.segments)] if args.segments is not None else r.log_paths(), default_mode=ReadMode.QLOG)
  CP = lr.first("carParams")
  assert CP is not None, "No carParams in route"

  carPlatform = MIGRATION.get(CP.carFingerprint, CP.carFingerprint)

  if args.platform is not None:
    platform = args.platform
  elif carPlatform != "MOCK":
    platform = carPlatform
  else:
    _, matches = match_fw_to_car(CP.carFw, CP.carVin, log=False)
    assert len(matches) == 1, f"Unable to auto-determine platform, matches: {matches}"
    platform = list(matches)[0]

  print("Attempting to add fw version for:", platform)

  fw_versions: dict[str, dict[tuple, list[bytes]]] = defaultdict(lambda: defaultdict(list))
  brand = MODEL_TO_BRAND[platform]
  print(brand)

  for fw in CP.carFw:
    print(fw)
    if fw.brand == brand and not fw.logging:
      addr = fw.address
      subAddr = None if fw.subAddress == 0 else fw.subAddress
      key = (fw.ecu.raw, addr, subAddr)

      fw_versions[platform][key].append(fw.fwVersion)
  format_brand_fw_versions(brand, fw_versions)
