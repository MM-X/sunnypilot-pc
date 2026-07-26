import os

from parameterized import parameterized

from openpilot.selfdrive.test.process_replay.regen import regen_segment
from openpilot.selfdrive.test.process_replay.process_replay import check_openpilot_enabled
from openpilot.tools.lib.local_route import local_route_path
from openpilot.tools.lib.openpilotci import get_url
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader

# Local mode: regress against the vendored demo route in tools/replay/data
# (no driver cam locally, so fcamera is reused as dummy dcam, same as regen).
_DEMO_ROUTE = "a2a0ccea32023010|00000004--9a1ce93c08"
if os.getenv("LOCAL_ROUTE_DIR"):
  TESTED_SEGMENTS = [("VW_DEMO", f"{_DEMO_ROUTE}--0")]
else:
  TESTED_SEGMENTS = [
    ("PRIUS_C2", "0982d79ebb0de295|2021-01-04--17-13-21--13"), # TOYOTA.TOYOTA_PRIUS:     NEO, pandaStateDEPRECATED, no peripheralState, sensorEventsDEPRECATED
    # Enable these once regen on CI becomes faster or use them for different tests running controlsd in isolation
    # ("MAZDA_C3", "bd6a637565e91581|2021-10-30--15-14-53--4"),  # MAZDA.CX9_2021:        TICI, incomplete managerState
    # ("FORD_C3", "54827bf84c38b14f|2023-01-26--21-59-07--4"),   # FORD.FORD_BRONCO_SPORT_MK1: TICI
  ]


def _src(route, sidx, fn):
  return local_route_path(route, str(sidx), fn) or get_url(route, sidx, fn)


def ci_setup_data_readers(route, sidx):
  local_rlog = local_route_path(route, str(sidx), "rlog.zst")
  lr = LogReader(local_rlog or get_url(route, sidx, "rlog.bz2"))
  frs = {
    'roadCameraState': FrameReader(_src(route, sidx, "fcamera.hevc")),
    'driverCameraState': FrameReader(_src(route, sidx, "fcamera.hevc")),
  }
  if next((True for m in lr if m.which() == "wideRoadCameraState"), False):
    frs["wideRoadCameraState"] = FrameReader(_src(route, sidx, "ecamera.hevc"))

  return lr, frs


class TestRegen:
  @parameterized.expand(TESTED_SEGMENTS)
  def test_engaged(self, case_name, segment):
    route, sidx = segment.rsplit("--", 1)
    lr, frs = ci_setup_data_readers(route, sidx)
    output_logs = regen_segment(lr, frs, disable_tqdm=True)

    engaged = check_openpilot_enabled(output_logs)
    assert engaged, f"openpilot not engaged in {case_name}"
