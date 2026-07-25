#!/usr/bin/env bash
# Replay a logged route through the full openpilot stack.
# Launches manager.py (background) then feeds messages via the replay binary.
#
# Usage:
#   ./tools/replay/replay.sh <route> [replay options]
#   ./tools/replay/replay.sh --demo
#   ./tools/replay/replay.sh 'a2a0ccea32023010|2023-07-27--13-01-19' --data_dir=./tools/replay/data
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# runtime env (venv if present, then launch_env.sh)
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi
source "$REPO_ROOT/launch_env.sh"

# FILLBACK present => current test config: no pandad/camerad, lenient safety;
# replay itself provides roadCameraState/can/pandaStates. SKIP_FW_QUERY skips
# the slow panda firmware probe (no panda connected in replay). REPLAY makes
# selfdrived drop camera/gps liveness checks.
export SKIP_FW_QUERY=1
export FINGERPRINT=VOLKSWAGEN_GOLF_MK7
# export REPLAY=1
export FILLBACK=1

if [ $# -eq 0 ]; then
  cat >&2 <<EOF
Usage: $(basename "$0") <route> [replay options]
  $(basename "$0") --demo
  $(basename "$0") 'a2a0ccea32023010|2023-07-27--13-01-19' --data_dir=./tools/replay/data
EOF
  exit 1
fi

# --demo => the bundled local route (no remote auth needed)
if [ "${1:-}" = "--demo" ]; then
  shift
  set -- 'a2a0ccea32023010|00000004--9a1ce93c08' --data_dir=./tools/replay/data "$@"
fi

# ponytail: fixed allow-list covers what replay must feed; everything else
# (modelV2, controlsState, ...) is produced by the running stack itself.
ALLOW="roadEncodeIdx,roadCameraState,can,pandaStates,peripheralState"

# manager in background, logs redirected so the terminal stays clean for
# replay's keyboard interaction (timeline seek/pause via keys).
python3 system/manager/manager.py > /tmp/replay_manager.log 2>&1 &
MANAGER_PID=$!
cleanup() { kill "$MANAGER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# ponytail: fixed 5s warmup; poll msgq sockets if manager startup latency varies
sleep 5

"$SCRIPT_DIR/replay" --allow "$ALLOW" "$@"
