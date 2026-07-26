#!/usr/bin/env python3
"""Local-first route data resolver.

Returns a local filesystem path for a route segment's file when LOCAL_ROUTE_DIR
points at a directory laid out like loggerd's realdata (one subdir per segment:
`<dongle_id>|<log_id>--<sidx>/<fn>`). Falls back to None so callers can use the
remote source (CI keeps using Azure). Used by process_replay / model_replay to
run regression offline against vendored routes (e.g. tools/replay/data).
"""
import os

LOCAL_ROUTE_DIR = os.getenv("LOCAL_ROUTE_DIR", "")


def local_route_path(route: str, sidx: int | str, fn: str) -> str | None:
  """Return local path for `<route>--<sidx>/<fn>` if it exists, else None.

  Tries two on-disk layouts: the full route name, and the log-id-only form
  (the part after `|`, which is how tools/replay/data lays segments out).
  """
  if not LOCAL_ROUTE_DIR:
    return None
  candidates = [
    os.path.join(LOCAL_ROUTE_DIR, f"{route}--{sidx}", fn),
    os.path.join(LOCAL_ROUTE_DIR, f"{route.split('|')[-1]}--{sidx}", fn),
  ]
  for p in candidates:
    if os.path.exists(p):
      return p
  return None
