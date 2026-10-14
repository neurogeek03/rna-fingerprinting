"""Progress logging: every line carries wall-clock time, elapsed time since import, and memory.

Format::

    [2026-10-14 12:34:56-0400 | +00:05:12 | rss/peak 29/58 GB] message

Lines are flushed immediately so Slurm ``.out`` files show progress in real time.
"""

from __future__ import annotations

import time
from datetime import datetime

_T0 = time.monotonic()


def rss_gb():
    """Current and peak resident memory of this process, in GB (Linux ``/proc``)."""
    cur = peak = float("nan")
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1e6
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1e6
    return cur, peak


def log(msg: str):
    """Print ``msg`` with a timestamp / elapsed / memory prefix, flushed."""
    elapsed = int(time.monotonic() - _T0)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    cur, peak = rss_gb()
    # Explicit UTC offset: containers often lack the host timezone (TZ is passed by the run scripts).
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{stamp} | +{h:02d}:{m:02d}:{s:02d} | rss/peak {cur:.0f}/{peak:.0f} GB] {msg}", flush=True)
