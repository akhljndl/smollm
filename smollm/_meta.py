"""Minimal eval-side metadata stub.

Attaches a checkpoint path and UTC timestamp to every evaluation output,
providing enough information to identify which model checkpoint produced
a given analysis result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_eval_meta(checkpoint: str | Path) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
    }
