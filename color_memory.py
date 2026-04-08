"""
Persistent color memory — online learning from usage.

Every time Claude Vision API corrects the local pipeline's result, that
correction is stored as a (Oklab coords, color name) sample. On the next
identification, the memory is checked first: if a perceptually close prior
sample exists with enough confirmations, its label is returned immediately
without running the full pipeline.

This is incremental k-NN learning in Oklab space, with Claude acting as
the "teacher" that labels corrections, and the user's Y-key confirmations
acting as reinforcement.

Storage: ~/.colorblindcam/memory.json  (persists across sessions)
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

_log = logging.getLogger("color_memory")

_DB_PATH = Path.home() / ".colorblindcam" / "memory.json"

# Oklab distance below which two colors are considered "the same region"
# Oklab is perceptually uniform — 0.04 is roughly the just-noticeable difference
_MATCH_RADIUS = 0.04

# Minimum number of times a sample must be seen/confirmed before it's trusted
_MIN_COUNT = 2


class ColorMemory:
    def __init__(self) -> None:
        # Each sample: {"oklab": [L, a, b], "color": str, "count": int}
        self._samples: list[dict] = []
        self._tree: cKDTree | None = None
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, oklab: np.ndarray) -> str | None:
        """
        Return a stored color name if a sufficiently close, trusted sample
        exists. Returns None if no match or confidence too low.
        """
        if self._tree is None or not self._samples:
            return None
        dist, idx = self._tree.query(oklab)
        sample = self._samples[idx]
        if dist <= _MATCH_RADIUS and sample["count"] >= _MIN_COUNT:
            _log.debug("Memory hit (dist=%.4f, count=%d): %s", dist, sample["count"], sample["color"])
            return sample["color"]
        return None

    def add_correction(self, oklab: np.ndarray, color: str) -> None:
        """
        Store a Claude-derived correction. If a very close sample already
        exists with the same label, reinforce it (increment count).
        If the label conflicts, replace it — Claude is authoritative.
        """
        self._upsert(oklab, color, source="claude")
        self._maybe_save()

    def confirm(self, oklab: np.ndarray, color: str) -> None:
        """
        Reinforce a correct prediction (user pressed Y, or local + Claude agreed).
        """
        self._upsert(oklab, color, source="confirm")
        self._maybe_save()

    def sample_count(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _upsert(self, oklab: np.ndarray, color: str, source: str) -> None:
        close_radius = _MATCH_RADIUS * 0.5  # must be very close to merge

        if self._tree is not None and self._samples:
            dist, idx = self._tree.query(oklab)
            if dist <= close_radius:
                existing = self._samples[idx]
                if existing["color"] == color:
                    existing["count"] += 1
                    _log.debug("[%s] Reinforced '%s' (count=%d)", source, color, existing["count"])
                else:
                    # Conflicting label — Claude/confirm wins; reset count
                    _log.info("[%s] Overwriting '%s' → '%s' at dist=%.4f",
                              source, existing["color"], color, dist)
                    existing["color"] = color
                    existing["count"] = 1
                self._rebuild_tree()
                self._dirty = True
                return

        self._samples.append({"oklab": list(float(x) for x in oklab), "color": color, "count": 1})
        _log.info("[%s] New sample: '%s' (total=%d)", source, color, len(self._samples))
        self._rebuild_tree()
        self._dirty = True

    def _rebuild_tree(self) -> None:
        if self._samples:
            points = np.array([s["oklab"] for s in self._samples], dtype=np.float64)
            self._tree = cKDTree(points)
        else:
            self._tree = None

    def _maybe_save(self) -> None:
        if not self._dirty:
            return
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_DB_PATH, "w") as f:
                json.dump(self._samples, f, indent=2)
            self._dirty = False
        except Exception as e:
            _log.warning("Could not save color memory: %s", e)

    def _load(self) -> None:
        if not _DB_PATH.exists():
            return
        try:
            with open(_DB_PATH) as f:
                self._samples = json.load(f)
            self._rebuild_tree()
            _log.info("Loaded %d samples from memory", len(self._samples))
        except Exception as e:
            _log.warning("Could not load color memory (%s) — starting fresh", e)
            self._samples = []
