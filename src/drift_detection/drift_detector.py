"""Confidence-based drift detector for YOLOv8 on-device monitoring.

Detects distribution shift by tracking a rolling window of inference
confidence scores. A drop below a configured threshold signals that the
model may have drifted relative to new deployment data.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """Describes a detected drift event."""
    timestamp: float
    rolling_avg_confidence: float
    threshold: float
    window_size: int
    num_detections: int


class DriftDetector:
    """Monitors inference confidence to detect data/concept drift.

    How it works:
      1. Each time the model runs inference, call ``update(confidences)``.
      2. The detector maintains a rolling window of all confidence scores.
      3. If the rolling mean drops below ``threshold``, drift is flagged.

    Example::

        detector = DriftDetector(threshold=0.45, window_size=50)
        for frame in video_stream:
            results = model(frame)
            confs = [box.conf.item() for box in results[0].boxes]
            if detector.update(confs):
                print("Drift detected! Triggering retraining.")
    """

    def __init__(
        self,
        threshold: float = 0.45,
        window_size: int = 50,
        min_detections: int = 10,
    ) -> None:
        """Initialise the drift detector.

        Args:
            threshold:       Minimum acceptable average confidence.
                             Calibrate by running on known-good data first.
            window_size:     Number of recent confidence scores to average.
            min_detections:  Don't evaluate drift until at least this many
                             scores have been collected (avoids false alarms
                             at startup).
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_detections = min_detections

        self._window: deque[float] = deque(maxlen=window_size)
        self._total_updates = 0
        self.drift_events: List[DriftEvent] = []
        self._last_drift_at: int = -1  # update index of last drift

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, confidences: List[float], timestamp: float = 0.0) -> bool:
        """Record new confidence scores and check for drift.

        Args:
            confidences: List of per-detection confidence values from one
                         inference call. Pass empty list for frames with no
                         detections (they don't count toward the window).
            timestamp:   Unix timestamp of this inference (optional).

        Returns:
            True if drift is currently detected, False otherwise.
        """
        # Only add frames that have at least one detection
        if confidences:
            for c in confidences:
                self._window.append(float(c))
            self._total_updates += len(confidences)

        return self._is_drifting(timestamp)

    def is_drifting(self) -> bool:
        """Check current drift status without updating the window."""
        return self._is_drifting()

    def rolling_mean(self) -> float:
        """Return the current rolling average confidence."""
        if not self._window:
            return 1.0
        return float(np.mean(list(self._window)))

    def reset(self) -> None:
        """Clear the window (call after retraining completes)."""
        self._window.clear()
        self._total_updates = 0
        logger.info("[DriftDetector] Window reset after retraining.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dictionary summary of current detector state."""
        return {
            "rolling_mean_confidence": self.rolling_mean(),
            "threshold": self.threshold,
            "window_fill": len(self._window),
            "window_size": self.window_size,
            "total_scores_seen": self._total_updates,
            "drift_events_logged": len(self.drift_events),
            "is_drifting": self.is_drifting(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_drifting(self, timestamp: float = 0.0) -> bool:
        if self._total_updates < self.min_detections:
            return False
        if len(self._window) < min(self.min_detections, self.window_size):
            return False

        mean_conf = self.rolling_mean()
        drifting = mean_conf < self.threshold

        if drifting:
            event = DriftEvent(
                timestamp=timestamp,
                rolling_avg_confidence=mean_conf,
                threshold=self.threshold,
                window_size=self.window_size,
                num_detections=self._total_updates,
            )
            # Only log a new event if we haven't already flagged this drift
            if self._total_updates != self._last_drift_at:
                self.drift_events.append(event)
                self._last_drift_at = self._total_updates
                logger.warning(
                    "[DriftDetector] DRIFT DETECTED — rolling confidence: %.4f < threshold: %.4f",
                    mean_conf,
                    self.threshold,
                )

        return drifting
