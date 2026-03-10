"""Retraining trigger manager: combines confidence, data count, and schedule.

Each trigger operates independently. The manager fires whenever ANY enabled
trigger condition is satisfied. All triggers are configurable via YAML.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    CONFIDENCE_DROP = "confidence_drop"
    NEW_DATA_ACCUMULATED = "new_data_accumulated"
    SCHEDULED_INTERVAL = "scheduled_interval"
    MANUAL = "manual"


@dataclass
class TriggerEvent:
    reason: TriggerReason
    timestamp: float
    details: str = ""


class TriggerManager:
    """Manages all retraining trigger conditions.

    Usage::

        tm = TriggerManager(
            confidence_threshold=0.45,
            new_data_threshold=100,
            schedule_interval_hours=24,
        )
        # In your inference loop:
        if tm.check(avg_confidence=0.38, new_sample_count=45):
            trigger = tm.last_trigger
            print(f"Retrain triggered: {trigger.reason}")
    """

    def __init__(
        self,
        # Confidence trigger
        confidence_enabled: bool = True,
        confidence_threshold: float = 0.45,
        # Data accumulation trigger
        new_data_enabled: bool = True,
        new_data_threshold: int = 100,
        # Schedule trigger
        schedule_enabled: bool = True,
        schedule_interval_hours: float = 24.0,
    ) -> None:
        self.confidence_enabled = confidence_enabled
        self.confidence_threshold = confidence_threshold
        self.new_data_enabled = new_data_enabled
        self.new_data_threshold = new_data_threshold
        self.schedule_enabled = schedule_enabled
        self.schedule_interval_seconds = schedule_interval_hours * 3600.0

        self._last_retrain_time: float = time.time()
        self.last_trigger: Optional[TriggerEvent] = None
        self.trigger_history: list[TriggerEvent] = []

    # ------------------------------------------------------------------
    # Main check method
    # ------------------------------------------------------------------

    def check(
        self,
        avg_confidence: Optional[float] = None,
        new_sample_count: int = 0,
    ) -> bool:
        """Evaluate all enabled triggers.

        Args:
            avg_confidence:  Current rolling average inference confidence
                             (from DriftDetector). None to skip confidence check.
            new_sample_count: Number of new labeled samples accumulated
                              since last retraining.

        Returns:
            True if retraining should be triggered now.
        """
        now = time.time()

        # --- Trigger 1: Confidence drop ---
        if self.confidence_enabled and avg_confidence is not None:
            if avg_confidence < self.confidence_threshold:
                return self._fire(
                    TriggerReason.CONFIDENCE_DROP,
                    f"confidence={avg_confidence:.4f} < threshold={self.confidence_threshold}",
                )

        # --- Trigger 2: New data accumulation ---
        if self.new_data_enabled:
            if new_sample_count >= self.new_data_threshold:
                return self._fire(
                    TriggerReason.NEW_DATA_ACCUMULATED,
                    f"samples={new_sample_count} >= threshold={self.new_data_threshold}",
                )

        # --- Trigger 3: Scheduled interval ---
        if self.schedule_enabled:
            elapsed = now - self._last_retrain_time
            if elapsed >= self.schedule_interval_seconds:
                return self._fire(
                    TriggerReason.SCHEDULED_INTERVAL,
                    f"elapsed={elapsed/3600:.1f}h >= interval={self.schedule_interval_seconds/3600:.1f}h",
                )

        return False

    def acknowledge_retrain(self) -> None:
        """Call this after retraining completes to reset time-based trigger."""
        self._last_retrain_time = time.time()
        logger.info("[TriggerManager] Retrain acknowledged; schedule reset.")

    # ------------------------------------------------------------------
    # From config dict
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "TriggerManager":
        """Instantiate from the ``retraining_triggers`` section of pipeline_config.yaml."""
        c = cfg.get("confidence_drop", {})
        d = cfg.get("new_data_accumulation", {})
        s = cfg.get("scheduled_interval", {})
        return cls(
            confidence_enabled=c.get("enabled", True),
            confidence_threshold=c.get("threshold", 0.45),
            new_data_enabled=d.get("enabled", True),
            new_data_threshold=d.get("sample_count", 100),
            schedule_enabled=s.get("enabled", True),
            schedule_interval_hours=s.get("interval_hours", 24.0),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self, reason: TriggerReason, details: str) -> bool:
        event = TriggerEvent(reason=reason, timestamp=time.time(), details=details)
        self.last_trigger = event
        self.trigger_history.append(event)
        logger.warning(
            "[TriggerManager] RETRAIN TRIGGERED — reason=%s | %s",
            reason.value,
            details,
        )
        return True
