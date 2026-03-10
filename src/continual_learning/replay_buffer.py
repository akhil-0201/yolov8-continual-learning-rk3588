"""Experience replay buffer for preventing catastrophic forgetting.

Stores a small subset of previously seen samples and mixes them into
every training batch. The reservoir sampling strategy ensures each past
sample has equal probability of surviving, preventing class imbalance.

Why replay on RK3588:
  - Pure EWC alone can fail when the distribution shift is large.
  - Storing 500 image paths (not raw tensors) costs ~40 KB on disk.
  - During training we load images on-the-fly, adding only ~4 images/batch overhead.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer with multiple eviction strategies.

    Supported strategies:
      - ``reservoir``      : Reservoir sampling (uniform random over all seen)
      - ``fifo``           : First-in-first-out (most recent N samples)
      - ``class_balanced`` : Keep equal samples per class
    """

    def __init__(
        self,
        capacity: int = 500,
        strategy: str = "reservoir",
    ) -> None:
        if strategy not in {"reservoir", "fifo", "class_balanced"}:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.capacity = capacity
        self.strategy = strategy

        # Internal storage: list of (image_path, label_path, class_id) tuples
        self._buffer: List[Tuple[str, str, int]] = []
        self._class_buckets: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        self._total_seen = 0  # for reservoir sampling probability

    # ------------------------------------------------------------------
    # Adding samples
    # ------------------------------------------------------------------

    def add(
        self,
        image_path: str,
        label_path: str,
        class_id: int = -1,
    ) -> None:
        """Add a new sample to the buffer.

        Args:
            image_path: Absolute or relative path to the image file.
            label_path: Path to corresponding YOLO-format label (.txt).
            class_id:   Primary class in this image (-1 if unknown).
        """
        self._total_seen += 1
        entry = (image_path, label_path, class_id)

        if self.strategy == "reservoir":
            self._reservoir_add(entry)
        elif self.strategy == "fifo":
            self._fifo_add(entry)
        elif self.strategy == "class_balanced":
            self._balanced_add(entry)

    def add_batch(
        self,
        image_paths: List[str],
        label_paths: List[str],
        class_ids: Optional[List[int]] = None,
    ) -> None:
        """Add multiple samples at once."""
        if class_ids is None:
            class_ids = [-1] * len(image_paths)
        for img, lbl, cls in zip(image_paths, label_paths, class_ids):
            self.add(img, lbl, cls)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> List[Tuple[str, str, int]]:
        """Sample n entries from the buffer (with replacement if needed).

        Args:
            n: Number of samples requested.

        Returns:
            List of (image_path, label_path, class_id) tuples.
        """
        if not self._buffer:
            return []
        n = min(n, len(self._buffer))
        return random.sample(self._buffer, n)

    def get_replay_paths(
        self, n: int
    ) -> Tuple[List[str], List[str]]:
        """Return (image_paths, label_paths) for n replay samples."""
        samples = self.sample(n)
        imgs = [s[0] for s in samples]
        lbls = [s[1] for s in samples]
        return imgs, lbls

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist buffer state to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "buffer": self._buffer,
                    "class_buckets": dict(self._class_buckets),
                    "total_seen": self._total_seen,
                    "capacity": self.capacity,
                    "strategy": self.strategy,
                },
                f,
            )
        logger.info("[ReplayBuffer] Saved %d entries to %s", len(self._buffer), path)

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        """Load buffer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        buf = cls(capacity=data["capacity"], strategy=data["strategy"])
        buf._buffer = data["buffer"]
        buf._class_buckets = defaultdict(list, data["class_buckets"])
        buf._total_seen = data["total_seen"]
        logger.info(
            "[ReplayBuffer] Loaded %d entries from %s", len(buf._buffer), path
        )
        return buf

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    # ------------------------------------------------------------------
    # Internal strategies
    # ------------------------------------------------------------------

    def _reservoir_add(self, entry: Tuple[str, str, int]) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(entry)
        else:
            # Replace a random existing entry with probability capacity/total_seen
            idx = random.randint(0, self._total_seen - 1)
            if idx < self.capacity:
                self._buffer[idx] = entry

    def _fifo_add(self, entry: Tuple[str, str, int]) -> None:
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(entry)

    def _balanced_add(self, entry: Tuple[str, str, int]) -> None:
        img, lbl, cls = entry
        per_class = max(1, self.capacity // max(len(self._class_buckets), 1))
        bucket = self._class_buckets[cls]
        if len(bucket) >= per_class:
            bucket.pop(0)
        bucket.append((img, lbl))
        # Rebuild flat buffer
        self._buffer = [
            (img, lbl, cid)
            for cid, bkt in self._class_buckets.items()
            for img, lbl in bkt
        ]
