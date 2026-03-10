"""Edge data collector: manages incoming images and pseudo-labels on RK3588.

The collector watches an incoming directory, optionally applies pseudo-labeling
via the current model, and prepares data for the next retraining cycle.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class EdgeDataCollector:
    """Watches a directory for new images and builds labeled datasets.

    Directory layout after collection::

        data/incoming/          <- drop raw images here (or camera writes here)
        data/labeled/
            images/train/       <- images ready for training
            labels/train/       <- YOLO-format .txt labels
        data/replay/            <- managed by ReplayBuffer

    Usage::

        collector = EdgeDataCollector(
            incoming_dir="data/incoming",
            labeled_dir="data/labeled",
        )
        collector.scan()                          # find new images
        count = collector.new_sample_count        # check accumulation trigger
        imgs, lbls = collector.prepare_batch()   # get paths for training
        collector.commit()                        # move to labeled, clear incoming
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        incoming_dir: str = "data/incoming",
        labeled_dir: str = "data/labeled",
        model=None,                      # Optional YOLOv8 model for pseudo-labeling
        conf_threshold: float = 0.5,    # Min confidence to accept pseudo-label
        imgsz: int = 640,
    ) -> None:
        self.incoming_dir = Path(incoming_dir)
        self.labeled_dir = Path(labeled_dir)
        self.model = model
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        # Subdirs for labeled data
        self._img_train = self.labeled_dir / "images" / "train"
        self._lbl_train = self.labeled_dir / "labels" / "train"
        for d in [self.incoming_dir, self._img_train, self._lbl_train]:
            d.mkdir(parents=True, exist_ok=True)

        self._pending: List[Path] = []   # new images found by scan()
        self._committed_count = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def scan(self) -> int:
        """Scan incoming_dir for new images. Returns count found."""
        self._pending = [
            p for p in self.incoming_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXTS
        ]
        logger.info(
            "[EdgeCollector] Found %d pending images in %s",
            len(self._pending),
            self.incoming_dir,
        )
        return len(self._pending)

    @property
    def new_sample_count(self) -> int:
        """Number of pending images (trigger the 'new_data' trigger)."""
        return len(self._pending)

    def pseudo_label_all(
        self, model, conf: Optional[float] = None
    ) -> Tuple[List[str], List[str]]:
        """Generate YOLO-format pseudo-labels for all pending images.

        Args:
            model: Loaded ultralytics YOLO model.
            conf:  Confidence threshold override.

        Returns:
            (image_paths, label_paths) for accepted pseudo-labeled samples.
        """
        conf = conf or self.conf_threshold
        accepted_imgs, accepted_lbls = [], []

        for img_path in self._pending:
            results = model.predict(str(img_path), imgsz=self.imgsz, verbose=False)
            result = results[0]

            # Skip if no detections above threshold
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Filter by confidence
            mask = boxes.conf >= conf
            if not mask.any():
                continue

            # Write YOLO label file
            label_path = self._lbl_train / (img_path.stem + ".txt")
            h, w = result.orig_shape
            with open(label_path, "w") as f:
                for box in boxes[mask]:
                    cls = int(box.cls.item())
                    x, y, bw, bh = box.xywhn[0].tolist()
                    f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

            # Copy image to labeled dir
            dest_img = self._img_train / img_path.name
            shutil.copy2(img_path, dest_img)
            accepted_imgs.append(str(dest_img))
            accepted_lbls.append(str(label_path))

        logger.info(
            "[EdgeCollector] Pseudo-labeled %d / %d images (conf >= %.2f)",
            len(accepted_imgs),
            len(self._pending),
            conf,
        )
        return accepted_imgs, accepted_lbls

    def prepare_batch(self) -> Tuple[List[str], List[str]]:
        """Return all currently labeled image and label paths for training."""
        imgs = sorted(str(p) for p in self._img_train.glob("*") if p.suffix.lower() in self.SUPPORTED_EXTS)
        lbls = sorted(str(p) for p in self._lbl_train.glob("*.txt"))
        return imgs, lbls

    def commit(self) -> int:
        """Move processed incoming images out of the queue.

        Returns:
            Number of images committed.
        """
        count = len(self._pending)
        for img in self._pending:
            try:
                img.unlink()  # delete from incoming (already copied to labeled)
            except FileNotFoundError:
                pass
        self._committed_count += count
        self._pending.clear()
        logger.info("[EdgeCollector] Committed %d images.", count)
        return count

    def dataset_size(self) -> int:
        """Total images currently in the labeled training set."""
        return sum(1 for _ in self._img_train.glob("*") if _.suffix.lower() in self.SUPPORTED_EXTS)
