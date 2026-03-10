"""Elastic Weight Consolidation (EWC) for YOLOv8 continual learning.

EWC penalises changes to weights that were important for the previous task
by adding a quadratic regularisation term to the loss, weighted by the
Fisher Information Matrix diagonal.

Reference: Kirkpatrick et al., 2017 — https://arxiv.org/abs/1612.00796
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EWC:
    """Elastic Weight Consolidation regulariser.

    Usage::

        ewc = EWC(model, lambda_ewc=5000)
        ewc.compute_fisher(dataloader, device)
        # ... train model on new task ...
        loss = criterion(output, target) + ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0) -> None:
        """Initialise EWC.

        Args:
            model:       The YOLOv8 model AFTER training on the anchor task.
            lambda_ewc:  Regularisation strength. Higher = less forgetting,
                         lower = more plasticity. 5000 is a practical default
                         for YOLOv8n on edge devices.
        """
        self.lambda_ewc = lambda_ewc
        # Snapshot of anchor-task parameters (theta*)
        self.anchor_params: dict[str, torch.Tensor] = {}
        # Diagonal Fisher Information Matrix estimates
        self.fisher: dict[str, torch.Tensor] = {}
        # Flag: True once Fisher has been estimated
        self._fisher_computed = False

        # Store anchor parameters immediately
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.anchor_params[name] = param.data.clone().detach()

    # ------------------------------------------------------------------
    # Fisher estimation
    # ------------------------------------------------------------------

    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
        n_samples: int = 200,
    ) -> None:
        """Estimate the diagonal Fisher Information Matrix.

        We use the empirical Fisher approximation: accumulate squared
        gradients of the log-likelihood over a sample of anchor-task data.

        Args:
            model:      The trained YOLOv8 model.
            dataloader: DataLoader yielding anchor-task samples.
            device:     Torch device string.
            n_samples:  Number of samples to average over (more = better
                        estimate; 200 is sufficient for RK3588 memory budget).
        """
        model.eval()
        model.to(device)

        # Initialise Fisher accumulators to zero
        fisher_acc: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        total_samples = 0
        for batch in tqdm(dataloader, desc="[EWC] Estimating Fisher"):
            if total_samples >= n_samples:
                break

            # YOLOv8 training batches are dicts with keys: img, cls, bboxes ...
            images = batch["img"].to(device).float() / 255.0
            batch_size = images.size(0)

            model.zero_grad()
            # Run forward pass; YOLOv8 returns (preds, loss) in train mode
            try:
                loss, _ = model(batch)  # ultralytics train-mode call
            except Exception:
                # Fallback: just run detection head
                preds = model(images)
                # Use sum of predictions as a proxy loss for Fisher
                if isinstance(preds, (list, tuple)):
                    proxy = sum(p.sum() for p in preds if isinstance(p, torch.Tensor))
                else:
                    proxy = preds.sum()
                proxy.backward()
            else:
                loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_acc[name] += param.grad.detach().pow(2) * batch_size

            total_samples += batch_size

        # Normalise by number of samples
        for name in fisher_acc:
            fisher_acc[name] /= max(total_samples, 1)

        self.fisher = fisher_acc
        self._fisher_computed = True
        logger.info(
            "[EWC] Fisher matrix estimated over %d samples.", total_samples
        )

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC regularisation penalty.

        penalty = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

        Args:
            model: Current model (being trained on new task).

        Returns:
            Scalar tensor; add this to your task loss before back-prop.
        """
        if not self._fisher_computed:
            logger.warning(
                "[EWC] Fisher not computed yet — returning zero penalty."
            )
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, requires_grad=True)
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.anchor_params:
                diff = param - self.anchor_params[name].to(param.device)
                penalty = penalty + (
                    self.fisher[name].to(param.device) * diff.pow(2)
                ).sum()

        return (self.lambda_ewc / 2.0) * penalty

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save Fisher matrix and anchor params to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "fisher": self.fisher,
                "anchor_params": self.anchor_params,
                "lambda_ewc": self.lambda_ewc,
            },
            path,
        )
        logger.info("[EWC] Saved to %s", path)

    @classmethod
    def load(cls, path: str, model: nn.Module) -> "EWC":
        """Load EWC state from disk.

        Args:
            path:  Path to saved .pt file.
            model: The model whose lambda will default the EWC instance.
        """
        data = torch.load(path, map_location="cpu")
        instance = cls(model, lambda_ewc=data["lambda_ewc"])
        instance.fisher = data["fisher"]
        instance.anchor_params = data["anchor_params"]
        instance._fisher_computed = True
        logger.info("[EWC] Loaded from %s", path)
        return instance
