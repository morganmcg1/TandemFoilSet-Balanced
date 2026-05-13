"""Lookahead optimizer wrapper (Zhang et al., 2019; arXiv:1907.08610).

Maintains slow weights alongside an inner optimizer's fast weights. Every k
inner steps, blends slow toward fast and resets fast to the new slow point:

    slow += alpha * (fast - slow)
    fast.data.copy_(slow)

All parameter writes are in-place ``.data`` mutations. The slow buffer is
stored Python-side on the wrapper, never registered as an ``nn.Parameter``,
so it is invisible to ``torch.compile`` graph capture and the sync does not
trigger Dynamo recompilation.

``param_groups`` is aliased to the inner optimizer's list, so any
``torch.optim.lr_scheduler`` mutating ``optimizer.param_groups[i]['lr']``
propagates correctly. ``GradScaler.unscale_`` / ``GradScaler.step`` work
transparently because they consume ``optimizer.param_groups`` and ``.grad``
tensors that the inner optimizer also sees.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    # Inherits Optimizer purely for ``isinstance`` compatibility with
    # ``torch.optim.lr_scheduler.*`` (which type-checks its argument). We do
    # not call ``super().__init__`` — that would require synthesizing params
    # and defaults; instead we forward the relevant attributes/methods to the
    # inner optimizer, and store slow buffers in ``self.state`` ourselves.
    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        if k < 1:
            raise ValueError(f"Lookahead k must be >= 1, got {k}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Lookahead alpha must be in (0, 1], got {alpha}")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step_counter = 0
        self.state: dict = {}
        # Slow-fast drift diagnostic, accumulated across sync events
        # (reset at epoch boundaries via reset_diag()).
        self._sync_count = 0
        self._sync_drift_norm_sum = 0.0
        self._sync_slow_norm_sum = 0.0
        self._sync_ratio_max = 0.0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.state[p] = {"slow_buffer": p.data.detach().clone()}

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        return {
            "k": self.k,
            "alpha": self.alpha,
            "_step_counter": self._step_counter,
            "inner": self.optimizer.state_dict(),
            "slow_buffers": [self.state[p]["slow_buffer"].clone() for p in params],
        }

    def load_state_dict(self, state_dict):
        self.k = state_dict["k"]
        self.alpha = state_dict["alpha"]
        self._step_counter = state_dict["_step_counter"]
        self.optimizer.load_state_dict(state_dict["inner"])
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        for p, sb in zip(params, state_dict["slow_buffers"]):
            self.state[p] = {"slow_buffer": sb.to(device=p.device, dtype=p.dtype)}

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_counter += 1
        if self._step_counter >= self.k:
            self._step_counter = 0
            self._sync()
        return loss

    @torch.no_grad()
    def _sync(self):
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        if not params:
            return
        dev = params[0].device
        drift_sq = torch.zeros((), device=dev)
        slow_sq = torch.zeros((), device=dev)
        for p in params:
            slow = self.state[p]["slow_buffer"]
            diff = p.data - slow
            drift_sq += diff.pow(2).sum()
            slow_sq += slow.pow(2).sum()
            # slow += alpha * (fast - slow)
            slow.add_(diff, alpha=self.alpha)
            # fast.data = slow (in-place; no rebinding, no recompile)
            p.data.copy_(slow)
        drift_norm = float(drift_sq.sqrt().item())
        slow_norm = float(slow_sq.sqrt().item())
        self._sync_count += 1
        self._sync_drift_norm_sum += drift_norm
        self._sync_slow_norm_sum += slow_norm
        ratio = drift_norm / max(slow_norm, 1e-12)
        if ratio > self._sync_ratio_max:
            self._sync_ratio_max = ratio

    def reset_diag(self):
        self._sync_count = 0
        self._sync_drift_norm_sum = 0.0
        self._sync_slow_norm_sum = 0.0
        self._sync_ratio_max = 0.0

    def get_diag(self) -> dict:
        n = max(self._sync_count, 1)
        mean_drift = self._sync_drift_norm_sum / n
        mean_slow = self._sync_slow_norm_sum / n
        return {
            "sync_count": self._sync_count,
            "drift_norm_mean": mean_drift,
            "slow_norm_mean": mean_slow,
            "drift_slow_ratio_mean": mean_drift / max(mean_slow, 1e-12),
            "drift_slow_ratio_max": self._sync_ratio_max,
        }
