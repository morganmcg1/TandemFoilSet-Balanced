"""Local Lion optimizer override with LR-scaled gradient-noise injection.

Copied from ``lion_pytorch.lion_pytorch.Lion`` (BSD-2-Clause, Phil Wang) and
extended with a single new lever:

  ``noise_sigma`` — base standard deviation of Gaussian perturbation added to
  the post-sign update vector. Effective noise scales with the current
  learning rate: ``sigma_effective = noise_sigma * (current_lr / peak_lr)``.

This implements gradient-noise injection (Neelakantan et al. 2017) /
Langevin-style perturbation on Lion's sign-momentum step. LR-scaling matches
the cosine schedule, so noise decays toward zero in the converged tail and
does not re-perturb the final solution.

When ``noise_sigma == 0.0`` the math reduces exactly to upstream Lion
(``torch.randn_like`` is never called).
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.optim.optimizer import Optimizer


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, noise_sigma, peak_lr):
    # stepweight decay
    p.data.mul_(1.0 - lr * wd)

    # sign-momentum update (Lion baseline)
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1.0 - beta1).sign_()

    # LR-scaled Gaussian noise on the post-sign update vector.
    # noise_scale -> 0 as lr -> 0 in the cosine tail (no re-perturbation of
    # the converged solution; see PR #2326 closure / Finding #41).
    if noise_sigma > 0.0:
        noise_scale = noise_sigma * (lr / peak_lr)
        update = update + torch.randn_like(update) * noise_scale

    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)


class Lion(Optimizer):
    """Lion optimizer with optional LR-scaled gradient-noise injection.

    Identical to ``lion_pytorch.Lion`` when ``noise_sigma == 0.0``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        noise_sigma: float = 0.0,
        peak_lr: float | None = None,
    ):
        assert lr > 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        assert noise_sigma >= 0.0

        self._init_lr = lr
        # peak_lr defaults to the initial lr (cosine schedule peak == cfg.lr)
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            noise_sigma=noise_sigma,
            peak_lr=peak_lr if peak_lr is not None else lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            noise_sigma = group["noise_sigma"]
            peak_lr = group["peak_lr"]

            for p in filter(lambda p: p.grad is not None, group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                update_fn(
                    p,
                    p.grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    noise_sigma,
                    peak_lr,
                )

        return loss
