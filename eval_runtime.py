"""Runtime contracts injected by the Senpai evaluation harness."""

from __future__ import annotations

import math
import os
import threading
from typing import Any


TIMEOUT_EXIT_CODE = 124


def timeout_minutes_from_env(default: float = 30.0) -> float:
    """Return the positive, finite wall-clock limit for this process."""
    raw_value = os.environ.get("SENPAI_TIMEOUT_MINUTES", str(default))
    minutes = float(raw_value)
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("SENPAI_TIMEOUT_MINUTES must be a positive finite number")
    return minutes


def resolve_wandb_group(cli_group: str | None) -> str | None:
    """Use the harness group when present so CLI args cannot split an eval."""
    env_group = os.environ.get("WANDB_RUN_GROUP")
    if env_group is None:
        return cli_group
    if not env_group.strip():
        raise ValueError("WANDB_RUN_GROUP must not be empty")
    return env_group


def resolve_trial_seed(cli_seed: int) -> int:
    """Use the harness seed when present so every outer trial is reproducible."""
    seed = int(os.environ.get("SENPAI_TRIAL_SEED", str(cli_seed)))
    if seed < 0:
        raise ValueError("SENPAI_TRIAL_SEED must be non-negative")
    return seed


def apply_torch_seed(torch_module: Any, seed: int) -> None:
    """Seed CPU and CUDA RNGs with the authoritative trial seed."""
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def is_complete_test_result(
    split_metrics: dict[str, dict[str, float]] | None,
    averages: dict[str, float] | None,
    expected_splits: list[str],
) -> bool:
    """Confirm exact split coverage, finite metrics, and the equal primary mean."""
    if split_metrics is None or averages is None:
        return False
    if set(split_metrics) != set(expected_splits):
        return False
    values = [value for metrics in split_metrics.values() for value in metrics.values()]
    values.extend(averages.values())
    if any(not math.isfinite(value) or value < 0 for value in values):
        return False
    expected_primary = sum(
        split_metrics[name]["mae_surf_p"] for name in expected_splits
    ) / len(expected_splits)
    return math.isclose(
        averages["avg/mae_surf_p"], expected_primary, rel_tol=1e-9, abs_tol=1e-9
    )


def arm_hard_timeout(minutes: float) -> threading.Timer:
    """Hard-exit this process at the wall-clock limit, even inside GPU work."""
    seconds = minutes * 60.0

    def terminate() -> None:
        message = f"SENPAI_TIMEOUT_MINUTES={minutes:g} expired; terminating process\n"
        try:
            os.write(2, message.encode())
        finally:
            os._exit(TIMEOUT_EXIT_CODE)

    timer = threading.Timer(seconds, terminate)
    timer.daemon = True
    timer.start()
    return timer
