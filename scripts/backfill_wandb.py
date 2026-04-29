"""Back-fill `test_avg/mae_surf_p` and per-split test surf_p in each W&B
run's summary using the NaN-safe re-eval results. Helpful for the dashboard.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import wandb


def backfill():
    api = wandb.Api()
    runs = api.runs(
        path=f"{os.environ.get('WANDB_ENTITY','wandb-applied-ai-team')}/{os.environ.get('WANDB_PROJECT','senpai-v1-ml-intern')}",
        filters={"group": "mlintern-pai2-r1"},
    )
    base = Path("research/reeval")
    name_to_path = {p.stem: p for p in base.glob("*.json")}

    n_updated = 0
    for run in runs:
        run_short = run.name.split("/")[-1] if run.name else None
        if run_short is None:
            continue
        candidates = [run_short, run_short.replace("-bf16", ""), run_short.replace("-default", "")]
        match = next((c for c in candidates if c in name_to_path), None)
        if match is None:
            continue
        d = json.loads(name_to_path[match].read_text())
        tavg = d["test_avg"].get("avg/mae_surf_p")
        per = {k: v["mae_surf_p"] for k, v in d["test"].items()}
        non_finite = {k: v.get("n_nonfinite_y_samples", 0) for k, v in d["test"].items()}
        if tavg is None:
            continue
        # Update only if missing or NaN
        existing = run.summary.get("test_avg/mae_surf_p")
        if existing is not None and existing == existing:  # not NaN
            print(f"{run_short}: existing test_avg/mae_surf_p={existing} — skipping")
            continue
        # Update summary
        for k, v in per.items():
            run.summary[f"test/{k}/mae_surf_p_safe"] = v
        run.summary["test_avg/mae_surf_p_safe"] = tavg
        run.summary["test_avg/mae_surf_p"] = tavg
        run.summary["test_n_nonfinite_y_samples"] = non_finite
        run.update()
        n_updated += 1
        print(f"{run_short}: updated test_avg/mae_surf_p = {tavg:.4f}")
    print(f"\nUpdated {n_updated} runs")


if __name__ == "__main__":
    backfill()
