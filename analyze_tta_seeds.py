"""Aggregate TTA per-epoch checkpoint-selection results across seeds.

Reads `metrics.jsonl` from one-or-more run directories and produces a unified
report:
  * per-epoch checkpoint selection: did the no_tta and tta-criterion picks land
    on the same epoch?
  * 4-arm TTA test sweep on each criterion's chosen checkpoint
  * mean ± std across seeds for the headline val_avg/mae_surf_p and
    test_avg/mae_surf_p under {no_tta, tta_0.05} (the two arms most relevant
    to the advisor's question).

Handles both naming conventions in the JSONL:
  * "new" (this branch's reimplementation): ``val_avg/mae_surf_p_tta``,
    ``checkpoint_label``.
  * "old" (the version running on seed43 from a prior session):
    ``val_avg_tta_0.05/mae_surf_p``, ``checkpoint_selection``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_per_epoch_val_tta(epoch_rec: dict) -> float | None:
    """Return TTA val_avg/mae_surf_p for the per-epoch arm, handling both naming styles."""
    if "val_avg/mae_surf_p_tta" in epoch_rec:
        return epoch_rec["val_avg/mae_surf_p_tta"]
    # Old naming: val_avg_tta_{delta}/mae_surf_p
    for k, v in epoch_rec.items():
        if k.startswith("val_avg_tta_") and k.endswith("/mae_surf_p"):
            return v
    return None


def find_per_epoch_val_no_tta(epoch_rec: dict) -> float | None:
    return epoch_rec.get("val_avg/mae_surf_p")


def select_criterion_field(record: dict) -> str | None:
    """Return the checkpoint criterion label from a tta_val/tta_test/tta_summary record."""
    if "checkpoint_label" in record:
        return record["checkpoint_label"]
    if "checkpoint_selection" in record:
        return record["checkpoint_selection"]
    return None


def summarize_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl missing in {run_dir}")
    records = load_jsonl(metrics_path)
    epochs = [r for r in records if r.get("event") == "epoch"]

    # Best epoch by each criterion (computed from per-epoch records, regardless
    # of which "is_best_*" flag the run wrote).
    no_tta_pairs = [
        (e["epoch"], v)
        for e in epochs
        if (v := find_per_epoch_val_no_tta(e)) is not None
    ]
    tta_pairs = [
        (e["epoch"], v)
        for e in epochs
        if (v := find_per_epoch_val_tta(e)) is not None
    ]
    best_no_tta_epoch, best_no_tta_val = min(no_tta_pairs, key=lambda p: p[1]) if no_tta_pairs else (None, None)
    best_tta_epoch, best_tta_val = min(tta_pairs, key=lambda p: p[1]) if tta_pairs else (None, None)

    # 4-arm TTA test/val sweep — there should be one tta_val + tta_test event
    # per criterion. We index by criterion label.
    val_avg_by_arm: dict[str, dict[str, float]] = {}   # criterion -> {arm: val_avg/mae_surf_p}
    test_avg_by_arm: dict[str, dict[str, float]] = {}  # criterion -> {arm: test_avg/mae_surf_p}
    for rec in records:
        ev = rec.get("event")
        crit = select_criterion_field(rec)
        if ev == "tta_val" and crit:
            val_avg_by_arm[crit] = {
                arm: rec["val_tta_avg"][arm].get("avg/mae_surf_p")
                for arm in rec["val_tta_avg"]
            }
        elif ev == "tta_test" and crit:
            test_avg_by_arm[crit] = {
                arm: rec["test_tta_avg"][arm].get("avg/mae_surf_p")
                for arm in rec["test_tta_avg"]
            }
        elif ev == "tta_summary":
            # Old run wrote a single combined summary with summary_by_selection
            sbs = rec.get("summary_by_selection")
            if sbs:
                for crit_, by_arm in sbs.items():
                    if "val_avg_by_arm" in by_arm:
                        val_avg_by_arm.setdefault(crit_, by_arm["val_avg_by_arm"])
                    if "test_avg_by_arm" in by_arm:
                        test_avg_by_arm.setdefault(crit_, by_arm["test_avg_by_arm"])

    # When both selection criteria picked the same epoch the run wrote a
    # single `shared_both_criteria` label. Mirror it into both
    # `no_tta_best` and `tta_best` so the cross-seed aggregation aligns
    # apples-to-apples with runs that wrote the two labels separately.
    if "shared_both_criteria" in val_avg_by_arm:
        shared_val = val_avg_by_arm["shared_both_criteria"]
        shared_test = test_avg_by_arm.get("shared_both_criteria", {})
        val_avg_by_arm.setdefault("no_tta_best", shared_val)
        val_avg_by_arm.setdefault("tta_best", shared_val)
        test_avg_by_arm.setdefault("no_tta_best", shared_test)
        test_avg_by_arm.setdefault("tta_best", shared_test)

    return {
        "run_dir": str(run_dir),
        "n_epochs_completed": len(epochs),
        "best_no_tta_epoch": best_no_tta_epoch,
        "best_no_tta_val": best_no_tta_val,
        "best_tta_epoch": best_tta_epoch,
        "best_tta_val": best_tta_val,
        "same_epoch": best_no_tta_epoch == best_tta_epoch,
        "val_avg_by_arm": val_avg_by_arm,
        "test_avg_by_arm": test_avg_by_arm,
    }


def mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="Run directories containing metrics.jsonl")
    args = ap.parse_args()

    summaries = []
    for d in args.run_dirs:
        s = summarize_run(Path(d))
        summaries.append(s)
        print(f"\n=== {Path(d).name} ===")
        print(f"  epochs completed: {s['n_epochs_completed']}")
        print(f"  best no_tta epoch: {s['best_no_tta_epoch']}, val={s['best_no_tta_val']:.4f}" if s['best_no_tta_val'] is not None else "  best no_tta: n/a")
        print(f"  best tta    epoch: {s['best_tta_epoch']}, val={s['best_tta_val']:.4f}" if s['best_tta_val'] is not None else "  best tta: n/a")
        print(f"  same_epoch: {s['same_epoch']}")
        for crit in sorted(s["val_avg_by_arm"]):
            v = s["val_avg_by_arm"][crit]
            t = s["test_avg_by_arm"].get(crit, {})
            print(f"  criterion={crit}:")
            for arm in sorted(v):
                val = v.get(arm)
                test = t.get(arm)
                val_s = f"{val:.4f}" if val is not None else "    n/a"
                test_s = f"{test:.4f}" if test is not None else "    n/a"
                print(f"    {arm:10s}  val={val_s}   test={test_s}")

    # Cross-seed aggregation: only meaningful if both seeds completed final eval.
    print(f"\n=== Cross-seed aggregate ({len(summaries)} seeds) ===")
    crits = sorted({c for s in summaries for c in s["val_avg_by_arm"]})
    arms = sorted({a for s in summaries for c in s["val_avg_by_arm"] for a in s["val_avg_by_arm"][c]})
    for crit in crits:
        print(f"\n  criterion={crit}:")
        for arm in arms:
            v_vals = [s["val_avg_by_arm"].get(crit, {}).get(arm) for s in summaries]
            t_vals = [s["test_avg_by_arm"].get(crit, {}).get(arm) for s in summaries]
            vm, vs = mean_std(v_vals)
            tm, ts = mean_std(t_vals)
            v_raw = ",".join(f"{x:.4f}" if x is not None else "na" for x in v_vals)
            t_raw = ",".join(f"{x:.4f}" if x is not None else "na" for x in t_vals)
            print(f"    {arm:10s}  val mean={vm:.4f} ± {vs:.4f}  [{v_raw}]   test mean={tm:.4f} ± {ts:.4f}  [{t_raw}]")

    # Hypothesis check: does the TTA-criterion + tta_0.05 arm beat the
    # baseline 28.8762 (val) / 24.9992 (test) reliably?
    print("\n  Baseline ref (PR #2011): val_avg/mae_surf_p = 28.8762, test_avg/mae_surf_p = 24.9992")
    for crit in crits:
        for arm in ["no_tta", "tta_0.05"]:
            t_vals = [s["test_avg_by_arm"].get(crit, {}).get(arm) for s in summaries]
            tm, ts = mean_std(t_vals)
            print(f"  criterion={crit} arm={arm}: test mean ± std = {tm:.4f} ± {ts:.4f}  (Δ vs baseline = {tm - 24.9992:+.4f})")


if __name__ == "__main__":
    main()
