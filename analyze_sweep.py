#!/usr/bin/env python3
"""Analyse the 4-arm EMA-decay sweep produced by run_sweep_bcd.sh.

For each arm load metrics.jsonl, summarise validation/test metrics and the
EMA <-> raw diagnostics requested by the PR body.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ARMS = [
    ("A", 0.999,  "models/model-charliepai2i48h4-tanjiro-sf-ema-r1-arma-d0_999-20260516-202503"),
    ("B", 0.99,   "models/model-charliepai2i48h4-tanjiro-sf-ema-r1-armb-d0_99-20260516-205953"),
    ("C", 0.9995, "models/model-charliepai2i48h4-tanjiro-sf-ema-r1-armc-d0_9995-20260516-213442"),
    ("D", 0.9999, "models/model-charliepai2i48h4-tanjiro-sf-ema-r1-armd-d0_9999-20260516-220927"),
]


def load(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def half_life_steps(decay: float) -> float:
    return math.log(0.5) / math.log(decay)


def summarise(arm_letter: str, decay: float, model_dir: Path) -> dict:
    events = load(model_dir / "metrics.jsonl")
    epochs = [e for e in events if e.get("event") == "epoch"]
    tests = [e for e in events if e.get("event") == "test"]

    val_trajectory = [(e.get("epoch"), e.get("val_avg/mae_surf_p")) for e in epochs]
    best = min(epochs, key=lambda e: e.get("val_avg/mae_surf_p", float("inf")))

    diag = {
        "epoch_count": len(epochs),
        "best_epoch": best.get("epoch"),
        "best_val_avg/mae_surf_p": best.get("val_avg/mae_surf_p"),
        "val_trajectory": val_trajectory,
        "best_val_splits": best.get("val_splits"),
        "best_ema_effective_decay": best.get("ema_effective_decay"),
        "best_ema_raw_diff_rel": best.get("ema_raw_diff_rel"),
        "best_ema_raw_l2_ratio": best.get("ema_raw_l2_ratio"),
        "best_grad_norm_p99": best.get("grad_norm/p99"),
        "peak_memory_gb": max(e.get("peak_memory_gb", 0) for e in epochs),
        "half_life_steps": half_life_steps(decay),
    }
    if tests:
        t = tests[-1]
        ts = t.get("test_splits", {})
        try:
            mean3 = sum(ts[k]["mae_surf_p"] for k in ("test_single_in_dist", "test_geom_camber_rc", "test_re_rand")) / 3.0
        except Exception:
            mean3 = float("nan")
        diag.update({
            "test_best_epoch": t.get("best_epoch"),
            "test_mean3_surf_p": mean3,
            "test_splits": {k: v.get("mae_surf_p") for k, v in ts.items()},
        })
    diag["arm"] = arm_letter
    diag["ema_decay"] = decay
    diag["model_dir"] = str(model_dir)
    return diag


def main():
    root = Path(__file__).resolve().parent
    summaries = []
    for letter, decay, rel in ARMS:
        summaries.append(summarise(letter, decay, root / rel))

    control = summaries[0]
    print("=" * 100)
    print(f"EMA-decay sweep summary (best ckpt = lowest val_avg/mae_surf_p)")
    print("=" * 100)
    print(f"Note: every arm trained {control['epoch_count']} epochs (timeout-limited at SENPAI_TIMEOUT_MINUTES=30).")
    print(f"Effective EMA half-life is reported in optimiser steps (375 steps / epoch).")
    print()
    header = ("arm", "decay", "half_life_steps", "best_epoch", "val_avg/mae_surf_p", "delta_vs_A", "delta_pct")
    print("{:>3} {:>8} {:>16} {:>10} {:>20} {:>12} {:>10}".format(*header))
    base = control["best_val_avg/mae_surf_p"]
    for s in summaries:
        val = s["best_val_avg/mae_surf_p"]
        d = val - base
        pct = 100 * d / base
        print("{arm:>3} {decay:>8} {hl:>16.2f} {ep:>10} {v:>20.4f} {d:>12.4f} {p:>10.3f}".format(
            arm=s["arm"], decay=s["ema_decay"], hl=s["half_life_steps"],
            ep=s["best_epoch"], v=val, d=d, p=pct))
    print()
    print("Test 3-split surf_p mean (single_in_dist + geom_camber_rc + re_rand) / 3:")
    base_test = control.get("test_mean3_surf_p")
    for s in summaries:
        val = s.get("test_mean3_surf_p")
        d = (val - base_test) if val is not None and base_test is not None else float("nan")
        print(f"  Arm {s['arm']} decay={s['ema_decay']}: {val:.4f}   delta={d:+.4f}")
    print()
    print("Per-split val (best epoch):")
    cols = ("val_single_in_dist", "val_geom_camber_rc", "val_geom_camber_cruise", "val_re_rand")
    print("{:>3} {:>8}".format("arm", "decay") + " ".join(f"{c:>30}" for c in cols))
    for s in summaries:
        row = [f"{s['best_val_splits'][c]['mae_surf_p']:>30.4f}" for c in cols]
        print("{:>3} {:>8} ".format(s["arm"], s["ema_decay"]) + " ".join(row))
    print()
    print("Per-split test (best epoch):")
    test_cols = ("test_single_in_dist", "test_geom_camber_rc", "test_geom_camber_cruise", "test_re_rand")
    print("{:>3} {:>8}".format("arm", "decay") + " ".join(f"{c:>30}" for c in test_cols))
    for s in summaries:
        row = [f"{s['test_splits'].get(c, float('nan')):>30.4f}" for c in test_cols]
        print("{:>3} {:>8} ".format(s["arm"], s["ema_decay"]) + " ".join(row))
    print()
    print("EMA <-> raw diagnostics (best epoch):")
    print("{:>3} {:>8} {:>22} {:>22} {:>20}".format("arm", "decay", "ema_effective_decay", "ema_raw_l2_ratio", "ema_raw_diff_rel"))
    def fmt6(v):
        if v is None:
            return "                   N/A"
        return f"{v:>22.6f}"
    for s in summaries:
        print("{:>3} {:>8}".format(s["arm"], s["ema_decay"])
              + fmt6(s["best_ema_effective_decay"])
              + fmt6(s["best_ema_raw_l2_ratio"])
              + fmt6(s["best_ema_raw_diff_rel"]))
    print()
    print("Per-epoch val trajectory:")
    epochs = sorted({e for s in summaries for e, _ in s["val_trajectory"]})
    print("epoch " + " ".join(f"{s['arm']:>11}" for s in summaries))
    for ep in epochs:
        cells = []
        for s in summaries:
            v = next((v for e, v in s["val_trajectory"] if e == ep), None)
            cells.append(f"{v:>11.4f}" if v is not None else " " * 11)
        print(f"{ep:>5} " + " ".join(cells))
    print()
    print("Peak memory GB:")
    for s in summaries:
        print(f"  Arm {s['arm']}: {s['peak_memory_gb']:.2f} GB")
    print()
    out_path = root / "logs" / "sweep_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
