"""End-of-run finalisation pipeline.

For each R*/baseline log, find the matching wandb run id and the on-disk
``models/model-<run_id>/`` checkpoint. Evaluate every checkpoint on val + test,
write per-checkpoint eval JSON, and update ``MLINTERN_RESULTS.jsonl`` with the
test_avg/mae_surf_p column. Then build several ensembles and report which one
wins.

Usage:
    python scripts/finalize_results.py [--top_k 10]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
EVALS = REPO / "research"
EVALS.mkdir(parents=True, exist_ok=True)


def parse_logs() -> list[dict]:
    """Parse logs/*.log and return one dict per run with best_val and run_id."""
    pat = re.compile(
        r"Epoch\s+(\d+)\s*\((\d+)s\)\s*\[([\d.]+)GB\]\s*"
        r"train\[vol=([\d.]+)\s+surf=([\d.]+)\]\s*val_avg_surf_p=([\d.]+)"
    )
    runid_pat = re.compile(r"runs/([a-z0-9]+)")

    results = []
    for log in sorted(LOGS.glob("*.log")):
        text = log.read_text(errors="ignore")
        epochs = []
        for line in re.split(r"[\r\n]+", text):
            m = pat.search(line.strip())
            if m:
                epochs.append({
                    "epoch": int(m.group(1)),
                    "epoch_time_s": int(m.group(2)),
                    "peak_mem_gb": float(m.group(3)),
                    "train_vol": float(m.group(4)),
                    "train_surf": float(m.group(5)),
                    "val_avg_mae_surf_p": float(m.group(6)),
                })
        if not epochs:
            continue
        rid = runid_pat.search(text)
        run_id = rid.group(1) if rid else None
        best = min(epochs, key=lambda r: r["val_avg_mae_surf_p"])
        params_M = None
        pm = re.search(r"Transolver\s+\(([\d.]+)M\s+params\)", text)
        if pm:
            params_M = float(pm.group(1))
        results.append({
            "log": log.name,
            "log_stem": log.stem,
            "run_id": run_id,
            "model_dir": f"models/model-{run_id}" if run_id else None,
            "n_params_M": params_M,
            "epochs_done": len(epochs),
            "best_epoch": best["epoch"],
            "best_val": best["val_avg_mae_surf_p"],
        })
    return results


def eval_checkpoint(model_dir: str, run_id: str) -> dict | None:
    """Run scripts/eval_test.py and return the JSON dict, or None if failed."""
    out_json = EVALS / f"eval_{run_id}.json"
    if out_json.exists():
        with open(out_json) as f:
            return json.load(f)
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_test.py"),
        "--model_dir", str(REPO / model_dir),
        "--out_json", str(out_json),
    ]
    print(f"  [eval] {model_dir}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"    FAILED: {e.stderr.decode()[-500:] if e.stderr else e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        return None
    if out_json.exists():
        with open(out_json) as f:
            return json.load(f)
    return None


def eval_ensemble(model_dirs: list[str], tag: str) -> dict | None:
    out_json = EVALS / f"eval_ensemble_{tag}.json"
    if out_json.exists():
        with open(out_json) as f:
            return json.load(f)
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_ensemble.py"),
        *[str(REPO / d) for d in model_dirs],
        "--out_json", str(out_json),
    ]
    print(f"  [ens]  {tag} ({len(model_dirs)} models)")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    except subprocess.CalledProcessError as e:
        print(f"    FAILED: {e.stderr.decode()[-500:] if e.stderr else e}")
        return None
    if out_json.exists():
        with open(out_json) as f:
            return json.load(f)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=8)
    args = p.parse_args()

    runs = parse_logs()
    print(f"Parsed {len(runs)} runs from logs.")

    # Sort by best_val and eval the top k single models
    runs.sort(key=lambda r: r["best_val"])
    top = [r for r in runs if r["model_dir"] and (REPO / r["model_dir"] / "checkpoint.pt").exists()]
    print(f"\nEvaluating top {args.top_k} checkpoints…")
    for r in top[:args.top_k]:
        ev = eval_checkpoint(r["model_dir"], r["run_id"])
        if ev:
            r["test_avg_mae_surf_p"] = ev["test_avg"]["avg/mae_surf_p"]
            r["val_avg_eval"] = ev["val_avg"]["avg/mae_surf_p"]
            r["test_per_split"] = ev["test_per_split"]

    # Ensembles: top-3, top-5, top-7 of default-arch (n_h=128) only.
    default_arch_top = [
        r for r in top
        if r.get("n_params_M") is not None and abs(r["n_params_M"] - 0.66) < 0.05
    ]
    ensembles_to_try = []
    for k in (3, 5, 7, 8, 10):
        if len(default_arch_top) >= k:
            ensembles_to_try.append((k, default_arch_top[:k]))

    print(f"\nRunning ensembles…")
    ens_results = []
    for k, members in ensembles_to_try:
        tag = f"top{k}_default"
        ev = eval_ensemble([m["model_dir"] for m in members], tag)
        if ev:
            ens_results.append({
                "tag": tag,
                "n_members": k,
                "members": [m["log_stem"] for m in members],
                "val_avg": ev["val_avg"]["avg/mae_surf_p"],
                "test_avg": ev["test_avg"]["avg/mae_surf_p"],
            })

    # Write the master JSONL.
    with open(REPO / "research" / "MLINTERN_RESULTS.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")

    # Print final tables.
    print(f"\n{'='*70}\nTOP {args.top_k} SINGLE MODELS\n{'='*70}")
    print(f"{'Rank':<4} {'BestVal':>8} {'Test':>8} {'Ep':>3} {'Params':>7} {'Run':<40}")
    for i, r in enumerate(top[:args.top_k], 1):
        test = r.get("test_avg_mae_surf_p")
        test_s = f"{test:>8.3f}" if isinstance(test, (int, float)) else f"{'—':>8s}"
        print(f"{i:<4} {r['best_val']:>8.3f} {test_s} {r['epochs_done']:>3d} {r['n_params_M']:>5.2f}M  {r['log']}")

    if ens_results:
        print(f"\n{'='*70}\nENSEMBLES (default-arch only)\n{'='*70}")
        print(f"{'Tag':<20} {'Val':>8} {'Test':>8}")
        for er in ens_results:
            print(f"{er['tag']:<20} {er['val_avg']:>8.3f} {er['test_avg']:>8.3f}")

    summary = {
        "n_runs": len(runs),
        "top_single": top[0] if top else None,
        "ensembles": ens_results,
    }
    with open(REPO / "research" / "MLINTERN_FINAL.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote research/MLINTERN_FINAL.json")


if __name__ == "__main__":
    main()
