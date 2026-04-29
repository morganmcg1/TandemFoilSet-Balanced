"""Collect training run results from session_logs/*.log into MLINTERN_RESULTS.jsonl.

Each line is a JSON object with:
  run_name, log_file, status (running|finished|crashed|timeout),
  best_val_avg_mae_surf_p, best_epoch, last_epoch_metrics{}, test_avg_metrics{},
  per_split_test{}, n_params, total_train_minutes, peak_gb, lr, weight_decay,
  ... extracted hyperparameters from config.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Anchored summary fields the trainer prints near the end of a run.
EPOCH_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s+\((?P<dt>\d+)s\)\s+\[(?P<gb>[\d.]+)GB\]\s+"
    r"train\[vol=(?P<train_vol>[\d.naN-]+)\s+surf=(?P<train_surf>[\d.naN-]+)\]\s+"
    r"val_avg_surf_p=(?P<vasp>[\d.naN-]+)(?P<best>\s\*)?\s+lr=(?P<lr>[\d.e+-]+)"
)
SPLIT_RE = re.compile(
    r"^\s+(?P<split>\S+)\s+loss=(?P<loss>[\d.naN-]+)\s+"
    r"surf\[p=(?P<sp>[\d.naN-]+)\s+Ux=(?P<sUx>[\d.naN-]+)\s+Uy=(?P<sUy>[\d.naN-]+)\]\s+"
    r"vol\[p=(?P<vp>[\d.naN-]+)\s+Ux=(?P<vUx>[\d.naN-]+)\s+Uy=(?P<vUy>[\d.naN-]+)\]"
)
TEST_AVG_RE = re.compile(r"TEST\s+avg_surf_p=(?P<tasp>[\d.naN-]+)")
BEST_VAL_RE = re.compile(r"Best val: epoch (?P<epoch>\d+),\s+val_avg/mae_surf_p\s*=\s*(?P<v>[\d.naN-]+)")
TOTAL_TIME_RE = re.compile(r"Training done in (?P<m>[\d.]+) min")
MODEL_PARAMS_RE = re.compile(r"Model: Transolver \((?P<p>[\d.]+)M params\)")
WANDB_RUN_RE = re.compile(r"View run at https?://wandb.ai/[^/]+/[^/]+/runs/(?P<rid>\w+)")
WANDB_NAME_RE = re.compile(r"Syncing run\s+(?P<n>\S+)")


def parse_log(log: Path) -> dict:
    text = log.read_text(errors="ignore")
    lines = text.splitlines()
    info: dict = {"log_file": str(log), "run_name": log.stem}

    # Extract from CLI arguments line if present (it's in the wandb config)
    # Or just parse our printed messages.

    # CLI args from filename pattern wave?-gpu?-NAME
    m = re.match(r"^wave[a-z0-9]+-gpu(\d+)-(.+)$", log.stem)
    if m:
        info["gpu_idx"] = int(m.group(1))
        info["short_name"] = m.group(2)

    if (m := MODEL_PARAMS_RE.search(text)):
        info["n_params_m"] = float(m.group("p"))
    if (m := WANDB_RUN_RE.search(text)):
        info["wandb_run_id"] = m.group("rid")
    if (m := WANDB_NAME_RE.search(text)):
        info["wandb_run_name"] = m.group("n")

    # All epoch lines
    epochs: list[dict] = []
    for line in lines:
        m = EPOCH_RE.match(line)
        if m:
            ep = int(m.group("epoch"))
            d = m.groupdict()
            epochs.append({
                "epoch": ep,
                "epoch_time_s": int(d["dt"]),
                "peak_gb": float(d["gb"]),
                "train_vol_loss": _f(d["train_vol"]),
                "train_surf_loss": _f(d["train_surf"]),
                "val_avg_mae_surf_p": _f(d["vasp"]),
                "is_best": d["best"] is not None,
                "lr": _f(d["lr"]),
            })
    info["n_epochs_completed"] = len(epochs)
    info["epochs"] = epochs

    # Best val
    best = None
    for e in epochs:
        v = e["val_avg_mae_surf_p"]
        if v is not None and (best is None or v < best["val_avg_mae_surf_p"]):
            best = {"epoch": e["epoch"], "val_avg_mae_surf_p": v, "lr": e["lr"]}
    if (m := BEST_VAL_RE.search(text)):
        # printed at end of run if completed
        ep = int(m.group("epoch"))
        v = _f(m.group("v"))
        if best is None or v != best.get("val_avg_mae_surf_p"):
            best = {"epoch": ep, "val_avg_mae_surf_p": v}
    info["best"] = best

    if (m := TOTAL_TIME_RE.search(text)):
        info["total_train_minutes"] = float(m.group("m"))

    # Test results (only present if run completed test eval)
    if (m := TEST_AVG_RE.search(text)):
        info["test_avg_mae_surf_p"] = _f(m.group("tasp"))
        # Find per-split test lines after that
        idx = text.index(m.group(0))
        block = text[idx:idx + 2000]
        per_split = {}
        for line in block.splitlines():
            sm = SPLIT_RE.match(line)
            if sm:
                d = sm.groupdict()
                per_split[d["split"]] = {
                    "loss": _f(d["loss"]),
                    "mae_surf_p": _f(d["sp"]),
                    "mae_surf_Ux": _f(d["sUx"]),
                    "mae_surf_Uy": _f(d["sUy"]),
                    "mae_vol_p": _f(d["vp"]),
                    "mae_vol_Ux": _f(d["vUx"]),
                    "mae_vol_Uy": _f(d["vUy"]),
                }
        info["test_per_split"] = per_split

    # Heuristic status
    if "Logged model artifact" in text:
        info["status"] = "finished"
    elif "out of memory" in text.lower() or "OutOfMemoryError" in text:
        info["status"] = "oom"
    elif "Traceback" in text:
        info["status"] = "crashed"
    elif "Timeout" in text:
        info["status"] = "timeout"
    else:
        info["status"] = "running"
    return info


def _f(s: str | None):
    if s is None:
        return None
    s = s.strip()
    if s in {"nan", "NaN", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default="session_logs")
    p.add_argument("--out", default="research/MLINTERN_RESULTS.jsonl")
    p.add_argument("--include_pattern", default="wave*.log")
    args = p.parse_args()

    logs = sorted(Path(args.logs_dir).glob(args.include_pattern))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for log in logs:
        try:
            info = parse_log(log)
            rows.append(info)
        except Exception as e:
            print(f"warn: failed to parse {log}: {e}", file=sys.stderr)

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")

    # Print summary
    print("\nRun summary:")
    print(f"{'Run':50s}  {'Status':10s}  {'Best val_avg/mae_surf_p':>22s}  {'Test avg':>9s}  {'Epochs':>6s}")
    rows.sort(key=lambda r: (r.get("best") or {}).get("val_avg_mae_surf_p") or 1e18)
    for r in rows:
        best = r.get("best") or {}
        bvp = best.get("val_avg_mae_surf_p")
        bvp_s = f"{bvp:.3f}" if bvp is not None else "—"
        ep = best.get("epoch") or 0
        tasp = r.get("test_avg_mae_surf_p")
        tasp_s = f"{tasp:.3f}" if tasp is not None else "—"
        n_ep = r.get("n_epochs_completed") or 0
        name = r.get("short_name", r["run_name"])[:50]
        print(f"{name:50s}  {r['status']:10s}  {bvp_s:>14s} ep{ep:<3d}  {tasp_s:>9s}  {n_ep:>6d}")


if __name__ == "__main__":
    main()
