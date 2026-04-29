"""Harvest training-run results from logs/ directory and append JSON lines.

Parses the *final* lines of each train.py log to extract:
  - run_id (last 8 chars of wandb path)
  - best_val_avg_mae_surf_p (best epoch on val)
  - per-split val metrics (only the BEST epoch's per-split breakdown)
  - test metrics (only for runs that ran end-of-run test eval)
  - command line, config snapshot

Usage:
  python research/harvest.py [logs_dir] [out_jsonl]
"""
import json
import re
import sys
from pathlib import Path

LOGS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/ml-intern-benchmark/target/logs")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/ml-intern-benchmark/target/research/MLINTERN_RESULTS.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


SPLITS = ("val_single_in_dist", "val_geom_camber_rc", "val_geom_camber_cruise", "val_re_rand")
TEST_SPLITS = ("test_single_in_dist", "test_geom_camber_rc", "test_geom_camber_cruise", "test_re_rand")


def parse_metrics_line(line: str):
    """Parse `<split> loss=L surf[p=P Ux=U Uy=U] vol[p=P Ux=U Uy=U]` line."""
    m = re.match(r"\s*(\w+)\s+loss=([\d.nan]+)\s+surf\[p=([\d.nan]+) Ux=([\d.nan]+) Uy=([\d.nan]+)\]\s+vol\[p=([\d.nan]+) Ux=([\d.nan]+) Uy=([\d.nan]+)\]", line)
    if m:
        def f(s): 
            try: return float(s)
            except ValueError: return None
        return m.group(1), {
            "loss": f(m.group(2)),
            "mae_surf_p": f(m.group(3)),
            "mae_surf_Ux": f(m.group(4)),
            "mae_surf_Uy": f(m.group(5)),
            "mae_vol_p": f(m.group(6)),
            "mae_vol_Ux": f(m.group(7)),
            "mae_vol_Uy": f(m.group(8)),
        }
    return None


def parse_log(path: Path) -> dict:
    text = path.read_text()
    name = path.stem.replace("__", "/")  # restore "/" from "__"
    rec: dict = {"name": name, "log": str(path)}

    m = re.search(r"https://wandb\.ai/[^/]+/[^/]+/runs/([A-Za-z0-9]+)", text)
    if m:
        rec["wandb_run_id"] = m.group(1)
        rec["wandb_url"] = m.group(0)

    m = re.search(r"Model: Transolver \(([\d.]+)M params\) — (.+)", text)
    if m:
        rec["n_params_m"] = float(m.group(1))
        rec["model_str"] = m.group(2).strip()

    m = re.search(r"Best val: epoch (\d+), val_avg/mae_surf_p = ([\d.]+)", text)
    if m:
        rec["best_epoch"] = int(m.group(1))
        rec["best_val_avg_mae_surf_p"] = float(m.group(2))
    else:
        # Fall back to the running ``best so far`` mark (`* ` after the epoch).
        # Take the smallest val we've seen so far.
        vals = re.findall(r"val_avg_surf_p=([\d.]+) \*", text)
        if vals:
            v = min(float(x) for x in vals)
            rec["best_val_avg_mae_surf_p_running"] = v

    m = re.search(r"TEST  avg_surf_p=([\d.nan]+)", text)
    if m:
        try: rec["test_avg_mae_surf_p"] = float(m.group(1))
        except ValueError: rec["test_avg_mae_surf_p"] = None

    # Test per-split metrics
    test_block_match = re.search(r"TEST  avg_surf_p=[\d.nan]+\n((?:.+\n){4})", text)
    if test_block_match:
        for line in test_block_match.group(1).splitlines():
            r = parse_metrics_line(line)
            if r:
                split_name, metrics = r
                if split_name in TEST_SPLITS:
                    for k, v in metrics.items():
                        rec[f"test_{split_name}/{k}"] = v

    # Best-epoch val metrics (look for `*` tag's neighborhood)
    best_blocks = list(re.finditer(r"val_avg_surf_p=([\d.]+) \*\n((?:.+\n){4})", text))
    if best_blocks:
        last_best = best_blocks[-1]
        for line in last_best.group(2).splitlines():
            r = parse_metrics_line(line)
            if r:
                split_name, metrics = r
                if split_name in SPLITS:
                    for k, v in metrics.items():
                        rec[f"{split_name}/{k}"] = v

    m = re.search(r"Training done in ([\d.]+) min", text)
    if m:
        rec["total_train_minutes"] = float(m.group(1))

    epochs = re.findall(r"Epoch\s+(\d+)\s+\([\d.]+s\)", text)
    if epochs:
        rec["last_epoch"] = max(int(e) for e in epochs)

    if "Logged model artifact" in text:
        rec["status"] = "complete"
    elif "Training done" in text:
        rec["status"] = "trained_no_test"
    elif "Error" in text or "OutOfMemoryError" in text:
        rec["status"] = "error"
    else:
        rec["status"] = "running"

    return rec


def main():
    records = []
    for log_path in sorted(LOGS_DIR.glob("*.log")):
        try:
            rec = parse_log(log_path)
            records.append(rec)
        except Exception as e:
            records.append({"name": log_path.stem, "log": str(log_path), "parse_error": str(e)})

    # JSON spec disallows NaN — substitute null. ``json.dumps`` defaults to
    # writing ``NaN`` which breaks strict parsers (e.g. ``jq``, ``json.loads``
    # with ``allow_nan=False``).  Sanitise floats here so the file is valid.
    import math
    def _clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(_clean(r)) + "\n")

    complete = [r for r in records if "best_val_avg_mae_surf_p" in r]
    complete.sort(key=lambda r: r["best_val_avg_mae_surf_p"])
    print(f"\n{len(records)} runs / {len(complete)} with best val recorded:\n")
    for r in records:
        name = r["name"]
        status = r.get("status", "?")
        ep = r.get("last_epoch", "-")
        bv = r.get("best_val_avg_mae_surf_p", r.get("best_val_avg_mae_surf_p_running"))
        bv_s = f"{bv:7.3f}" if bv is not None else "      -"
        # Annotate running best with ~
        if "best_val_avg_mae_surf_p" not in r and "best_val_avg_mae_surf_p_running" in r:
            bv_s = bv_s.strip()
            bv_s = f"~{bv_s:>6}"
        ta = r.get("test_avg_mae_surf_p", None)
        ta_s = f"{ta:7.3f}" if isinstance(ta, float) else (f"   nan " if ta is None else "      -")
        if "test_avg_mae_surf_p" in r and r["test_avg_mae_surf_p"] is None:
            ta_s = "   nan "
        print(f"  {status:9s} ep={ep:>3}  val={bv_s}  test={ta_s}  {name}")

    if complete:
        print("\nSorted by best val:\n")
        for r in complete[:15]:
            name = r["name"]
            bv = r["best_val_avg_mae_surf_p"]
            ta = r.get("test_avg_mae_surf_p")
            ta_s = f"{ta:7.3f}" if isinstance(ta, float) else "      -"
            print(f"  val={bv:7.3f}  test={ta_s}  {name}")


if __name__ == "__main__":
    main()
