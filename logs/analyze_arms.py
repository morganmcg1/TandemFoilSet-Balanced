"""Aggregate per-arm diagnostics for the sf-bs-r1 sweep."""
import json
import glob
import os
import statistics
import yaml

ARMS = [
    ("Arm A bs=4",  4, glob.glob("models/model-charliepai2i48h4-thorfinn-sf-bs-r1-arma-bs4-r4-*")),
    ("Arm B bs=6",  6, glob.glob("models/model-charliepai2i48h4-thorfinn-sf-bs-r1-armb-bs6-r4-*")),
    ("Arm C bs=8",  8, glob.glob("models/model-charliepai2i48h4-thorfinn-sf-bs-r1-armc-bs8-r4-*")),
    ("Arm D bs=9",  9, glob.glob("models/model-charliepai2i48h4-thorfinn-sf-bs-r1-armd-bs9-r4-*")),
]

print(f"{'Arm':12s} {'best_ep':>7s} {'val_avg':>10s} {'sec/ep':>7s} {'epochs':>7s} {'peak_GB':>8s} {'steps/ep':>9s} {'g_mean':>8s} {'g_std':>8s} {'g_cv':>8s}")
results = {}
for name, bs, dirs in ARMS:
    if not dirs:
        print(f"{name:12s} -- not started --")
        continue
    d = sorted(dirs)[-1]
    jsonl_path = os.path.join(d, "metrics.jsonl")
    yaml_path = os.path.join(d, "metrics.yaml")
    if not os.path.exists(jsonl_path):
        print(f"{name:12s} -- no metrics.jsonl --")
        continue
    epoch_rows = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "epoch":
                epoch_rows.append(r)
    if not epoch_rows:
        print(f"{name:12s} -- no epochs yet --")
        continue
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            y = yaml.safe_load(f)
    else:
        y = {}
    n_epochs = len(epoch_rows)
    total_t = sum(r.get("seconds", 0) for r in epoch_rows)
    sec_ep = total_t / n_epochs
    peak_gb = max(r.get("peak_memory_gb", 0) for r in epoch_rows)
    g_means = [r.get("grad_norm/mean") for r in epoch_rows if r.get("grad_norm/mean") is not None]
    g_stds  = [r.get("grad_norm/std")  for r in epoch_rows if r.get("grad_norm/std")  is not None]
    g_cvs   = [r.get("grad_norm/cv")   for r in epoch_rows if r.get("grad_norm/cv")   is not None]
    n_steps_per_epoch = epoch_rows[0].get("grad_norm/n_steps", 0) if epoch_rows[0].get("grad_norm/n_steps") else (1499 + bs - 1)//bs
    best_epoch = y.get("best_epoch", "?")
    val_avg = y.get("best_val_avg/mae_surf_p", float("nan"))
    print(f"{name:12s} {best_epoch!s:>7s} {val_avg:>10.3f} {sec_ep:>7.1f} {n_epochs:>7d} {peak_gb:>8.2f} {n_steps_per_epoch:>9d} "
          f"{statistics.mean(g_means) if g_means else 0:>8.2f} {statistics.mean(g_stds) if g_stds else 0:>8.2f} {statistics.mean(g_cvs) if g_cvs else 0:>8.3f}")
    results[name] = dict(
        bs=bs, dir=d,
        best_epoch=best_epoch, val_avg=val_avg,
        sec_ep=sec_ep, n_epochs=n_epochs, peak_gb=peak_gb,
        steps_ep=n_steps_per_epoch,
        g_mean=statistics.mean(g_means) if g_means else None,
        g_std=statistics.mean(g_stds) if g_stds else None,
        g_cv=statistics.mean(g_cvs) if g_cvs else None,
        val_per_split={
            "in_dist":  y.get("best_val/val_single_in_dist/mae_surf_p"),
            "camber_rc": y.get("best_val/val_geom_camber_rc/mae_surf_p"),
            "camber_cruise": y.get("best_val/val_geom_camber_cruise/mae_surf_p"),
            "re_rand":  y.get("best_val/val_re_rand/mae_surf_p"),
        },
        test_per_split={
            "in_dist":  y.get("test/test_single_in_dist/mae_surf_p"),
            "camber_rc": y.get("test/test_geom_camber_rc/mae_surf_p"),
            "camber_cruise": y.get("test/test_geom_camber_cruise/mae_surf_p"),
            "re_rand":  y.get("test/test_re_rand/mae_surf_p"),
        },
        train_val_gap_at_best={
            "train_surf_loss": next((r["train/surf_loss"] for r in epoch_rows if r.get("epoch") == best_epoch), None),
            "val_loss_avg": None,  # noisy in jsonl
        },
        val_trajectory=[(r["epoch"], r["val_avg/mae_surf_p"]) for r in epoch_rows],
        grad_norm_trajectory=[(r["epoch"], r.get("grad_norm/mean"), r.get("grad_norm/std"), r.get("grad_norm/cv")) for r in epoch_rows],
    )

print("\n--- Per-split val at best (mae_surf_p) ---")
for name, info in results.items():
    p = info["val_per_split"]
    print(f"{name:12s}  in_dist={p['in_dist']:.3f}  camber_rc={p['camber_rc']:.3f}  camber_cruise={p['camber_cruise']:.3f}  re_rand={p['re_rand']:.3f}")

print("\n--- Per-split test (mae_surf_p) ---")
for name, info in results.items():
    p = info["test_per_split"]
    print(f"{name:12s}  in_dist={p['in_dist']}  camber_rc={p['camber_rc']}  camber_cruise={p['camber_cruise']}  re_rand={p['re_rand']}")

print("\n--- Test 3-split mean (excluding cruise NaN) ---")
for name, info in results.items():
    p = info["test_per_split"]
    if any(v is None for v in (p['in_dist'], p['camber_rc'], p['re_rand'])):
        print(f"{name:12s}  -- pending --")
        continue
    m = (p['in_dist'] + p['camber_rc'] + p['re_rand']) / 3
    print(f"{name:12s}  test3 = {m:.3f}")

print("\n--- Val trajectory (epoch -> val_avg/mae_surf_p) ---")
for name, info in results.items():
    print(f"{name}:")
    traj = info["val_trajectory"]
    print(f"  first 5: {[(e, round(v,2)) for e,v in traj[:5]]}")
    print(f"  last  5: {[(e, round(v,2)) for e,v in traj[-5:]]}")

print("\n--- ||g||_2 mean trajectory ---")
for name, info in results.items():
    print(f"{name}:")
    g = info["grad_norm_trajectory"]
    print(f"  first 5 (epoch, mean, std, cv): {[(e, round(m,1) if m else None, round(s,1) if s else None, round(c,2) if c else None) for e,m,s,c in g[:5]]}")
    print(f"  last  5: {[(e, round(m,1) if m else None, round(s,1) if s else None, round(c,2) if c else None) for e,m,s,c in g[-5:]]}")

print("\n--- val improvement per epoch (first 5 epochs) ---")
for name, info in results.items():
    traj = info["val_trajectory"][:6]
    deltas = [traj[i+1][1] - traj[i][1] for i in range(min(len(traj)-1, 5))]
    print(f"{name}: deltas = {[round(d,2) for d in deltas]}, total = {round(sum(deltas),2)}")
