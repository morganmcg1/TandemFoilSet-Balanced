"""Offline: fit a per-channel linear baseline mapping condition features to
per-sample surface-pressure (and Ux, Uy) DC offsets.

The baseline is subtracted from y before training (residual learning, PR #3820,
DeltaPhi-style preconditioning). It is added back at eval time, so MAE stays
on the original target scale.

Per-channel Ridge regression:
    cond_features ([B, F]) -> per-sample y-surface-mean ([B, 3])

We use the 11 per-sample scalar condition features `x[0, 13:24]`:
    13 log(Re)
    14 AoA foil 1
    15-17 NACA foil 1 (M, P, T)
    18 AoA foil 2
    19-21 NACA foil 2 (M, P, T)
    22 gap
    23 stagger

Plus a single physics-motivated extra feature: `log(Re)^2` (pressure scales
with Re^2 in kinematic units p/rho ~ U^2 ~ Re^2 at fixed chord/viscosity).
The squared term is included for all channels — Ridge will down-weight it
where it is not useful (e.g. Ux, Uy).

Outputs to `data/linear_baseline.pkl`:
    {
        "feature_indices": list[int],          # 11 raw indices, plus extras
        "extra_features": list[str],           # e.g. ["log_re_sq"]
        "feature_means": np.ndarray [F],       # for feature standardization
        "feature_stds":  np.ndarray [F],
        "coef_per_ch": np.ndarray [3, F],      # Ridge coefficients (std-feature space)
        "intercept_per_ch": np.ndarray [3],    # Ridge intercepts (raw y space)
        "alpha_per_ch": np.ndarray [3],        # CV-selected alpha per channel
        "r2_train_per_ch": np.ndarray [3],
        "residual_y_mean": np.ndarray [3],     # global mean of (y - baseline) over
                                               # all valid training nodes
        "residual_y_std":  np.ndarray [3],     # global std of (y - baseline)
    }

Run once. Output is tiny (~few hundred bytes).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import SPLITS_DIR, SplitDataset

# Per-sample condition feature indices in x[0, :]
COND_INDICES = list(range(13, 24))  # 11 features
N_RAW_FEATURES = len(COND_INDICES)
EXTRA_FEATURES = ["log_re_sq"]  # log(Re)^2
N_FEATURES = N_RAW_FEATURES + len(EXTRA_FEATURES)


def build_feature_row(cond: np.ndarray) -> np.ndarray:
    """cond: [F_raw] raw condition features. Returns [N_FEATURES] augmented row."""
    log_re = cond[0]  # index 13 → cond[0]
    extras = np.array([log_re ** 2], dtype=np.float64)
    return np.concatenate([cond.astype(np.float64), extras])


def fit_ridge_cv(X: np.ndarray, y: np.ndarray, alphas: list[float], n_folds: int = 5):
    """Simple K-fold Ridge with CV-selected alpha.

    X: [N, F] standardized features. y: [N] target. Returns (coef, intercept,
    best_alpha, r2_full).
    """
    n = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    best_alpha = alphas[0]
    best_cv_r2 = -np.inf

    for a in alphas:
        cv_r2 = []
        for fold in folds:
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            X_tr, X_va = X[mask], X[fold]
            y_tr, y_va = y[mask], y[fold]
            y_tr_mean = y_tr.mean()
            yc = y_tr - y_tr_mean
            # closed-form Ridge: (X'X + alpha I)^-1 X' y
            XtX = X_tr.T @ X_tr
            coef = np.linalg.solve(XtX + a * np.eye(X_tr.shape[1]), X_tr.T @ yc)
            pred = X_va @ coef + y_tr_mean
            ss_res = ((y_va - pred) ** 2).sum()
            ss_tot = ((y_va - y_va.mean()) ** 2).sum()
            cv_r2.append(1 - ss_res / max(ss_tot, 1e-12))
        mean_cv_r2 = float(np.mean(cv_r2))
        if mean_cv_r2 > best_cv_r2:
            best_cv_r2 = mean_cv_r2
            best_alpha = a

    # Refit on all data with best alpha
    y_mean = y.mean()
    yc = y - y_mean
    XtX = X.T @ X
    coef = np.linalg.solve(XtX + best_alpha * np.eye(X.shape[1]), X.T @ yc)
    pred = X @ coef + y_mean
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_full = float(1 - ss_res / max(ss_tot, 1e-12))
    return coef, float(y_mean), best_alpha, best_cv_r2, r2_full


def main():
    splits_dir = Path(SPLITS_DIR)
    train_ds = SplitDataset(splits_dir / "train")
    print(f"Loading {len(train_ds)} training samples...")

    # Pass 1: build per-sample feature matrix and per-sample surface y means
    cond_rows: list[np.ndarray] = []
    surf_means: list[np.ndarray] = []
    for i in tqdm(range(len(train_ds)), desc="pass 1 (fit)"):
        x, y, is_surface = train_ds[i]
        cond = x[0, COND_INDICES].numpy()  # [11]
        feat_row = build_feature_row(cond)  # [F]
        surf_y = y[is_surface]  # [N_surf, 3]
        # Guard against samples with zero surface nodes (shouldn't happen but be safe)
        if surf_y.shape[0] == 0:
            print(f"  WARN: sample {i} has no surface nodes")
            continue
        surf_mean = surf_y.mean(dim=0).numpy().astype(np.float64)  # [3]
        cond_rows.append(feat_row)
        surf_means.append(surf_mean)

    X_raw = np.stack(cond_rows, axis=0)  # [N, F]
    Y = np.stack(surf_means, axis=0)  # [N, 3]
    print(f"  X shape: {X_raw.shape}, Y shape: {Y.shape}")

    # Standardize features (zero-mean, unit-std)
    feature_means = X_raw.mean(axis=0)
    feature_stds = X_raw.std(axis=0)
    feature_stds_safe = np.where(feature_stds > 1e-8, feature_stds, 1.0)
    X = (X_raw - feature_means) / feature_stds_safe

    # Fit per-channel Ridge
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    coef_per_ch = np.zeros((3, X.shape[1]))
    intercept_per_ch = np.zeros(3)
    alpha_per_ch = np.zeros(3)
    r2_train_per_ch = np.zeros(3)
    cv_r2_per_ch = np.zeros(3)
    channel_names = ["Ux", "Uy", "p"]

    print("\nPer-channel Ridge fit:")
    for c in range(3):
        coef, intercept, alpha, cv_r2, r2 = fit_ridge_cv(X, Y[:, c], alphas)
        coef_per_ch[c] = coef
        intercept_per_ch[c] = intercept
        alpha_per_ch[c] = alpha
        r2_train_per_ch[c] = r2
        cv_r2_per_ch[c] = cv_r2
        print(f"  ch {c} ({channel_names[c]:2s}): alpha={alpha:>6.2f}  CV-R²={cv_r2:+.4f}  full-R²={r2:+.4f}")

    # Pass 2: compute residual (y - baseline) stats across all valid training
    # nodes (surface + volume), using mask=True for all nodes in saved samples.
    # We compute sum and sum-of-squares per channel using streaming reduction.
    sum_per_ch = np.zeros(3, dtype=np.float64)
    sum_sq_per_ch = np.zeros(3, dtype=np.float64)
    n_total = 0

    for i in tqdm(range(len(train_ds)), desc="pass 2 (residual stats)"):
        x, y, is_surface = train_ds[i]
        cond = x[0, COND_INDICES].numpy()
        feat_row = build_feature_row(cond)
        feat_std = (feat_row - feature_means) / feature_stds_safe
        # Per-sample baseline [3]
        baseline = coef_per_ch @ feat_std + intercept_per_ch  # [3]
        # Subtract from all nodes
        y_np = y.numpy().astype(np.float64)  # [N, 3]
        y_resid = y_np - baseline[None, :]
        sum_per_ch += y_resid.sum(axis=0)
        sum_sq_per_ch += (y_resid ** 2).sum(axis=0)
        n_total += y_np.shape[0]

    residual_y_mean = sum_per_ch / n_total
    residual_y_var = sum_sq_per_ch / n_total - residual_y_mean ** 2
    residual_y_std = np.sqrt(np.maximum(residual_y_var, 1e-12))

    print(f"\nResidual stats (all nodes, n={n_total}):")
    print(f"  residual_y_mean: {residual_y_mean.tolist()}")
    print(f"  residual_y_std:  {residual_y_std.tolist()}")
    print(f"  vs original y_mean (from stats.json):")
    print(f"    Ux: -> residual {residual_y_mean[0]:+.3f} std {residual_y_std[0]:.3f}")
    print(f"    Uy: -> residual {residual_y_mean[1]:+.3f} std {residual_y_std[1]:.3f}")
    print(f"    p:  -> residual {residual_y_mean[2]:+.3f} std {residual_y_std[2]:.3f}")

    # Save
    out_path = Path("data/linear_baseline.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_indices": COND_INDICES,
        "extra_features": EXTRA_FEATURES,
        "feature_means": feature_means.astype(np.float64),
        "feature_stds": feature_stds_safe.astype(np.float64),
        "coef_per_ch": coef_per_ch.astype(np.float64),
        "intercept_per_ch": intercept_per_ch.astype(np.float64),
        "alpha_per_ch": alpha_per_ch.astype(np.float64),
        "r2_train_per_ch": r2_train_per_ch.astype(np.float64),
        "cv_r2_per_ch": cv_r2_per_ch.astype(np.float64),
        "residual_y_mean": residual_y_mean.astype(np.float64),
        "residual_y_std": residual_y_std.astype(np.float64),
        "n_train_samples": int(X_raw.shape[0]),
        "n_train_nodes": int(n_total),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved baseline to {out_path}")


if __name__ == "__main__":
    main()
