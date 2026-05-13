"""Precompute per-node signed surface curvature and cache it on disk.

For each sample, this script:

1. Identifies surface nodes via the ``is_surface`` mask.
2. Splits surface nodes into 1 (single-foil) or 2 (tandem) foils via 2-means
   on the (x, z) positions. Tandem detection: gap, stagger, or NACA foil-2
   features non-zero in ``x``.
3. Per foil, fits a local parabola in tangent/normal coordinates at each
   surface node using K nearest same-foil neighbours; extracts the signed
   curvature κ = 2a (parabola coefficient). Sign is consistent with the
   outward-pointing normal (away from the foil centroid).
4. Propagates curvature to volume nodes: each volume node copies κ from its
   nearest surface node (single argmin per node via batched ``cdist`` on GPU).
5. Writes ``curvature_cache/splits_v2/{split}/{file}.pt`` with a 1D float tensor
   of length N.

Run once before training:

    cd target/
    python build_curvature_cache.py

After the per-sample caches are written, the script aggregates abs(κ) over the
training set surface nodes and writes ``curvature_cache/splits_v2/stats.json``
with the 99th-percentile value used to normalise κ at training time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
CACHE_DIR = Path("curvature_cache/splits_v2")

# Splits to process (mirror the on-PVC layout).
SPLITS = [
    "train",
    "val_single_in_dist",
    "val_geom_camber_rc",
    "val_geom_camber_cruise",
    "val_re_rand",
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]

K_NEIGHBORS = 5  # number of same-foil neighbours used in the local parabola fit
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kmeans_2d(points: torch.Tensor, k: int = 2, max_iter: int = 20) -> torch.Tensor:
    """Plain k-means on 2D points. Returns labels [N]."""
    N = points.shape[0]
    if N < k:
        return torch.zeros(N, dtype=torch.long, device=points.device)
    points_f = points.float()
    # Pick two seeds that are far apart (k-means++ style for k=2).
    if k == 2:
        d0 = ((points_f - points_f[0]) ** 2).sum(-1)
        idx1 = d0.argmax()
        d1 = ((points_f - points_f[idx1]) ** 2).sum(-1)
        idx0 = d1.argmax()
        centers = torch.stack([points_f[idx0], points_f[idx1]], dim=0).clone()
    else:
        centers = points_f[torch.randperm(N, device=points.device)[:k]].clone()
    for _ in range(max_iter):
        d = ((points_f.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        labels = d.argmin(dim=-1)
        new_centers = torch.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = points_f[mask].mean(dim=0)
            else:
                new_centers[j] = centers[j]
        if torch.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels


def compute_curvature_surface(surf_xy: torch.Tensor, foil_labels: torch.Tensor,
                              k: int = K_NEIGHBORS) -> torch.Tensor:
    """Signed curvature per surface node via per-foil local parabola fit.

    For each surface node:
      - K nearest same-foil neighbours
      - SVD on the K×2 displacement matrix → tangent / normal directions
      - Fit h = a · s² (least squares) → κ = 2a
      - Orient: flip κ sign so the (outward-pointing-from-centroid) normal
        points away from the foil centroid.
    """
    Ns = surf_xy.shape[0]
    kappa = torch.zeros(Ns, dtype=torch.float32, device=surf_xy.device)
    surf_xy_f = surf_xy.float()
    for fid in foil_labels.unique():
        foil_mask = foil_labels == fid
        foil_pos = surf_xy_f[foil_mask]
        Nf = foil_pos.shape[0]
        if Nf < 4:
            continue
        diffs = foil_pos.unsqueeze(0) - foil_pos.unsqueeze(1)
        dists_sq = (diffs ** 2).sum(-1)
        dists_sq.fill_diagonal_(float("inf"))
        kk = min(k, Nf - 1)
        _, knn_idx = torch.topk(dists_sq, k=kk, dim=-1, largest=False)
        nb_pos = foil_pos[knn_idx]
        rel = nb_pos - foil_pos.unsqueeze(1)  # [Nf, kk, 2]
        _, _, Vh = torch.linalg.svd(rel, full_matrices=False)  # [Nf, kk, 2], Vh: [Nf, 2, 2]
        tangent = Vh[:, 0, :]
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        s = (rel * tangent.unsqueeze(1)).sum(-1)
        h = (rel * normal.unsqueeze(1)).sum(-1)
        num = (s ** 2 * h).sum(-1)
        denom = (s ** 4).sum(-1).clamp(min=1e-12)
        a = num / denom
        kappa_f = 2 * a
        centroid = foil_pos.mean(dim=0)
        outward = foil_pos - centroid
        outward_dot_normal = (outward * normal).sum(-1)
        sign_flip = torch.where(outward_dot_normal < 0, -1.0, 1.0)
        kappa_f = kappa_f * sign_flip
        kappa[foil_mask] = kappa_f.float()
    return kappa


def nearest_surface_idx(vol_pos: torch.Tensor, surf_pos: torch.Tensor,
                        chunk: int = 16384) -> torch.Tensor:
    """For each volume node, return the index of its nearest surface node."""
    Nv = vol_pos.shape[0]
    indices = []
    for i in range(0, Nv, chunk):
        v_chunk = vol_pos[i:i + chunk]
        d = torch.cdist(v_chunk.unsqueeze(0), surf_pos.unsqueeze(0)).squeeze(0)
        indices.append(d.argmin(dim=-1))
    return torch.cat(indices)


def is_tandem_sample(x: torch.Tensor) -> bool:
    """Tandem detection from per-sample features (same for all nodes in a sample).

    Single-foil rows have gap, stagger, AoA2, and foil-2 NACA all zero.
    """
    return bool(
        x[0, 18].item() != 0.0  # AoA2
        or x[0, 19].item() != 0.0  # NACA2 camber
        or x[0, 20].item() != 0.0  # NACA2 position
        or x[0, 21].item() != 0.0  # NACA2 thickness
        or x[0, 22].item() != 0.0  # gap
        or x[0, 23].item() != 0.0  # stagger
    )


def compute_sample_curvature(x: torch.Tensor, is_surface: torch.Tensor) -> torch.Tensor:
    """Per-sample signed curvature for every node (surface + volume)."""
    x = x.to(DEVICE)
    is_surface = is_surface.to(DEVICE)
    pos = x[:, :2]
    N = pos.shape[0]
    kappa = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    if not is_surface.any():
        return kappa.cpu()

    surf_idx = is_surface.nonzero(as_tuple=True)[0]
    surf_pos = pos[is_surface]
    is_tandem = is_tandem_sample(x.cpu())
    if is_tandem:
        foil_labels = kmeans_2d(surf_pos, k=2)
    else:
        foil_labels = torch.zeros(surf_pos.shape[0], dtype=torch.long, device=DEVICE)

    kappa_surf = compute_curvature_surface(surf_pos, foil_labels)
    kappa[surf_idx] = kappa_surf

    vol_mask = ~is_surface
    if vol_mask.any():
        vol_pos = pos[vol_mask]
        nearest = nearest_surface_idx(vol_pos, surf_pos)
        vol_idx = vol_mask.nonzero(as_tuple=True)[0]
        kappa[vol_idx] = kappa_surf[nearest]

    return kappa.cpu()


def build_split(split: str, x_dir: Path, gt_dir: Path | None, out_dir: Path) -> list[float]:
    """Compute and write per-sample curvature for one split. Returns abs(κ) values
    over surface nodes (for p99 stats — only used by ``train`` split's caller)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(x_dir.glob("*.pt"))
    abs_kappa_surf: list[float] = []
    for f in tqdm(files, desc=split):
        sample = torch.load(f, weights_only=True)
        x = sample["x"]
        # Test splits store x under ``x_dir``; ``is_surface`` lives in the
        # ground-truth shard so we read both when present.
        if "is_surface" in sample:
            is_surface = sample["is_surface"]
        else:
            assert gt_dir is not None, f"is_surface missing for {f}"
            gt = torch.load(gt_dir / f.name, weights_only=True)
            is_surface = gt["is_surface"]
        kappa = compute_sample_curvature(x, is_surface)
        torch.save(kappa, out_dir / f.name)
        if split == "train":
            abs_kappa_surf.extend(kappa[is_surface].abs().tolist())
    return abs_kappa_surf


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    abs_kappa_train: list[float] = []
    for split in SPLITS:
        x_dir = SPLITS_DIR / split
        # Test splits keep is_surface in the hidden GT directory.
        gt_dir = (SPLITS_DIR / f".{split}_gt") if split.startswith("test_") else None
        out_dir = CACHE_DIR / split
        if not x_dir.exists():
            print(f"[skip] {x_dir} does not exist")
            continue
        vals = build_split(split, x_dir, gt_dir, out_dir)
        if split == "train":
            abs_kappa_train = vals

    if abs_kappa_train:
        arr = np.asarray(abs_kappa_train, dtype=np.float64)
        stats = {
            "kappa_abs_p50": float(np.quantile(arr, 0.50)),
            "kappa_abs_p90": float(np.quantile(arr, 0.90)),
            "kappa_abs_p95": float(np.quantile(arr, 0.95)),
            "kappa_abs_p99": float(np.quantile(arr, 0.99)),
            "kappa_abs_max": float(arr.max()),
            "kappa_abs_mean": float(arr.mean()),
            "n_surface_nodes": int(arr.size),
            "k_neighbors": K_NEIGHBORS,
        }
        with open(CACHE_DIR / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("\nCurvature cache stats (train surface nodes):")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
