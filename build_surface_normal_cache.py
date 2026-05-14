"""Precompute per-node surface-normal direction (n_x, n_z) and cache it on disk.

For each sample, this script:

1. Identifies surface nodes via the ``is_surface`` mask.
2. Splits surface nodes into 1 (single-foil) or 2 (tandem) foils via 2-means
   on the (x, z) positions. Tandem detection: gap, stagger, or NACA foil-2
   features non-zero in ``x``.
3. Per foil, computes a local tangent direction at each surface node via SVD
   on K nearest same-foil neighbours; the unit normal is the 90° rotation of
   the tangent. Sign is flipped where needed so the normal points outward
   (away from the per-foil centroid).
4. Propagates normals to volume nodes: each volume node copies the (n_x, n_z)
   pair from its nearest surface node (batched ``cdist`` argmin).
5. Writes ``surface_normal_cache/splits_v2/{split}/{file}.pt`` with a 2D float
   tensor of shape [N, 2] per sample.

Run once before training:

    cd target/
    python build_surface_normal_cache.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
CACHE_DIR = Path("surface_normal_cache/splits_v2")

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

K_NEIGHBORS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kmeans_2d(points: torch.Tensor, k: int = 2, max_iter: int = 20) -> torch.Tensor:
    """Plain k-means on 2D points. Returns labels [N]."""
    N = points.shape[0]
    if N < k:
        return torch.zeros(N, dtype=torch.long, device=points.device)
    points_f = points.float()
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


def compute_normals_surface(surf_xy: torch.Tensor, foil_labels: torch.Tensor,
                            k: int = K_NEIGHBORS) -> torch.Tensor:
    """Outward unit normals per surface node via per-foil local SVD tangent fit.

    Returns: ``[Ns, 2]`` (n_x, n_z), unit length, oriented away from per-foil
    centroid.
    """
    Ns = surf_xy.shape[0]
    normals = torch.zeros(Ns, 2, dtype=torch.float32, device=surf_xy.device)
    surf_xy_f = surf_xy.float()
    for fid in foil_labels.unique():
        foil_mask = foil_labels == fid
        foil_pos = surf_xy_f[foil_mask]
        Nf = foil_pos.shape[0]
        if Nf < 4:
            # Degenerate: leave normals as zeros; only happens on absurdly small
            # foils which don't appear in this corpus.
            continue
        diffs = foil_pos.unsqueeze(0) - foil_pos.unsqueeze(1)
        dists_sq = (diffs ** 2).sum(-1)
        dists_sq.fill_diagonal_(float("inf"))
        kk = min(k, Nf - 1)
        _, knn_idx = torch.topk(dists_sq, k=kk, dim=-1, largest=False)
        nb_pos = foil_pos[knn_idx]
        rel = nb_pos - foil_pos.unsqueeze(1)  # [Nf, kk, 2]
        # SVD on the per-node displacement matrix. ``Vh[:, 0, :]`` is the
        # primary direction of variation = unit tangent.
        _, _, Vh = torch.linalg.svd(rel, full_matrices=False)
        tangent = Vh[:, 0, :]
        # 90°-rotation of tangent → unit normal.
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
        centroid = foil_pos.mean(dim=0)
        outward = foil_pos - centroid
        outward_dot_normal = (outward * normal).sum(-1, keepdim=True)
        sign_flip = torch.where(outward_dot_normal < 0, -1.0, 1.0)
        normal = normal * sign_flip
        # Re-normalise (SVD output is already unit-length up to numerical noise,
        # but this guards against any drift from the sign flip).
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        normals[foil_mask] = normal.float()
    return normals


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
    """Single-foil rows have AoA2, NACA2, gap, stagger all zero."""
    return bool(
        x[0, 18].item() != 0.0
        or x[0, 19].item() != 0.0
        or x[0, 20].item() != 0.0
        or x[0, 21].item() != 0.0
        or x[0, 22].item() != 0.0
        or x[0, 23].item() != 0.0
    )


def compute_sample_normals(x: torch.Tensor, is_surface: torch.Tensor) -> torch.Tensor:
    """Per-sample (n_x, n_z) tensor for every node (surface + volume).

    Returns ``[N, 2]`` on CPU.
    """
    x = x.to(DEVICE)
    is_surface = is_surface.to(DEVICE)
    pos = x[:, :2]
    N = pos.shape[0]
    normals = torch.zeros(N, 2, dtype=torch.float32, device=DEVICE)
    if not is_surface.any():
        return normals.cpu()

    surf_idx = is_surface.nonzero(as_tuple=True)[0]
    surf_pos = pos[is_surface]
    is_tandem = is_tandem_sample(x.cpu())
    if is_tandem:
        foil_labels = kmeans_2d(surf_pos, k=2)
    else:
        foil_labels = torch.zeros(surf_pos.shape[0], dtype=torch.long, device=DEVICE)

    normals_surf = compute_normals_surface(surf_pos, foil_labels)
    normals[surf_idx] = normals_surf

    vol_mask = ~is_surface
    if vol_mask.any():
        vol_pos = pos[vol_mask]
        nearest = nearest_surface_idx(vol_pos, surf_pos)
        vol_idx = vol_mask.nonzero(as_tuple=True)[0]
        normals[vol_idx] = normals_surf[nearest]

    return normals.cpu()


def build_split(split: str, x_dir: Path, gt_dir: Path | None, out_dir: Path
                ) -> tuple[int, int, float]:
    """Compute and write per-sample normals for one split.

    Returns (n_samples, n_surface_nodes_total, surface_norm_mean_magnitude).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(x_dir.glob("*.pt"))
    n_surf = 0
    norm_mag_sum = 0.0
    for f in tqdm(files, desc=split):
        sample = torch.load(f, weights_only=True)
        x = sample["x"]
        if "is_surface" in sample:
            is_surface = sample["is_surface"]
        else:
            assert gt_dir is not None, f"is_surface missing for {f}"
            gt = torch.load(gt_dir / f.name, weights_only=True)
            is_surface = gt["is_surface"]
        normals = compute_sample_normals(x, is_surface)
        torch.save(normals, out_dir / f.name)
        if split == "train":
            surf_normals = normals[is_surface]
            n_surf += int(surf_normals.shape[0])
            norm_mag_sum += float(surf_normals.norm(dim=-1).sum())
    surf_mag_mean = (norm_mag_sum / n_surf) if n_surf > 0 else 0.0
    return len(files), n_surf, surf_mag_mean


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_stats = {}
    for split in SPLITS:
        x_dir = SPLITS_DIR / split
        gt_dir = (SPLITS_DIR / f".{split}_gt") if split.startswith("test_") else None
        out_dir = CACHE_DIR / split
        if not x_dir.exists():
            print(f"[skip] {x_dir} does not exist")
            continue
        n_samples, n_surf, surf_mag_mean = build_split(split, x_dir, gt_dir, out_dir)
        if split == "train":
            train_stats = {
                "n_train_samples": n_samples,
                "n_train_surface_nodes": n_surf,
                "train_surf_normal_magnitude_mean": surf_mag_mean,
                "k_neighbors": K_NEIGHBORS,
            }
    if train_stats:
        with open(CACHE_DIR / "stats.json", "w") as f:
            json.dump(train_stats, f, indent=2)
        print("\nSurface-normal cache stats:")
        for k, v in train_stats.items():
            print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
