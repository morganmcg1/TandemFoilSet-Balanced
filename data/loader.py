"""Dataset and loaders for the pre-materialized TandemFoil splits on PVC.

Run ``python data/prepare_splits.py`` once on the PVC to materialize
``splits_v2/`` from ``data/split_manifest.json``; this module then streams
per-sample ``.pt`` files on demand.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

X_DIM = 24

VAL_SPLIT_NAMES = [
    "val_single_in_dist",
    "val_geom_camber_rc",
    "val_geom_camber_cruise",
    "val_re_rand",
]

TEST_SPLIT_NAMES = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")

_MANIFEST_PATH = Path(__file__).parent / "split_manifest.json"

# Pickle-file-derived 3-way domain id (0=racecar_single, 1=racecar_tandem, 2=cruise).
# This is the *clean* per-sample domain — derived from the source pickle file via
# the manifest's ``pickle_files`` + ``file_sizes``. We deliberately do NOT use
# ``meta.json`` ``domain_groups`` because that field encodes train-local
# positions in the *unsorted* train list whereas ``manifest['splits']['train']``
# is sorted before .pt materialization, so a direct lookup gives wrong labels for
# many tandem samples (racecar_tandem ↔ cruise scramble).
DOMAIN_NAME_TO_ID = {"racecar_single": 0, "racecar_tandem": 1, "cruise": 2}
N_DOMAINS = 3
_TEST_SHUFFLE_SEED = 123  # matches prepare_splits.py


def _classify_pickle_file(name: str) -> int:
    n = name.lower()
    if "single" in n:
        return DOMAIN_NAME_TO_ID["racecar_single"]
    if n.startswith("racecar"):
        return DOMAIN_NAME_TO_ID["racecar_tandem"]
    if n.startswith("cruise"):
        return DOMAIN_NAME_TO_ID["cruise"]
    raise ValueError(f"Cannot classify pickle file: {name!r}")


def _build_global_to_domain(manifest: dict) -> list[int]:
    """Per-global-idx domain id, length = sum(file_sizes)."""
    domain_per_global: list[int] = []
    for fi, sz in enumerate(manifest["file_sizes"]):
        did = _classify_pickle_file(manifest["pickle_files"][fi])
        domain_per_global.extend([did] * sz)
    return domain_per_global


def _split_domain_ids(split_name: str, manifest: dict, global_to_domain: list[int]) -> list[int]:
    """Per-file-index domain id for files materialized under ``split_name``.

    Train and val splits are written in the order of ``manifest['splits'][name]``
    (sorted ascending). Test splits are written in shuffled order with seed
    ``_TEST_SHUFFLE_SEED`` to match ``prepare_splits.py``.
    """
    global_idxs = list(manifest["splits"][split_name])
    if split_name.startswith("test_"):
        global_idxs = list(global_idxs)
        np.random.default_rng(_TEST_SHUFFLE_SEED).shuffle(global_idxs)
    return [global_to_domain[g] for g in global_idxs]


def _load_manifest() -> dict:
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


class SplitDataset(Dataset):
    """Dataset backed by individual ``{x, y, is_surface}`` .pt files.

    ``domain_ids`` (optional) is a per-file int list; ``__getitem__`` returns
    ``(x, y, is_surface, domain_id)`` so the model can condition on domain.
    """

    def __init__(self, directory: str | Path, domain_ids: list[int] | None = None):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.pt"))
        if domain_ids is not None and len(domain_ids) != len(self.files):
            raise ValueError(
                f"domain_ids length {len(domain_ids)} != files length {len(self.files)} "
                f"for {self.directory}"
            )
        self.domain_ids = domain_ids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = torch.load(self.files[idx], weights_only=True)
        domain_id = self.domain_ids[idx] if self.domain_ids is not None else 0
        return s["x"], s["y"], s["is_surface"], domain_id


class TestDataset(Dataset):
    """Test split dataset: reads ``x`` from ``test_*/`` and ``y`` from ``.test_*_gt/``.

    Ground truth is held separately on PVC so prediction submissions can be
    scored blind. For in-repo training, we load both sides here.
    """

    def __init__(
        self,
        x_dir: str | Path,
        gt_dir: str | Path,
        domain_ids: list[int] | None = None,
    ):
        self.x_files = sorted(Path(x_dir).glob("*.pt"))
        self.gt_files = sorted(Path(gt_dir).glob("*.pt"))
        assert len(self.x_files) == len(self.gt_files), (
            f"Test split file-count mismatch: {len(self.x_files)} x-files vs "
            f"{len(self.gt_files)} gt-files"
        )
        if domain_ids is not None and len(domain_ids) != len(self.x_files):
            raise ValueError(
                f"domain_ids length {len(domain_ids)} != files length {len(self.x_files)} "
                f"for {x_dir}"
            )
        self.domain_ids = domain_ids

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        xs = torch.load(self.x_files[idx], weights_only=True)
        gt = torch.load(self.gt_files[idx], weights_only=True)
        domain_id = self.domain_ids[idx] if self.domain_ids is not None else 0
        return xs["x"], gt["y"], gt["is_surface"], domain_id


def pad_collate(batch):
    """Pad variable-length mesh samples into a batch.

    Returns ``(x, y, is_surface, mask, domain_ids)``: the first four are
    ``[B, N_max, ...]`` and ``domain_ids`` is ``[B]`` int64.
    """
    xs, ys, surfs, dids = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (x, y, sf) in enumerate(zip(xs, ys, surfs)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
    domain_ids = torch.tensor(dids, dtype=torch.long)
    return x_pad, y_pad, surf_pad, mask, domain_ids


def _load_stats(splits_dir: Path) -> dict[str, torch.Tensor]:
    with open(splits_dir / "stats.json") as f:
        raw = json.load(f)
    return {k: torch.tensor(raw[k], dtype=torch.float32) for k in ("x_mean", "x_std", "y_mean", "y_std")}


def load_data(
    splits_dir: str | Path = SPLITS_DIR,
    debug: bool = False,
) -> tuple[SplitDataset, dict[str, SplitDataset], dict[str, torch.Tensor], torch.Tensor]:
    """Train + val datasets, normalization stats, and balanced-domain weights.

    Returns ``(train_ds, val_splits, stats, sample_weights)``.
    """
    splits_dir = Path(splits_dir)

    with open(splits_dir / "meta.json") as f:
        meta = json.load(f)
    stats = _load_stats(splits_dir)

    manifest = _load_manifest()
    global_to_domain = _build_global_to_domain(manifest)

    train_domain_ids = _split_domain_ids("train", manifest, global_to_domain)
    train_ds = SplitDataset(splits_dir / "train", domain_ids=train_domain_ids)

    val_splits = {
        name: SplitDataset(
            splits_dir / name,
            domain_ids=_split_domain_ids(name, manifest, global_to_domain),
        )
        for name in VAL_SPLIT_NAMES
    }

    if debug:
        train_ds.files = train_ds.files[:6]
        train_ds.domain_ids = train_ds.domain_ids[:6]
        for ds in val_splits.values():
            ds.files = ds.files[:2]
            ds.domain_ids = ds.domain_ids[:2]

    # Domain-balanced sample weights for the WeightedRandomSampler.
    # We use the *clean* domain_ids derived above (not meta['domain_groups'],
    # which encodes train-local positions in unsorted order and so mislabels
    # racecar_tandem ↔ cruise for many tandem samples).
    counts = [0] * N_DOMAINS
    for d in train_domain_ids:
        counts[d] += 1
    sample_weights = torch.tensor(
        [1.0 / counts[d] for d in train_domain_ids],
        dtype=torch.float64,
    )

    print(
        f"Train: {len(train_ds)} (domain counts: {counts}), "
        + ", ".join(f"{k}: {len(v)}" for k, v in val_splits.items())
    )
    return train_ds, val_splits, stats, sample_weights


def load_test_data(
    splits_dir: str | Path = SPLITS_DIR,
    debug: bool = False,
) -> dict[str, TestDataset]:
    """Test datasets keyed by split name (with joined hidden ground truth)."""
    splits_dir = Path(splits_dir)
    manifest = _load_manifest()
    global_to_domain = _build_global_to_domain(manifest)

    test_splits: dict[str, TestDataset] = {}
    for name in TEST_SPLIT_NAMES:
        ds = TestDataset(
            splits_dir / name,
            splits_dir / f".{name}_gt",
            domain_ids=_split_domain_ids(name, manifest, global_to_domain),
        )
        if debug:
            ds.x_files = ds.x_files[:2]
            ds.gt_files = ds.gt_files[:2]
            ds.domain_ids = ds.domain_ids[:2]
        test_splits[name] = ds
    print("Test: " + ", ".join(f"{k}: {len(v)}" for k, v in test_splits.items()))
    return test_splits
