"""Integrity contract for materialized TandemFoilSet splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("split_manifest.json")


def manifest_sha256(manifest_path: str | Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def require_materialized_manifest(
    splits_dir: str | Path,
    manifest_path: str | Path = MANIFEST_PATH,
) -> str:
    """Require the PVC data to match the exact checked-in split manifest."""
    splits_dir = Path(splits_dir)
    manifest_path = Path(manifest_path)
    expected_digest = manifest_sha256(manifest_path)
    meta_path = splits_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"Materialized split metadata is missing at {meta_path}; "
            "run `python data/prepare_splits.py`"
        )

    manifest = json.loads(manifest_path.read_text())
    meta = json.loads(meta_path.read_text())
    materialized_digest = meta.get("split_manifest_sha256")
    if materialized_digest != expected_digest:
        raise RuntimeError(
            "Materialized splits are stale: meta.json records manifest "
            f"{materialized_digest!r}, but the checked-in manifest is {expected_digest}; "
            "run `python data/prepare_splits.py`"
        )
    if meta.get("split_counts") != manifest["split_counts"]:
        raise RuntimeError("Materialized split counts do not match the checked-in manifest")

    for split_name, expected_count in manifest["split_counts"].items():
        actual_count = len(list((splits_dir / split_name).glob("*.pt")))
        if actual_count != expected_count:
            raise RuntimeError(
                f"Materialized split {split_name!r} has {actual_count} files; "
                f"the checked-in manifest requires {expected_count}"
            )
        if split_name in manifest["test_splits"]:
            gt_count = len(list((splits_dir / f".{split_name}_gt").glob("*.pt")))
            if gt_count != expected_count:
                raise RuntimeError(
                    f"Materialized ground truth for {split_name!r} has {gt_count} files; "
                    f"the checked-in manifest requires {expected_count}"
                )
    return expected_digest
