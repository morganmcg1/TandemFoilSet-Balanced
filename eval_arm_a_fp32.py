"""Re-evaluate H123 checkpoints in fp32 to bypass bf16 overflow on test splits.

Both Arm A (K=0) and Arm B (K=1, scale=0.5) produced NaN/Inf in the
``test_geom_camber_cruise`` surface/volume pressure channels during the
end-of-training test pass (bf16 forward overflow on a few large pressure
samples). The Ux/Uy MAE remained finite. Re-running the saved checkpoint in
fp32 cleanly recovers ``mae_surf_p`` for that split.

We avoid importing ``train.py`` directly because ``train.py`` has no
``if __name__ == "__main__"`` guard and would re-run the whole training
pipeline on import. Instead we exec ``train.py`` source up to (but not
including) the ``cfg = sp.parse(Config)`` line — that gives us the model
classes and ``evaluate_split`` without launching training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    load_data,
    load_test_data,
    pad_collate,
)


def _load_train_module_lib():
    """Exec train.py source up to the training pipeline entry and return the module.

    Registers the new module in ``sys.modules`` so dataclasses can resolve
    ``cls.__module__`` while processing ``Config``.
    """
    import types
    src = Path(__file__).resolve().parent.joinpath("train.py").read_text()
    sentinel = "\ncfg = sp.parse(Config)"
    if sentinel not in src:
        raise RuntimeError("Could not find sentinel 'cfg = sp.parse(Config)' in train.py")
    lib_src = src.split(sentinel)[0]
    mod = types.ModuleType("train_lib")
    mod.__file__ = str(Path(__file__).resolve().parent / "train.py")
    sys.modules["train_lib"] = mod
    exec(compile(lib_src, "train.py", "exec"), mod.__dict__)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--out", default=None, help="Path to append a single JSONL event")
    args = parser.parse_args()

    train_lib = _load_train_module_lib()
    Transolver = train_lib.Transolver
    evaluate_split = train_lib.evaluate_split

    model_dir = Path(args.model_dir)
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = cfg["splits_dir"]

    # We only need stats for normalization; load_data also returns train/val datasets.
    _train, _val, stats, _ = load_data(splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    test_datasets = load_test_data(splits_dir, debug=False)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    fourier_pe_freqs = cfg["fourier_pe_freqs"] if cfg["fourier_pe"] else 0
    extra_pe_features = 4 * fourier_pe_freqs
    model_config = dict(
        space_dim=2,
        fun_dim=X_DIM - 2 + extra_pe_features,
        out_dim=3,
        n_hidden=cfg["model_config"]["n_hidden"],
        n_layers=cfg["n_layers"],
        n_head=cfg["n_head"],
        slice_num=cfg["slice_num"],
        mlp_ratio=cfg["model_config"]["mlp_ratio"],
        cond_dim=cfg["cond_dim"],
        ffn_act=cfg["ffn_act"],
        norm_type=cfg["norm_type"],
        fourier_pe_freqs=fourier_pe_freqs,
        fourier_pe_scale=cfg.get("fourier_pe_scale", 1.0),
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)
    state = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    surf_weight = cfg["surf_weight"]
    test_metrics = {
        name: evaluate_split(model, loader, stats, surf_weight, device, use_bf16=args.use_bf16)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)

    precision = "bf16" if args.use_bf16 else "fp32"
    print(f"\nRe-evaluated {model_dir.name} in {precision}")
    print(f"  test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(f"  {name:30s} surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}] "
              f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]")

    if args.out:
        record = {
            "event": "test_rerun",
            "precision": precision,
            "test_avg": test_avg,
            "test_splits": test_metrics,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
