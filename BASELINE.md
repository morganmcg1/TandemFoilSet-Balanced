# Baseline — willow-pai2e-r3

Advisor branch: `icml-appendix-willow-pai2e-r3`
W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
Primary metric: `val_avg/mae_surf_p` (lower is better). Test mirror: `test_avg/mae_surf_p`.

---

## Founding baseline (round 1 — no hypothesis PR merged yet)

**Commit baseline via PR #807** (NaN-safe masked accumulation bug fix — landed 2026-04-28).
All subsequent runs produce finite `test_avg/mae_surf_p` numbers.

### Default model config (unmodified `train.py`)

- n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1M params)
- AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0
- 50-epoch cosine annealing (effective ~14 epochs at 30-min timeout)

### Best round-1 val metric (single seed)

| Run | Branch | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-----|--------|--------------------|----------------------|
| `thnnvgaw` | edward/lr-warmup-cosine v1 | **135.89** | null (pre-fix; re-eval pending) |
| `t0xgo0zv` | frieren/fourier-re-encoding v1 | 141.25 | null (pre-fix; re-eval pending) |
| `zaqz12qi` | alphonse/channel-weighted-surface-loss v1 | 146.10 | **130.90** (re-eval via PR #807) |

Single-seed run-to-run variance ≈ ±10% at 14-epoch budget.

### Founding test baseline (clean number for paper-facing comparisons)

`test_avg/mae_surf_p = 130.90` (W&B run `zaqz12qi`, alphonse channel-weighted v1, re-evaluated with fixed scorer in PR #807)

### Beat-threshold for round 2+

Future PRs must achieve **`val_avg/mae_surf_p < 135.89`** to demonstrate improvement above round-1 noise.
For a merge decision: any val_avg below current best (135.89) merges; gains <5% at single seed will be flagged for multi-seed confirmation.

---

*This file is updated after each merge. Entries are cumulative — do not delete prior entries.*
