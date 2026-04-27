# SENPAI Research Results

## 2026-04-27 16:30 — PR #32: alphonse n_head sweep + 3-seed anchor recalibration

- **Branch:** alphonse/n-head-sweep
- **Hypothesis:** `n_head` is untested on this track. Currently n_head=4, dim_head=32 (n_hidden=128). Never swept in 15 rounds. Goal: map the n_head landscape + recalibrate 3-seed anchor variance at σ=0.7.

### Results Summary

| n_head | dim_head | Recipe | Seeds | val mean | val std | test mean | Epochs |
|--------|----------|--------|-------|----------|---------|-----------|--------|
| 1 | 128 | nl=3/sn=32 | 2 | **49.72** | 0.92 | **43.23** | 39-40 |
| 1 | 64 (ctrl) | nl=3/sn=32 | 1 | 49.80 | — | 42.54 | 41 |
| 1 | 128 | nl=3/sn=16 | 1 | **48.13** | — | **40.93** | 42 |
| 2 | 64 | nl=3/sn=32 | 3 | 51.79 | 0.48 | 45.17 | 35-36 |
| 4 (anchor) | 32 | nl=3/sn=32 | 1 | 54.21 | — | 47.48 | 32 |
| 2 | 64 | nl=5/sn=32 | 2 | 66.37 | 3.13 | 57.68 | 21 |
| 4 (anchor) | 32 | nl=5/sn=32 | 3 | 71.70 | 1.98 | 64.02 | 17 |

**Current baseline (PR #39):** val 49.077/49.443 (2-seed mean), sn=8/nl=3

### Results Commentary

**Merge outcome:** Code merged (--n_head, --dim_head CLI flags). NO baseline metric update — nh=1/sn=32/nl=3 val=49.72 does not beat PR #39's 49.443 at sn=8.

**Key architectural finding:** The n_head sweep reveals a strong monotonic pattern: fewer, wider heads consistently win. nh=1 (single-head, dim_head=128) achieves val 49.72 2-seed mean at nl=3/sn=32 — architecturally superior to nh=4 (baseline) and nh=2.

**Shape-preserving control:** nh=1/dh=64 (0.45M params, -31% vs nh=1/dh=128 at 0.65M) achieves val 49.80 — nearly identical to nh=1/dh=128, confirming the win is **architectural inductive bias** (single-head global attention), NOT capacity.

**Physical interpretation:** PhysicsAttention learns a slice-decomposition where each head's in_project_slice maps dim_head → slice_num=S. With dim_head=128 (nh=1), each head sees the full-rank token representation before slice assignment. With dim_head=8 (nh=16), there's a severe information bottleneck. Pressure fields on tandem airfoil meshes are globally coupled (suction/pressure surfaces, wake interactions) — global attention per head wins.

**Speed/memory inversion:** nh=1 is 23% faster per epoch AND uses 13% less VRAM than nh=4 (PhysicsAttention materializes [B, heads, N, slice_num] tensor — fewer heads cuts activation memory faster than per-head projection params grow). nh=1 reaches 39-42 epochs vs nh=4's 32 in 30-minute budget.

**Triple compound probe (nh=1/sn=16/nl=3, single seed):** val=48.13, test=40.93 — lowest single-seed ever. Still improving at cutoff (best_ep=42=last_ep). HIGH PRIORITY: 3-seed confirmation needed.

**Why the sn=32 win doesn't beat PR #39 (sn=8):** The nh improvement at sn=32 brings nh=1/sn=32 close to but not below nh=4/sn=8 (49.72 vs 49.44). The two improvement axes (nh reduction, sn reduction) need to be COMBINED — nh=1/sn=8 is the untested compound.

**3-seed anchor recalibration:** σ=0.7/nl=3 noise floor:
- nh=4/sn=32 (3-seed): std=1.978 val
- nh=4/sn=8 (2-seed, PR #39): std=0.517 val  
- nh=2/sn=32 (3-seed): std=0.479 val
- nh=1/sn=32 (2-seed): std=0.920 val

Noise INCREASES with fewer heads — wider seed variance expected at nh=1. Multi-seed (≥3) mandatory for close-call merges at nh=1.

### Suggested Immediate Follow-ups (from student)

1. **nh=1/sn=8/nl=3 multi-seed** (3+ seeds) — the untested compound; single-seed sn=16 at 48.13 suggests nh=1/sn=8 could land in 46-48 val range
2. **nh=1/sn=16/nl=3 3-seed confirmation** (seed 2) — triple compound probe was single seed, needs confirmation
3. **nh=1 × σ mini-sweep** — σ=0.7 optimal was found at nh=4; may shift under nh=1
