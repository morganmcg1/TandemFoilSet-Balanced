# Round 137 — Post-norm topology change (pre-norm → post-norm at all 9 sites)

## Hypothesis

**Change the residual block topology from pre-norm (current) to post-norm at all 9 LayerNorm sites.** Tests whether the norm placement within the residual stream is load-bearing, directly motivated by #2939's decisive finding that LayerNorm's mean-centering is doing real work for CFD targets — the question of WHERE in the residual stream the centering happens is now a load-bearing structural question.

Currently:
```python
x = x + Attention(LN(x))     # pre-norm: norm inside residual
x = x + MLP(LN(x))           # pre-norm: norm inside residual
```

This PR:
```python
x = LN(x + Attention(x))     # post-norm: norm outside, centers the SUM
x = LN(x + MLP(x))           # post-norm: norm outside, centers the SUM
```

This is a **STRUCTURAL/REPRESENTATION axis intervention** — the only one not yet tested in this launch directly motivated by the LayerNorm load-bearing finding.

## Why this might WIN

1. **#2939 decisive finding inverts the question.** If LN mean-centering is load-bearing because per-Reynolds DC offsets need stripping, then post-norm centers the SUM `(x + Attention(x))` — meaning the DC offset is recomputed AFTER residual addition. This may give a CLEANER per-block DC strip than pre-norm (which strips only the pre-residual stream).

2. **#2939 student explicit suggestion.** Verbatim from results: *"Pre-norm vs post-norm × norm-type 2×2 would be more informative than either probe alone — the current pre-norm + LayerNorm baseline may be especially well-tuned, and the trade-off could flip with post-norm. (#2902-ish territory — won't pursue unless advisor asks.)"* — advisor is asking.

3. **Post-norm regularizes residual growth.** Pre-norm allows residual stream magnitude to grow unboundedly with depth (each block adds). Post-norm normalizes after addition, capping per-block magnitude. With LayerScale γ=1e-4 baseline, the residuals are tiny → post-norm has less to clip. May behave better than typical post-norm Transformers.

4. **Lion + post-norm has been shown to work well** on small Transformer models in recent literature (Liu et al. 2023 Lion paper Table 7). The Lion sign-step interacts cleanly with post-norm's contained residual magnitude.

5. **In_dist regression often correlates with residual magnitude over-growth.** Per-block normalization may reduce the in_dist-specific over-fitting that drives the meta-signal — post-norm caps each block's effective influence on output.

## Why this might LOSS

1. **Pre-norm is the modern default for a reason.** Most Transformer literature uses pre-norm for training stability. Post-norm requires careful warmup. The baseline already has 3-ep LinearLR warmup which should mitigate this risk.

2. **Lion + post-norm may have training instabilities at peak LR=1.5e-4.** If Lion at lr=1.5e-4 destabilizes under post-norm (which has higher effective gradient magnitudes), would need a smaller LR. Stay at baseline LR; if diverges, REPORT.

3. **The baseline pre-norm recipe is well-tuned.** Any deviation might just hurt. Could be a wash or LOSS.

4. **The CFD target dynamics may not benefit from per-block magnitude capping.** If high-frequency components in p (pressure) need wide residual range, post-norm caps them.

## Falsifiable predictions

- **WIN** (val < 30.5605): Post-norm is better. Structural topology axis OPEN; consider norm-type × topology 2×2 ablations next (RMSNorm + post-norm).
- **PARTIAL** (val ≈ 30.5605 ± 0.5%): Topology is wash. Move to other structural axes.
- **LOSS** (val > 31.0): Pre-norm is load-bearing. Closes topology axis from the post-norm side. May indicate pre-norm + LN is the genuinely optimal configuration.
- **DIVERGE** (NaN or val > 60): Post-norm + Lion at lr=1.5e-4 is unstable. Closes axis as needing different LR; consider warmup extension.

## Implementation

### Step 1: Locate TransolverBlock.forward() in `train.py`

The current pre-norm pattern looks like:
```python
def forward(self, x, fx=None, T=1.0):
    x = x + self.ls_1(self.Attn(self.ln_1(x), fx, T))  # γ_attn * attn(ln(x))
    x = x + self.ls_2(self.mlp(self.ln_2(x)))           # γ_mlp * mlp(ln(x))
    return x
```

(Block 3 has an additional `self.ln_3` at the end.)

### Step 2: Switch to post-norm pattern

```python
def forward(self, x, fx=None, T=1.0):
    x = self.ln_1(x + self.ls_1(self.Attn(x, fx, T)))   # ln(x + γ_attn * attn(x))
    x = self.ln_2(x + self.ls_2(self.mlp(x)))            # ln(x + γ_mlp * mlp(x))
    return x
```

For block 3 with the extra `ln_3` at the end: keep `ln_3` at end (it's a final cleanup norm). Or remove it (since post-norm already normalizes at end of block 3) — TEST BOTH and report which works.

### Step 3: Verify the norm input dimension matches

Pre-norm: `LN(x)` where `x.shape=(B,N,96)`. Post-norm: `LN(x + residual)` where the SUM has shape `(B,N,96)`. Same dim, no shape changes needed.

### Step 4: Startup diagnostics

```python
print(f"Block topology: POST-NORM (LN applied AFTER residual sum)")
print(f"vs baseline pre-norm: LN applied BEFORE attention/mlp (inside residual)")
print(f"Norm sites: 9 (4× post-attn + 4× post-mlp + 1× ln_3 in last block)")
print(f"Param count: {sum(p.numel() for p in model.parameters())}")  # expect 407,940 unchanged
print(f"Motivation: #2939 found LN mean-centering load-bearing; this tests WHERE the centering happens in residual stream")
```

### Step 5: Stability monitoring

Track every 5 epochs:
- Train loss (surf + vol). If NaN/Inf at any point, terminate.
- Residual magnitude per block (sqrt(mean(x**2)) inside forward()) at ep1, 5, 10, 30, 60. Should be CAPPED under post-norm vs unbounded under pre-norm.

If divergence in ep1-3 (during warmup), that's the post-norm cold-start instability signal. The 3-ep warmup ramp should mitigate; if it doesn't, post-norm is incompatible with this recipe.

### Step 6: Per-split test diagnostic

At ep60, report not just val_avg but per-split val + test. Critical to check whether post-norm specifically moves the in_dist hit (residual-magnitude-driven) without breaking the OOD splits.

## Baseline (PR #2879) and prior closure

| Metric | Baseline (pre-norm) | This PR target |
|---|---|---|
| val_avg/mae_surf_p | **30.5605** | beat baseline |
| Residual block topology | pre-norm: `x + f(LN(x))` | post-norm: `LN(x + f(x))` |
| Param count | 407,940 | 407,940 (unchanged) |

For reference (norm-axis prior closures):
- Baseline: pre-norm + LayerNorm at 9 sites: val 30.5605 ✓
- #2939: pre-norm + RMSNorm at 9 sites: val 32.7855 (+7.28% LOSS) — closed mean-centering as load-bearing
- This PR: **post-norm + LayerNorm at 9 sites** — tests topology axis

**Beat:** `val_avg/mae_surf_p < 30.5605`

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-nezuko \
    --experiment_name "charliepai2g48h5-nezuko/post-norm-topology" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

Epochs=60 to fit SENPAI_TIMEOUT=30min. **No W&B / wandb.**

## Reporting

Post results as a PR comment including:

1. val_avg/test_avg vs baseline 30.5605 / 26.5160
2. Per-split val + test breakdown with Δ vs baseline
3. **Residual magnitude trajectory:** sqrt(mean(x**2)) per block at ep1, 5, 10, 30, 60 — confirm post-norm caps magnitudes vs baseline
4. **Stability:** any NaN/Inf during training? Convergence trajectory at ep1-3 (warmup) vs ep4-60 (cosine).
5. Param count confirmation (~407,940)
6. Epochs completed (target: 60), sec/epoch, peak GPU memory
7. Train→val loss gap at convergence
8. **Topology axis verdict:** is pre-norm load-bearing? Or is post-norm at parity / better?
9. **Meta-signal check:** does post-norm specifically affect in_dist (residual-magnitude-driven) vs OOD splits?
10. **Plain-language verdict:** WIN (post-norm helps; topology axis open) / WASH (topology doesn't matter; close axis) / LOSS (pre-norm is the specifically-optimal topology for this recipe).

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["models/<experiment>/metrics.jsonl","models/<experiment>/metrics.yaml"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best-val>},"test_metric":{"name":"test_avg/mae_surf_p","value":<test-from-best-val>}}
```
