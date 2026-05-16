## Hypothesis

**H48: Replace MLP's GELU activation with GEGLU gating in the Transolver FFN — gated linear units for surface pressure gradients.**

The current Transolver FFN uses `Linear → GELU → Linear` pattern (per-layer width `mlp_ratio=2 × n_hidden = 256`). For surface pressure fields where near-wall gradients are orders of magnitude larger than far-field values, a **gated** linear unit could let the FFN learn per-neuron spatial selectivity without widening n_hidden (which already failed in H33).

**GEGLU formulation** (Shazeer 2020, "GLU Variants Improve Transformer"):
```
GEGLU(x) = (xW₁ + b₁) ⊙ gelu(xW₂ + b₂)
```
where ⊙ is element-wise product. The gate path `gelu(xW₂+b₂)` selectively amplifies or suppresses each channel of the value path `(xW₁+b₁)`. Architecturally this is two parallel linear projections from input dim to FFN inner dim, multiplied element-wise after the gate is GELU'd.

**Why this might help in CFD surrogate context:**
1. Pressure gradients near boundary layers are sharp; the gate can learn to *boost* attention on those nodes' channels while suppressing far-field noise.
2. Total parameter change is small relative to model size — n_hidden×n_hidden×2 vs n_hidden×n_hidden×1 (one extra n_hidden² block per FFN). At n_hidden=128, mlp_ratio=2, that's 32K extra params per Transolver block, ~160K for the full 5-block model (vs 891K baseline H37b). Modest +18%.
3. Validated mechanism for mesh-CFD by Mines Paris (2025) — gating mechanism specifically targets spatially heterogeneous gradient fields.
4. Stackable with all current best knobs: lr=1e-3, wd=5e-5, n_head=2, T_max=15.

**Two arms to compare gate functions:**

- **Arm A — GEGLU**: gate is `gelu(xW_gate + b)`. The original Shazeer formulation. Predicted strongest at this scale.
- **Arm B — SwiGLU**: gate is `silu(xW_gate + b)` (also called Swish-GLU). Same structural change; isolates whether the gate *shape* or the gate *mechanism* matters. SwiGLU is the default in LLaMA-2/3 and recent CFD work.

If Arm A > Arm B, the GELU gate is preferred (slightly less aggressive nonlinearity than SiLU). If Arm B > Arm A, the SiLU gate's stronger gradient flow at small magnitudes is the lever. Either way we learn what's mechanistically responsible.

## Instructions

Code change is **in `train.py` Transolver MLP class only** (line ~65). Do NOT modify the model architecture beyond the FFN block. Other components (attention, FiLM, slice projection, output head) stay untouched.

**Step 1: Add GEGLU/SwiGLU module classes after the MLP class (line ~82):**

```python
class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation on the gate path."""
    def __init__(self, dim_in, dim_out, gate_act="gelu"):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2 * dim_out)
        self.gate_act = nn.GELU() if gate_act == "gelu" else nn.SiLU()
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.gate_act(gate)


class GatedMLP(nn.Module):
    """MLP variant where the first hidden layer is a GEGLU/SwiGLU gated unit.
    
    Replaces the linear_pre + activation in the standard MLP. The hidden width
    is doubled internally for the gate projection but the user-visible
    hidden dimension is unchanged (so n_hidden stays the same downstream).
    """
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, gate_act="gelu", res=True):
        super().__init__()
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = GEGLU(n_input, n_hidden, gate_act=gate_act)
        self.linear_post = nn.Linear(n_hidden, n_output)
        # Subsequent layers (if any) use the same gated structure
        self.linears = nn.ModuleList(
            [GEGLU(n_hidden, n_hidden, gate_act=gate_act) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)
```

**Step 2: Add Config field (after `clip_grad_norm` line ~404):**

```python
ffn_act: str = "gelu"   # FFN activation: 'gelu' (default), 'geglu', 'swiglu'
```

**Step 3: Wire it into TransolverBlock (line ~175 where self.mlp is constructed):**

Replace:
```python
self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
               n_layers=1, act=act, res=False)
```

With:
```python
if cfg.ffn_act == "geglu":
    self.mlp = GatedMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, gate_act="gelu", res=False)
elif cfg.ffn_act == "swiglu":
    self.mlp = GatedMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, gate_act="silu", res=False)
else:
    self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                   n_layers=1, act=act, res=False)
```

Note: `n_layers=0` for GatedMLP because GEGLU already does the nonlinearity — we don't want an additional gated block on top. The structure is: `Linear(dim, 2*mlp_dim) → split → x * act(gate) → Linear(mlp_dim, dim)`. This matches Shazeer's GEGLU FFN.

You'll need to pass `cfg` (or just `cfg.ffn_act`) into TransolverBlock — add it as a constructor arg. Inspect how `cond_dim` is passed (around line 165) and follow that pattern.

**Step 4: Add `--ffn_act` CLI flag** following the pattern of other Config flags.

**Arm A — GEGLU (gate = GELU):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h48-geglu-nhead2-wd5e5 \
  --agent charliepai2i48h3-askeladd \
  --ffn_act geglu --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

**Arm B — SwiGLU (gate = SiLU):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h48-swiglu-nhead2-wd5e5 \
  --agent charliepai2i48h3-askeladd \
  --ffn_act swiglu --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs before wall, best epoch
- **Total parameter count** (expect ~1.05M vs 891K baseline — confirm GEGLU is wired in)
- Per-epoch val_avg trajectory
- Peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if Arm A diverges**: if val_avg at epoch 3 exceeds 200, kill and check weight init scale for the gate projection (standard nn.Linear init should be fine; if not, try `nn.init.xavier_uniform_(self.proj.weight)`).

## Baseline

Current best — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0 (tanjiro)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **66.1060** |
| val_single_in_dist/mae_surf_p | 74.3956 |
| val_geom_camber_rc/mae_surf_p | 78.9959 |
| val_geom_camber_cruise/mae_surf_p | 46.4384 |
| val_re_rand/mae_surf_p | 64.5940 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **64.4522** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + **n_head=2 (head_dim=64)** + wd=1e-4.

**Beat this: val_avg/mae_surf_p < 66.1060**

This experiment ALSO stacks wd=5e-5 (H38 orthogonal win), so both arms test architecture × optimizer compound. Predicted val_avg if GEGLU works: 64–65.5.

⚠ `test_avg/mae_surf_p` will appear NaN (pre-existing scoring bug). Report 3-split excl. cruise.

**Reproduce baseline:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h37b-nhead2-lr1e3-clip1 \
  --agent charliepai2i48h3-askeladd \
  --n_head 2 --lr 1e-3 --clip_grad_norm 1.0
```
