# SENPAI Research Results — icml-appendix-willow-pai2c-r4

This log records every reviewed experiment. New entries appended at the bottom.

---

## 2026-04-27 18:30 — Launch: Round 1 PRs Dispatched

Round 1: 8 hypotheses across 5 strategy tiers assigned. No results yet — entries will be added on PR review.

| PR | Student | Hypothesis | Tier | Predicted Δ val_avg/mae_surf_p |
|---|---|---|---|---|
| #197 | willowpai2c4-alphonse | H1: Re-conditional loss reweighting (per-sample y_std) | loss | −5 to −12% |
| #201 | willowpai2c4-askeladd | H7: Deeper Transolver (10L×128 + 8L×192) | architecture | −4 to −10% |
| #205 | willowpai2c4-edward | H10: EMA weights (decay 0.9999 + 0.999) | optimizer | −2 to −5% |
| #212 | willowpai2c4-fern | H5: Physics features (1/√Re, sin/cos AoA) | features | −4 to −9% |
| #217 | willowpai2c4-frieren | H9: OneCycleLR warmup + grad clip | optimizer | −3 to −7% |
| #222 | willowpai2c4-nezuko | H12: Surface-aware attention bias | architecture | −5 to −10% |
| #231 | willowpai2c4-tanjiro | H11: Cruise z-reflection augmentation | data | −3 to −9% |
| #239 | willowpai2c4-thorfinn | H4: Per-channel uncertainty weighting (Kendall 2017) | loss | −3 to −7% |
