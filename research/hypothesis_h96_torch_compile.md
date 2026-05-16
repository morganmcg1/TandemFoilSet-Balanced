## Hypothesis

**H96: Enable torch.compile — efficiency unlock orthogonal to mixed precision.**

Same motivation as H95 (wall-cut-bound budget), different lever. torch.compile fuses ops, eliminates Python overhead, and produces optimized CUDA kernels. Typical wins: 15-30% s/epoch reduction with no semantic change.

Three modes to try (single arm = "default", separate arm = "reduce-overhead"):
- **Arm A: torch.compile(mode='default')** — full ahead-of-time graph capture + kernel fusion
- **Arm B: torch.compile(mode='reduce-overhead')** — CUDA graphs + lower per-step latency, helps when small-batch overhead dominates

**Predicted:**
- Arm A: 15-25% s/epoch reduction → val ~40-42 at unlocked epoch count
- Arm B: 20-30% reduction if batch overhead is significant → val ~40-42

**Risks:**
- Compile failures: model has dynamic shapes (variable point counts per sample), which can trigger recompiles. If recompiles dominate, compile mode is slower. Mitigation: try `dynamic=True` or fall back to `mode='reduce-overhead'`.
- First-epoch warm-up: compile adds 30-60s first-epoch overhead. Worth it if subsequent epochs save 20-30s each.

**Orthogonal to H95 (bf16):** if both win, they stack. Each is testable independently to attribute the speedup.

## Baseline

H78 Arm B val=42.3048 / test=40.5564 (PR #4097, MERGED). ~120 s/epoch, ~15 epochs/30-min budget.
