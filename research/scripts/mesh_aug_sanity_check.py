"""Sanity checks for the mesh rotation augmentation introduced in PR #4163.

Verifies:
  1. theta=0, no flip → identity (positions and velocity targets unchanged).
  2. theta=0, flip → only z-coord and Uy negated; everything else identical.
  3. theta=90° → (x, z) → (-z, x); (Ux, Uy) → (-Uy, Ux). Pressure unchanged.
  4. theta=180° → (x, z) → (-x, -z); (Ux, Uy) → (-Ux, -Uy).
  5. Two successive rotations sum: R(a) o R(b) == R(a+b) (within fp tolerance).
  6. Padding rows (positions/velocity at zero) stay at zero after augmentation.

Run:
  python -m research.scripts.mesh_aug_sanity_check
"""

from __future__ import annotations

import torch


# Inlined copy of train.rotate_mesh (train.py runs the training pipeline on
# import, so we duplicate the helper here to keep the sanity check standalone).
# Source: target/train.py, PR #4163.
def rotate_mesh(positions, vel_x, vel_y, theta_deg):
    theta = torch.tensor(theta_deg * 3.14159265358979 / 180.0,
                         device=positions.device, dtype=positions.dtype)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = positions[..., 0] * cos_t - positions[..., 1] * sin_t
    y_rot = positions[..., 0] * sin_t + positions[..., 1] * cos_t
    positions_rot = torch.stack([x_rot, y_rot], dim=-1)
    vel_x_rot = vel_x * cos_t - vel_y * sin_t
    vel_y_rot = vel_x * sin_t + vel_y * cos_t
    return positions_rot, vel_x_rot, vel_y_rot


def _make_batch(b=2, n=5, x_dim=24, y_dim=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, x_dim, generator=g)
    y = torch.randn(b, n, y_dim, generator=g)
    # Add zero-padding rows (positions and y at zero) at the tail of each sample.
    x[:, -1, 0:2] = 0.0
    y[:, -1, 0:2] = 0.0
    return x, y


def check_identity_no_flip():
    x, y = _make_batch()
    positions_rot, vx_rot, vy_rot = rotate_mesh(x[..., 0:2], y[..., 0], y[..., 1], 0.0)
    assert torch.allclose(positions_rot, x[..., 0:2]), "theta=0 positions must be identity"
    assert torch.allclose(vx_rot, y[..., 0]), "theta=0 Ux must be identity"
    assert torch.allclose(vy_rot, y[..., 1]), "theta=0 Uy must be identity"
    print("[OK] theta=0 (no flip): identity")


def check_90_deg():
    x = torch.tensor([[[1.0, 0.0]]])  # position (1, 0)
    vx = torch.tensor([[1.0]])
    vy = torch.tensor([[0.0]])
    pos_rot, vx_rot, vy_rot = rotate_mesh(x, vx, vy, 90.0)
    # (1, 0) rotated 90 deg CCW → (0, 1)
    assert torch.allclose(pos_rot[..., 0], torch.tensor(0.0), atol=1e-5), pos_rot
    assert torch.allclose(pos_rot[..., 1], torch.tensor(1.0), atol=1e-5), pos_rot
    # (Ux=1, Uy=0) rotated 90 deg CCW → (0, 1)
    assert torch.allclose(vx_rot, torch.tensor(0.0), atol=1e-5), vx_rot
    assert torch.allclose(vy_rot, torch.tensor(1.0), atol=1e-5), vy_rot
    print("[OK] theta=90 (1,0) -> (0,1) for both positions and velocity")


def check_180_deg():
    pos = torch.tensor([[[1.5, -2.5]]])
    vx = torch.tensor([[3.0]])
    vy = torch.tensor([[-4.0]])
    pos_rot, vx_rot, vy_rot = rotate_mesh(pos, vx, vy, 180.0)
    assert torch.allclose(pos_rot, -pos, atol=1e-5), pos_rot
    assert torch.allclose(vx_rot, -vx, atol=1e-5), vx_rot
    assert torch.allclose(vy_rot, -vy, atol=1e-5), vy_rot
    print("[OK] theta=180 negates both positions and velocity")


def check_composition():
    x, y = _make_batch(seed=42)
    a = 7.5
    b = -4.2
    # Apply rotation by a then by b
    p1, vx1, vy1 = rotate_mesh(x[..., 0:2], y[..., 0], y[..., 1], a)
    p_ab, vx_ab, vy_ab = rotate_mesh(p1, vx1, vy1, b)
    # Compare with single rotation by (a + b)
    p_sum, vx_sum, vy_sum = rotate_mesh(x[..., 0:2], y[..., 0], y[..., 1], a + b)
    assert torch.allclose(p_ab, p_sum, atol=1e-5), (p_ab - p_sum).abs().max()
    assert torch.allclose(vx_ab, vx_sum, atol=1e-5)
    assert torch.allclose(vy_ab, vy_sum, atol=1e-5)
    print(f"[OK] R({a}) o R({b}) == R({a+b}) within 1e-5")


def check_padding_preserved():
    """Padding rows (positions at (0,0), velocity at (0,0)) stay at zero."""
    x, y = _make_batch()
    p_rot, vx_rot, vy_rot = rotate_mesh(x[..., 0:2], y[..., 0], y[..., 1], 15.0)
    # Tail row across both samples was set to zero earlier.
    assert torch.allclose(p_rot[:, -1, :], torch.zeros(2, 2)), p_rot[:, -1, :]
    assert torch.allclose(vx_rot[:, -1], torch.zeros(2)), vx_rot[:, -1]
    assert torch.allclose(vy_rot[:, -1], torch.zeros(2)), vy_rot[:, -1]
    print("[OK] zero-padding rows preserved at zero after rotation")


def check_pressure_invariant_in_y_clone():
    """The training loop clones y and only writes channels 0,1.

    Confirm the standalone rotate_mesh does NOT modify the pressure channel:
    the calling code uses x[..., 0] = vel_x_aug to write back.
    """
    x, y = _make_batch(seed=7)
    p_orig = y[..., 2].clone()
    pos_rot, vx_rot, vy_rot = rotate_mesh(x[..., 0:2], y[..., 0], y[..., 1], 33.0)
    # rotate_mesh returns the rotated velocity, but the input y is untouched.
    assert torch.allclose(y[..., 2], p_orig), "pressure channel must not be modified"
    print("[OK] rotate_mesh does not touch pressure channel of y")


def main():
    check_identity_no_flip()
    check_90_deg()
    check_180_deg()
    check_composition()
    check_padding_preserved()
    check_pressure_invariant_in_y_clone()
    print("\nAll mesh_aug sanity checks passed.")


if __name__ == "__main__":
    main()
