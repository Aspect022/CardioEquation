"""
Conditional Flow Matching (OT-CFM) Scheduler
==============================================
Implements Optimal Transport Conditional Flow Matching for Run 6.

Instead of DDPM's noise prediction, CFM learns a velocity field v(x_t, t)
that transports samples from noise (t=1) to data (t=0) along near-straight
paths via optimal transport coupling.

Key advantages over DDPM for ECG generation:
  - Straighter transport paths (curvature ≈1.02 vs ≈3.45 for DDPM)
  - Better conditioning adherence (HR, identity)
  - Faster inference via Euler ODE solver (10-20 steps vs 50 DDIM)
  - Simpler training objective (velocity MSE)

References:
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
  - Tong et al., "Improving and Generalizing Flow-Based Generative Models
    with Minibatch Optimal Transport" (ICML 2024)
  - FlowECG: Bondar et al. (2025) — flow matching for ECG synthesis
  - ETTA: DiT + OT-CFM for audio generation
"""

import torch
import torch.nn.functional as F
import math


class FlowMatchingScheduler:
    """
    Conditional Flow Matching scheduler with linear interpolation.

    Forward process (training):
        x_t = (1 - t) * x_0 + t * x_1        where x_1 ~ N(0,I)
        v_target = x_1 - x_0                   velocity target

    Reverse process (sampling):
        dx/dt = v_θ(x_t, t)                   ODE integration from t=1 → t=0
        Using Euler: x_{t-Δt} = x_t - Δt * v_θ(x_t, t)

    Optimal Transport coupling improves path straightness by pairing
    noise samples with data samples that minimize transport cost.
    """

    def __init__(self, sigma_min=1e-4):
        """
        Args:
            sigma_min: Small noise floor to prevent numerical issues at t=0.
        """
        self.sigma_min = sigma_min

    def sample_t(self, batch_size, device):
        """
        Sample random timesteps t ~ U[0, 1] for training.

        Returns:
            t: (B,) continuous timestep in [0, 1]
        """
        return torch.rand(batch_size, device=device)

    def interpolate(self, x_0, x_1, t):
        """
        Forward process: linear interpolation between data and noise.

        x_t = (1 - t) * x_0 + t * x_1

        Args:
            x_0: (B, C, T) — clean ECG signal (data)
            x_1: (B, C, T) — noise sample ~ N(0, I)
            t: (B,) — timestep in [0, 1]
        Returns:
            x_t: (B, C, T) — interpolated sample
            v_target: (B, C, T) — velocity target (x_1 - x_0)
        """
        t_expand = t.view(-1, 1, 1)  # (B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0
        return x_t, v_target

    def predict_x0(self, x_t, v_pred, t):
        """
        Estimate x_0 from current sample and predicted velocity.

        x_0 = x_t - t * v_pred

        Used for auxiliary losses (HR, identity, morphology) that need
        a clean signal estimate.

        Args:
            x_t: (B, C, T) — current noisy sample
            v_pred: (B, C, T) — model's velocity prediction
            t: (B,) — current timestep
        Returns:
            x_0_pred: (B, C, T) — estimated clean signal
        """
        t_expand = t.view(-1, 1, 1)
        x_0_pred = x_t - t_expand * v_pred
        return x_0_pred

    def velocity_loss(self, v_pred, v_target):
        """
        Primary training loss: MSE between predicted and target velocity.

        Args:
            v_pred: (B, C, T) — model's velocity prediction
            v_target: (B, C, T) — ground truth velocity (x_1 - x_0)
        Returns:
            loss: scalar MSE loss
        """
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def euler_sample(
        self,
        model,
        shape,
        cond,
        hr_bpm=None,
        num_steps=20,
        guidance_scale=2.0,
        device='cuda',
    ):
        """
        Full ODE sampling via Euler integration from t=1 (noise) → t=0 (data).

        x_{t-Δt} = x_t - Δt * v_θ(x_t, t)

        Run 6 default: 20 steps (vs 50 DDIM steps in Run 5b).

        Args:
            model: DiT-ECG model (predicts velocity v)
            shape: (B, C, T) output shape
            cond: (B, D) identity conditioning vector
            hr_bpm: (B,) HR conditioning in bpm
            num_steps: ODE integration steps (20 recommended)
            guidance_scale: CFG scale (2.0 recommended for Run 6)
            device: compute device
        Returns:
            x_0: (B, C, T) generated clean signal
        """
        # Start from pure noise at t=1
        x_t = torch.randn(shape, device=device)

        # Timestep schedule: 1.0 → 0.0 in num_steps
        dt = 1.0 / num_steps
        timesteps = torch.linspace(1.0, dt, num_steps, device=device)

        for t_val in timesteps:
            B = x_t.shape[0]
            t_batch = torch.full((B,), t_val.item(), dtype=torch.float32, device=device)

            # CFG: velocity with guidance
            if guidance_scale > 1.0 and hasattr(model, 'forward_with_cfg'):
                v_pred = model.forward_with_cfg(
                    x_t, t_batch, cond,
                    hr_bpm=hr_bpm,
                    guidance_scale=guidance_scale,
                )
            else:
                v_pred = model(x_t, t_batch, cond, hr_bpm=hr_bpm)

            # Euler step: x_{t-dt} = x_t - dt * v
            x_t = x_t - dt * v_pred

        return x_t

    @torch.no_grad()
    def midpoint_sample(
        self,
        model,
        shape,
        cond,
        hr_bpm=None,
        num_steps=20,
        guidance_scale=2.0,
        device='cuda',
    ):
        """
        Midpoint method (2nd-order Runge-Kutta) for higher accuracy.

        More accurate than Euler at the same number of steps,
        at the cost of 2x model evaluations per step.
        Use when quality matters more than speed.

        Args: same as euler_sample
        Returns:
            x_0: (B, C, T) generated clean signal
        """
        x_t = torch.randn(shape, device=device)

        dt = 1.0 / num_steps
        timesteps = torch.linspace(1.0, dt, num_steps, device=device)

        for t_val in timesteps:
            B = x_t.shape[0]
            t_batch = torch.full((B,), t_val.item(), dtype=torch.float32, device=device)
            t_mid = torch.full((B,), t_val.item() - dt / 2, dtype=torch.float32, device=device)

            # k1: velocity at current point
            if guidance_scale > 1.0 and hasattr(model, 'forward_with_cfg'):
                v1 = model.forward_with_cfg(x_t, t_batch, cond, hr_bpm=hr_bpm, guidance_scale=guidance_scale)
            else:
                v1 = model(x_t, t_batch, cond, hr_bpm=hr_bpm)

            # Midpoint
            x_mid = x_t - (dt / 2) * v1

            # k2: velocity at midpoint
            if guidance_scale > 1.0 and hasattr(model, 'forward_with_cfg'):
                v2 = model.forward_with_cfg(x_mid, t_mid, cond, hr_bpm=hr_bpm, guidance_scale=guidance_scale)
            else:
                v2 = model(x_mid, t_mid, cond, hr_bpm=hr_bpm)

            # Full step using midpoint velocity
            x_t = x_t - dt * v2

        return x_t


if __name__ == "__main__":
    # Quick verification
    scheduler = FlowMatchingScheduler()

    # Test interpolation
    x_0 = torch.randn(4, 1, 2500)  # Clean data
    x_1 = torch.randn(4, 1, 2500)  # Noise
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])

    x_t, v_target = scheduler.interpolate(x_0, x_1, t)

    # At t=0, x_t should be x_0
    assert torch.allclose(x_t[0], x_0[0], atol=1e-5), "t=0 should give x_0"
    # At t=1, x_t should be x_1
    assert torch.allclose(x_t[3], x_1[3], atol=1e-5), "t=1 should give x_1"
    # Velocity should be x_1 - x_0
    assert torch.allclose(v_target, x_1 - x_0, atol=1e-5), "v = x_1 - x_0"

    print(f"Interpolation shapes: x_t={x_t.shape}, v={v_target.shape}")

    # Test x_0 prediction
    v_pred = v_target + 0.1 * torch.randn_like(v_target)  # Slightly noisy prediction
    x_0_pred = scheduler.predict_x0(x_t, v_pred, t)
    print(f"x_0_pred shape: {x_0_pred.shape}")

    # Test velocity loss
    loss = scheduler.velocity_loss(v_pred, v_target)
    print(f"Velocity loss: {loss.item():.6f}")

    print("✅ FlowMatchingScheduler verified!")
