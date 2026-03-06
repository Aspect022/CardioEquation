"""
Diffusion Noise Scheduler
==========================
Cosine noise schedule for forward diffusion process.
Supports both DDPM training and DDIM sampling.
"""

import torch
import math


class CosineNoiseScheduler:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Provides smoother signal-to-noise transitions than linear schedule,
    especially important for ECG where fine morphological details
    (P-waves, QRS complex) need to be learned at lower noise levels.
    """

    def __init__(self, num_train_timesteps=1000, s=0.008):
        """
        Args:
            num_train_timesteps: Total discrete timesteps for training
            s: Offset to prevent β_t from being too small near t=0
        """
        self.num_train_timesteps = num_train_timesteps

        # Compute alpha_bar schedule
        steps = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
        f_t = torch.cos(((steps / num_train_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]

        # Clip to prevent numerical issues
        self.alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=0.9999)

        # Compute betas from alpha_bar
        self.betas = 1.0 - (self.alpha_bar[1:] / self.alpha_bar[:-1])
        self.betas = torch.clamp(self.betas, min=1e-5, max=0.999)

        self.alphas = 1.0 - self.betas
        self.alpha_bar_t = self.alpha_bar[:-1]  # alpha_bar at each timestep

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to clean signal.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: (B, C, T) clean signal
            t: (B,) integer timesteps
            noise: (B, C, T) optional pre-sampled noise
        Returns:
            x_t: (B, C, T) noisy signal at timestep t
            noise: (B, C, T) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bar_t.to(x_0.device)[t]

        # Reshape for broadcasting: (B,) → (B, 1, 1)
        sqrt_alpha = alpha_bar_t.sqrt().view(-1, 1, 1)
        sqrt_one_minus_alpha = (1 - alpha_bar_t).sqrt().view(-1, 1, 1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    @torch.no_grad()
    def ddim_sample_step(self, model, x_t, t, t_prev, cond, guidance_scale=1.0):
        """
        Single DDIM deterministic sampling step.

        Args:
            model: DiT-ECG model
            x_t: (B, C, T) current noisy state
            t: current timestep (int)
            t_prev: previous timestep (int)
            cond: (B, D) conditioning vector
            guidance_scale: CFG scale (1.0 = no guidance)
        Returns:
            x_t_prev: (B, C, T) denoised state at t_prev
        """
        B = x_t.shape[0]
        device = x_t.device

        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict noise (with optional CFG)
        if guidance_scale > 1.0:
            eps_pred = model.forward_with_cfg(
                x_t, t_tensor.float() / self.num_train_timesteps,
                cond, guidance_scale
            )
        else:
            eps_pred = model(
                x_t, t_tensor.float() / self.num_train_timesteps, cond
            )

        # DDIM update
        alpha_bar_t = self.alpha_bar_t[t].to(device)
        alpha_bar_t_prev = self.alpha_bar_t[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

        # Estimate x_0
        sqrt_alpha = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_bar_t).sqrt()
        x_0_pred = (x_t - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha.clamp(min=1e-8)

        # DDIM (deterministic, η=0)
        sqrt_alpha_prev = alpha_bar_t_prev.sqrt()
        sqrt_one_minus_alpha_prev = (1 - alpha_bar_t_prev).sqrt()

        x_t_prev = sqrt_alpha_prev * x_0_pred + sqrt_one_minus_alpha_prev * eps_pred
        return x_t_prev

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, num_steps=50, guidance_scale=3.0, device='cuda'):
        """
        Full DDIM sampling loop.

        Args:
            model: DiT-ECG model
            shape: (B, C, T) output shape
            cond: (B, D) conditioning vector
            num_steps: Number of sampling steps (50 recommended)
            guidance_scale: CFG scale (3.0 recommended)
        Returns:
            x_0: (B, C, T) generated clean signal
        """
        # Create evenly spaced timestep schedule
        step_size = self.num_train_timesteps // num_steps
        timesteps = list(range(0, self.num_train_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            x_t = self.ddim_sample_step(model, x_t, t, t_prev, cond, guidance_scale)

        return x_t
