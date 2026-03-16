"""
V2 Loss Functions for CardioEquation (PyTorch)
================================================
Multi-component loss for diffusion training + forecasting.

Changes in V2.1:
  - combined_diffusion_loss now returns (total_loss, loss_dict) for W&B logging
  - Added morphology_gradient_loss (1st + 2nd derivative matching)

Changes in V2.2 (Run 5):
  - Added DifferentiableHRLoss (FFT autocorrelation + soft argmax)
  - Added hr_variance_loss (penalizes HR diversity collapse)
  - Fixed w_spectral default: 0.1 → 0.0001
"""

import torch
import torch.nn.functional as F
import math


def noise_mse_loss(noise_true, noise_pred):
    """Standard diffusion denoising loss."""
    return F.mse_loss(noise_pred, noise_true)


def signal_mse_loss(clean, reconstructed):
    """Reconstruction fidelity in signal space."""
    return F.mse_loss(reconstructed, clean)


def cosine_identity_loss(feature_extractor, clean_signal, recon_signal):
    """
    Identity preservation loss via cosine similarity.
    Ensures generated ECG has the same patient identity as the ground truth.

    Args:
        feature_extractor: Pre-trained identity encoder (B, 1, T) → (B, 512)
        clean_signal: Ground truth clean ECG (B, 1, T)
        recon_signal: Reconstructed/generated ECG (B, 1, T)
    """
    with torch.no_grad():
        feat_true = feature_extractor(clean_signal)
    feat_pred = feature_extractor(recon_signal)

    # Cosine similarity → loss = 1 - cos_sim
    cos_sim = F.cosine_similarity(feat_true, feat_pred, dim=-1)
    return (1.0 - cos_sim).mean()


def spectral_loss(clean, reconstructed):
    """
    Spectral loss in frequency domain.
    Preserves frequency characteristics (heart rate, wave morphology).
    """
    fft_clean = torch.fft.rfft(clean.squeeze(1))
    fft_recon = torch.fft.rfft(reconstructed.squeeze(1))
    return F.mse_loss(fft_recon.abs(), fft_clean.abs())


def correlation_loss(clean, reconstructed):
    """
    Pearson correlation loss.
    Scale-invariant measure of waveform similarity.
    """
    # Flatten to (B, T)
    c = clean.view(clean.shape[0], -1)
    r = reconstructed.view(reconstructed.shape[0], -1)

    # Center
    c_centered = c - c.mean(dim=1, keepdim=True)
    r_centered = r - r.mean(dim=1, keepdim=True)

    # Pearson correlation
    cov = (c_centered * r_centered).mean(dim=1)
    std_c = c.std(dim=1)
    std_r = r.std(dim=1)

    corr = cov / (std_c * std_r + 1e-8)
    return (1.0 - corr).mean()


def morphology_gradient_loss(clean, reconstructed):
    """
    Morphology loss via 1st and 2nd derivative matching.

    Preserves sharp QRS slopes and P/T-wave curvatures by penalizing
    differences in the signal's first derivative (slope) and second
    derivative (curvature). This is more effective than MSE for
    high-frequency transients like the QRS complex.

    Ref: MIDT-ECG (2025), Gemini research §3.
    """
    # Squeeze channel dim: (B, 1, T) → (B, T)
    c = clean.squeeze(1)
    r = reconstructed.squeeze(1)

    # 1st derivative (slope) — preserves QRS sharpness
    gen_grad = torch.diff(r, dim=-1)
    real_grad = torch.diff(c, dim=-1)
    grad_loss = F.mse_loss(gen_grad, real_grad)

    # 2nd derivative (curvature) — preserves P/T wave shapes
    gen_curv = torch.diff(gen_grad, dim=-1)
    real_curv = torch.diff(real_grad, dim=-1)
    curv_loss = F.mse_loss(gen_curv, real_curv)

    return grad_loss + 0.1 * curv_loss


def combined_diffusion_loss(
    noise_true, noise_pred, clean_signal, recon_signal,
    feature_extractor=None,
    alpha_bar_t=None,
    w_noise=1.0, w_signal=1.0, w_identity=0.5,
    w_spectral=0.0001, w_correlation=0.2, w_morphology=0.3,
):
    """
    Full multi-component loss for CardioEquation diffusion training.

    Components:
        1. Noise MSE (standard diffusion) — weight 1.0  (always active)
        2. Signal MSE (reconstruction fidelity) — weight 1.0  (SNR-weighted)
        3. Identity cosine loss (personalization) — weight 0.5  (SNR-weighted)
        4. Spectral FFT loss (frequency preservation) — weight 0.1  (SNR-weighted)
        5. Correlation loss (shape preservation) — weight 0.2  (SNR-weighted)
        6. Morphology gradient loss (QRS slope) — weight 0.3  (SNR-weighted)

    Auxiliary losses (2-6) are weighted by alpha_bar_t so they only contribute
    meaningfully at low-noise timesteps where x_0_pred is reliable.
    At high noise (large t), alpha_bar_t ≈ 0 → auxiliary losses are suppressed.

    Returns:
        total_loss: scalar tensor
        loss_dict: dict of individual loss components (detached scalars) for logging
    """
    # Individual losses
    l_noise = noise_mse_loss(noise_true, noise_pred)
    l_signal = signal_mse_loss(clean_signal, recon_signal)
    l_spectral = spectral_loss(clean_signal, recon_signal)
    l_correlation = correlation_loss(clean_signal, recon_signal)
    l_morphology = morphology_gradient_loss(clean_signal, recon_signal)

    l_identity = torch.tensor(0.0, device=noise_true.device)
    if feature_extractor is not None:
        l_identity = cosine_identity_loss(
            feature_extractor, clean_signal, recon_signal
        )

    # SNR weight: mean of alpha_bar_t across the batch (scalar)
    # Falls off naturally: t=0 → snr_w≈1.0, t=500 → ~0.5, t=900 → ~0.01
    if alpha_bar_t is not None:
        snr_w = alpha_bar_t.mean().clamp(min=0.0, max=1.0)
    else:
        snr_w = 1.0  # Fallback: no weighting

    # Primary diffusion loss — always active at full weight
    total = w_noise * l_noise

    # Auxiliary losses — SNR-weighted
    total = total + snr_w * w_signal * l_signal
    total = total + snr_w * w_identity * l_identity
    total = total + snr_w * w_spectral * l_spectral
    total = total + snr_w * w_correlation * l_correlation
    total = total + snr_w * w_morphology * l_morphology

    # Build loss dict for W&B/TensorBoard logging
    loss_dict = {
        'noise_mse': l_noise.detach(),
        'signal_mse': l_signal.detach(),
        'identity': l_identity.detach(),
        'spectral': l_spectral.detach(),
        'correlation': l_correlation.detach(),
        'morphology': l_morphology.detach(),
        'snr_weight': snr_w if isinstance(snr_w, float) else snr_w.detach(),
    }

    return total, loss_dict


def info_nce_loss(z_i, z_j, temperature=0.07):
    """
    InfoNCE contrastive loss for patient identity pre-training.

    Args:
        z_i, z_j: (B, D) L2-normalized projection vectors
                  from two augmented views of the same patient segments
        temperature: Sharpness of distribution (0.07 default for ECG)
    """
    # Already L2-normalized
    logits = torch.matmul(z_i, z_j.T) / temperature  # (B, B)
    labels = torch.arange(len(z_i), device=z_i.device)

    loss_ij = F.cross_entropy(logits, labels)
    loss_ji = F.cross_entropy(logits.T, labels)

    return (loss_ij + loss_ji) / 2.0


# ── Run 5: HR-specific losses ──────────────────────────────────────────────

class DifferentiableHRLoss(torch.nn.Module):
    """
    Differentiable HR estimation via FFT autocorrelation + soft argmax.

    Research-confirmed: Most stable method for backprop through HR estimation.
    Smooth gradients even when generated ECG is noisy (early training).

    Pipeline:
        1. Zero-mean signal
        2. FFT autocorrelation (Wiener-Khinchin theorem)
        3. Soft argmax over physiological lag window [lag_min, lag_max]
        4. Convert soft lag → HR (bpm)
        5. L1 loss against target HR

    Args:
        fs: Sampling frequency (Hz)
        hr_min: Minimum physiological HR (bpm)
        hr_max: Maximum physiological HR (bpm)
        softmax_temp: Temperature for soft argmax (higher = sharper)
    """

    def __init__(self, fs=500.0, hr_min=30.0, hr_max=200.0, softmax_temp=10.0):
        super().__init__()
        self.fs = fs
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.temp = softmax_temp
        self.lag_min = int(fs * 60.0 / hr_max)   # ~150 samples for 200bpm
        self.lag_max = int(fs * 60.0 / hr_min)    # ~1000 samples for 30bpm

    def forward(self, ecg, hr_target, weight=1.0):
        """
        Args:
            ecg: (B, 1, T) generated/predicted ECG signal
            hr_target: (B,) target HR in bpm
            weight: scalar weight for the loss (used for warmup ramp)

        Returns:
            loss: scalar
            hr_estimated: (B,) estimated HR in bpm (detached)
        """
        # Squeeze channel: (B, 1, T) → (B, T)
        x = ecg.squeeze(1)
        x = x - x.mean(dim=-1, keepdim=True)  # Zero-mean

        B, T = x.shape

        # FFT autocorrelation (Wiener-Khinchin)
        n_fft = 2 ** math.ceil(math.log2(2 * T - 1))
        X = torch.fft.rfft(x, n=n_fft)
        acf = torch.fft.irfft(X.real**2 + X.imag**2, n=n_fft)[:, :T]
        acf = acf / (acf[:, 0:1] + 1e-8)  # Normalize by zero-lag

        # Extract window in physiological lag range
        lag_max = min(self.lag_max, T - 1)
        window = acf[:, self.lag_min:lag_max]  # (B, W)

        # Soft argmax: weighted sum of lag indices
        lags = torch.arange(
            window.shape[-1], device=x.device, dtype=x.dtype
        ) + self.lag_min  # (W,)
        weights = F.softmax(self.temp * window, dim=-1)  # (B, W)
        soft_lag = (weights * lags).sum(dim=-1)  # (B,)

        # Convert lag → HR (bpm)
        hr_est = (60.0 * self.fs / (soft_lag + 1e-6)).clamp(
            self.hr_min, self.hr_max
        )

        # L1 loss (more robust than L2 for HR)
        loss = F.l1_loss(hr_est, hr_target.float())

        return weight * loss, hr_est.detach()


def hr_variance_loss(hr_estimated, target_std=47.0, weight=1.0):
    """
    Penalizes when batch HR std drops below target.

    Real ECG HR std = 47.4 bpm. When generated ECGs collapse to
    similar heart rates, this loss pushes the model to diversify.

    Args:
        hr_estimated: (B,) estimated HR values in bpm
        target_std: target standard deviation (bpm)
        weight: scalar weight

    Returns:
        loss: scalar (0 if std >= target_std)
    """
    return weight * torch.relu(target_std - hr_estimated.std())
