"""
V2 Loss Functions for CardioEquation (PyTorch)
================================================
Multi-component loss for diffusion training + forecasting.
"""

import torch
import torch.nn.functional as F


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


def combined_diffusion_loss(
    noise_true, noise_pred, clean_signal, recon_signal,
    feature_extractor=None,
    alpha_bar_t=None,
    w_noise=1.0, w_signal=1.0, w_identity=0.5,
    w_spectral=0.1, w_correlation=0.2,
):
    """
    Full multi-component loss for CardioEquation diffusion training.

    Components:
        1. Noise MSE (standard diffusion) — weight 1.0  (always active)
        2. Signal MSE (reconstruction fidelity) — weight 1.0  (SNR-weighted)
        3. Identity cosine loss (personalization) — weight 0.5  (SNR-weighted)
        4. Spectral FFT loss (frequency preservation) — weight 0.1  (SNR-weighted)
        5. Correlation loss (shape preservation) — weight 0.2  (SNR-weighted)

    Auxiliary losses (2-5) are weighted by alpha_bar_t so they only contribute
    meaningfully at low-noise timesteps where x_0_pred is reliable.
    At high noise (large t), alpha_bar_t ≈ 0 → auxiliary losses are suppressed.
    """
    # Primary diffusion loss — always active at full weight
    loss = w_noise * noise_mse_loss(noise_true, noise_pred)

    # SNR weight: mean of alpha_bar_t across the batch (scalar)
    # Falls off naturally: t=0 → snr_w≈1.0, t=500 → ~0.5, t=900 → ~0.01
    if alpha_bar_t is not None:
        snr_w = alpha_bar_t.mean().clamp(min=0.0, max=1.0)
    else:
        snr_w = 1.0  # Fallback: no weighting

    loss = loss + snr_w * w_signal * signal_mse_loss(clean_signal, recon_signal)

    if feature_extractor is not None:
        loss = loss + snr_w * w_identity * cosine_identity_loss(
            feature_extractor, clean_signal, recon_signal
        )

    loss = loss + snr_w * w_spectral * spectral_loss(clean_signal, recon_signal)
    loss = loss + snr_w * w_correlation * correlation_loss(clean_signal, recon_signal)

    return loss


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
