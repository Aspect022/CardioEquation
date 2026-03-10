"""
V2 Loss Functions for CardioEquation (PyTorch)
================================================
Multi-component loss for diffusion training + forecasting.

Changes in V2.1:
  - combined_diffusion_loss now returns (total_loss, loss_dict) for W&B logging
  - Added morphology_gradient_loss (1st + 2nd derivative matching)
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
    w_spectral=0.1, w_correlation=0.2, w_morphology=0.3,
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
