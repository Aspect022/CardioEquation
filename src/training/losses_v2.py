"""
V3 Loss Functions for CardioEquation (PyTorch)
================================================
Multi-component loss for diffusion / flow matching training.

Run 6 Changes:
  - Added SoftDTW loss for QRS temporal fidelity (addresses 45% compression)
  - Added qrs_segment_dtw_loss: extracts QRS segments + applies Soft-DTW
  - Identity SNR floor: identity always gets >= 30% gradient even at high noise
  - HR loss restructured: only fires at low-noise timesteps (t < 0.2)
  - combined_diffusion_loss signature updated for Run 6 weights
  - w_identity: 0.5 → 1.5
  - w_spectral: 0.0001 → 0.00001
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


# ── Run 6: Soft-DTW Loss for QRS Temporal Fidelity ───────────────────────

# Try to import GPU-accelerated Soft-DTW (pip install pysdtw)
_SOFT_DTW_CUDA_AVAILABLE = False
try:
    import pysdtw
    _SOFT_DTW_CUDA_AVAILABLE = True
except ImportError:
    pass


def _soft_dtw_forward_vectorized(D, gamma=0.1):
    """
    Compute Soft-DTW distance using anti-diagonal vectorized DP.

    Instead of O(N*M) sequential Python iterations, this processes
    each anti-diagonal in a single batched tensor operation.
    For an N×M matrix there are N+M-1 anti-diagonals, each computed
    in one vectorized step → ~50× faster than nested loops.

    Fallback for when soft-dtw-cuda is not installed.

    Args:
        D: (B, N, M) pairwise distance matrix between sequences
        gamma: smoothing parameter (smaller = more like hard DTW)
    Returns:
        sdtw: (B,) soft DTW distances
    """
    B, N, M = D.shape
    device = D.device
    INF = 1e9

    # Pad R with an extra row and column of infinity
    R = torch.full((B, N + 1, M + 1), INF, device=device, dtype=D.dtype)
    R[:, 0, 0] = 0.0

    # Process anti-diagonals: for diagonal d, all (i, j) where i+j = d
    for d in range(1, N + M + 1):
        # Valid (i, j) indices on this diagonal
        i_start = max(1, d - M)
        i_end = min(N, d - 1)

        if i_start > i_end:
            continue

        i_indices = torch.arange(i_start, i_end + 1, device=device)
        j_indices = d - i_indices  # j = d - i, guaranteed 1 <= j <= M

        # Gather the three predecessors for all cells on this diagonal
        r_diag = R[:, i_indices - 1, j_indices - 1]  # (B, num_cells)
        r_up   = R[:, i_indices - 1, j_indices]       # (B, num_cells)
        r_left = R[:, i_indices,     j_indices - 1]    # (B, num_cells)

        # Stack and compute soft-min
        costs = torch.stack([r_diag, r_up, r_left], dim=-1)  # (B, num_cells, 3)
        soft_min = -gamma * torch.logsumexp(-costs / gamma, dim=-1)  # (B, num_cells)

        # Add distance and store
        d_vals = D[:, i_indices - 1, j_indices - 1]  # (B, num_cells)
        R[:, i_indices, j_indices] = d_vals + soft_min

    return R[:, N, M]


def soft_dtw_loss(x, y, gamma=0.1):
    """
    Differentiable Soft-DTW distance between two batched 1D signals.

    Uses GPU-accelerated soft-dtw-cuda when available (pip install soft-dtw-cuda),
    falls back to anti-diagonal vectorized PyTorch implementation.

    For QRS segments (~100 samples), this is fast enough for training on A100.

    Args:
        x: (B, L1) — generated QRS segments
        y: (B, L2) — reference QRS segments
        gamma: smoothing temperature (0.1 recommended for ECG)
    Returns:
        loss: scalar — mean Soft-DTW distance across batch
    """
    if _SOFT_DTW_CUDA_AVAILABLE and x.is_cuda:
        # GPU-accelerated path (pysdtw)
        # SoftDTW expects (B, L, 1) for 1D signals
        sdtw_fn = pysdtw.SoftDTW(gamma=gamma, use_cuda=True)
        return sdtw_fn(x.unsqueeze(-1), y.unsqueeze(-1)).mean()

    # Vectorized fallback (anti-diagonal DP)
    D = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2)  # (B, L1, L2)
    sdtw = _soft_dtw_forward_vectorized(D, gamma)
    return sdtw.mean()


def _detect_r_peaks_simple(signal, fs=500.0, min_distance_ms=300):
    """
    Simple differentiable-friendly R-peak detection for QRS extraction.

    Not used in gradient computation — only for indexing QRS segments.

    Args:
        signal: (T,) 1D numpy or tensor — single ECG signal
        fs: sampling frequency
        min_distance_ms: minimum RR interval in ms
    Returns:
        peaks: list of R-peak indices
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    min_dist = int(fs * min_distance_ms / 1000)

    # Simple peak detection: find local maxima above threshold
    threshold = signal.mean() + 0.6 * signal.std()
    peaks = []

    for i in range(1, len(signal) - 1):
        if (signal[i] > signal[i-1] and
            signal[i] > signal[i+1] and
            signal[i] > threshold):
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)

    return peaks


def qrs_segment_dtw_loss(clean, recon, fs=500.0, gamma=0.1, window_ms=100):
    """
    GPU-only Soft-DTW loss on fixed-stride QRS-width windows.

    Run 6 optimization: Instead of per-sample CPU R-peak detection
    (which causes GPU→CPU sync stalls), we extract overlapping windows
    at fixed cardiac intervals and compute DTW on a random subset.

    This stays entirely on GPU and eliminates the CPU bottleneck.

    Args:
        clean: (B, 1, T) — ground truth clean ECG
        recon: (B, 1, T) — reconstructed/predicted ECG (x_0_pred)
        fs: sampling frequency
        gamma: Soft-DTW smoothing parameter
        window_ms: window width in ms (±window_ms around center)
    Returns:
        loss: scalar DTW loss averaged across sampled windows
    """
    B, _, T = clean.shape
    window_samples = int(fs * window_ms / 1000)  # 50 samples for 100ms
    window_size = 2 * window_samples  # 100 samples total
    stride = int(fs * 0.2)  # 200ms stride ≈ half a typical QRS interval

    if T < window_size + stride:
        return torch.tensor(0.0, device=clean.device)

    # Extract all windows using unfold (stays on GPU, no CPU sync)
    clean_1d = clean[:, 0, :]  # (B, T)
    recon_1d = recon[:, 0, :]  # (B, T)

    # Unfold into overlapping windows: (B, num_windows, window_size)
    clean_windows = clean_1d.unfold(1, window_size, stride)  # (B, W, L)
    recon_windows = recon_1d.unfold(1, window_size, stride)

    num_windows = clean_windows.shape[1]

    # Subsample to limit compute: max 4 windows per sample
    max_windows = min(4, num_windows)
    if num_windows > max_windows:
        indices = torch.randperm(num_windows, device=clean.device)[:max_windows]
        clean_windows = clean_windows[:, indices, :]  # (B, 4, L)
        recon_windows = recon_windows[:, indices, :]

    # Reshape to batch all windows together: (B*W, L)
    Bw = clean_windows.shape[0] * clean_windows.shape[1]
    clean_flat = clean_windows.reshape(Bw, window_size)
    recon_flat = recon_windows.reshape(Bw, window_size)

    # Normalize each window
    clean_flat = (clean_flat - clean_flat.mean(dim=-1, keepdim=True)) / (clean_flat.std(dim=-1, keepdim=True) + 1e-8)
    recon_flat = (recon_flat - recon_flat.mean(dim=-1, keepdim=True)) / (recon_flat.std(dim=-1, keepdim=True) + 1e-8)

    # Single batched Soft-DTW call
    return soft_dtw_loss(recon_flat, clean_flat, gamma=gamma)


def combined_diffusion_loss(
    noise_true, noise_pred, clean_signal, recon_signal,
    feature_extractor=None,
    alpha_bar_t=None,
    t_normalized=None,
    w_noise=1.0, w_signal=1.0, w_identity=1.5,
    w_spectral=0.00001, w_correlation=0.2, w_morphology=0.3,
    w_soft_dtw=0.0,
    identity_snr_floor=0.3,
    use_flow_matching=False,
):
    """
    Full multi-component loss for CardioEquation Run 6.

    Run 6 Changes:
        - w_identity: 0.5 → 1.5 (3× increase for ReID)
        - w_spectral: 0.0001 → 0.00001 (reduce gradient consumption)
        - identity_snr_floor: new, ensures identity >= 30% gradient
        - w_soft_dtw: Soft-DTW loss for QRS temporal fidelity
        - t_normalized: used for timestep-aware auxiliary losses
        - use_flow_matching: when True, noise_true/noise_pred are velocities

    Components:
        1. Primary loss (noise MSE or velocity MSE) — weight 1.0
        2. Signal MSE (reconstruction fidelity) — weight 1.0  (SNR-weighted)
        3. Identity cosine loss — weight 1.5  (SNR-weighted with floor 0.3)
        4. Spectral FFT loss — weight 1e-5  (SNR-weighted)
        5. Correlation loss (shape preservation) — weight 0.2  (SNR-weighted)
        6. Morphology gradient loss (QRS slope) — weight 0.3  (SNR-weighted)
        7. Soft-DTW loss (QRS temporal fidelity) — weight 0.0-0.3 (warmup)

    Returns:
        total_loss: scalar tensor
        loss_dict: dict of individual loss components for logging
    """
    # Individual losses
    l_noise = F.mse_loss(noise_pred, noise_true)  # Works for both ε and v
    l_signal = signal_mse_loss(clean_signal, recon_signal)
    l_spectral = spectral_loss(clean_signal, recon_signal)
    l_correlation = correlation_loss(clean_signal, recon_signal)
    l_morphology = morphology_gradient_loss(clean_signal, recon_signal)

    l_identity = torch.tensor(0.0, device=noise_true.device)
    if feature_extractor is not None:
        l_identity = cosine_identity_loss(
            feature_extractor, clean_signal, recon_signal
        )

    # Soft-DTW loss (only when weight > 0 — has overhead from R-peak detection)
    l_soft_dtw = torch.tensor(0.0, device=noise_true.device)
    if w_soft_dtw > 0:
        l_soft_dtw = qrs_segment_dtw_loss(
            clean_signal, recon_signal, fs=500.0, gamma=0.1
        )

    # SNR weight: mean of alpha_bar_t across the batch (scalar)
    # For flow matching: use (1 - t) as proxy for signal clarity
    if alpha_bar_t is not None:
        snr_w = alpha_bar_t.mean().clamp(min=0.0, max=1.0)
    elif t_normalized is not None:
        # Flow matching: signal clarity ≈ (1 - t)
        snr_w = (1.0 - t_normalized.mean()).clamp(min=0.0, max=1.0)
    else:
        snr_w = 1.0  # Fallback: no weighting

    # Identity SNR weight: apply FLOOR so identity always gets gradient
    identity_snr = max(snr_w.item() if isinstance(snr_w, torch.Tensor) else snr_w,
                       identity_snr_floor)

    # Primary loss — always active at full weight
    total = w_noise * l_noise

    # Auxiliary losses — SNR-weighted
    total = total + snr_w * w_signal * l_signal
    total = total + identity_snr * w_identity * l_identity  # FLOOR applied
    total = total + snr_w * w_spectral * l_spectral
    total = total + snr_w * w_correlation * l_correlation
    total = total + snr_w * w_morphology * l_morphology
    total = total + snr_w * w_soft_dtw * l_soft_dtw

    # Build loss dict for W&B/TensorBoard logging
    loss_dict = {
        'primary_loss': l_noise.detach(),
        'signal_mse': l_signal.detach(),
        'identity': l_identity.detach(),
        'spectral': l_spectral.detach(),
        'correlation': l_correlation.detach(),
        'morphology': l_morphology.detach(),
        'soft_dtw': l_soft_dtw.detach(),
        'snr_weight': snr_w if isinstance(snr_w, float) else snr_w.detach(),
        'identity_snr': identity_snr,
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


# ── Run 6: HR-specific losses (restructured) ──────────────────────────────

class DifferentiableHRLoss(torch.nn.Module):
    """
    Differentiable HR estimation via FFT autocorrelation + soft argmax.

    Run 6 Change: Added timestep gating — HR loss only fires at
    low-noise timesteps where x_0_pred is reliable enough to extract HR.
    At high noise (t > t_gate), the HR estimate is just noise → useless gradient.

    Pipeline:
        1. Check timestep gate (skip if t > t_gate)
        2. Zero-mean signal
        3. FFT autocorrelation (Wiener-Khinchin theorem)
        4. Soft argmax over physiological lag window
        5. Convert soft lag → HR (bpm)
        6. L1 loss against target HR

    Args:
        fs: Sampling frequency (Hz)
        hr_min: Minimum physiological HR (bpm)
        hr_max: Maximum physiological HR (bpm)
        softmax_temp: Temperature for soft argmax (higher = sharper)
        t_gate: Timestep threshold — only compute HR loss when t < t_gate
                For DDPM: t < 200/1000 = 0.2. For CFM: t < 0.2 (same meaning:
                low noise region where x_0_pred is clean enough).
    """

    def __init__(self, fs=500.0, hr_min=30.0, hr_max=200.0, softmax_temp=10.0,
                 t_gate=0.2):
        super().__init__()
        self.fs = fs
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.temp = softmax_temp
        self.t_gate = t_gate
        self.lag_min = int(fs * 60.0 / hr_max)   # ~150 samples for 200bpm
        self.lag_max = int(fs * 60.0 / hr_min)    # ~1000 samples for 30bpm

    def forward(self, ecg, hr_target, t_normalized, weight=1.0):
        """
        Args:
            ecg: (B, 1, T) generated/predicted ECG signal (x_0_pred)
            hr_target: (B,) target HR in bpm
            t_normalized: (B,) normalized timestep [0, 1]
            weight: scalar weight for the loss (used for warmup ramp)

        Returns:
            loss: scalar
            hr_estimated: (B,) estimated HR in bpm (detached)
        """
        # ── Timestep gating (Run 6) ──
        # Only compute HR loss for samples where t < t_gate (low noise)
        low_noise_mask = t_normalized < self.t_gate  # (B,)

        if not low_noise_mask.any():
            # All samples are at high noise — return zero loss
            dummy_hr = torch.full_like(hr_target, 75.0)
            return torch.tensor(0.0, device=ecg.device, requires_grad=True), dummy_hr

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

        # L1 loss — ONLY for low-noise samples (gated)
        # Mask the loss: only compute for samples where t < t_gate
        hr_error = (hr_est - hr_target.float()).abs()
        gated_error = hr_error * low_noise_mask.float()
        n_valid = low_noise_mask.sum().clamp(min=1)
        loss = gated_error.sum() / n_valid

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
