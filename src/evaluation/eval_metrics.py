"""
Evaluation Metrics for Generated ECG
=======================================
Comprehensive evaluation toolkit following the ECG-Bench three-level protocol
(arXiv:2507.14206).

Level 1: Distribution Quality (FFD, MMD)
Level 2: Morphological Fidelity (HR-MAE, QRS duration, QT interval)
Level 3: Downstream Utility (Patient Re-ID)

Usage:
    from src.evaluation.eval_metrics import ECGEvaluator
    evaluator = ECGEvaluator(encoder_path="checkpoints/feature_extractor_contrastive.pt")
    results = evaluator.evaluate_all(real_ecgs, generated_ecgs)
"""

import torch
import numpy as np
from scipy.linalg import sqrtm
import warnings


def compute_ffd(real_features, fake_features):
    """
    Fréchet ECG Distance (analogous to FID for images).

    Computes the Fréchet distance between the feature distributions
    of real and generated ECGs.

    Args:
        real_features: (N, D) numpy array — encoder features of real ECGs
        fake_features: (N, D) numpy array — encoder features of generated ECGs
    Returns:
        ffd: float — lower is better
    """
    mu_r = real_features.mean(axis=0)
    mu_f = fake_features.mean(axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f

    # Matrix square root
    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    ffd = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(ffd)


def compute_mmd(real_features, fake_features, kernel='rbf', gamma=None):
    """
    Maximum Mean Discrepancy with RBF kernel.

    Args:
        real_features: (N, D) numpy array
        fake_features: (M, D) numpy array
        gamma: RBF kernel bandwidth (auto if None)
    Returns:
        mmd: float — lower is better
    """
    from scipy.spatial.distance import cdist

    if gamma is None:
        # Median heuristic
        all_features = np.vstack([real_features, fake_features])
        dists = cdist(all_features, all_features, 'sqeuclidean')
        gamma = 1.0 / np.median(dists[dists > 0])

    def rbf_kernel(X, Y):
        dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * dists)

    K_rr = rbf_kernel(real_features, real_features)
    K_ff = rbf_kernel(fake_features, fake_features)
    K_rf = rbf_kernel(real_features, fake_features)

    mmd = K_rr.mean() + K_ff.mean() - 2 * K_rf.mean()
    return float(mmd)


def compute_morphological_metrics(signals, fs=500):
    """
    Compute morphological ECG metrics using neurokit2.

    Args:
        signals: (N, T) numpy array — ECG signals
        fs: sampling frequency
    Returns:
        dict with HR, QRS duration, etc. (means and stds)
    """
    try:
        import neurokit2 as nk
    except ImportError:
        warnings.warn("neurokit2 not installed. Install with: pip install neurokit2")
        return {}

    heart_rates = []
    qrs_durations = []

    for i in range(len(signals)):
        try:
            sig = signals[i]
            if np.isnan(sig).any() or np.std(sig) < 1e-6:
                continue

            # Process ECG
            ecg_cleaned = nk.ecg_clean(sig, sampling_rate=fs)
            _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)

            r_peaks = info.get('ECG_R_Peaks', [])
            if len(r_peaks) >= 2:
                rr_intervals = np.diff(r_peaks) / fs
                hr = 60.0 / rr_intervals.mean()
                heart_rates.append(hr)

            # QRS duration via delineation
            try:
                _, waves = nk.ecg_delineate(ecg_cleaned, r_peaks, sampling_rate=fs)
                q_onsets = waves.get('ECG_Q_Peaks', [])
                s_offsets = waves.get('ECG_S_Peaks', [])
                if q_onsets and s_offsets:
                    valid_pairs = [(q, s) for q, s in zip(q_onsets, s_offsets)
                                   if q is not None and s is not None and not np.isnan(q) and not np.isnan(s)]
                    if valid_pairs:
                        durations = [(s - q) / fs * 1000 for q, s in valid_pairs]  # ms
                        qrs_durations.extend(durations)
            except Exception:
                pass

        except Exception:
            continue

    results = {}
    if heart_rates:
        results['hr_mean'] = float(np.mean(heart_rates))
        results['hr_std'] = float(np.std(heart_rates))
    if qrs_durations:
        results['qrs_duration_mean_ms'] = float(np.mean(qrs_durations))
        results['qrs_duration_std_ms'] = float(np.std(qrs_durations))

    return results


def compute_hr_mae(real_signals, fake_signals, fs=500):
    """
    Heart Rate Mean Absolute Error between real and generated signals.
    Target: < 5 BPM.
    """
    real_metrics = compute_morphological_metrics(real_signals, fs)
    fake_metrics = compute_morphological_metrics(fake_signals, fs)

    if 'hr_mean' in real_metrics and 'hr_mean' in fake_metrics:
        return abs(real_metrics['hr_mean'] - fake_metrics['hr_mean'])
    return float('nan')


def compute_patient_reid(encoder, real_signals, fake_signals, top_k=5):
    """
    Patient Re-Identification accuracy.

    For each generated ECG, check if the encoder maps it closest
    to the correct patient's real ECG.

    Args:
        encoder: FeatureExtractorPT model
        real_signals: (N, 1, T) tensor — real ECGs (one per patient)
        fake_signals: (N, 1, T) tensor — generated ECGs (one per patient, same order)
        top_k: Top-K accuracy threshold
    Returns:
        top1_acc, topk_acc: float
    """
    encoder.eval()
    device = next(encoder.parameters()).device

    with torch.no_grad():
        real_feats = encoder(real_signals.to(device))  # (N, 512)
        fake_feats = encoder(fake_signals.to(device))  # (N, 512)

    # Normalize
    real_feats = torch.nn.functional.normalize(real_feats, dim=-1)
    fake_feats = torch.nn.functional.normalize(fake_feats, dim=-1)

    # Cosine similarity matrix: (N, N)
    sim_matrix = torch.matmul(fake_feats, real_feats.T)

    # Top-1 and Top-K accuracy
    N = sim_matrix.shape[0]
    labels = torch.arange(N, device=device)

    top1_correct = (sim_matrix.argmax(dim=1) == labels).sum().item()
    topk_correct = 0
    for i in range(N):
        top_k_indices = sim_matrix[i].topk(min(top_k, N)).indices
        if labels[i] in top_k_indices:
            topk_correct += 1

    return top1_correct / N, topk_correct / N


class ECGEvaluator:
    """
    Comprehensive ECG generation evaluator.

    Follows the ECG-Bench three-level protocol.
    """

    def __init__(self, encoder=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        if encoder is not None:
            self.encoder = encoder.to(self.device)
            self.encoder.eval()

    @torch.no_grad()
    def _extract_features(self, signals):
        """Extract features using the encoder. signals: (N, 1, T) tensor."""
        if self.encoder is None:
            raise ValueError("Encoder required for FFD/MMD computation")
        return self.encoder(signals.to(self.device)).cpu().numpy()

    def evaluate_all(self, real_signals_np, fake_signals_np, fs=500):
        """
        Run all evaluation metrics.

        Args:
            real_signals_np: (N, T) numpy array — real ECGs
            fake_signals_np: (N, T) numpy array — generated ECGs
            fs: sampling rate
        Returns:
            dict of all metrics
        """
        results = {}

        # Level 1: Distribution metrics (if encoder available)
        if self.encoder is not None:
            real_tensor = torch.from_numpy(real_signals_np[:, np.newaxis, :]).float()
            fake_tensor = torch.from_numpy(fake_signals_np[:, np.newaxis, :]).float()

            real_feats = self._extract_features(real_tensor)
            fake_feats = self._extract_features(fake_tensor)

            results['FFD'] = compute_ffd(real_feats, fake_feats)
            results['MMD'] = compute_mmd(real_feats, fake_feats)

            # Level 3: Patient Re-ID
            top1, top5 = compute_patient_reid(self.encoder, real_tensor, fake_tensor)
            results['ReID_Top1'] = top1
            results['ReID_Top5'] = top5

        # Level 2: Morphological metrics
        real_morph = compute_morphological_metrics(real_signals_np, fs)
        fake_morph = compute_morphological_metrics(fake_signals_np, fs)

        results['real_morphology'] = real_morph
        results['fake_morphology'] = fake_morph

        if 'hr_mean' in real_morph and 'hr_mean' in fake_morph:
            results['HR_MAE'] = abs(real_morph['hr_mean'] - fake_morph['hr_mean'])

        return results
