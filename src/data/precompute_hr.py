"""
GPU-Batched HR Precomputation for CardioEquation
==================================================
Run once before training to add HR labels to all .npz files.

Usage:
    python src/data/precompute_hr.py

Pipeline:
    1. Pan-Tompkins preprocessing via GPU conv1d (bandpass → derivative → square → integrate)
    2. FFT autocorrelation for dominant lag detection
    3. Convert lag → bpm, clamp to physiological range [30, 200]

Throughput: ~20k-50k segments/sec on GPU (~5-10s for 75k total)

References:
    - Research_report.md Part 3: Fast HR Precomputation
    - Pan & Tompkins (1985): real-time QRS detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm


class GPUBatchedHRExtractor:
    """
    Extract heart rate from ECG segments using GPU-accelerated pipeline.

    Steps:
        1. Bandpass filter (5-15 Hz) to isolate QRS energy
        2. Derivative (5-point) to enhance QRS slopes
        3. Square to make all values positive
        4. Moving average integration (150ms window)
        5. FFT autocorrelation to find dominant RR interval
        6. Convert dominant lag to bpm
    """

    def __init__(self, fs=500.0, batch_size=512, hr_min=30.0, hr_max=200.0):
        self.fs = fs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hr_min = hr_min
        self.hr_max = hr_max
        self._build_filters()

    def _build_filters(self):
        """Pre-build convolution kernels for Pan-Tompkins preprocessing."""
        # Bandpass filter: 5–15 Hz via FIR (Hamming window)
        n_taps = 33
        fc_low = 5.0 / (self.fs / 2)
        fc_high = 15.0 / (self.fs / 2)
        t = torch.linspace(-(n_taps // 2), n_taps // 2, n_taps)

        # sinc bandpass
        h_high = torch.sinc(2 * fc_high * t) * 2 * fc_high
        h_low = torch.sinc(2 * fc_low * t) * 2 * fc_low
        h_bp = (h_high - h_low) * torch.hamming_window(n_taps)
        h_bp = h_bp / h_bp.abs().sum()

        # Moving average integration window (150ms)
        win_size = int(0.15 * self.fs)  # 75 samples at 500Hz

        self.bp_weight = h_bp.view(1, 1, -1).to(self.device)
        self.int_weight = (torch.ones(win_size) / win_size).view(1, 1, -1).to(self.device)
        self.n_taps = n_taps
        self.win_size = win_size

    @torch.no_grad()
    def _preprocess(self, x):
        """Pan-Tompkins preprocessing: bandpass → derivative → square → integrate."""
        B, T = x.shape
        x3 = x.unsqueeze(1)  # (B, 1, T)

        # Step 1: Bandpass filter (5–15 Hz)
        x_bp = F.conv1d(x3, self.bp_weight, padding=self.n_taps // 2)[:, 0, :T]

        # Step 2: 5-point derivative (Pan-Tompkins)
        dk = torch.tensor(
            [-1., -2., 0., 2., 1.], device=self.device
        ).view(1, 1, -1) * (self.fs / 8.0)
        x_diff = F.conv1d(x_bp.unsqueeze(1), dk, padding=2)[:, 0, :]

        # Step 3: Square
        x_sq = x_diff ** 2

        # Step 4: Moving average integration
        x_integ = F.conv1d(
            x_sq.unsqueeze(1), self.int_weight, padding=self.win_size // 2
        )[:, 0, :T]

        return x_integ

    @torch.no_grad()
    def _fft_hr(self, x_integ):
        """FFT autocorrelation → dominant lag → bpm."""
        B, T = x_integ.shape

        # Zero-mean
        x_n = x_integ - x_integ.mean(dim=-1, keepdim=True)

        # FFT autocorrelation (Wiener-Khinchin)
        n_fft = 2 ** math.ceil(math.log2(2 * T - 1))
        X = torch.fft.rfft(x_n, n=n_fft)
        acf = torch.fft.irfft(X.real**2 + X.imag**2, n=n_fft)[:, :T]

        # Normalize by zero-lag
        acf = acf / (acf[:, 0:1].abs() + 1e-8)

        # Find peak in physiological RR range
        lag_min = int(self.fs * 60.0 / self.hr_max)  # ~150 samples for 200 bpm
        lag_max = min(int(self.fs * 60.0 / self.hr_min), T - 1)  # ~1000 for 30 bpm

        peak_lags = acf[:, lag_min:lag_max].argmax(dim=-1) + lag_min

        # Convert lag → HR (bpm)
        hr = (60.0 * self.fs / peak_lags.float()).clamp(self.hr_min, self.hr_max)

        return hr.cpu()

    def extract_dataset(self, ecg_array):
        """
        Extract HR for entire dataset.

        Args:
            ecg_array: (N, T) numpy array of ECG segments

        Returns:
            hr_labels: (N,) numpy array of HR values in bpm
        """
        N = ecg_array.shape[0]
        hr_all = np.zeros(N, dtype=np.float32)

        for start in tqdm(range(0, N, self.batch_size), desc="Extracting HR (GPU)"):
            end = min(start + self.batch_size, N)
            x = torch.from_numpy(ecg_array[start:end]).float().to(self.device)
            x_integ = self._preprocess(x)
            hr_all[start:end] = self._fft_hr(x_integ).numpy()

        return hr_all


def process_npz_file(npz_path, extractor):
    """Process a single .npz file: extract HR + save back."""
    p = Path(npz_path)
    if not p.exists():
        print(f"  ⏭️  Skipping {npz_path} (not found)")
        return

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    # Find the ECG signal key
    ecg_key = None
    for candidate in ['signals', 'context', 'ecg', 'data']:
        if candidate in keys:
            ecg_key = candidate
            break
    if ecg_key is None:
        ecg_key = keys[0]

    ecg = data[ecg_key].astype(np.float32)

    # Handle channel dim: (N, C, T) → (N, T) by taking lead I
    if ecg.ndim == 3:
        if ecg.shape[1] == 1:
            ecg = ecg[:, 0, :]
        elif ecg.shape[-1] == 1:
            ecg = ecg[:, :, 0]
        else:
            ecg = ecg[:, 0, :]  # Lead I

    print(f"\n📊 Processing {npz_path}: {ecg.shape}")
    hr = extractor.extract_dataset(ecg)
    print(f"   HR stats: mean={hr.mean():.1f}, std={hr.std():.1f}, "
          f"range=[{hr.min():.1f}, {hr.max():.1f}] bpm")

    # Save new npz with hr_labels added
    out = dict(data)
    out["hr_labels"] = hr
    np.savez_compressed(str(p), **out)
    print(f"   ✅ Saved with hr_labels → {npz_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("CardioEquation — GPU HR Precompute")
    print("=" * 60)

    extractor = GPUBatchedHRExtractor(fs=500.0, batch_size=512)
    print(f"Device: {extractor.device}")

    # Process all dataset files
    npz_files = [
        "data/mitbih_forecasting.npz",
        "data/ptbxl_processed.npz",
        "data/chapman_processed.npz",
    ]

    for path in npz_files:
        process_npz_file(path, extractor)

    print("\n✅ All HR labels precomputed!")
