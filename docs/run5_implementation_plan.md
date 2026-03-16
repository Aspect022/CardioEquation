# CardioEquation — Run 5 Implementation Plan
**Research-Backed, Ready to Code**

> All design decisions below are grounded in the completed research (see `Research/Research_report.md` and `Research/deep-research-report (1).md`).

---

## Summary of Research Findings

### Finding 1 — HR Conditioning: Option B (Separate AdaLN Streams) ✅
Both research reports unanimously confirm **Option B** as optimal:
- Option A (fuse HR into timestep): HR gradients smothered by timestep signal — proven failure mode
- Option B (separate MLP per signal): Dedicated gradient pathway for HR, additive combination before AdaLN. Mirrors how DiT-XL handles class conditioning on ImageNet
- Option C (cross-attention): Overkill for a scalar; causes instability, adds O(seq_len) overhead

**Critical insight**: HR head must be **zero-initialized** so training starts from pure timestep conditioning and gradually learns HR — prevents early gradient explosions.

### Finding 2 — Differentiable HR Loss: FFT Autocorrelation ✅
Research verdict: **FFT autocorrelation + soft argmax** over physiological lag range is the most stable method:
- Soft R-peak/SoftArgmax: Fails early in training when ECG is still noisy — sparse, contradictory gradients
- CNN estimator: Biased by pretraining data; co-adaptation failure if fine-tuned jointly
- FFT autocorrelation: Smooth, differentiable via `torch.fft`, gracefully degrades under noise

**Implementation**: `DifferentiableHRLoss` class — autocorr → soft argmax over lag range [150, 1000] samples → convert to bpm → L1 loss. Ramp weight from 0→1 over 5000 warmup steps.

### Finding 3 — HR Precomputation: GPU-Batched FFT Pipeline ✅
For 75k segments at 500Hz:
- NeuroKit2 (baseline): ~1 seg/sec = 10–40 hours ❌
- **GPU FFT pipeline: ~20,000–50,000 seg/sec = 3–8 seconds** ✅

**Implementation**: `GPUBatchedHRExtractor` — Pan-Tompkins preprocessing (GPU conv1d) → FFT autocorrelation for dominant lag → HR in bpm. Output saved as `dataset.npz`.

---

## What Changed Since Previous Plan

| Previous Plan | Updated Plan (Research-Backed) |
|---------------|-------------------------------|
| `HRConditioner`: simple Linear(1→d_model) | `HREmbedding`: sinusoidal encoding → MLP, normalized [30–200 bpm] |
| HR fused additively into combined cond | **Separate ConditioningProjector** with 3 independent MLP heads (t, hr, patient_id) |
| `rr_interval_loss` using argmax (not differentiable) | `DifferentiableHRLoss` using soft argmax (fully differentiable) |
| Precompute HR with NeuroKit2 (slow, CPU) | `GPUBatchedHRExtractor` (GPU, ~5 seconds for 75k) |
| HR loss: immediate full weight | HR loss: warmup ramp over 5000 steps |
| Zero-init not mentioned | **Zero-init HR head** — critical for training stability |

---

## File Changes for Run 5

### Step 0: Pre-compute HR Labels [DO FIRST]

**New file**: `src/data/precompute_hr.py`

```python
"""
Run once before training: python src/data/precompute_hr.py
Adds HR labels to existing .npz files.
Throughput: ~20k-50k segments/sec on GPU (~5s for 75k total).
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm


class GPUBatchedHRExtractor:
    def __init__(self, fs=500.0, batch_size=512, hr_min=30.0, hr_max=200.0):
        self.fs = fs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hr_min, self.hr_max = hr_min, hr_max
        self._build_filters()

    def _build_filters(self):
        n_taps = 33
        fc_low  = 5.0  / (self.fs / 2)
        fc_high = 15.0 / (self.fs / 2)
        t = torch.linspace(-(n_taps // 2), n_taps // 2, n_taps)
        h_bp = torch.sinc(2 * fc_high * t) * 2 * fc_high - torch.sinc(2 * fc_low * t) * 2 * fc_low
        h_bp = h_bp * torch.hamming_window(n_taps) / h_bp.abs().sum()
        win_size = int(0.15 * self.fs)
        self.bp_weight   = h_bp.view(1, 1, -1).to(self.device)
        self.int_weight  = (torch.ones(win_size) / win_size).view(1, 1, -1).to(self.device)
        self.n_taps, self.win_size = n_taps, win_size

    @torch.no_grad()
    def _preprocess(self, x):
        B, T = x.shape
        x3 = x.unsqueeze(1)
        x_bp    = F.conv1d(x3, self.bp_weight, padding=self.n_taps//2)[:, 0, :T]
        dk      = torch.tensor([-1., -2., 0., 2., 1.], device=self.device).view(1,1,-1) * (self.fs / 8.0)
        x_diff  = F.conv1d(x_bp.unsqueeze(1), dk, padding=2)[:, 0, :]
        x_sq    = x_diff ** 2
        x_integ = F.conv1d(x_sq.unsqueeze(1), self.int_weight, padding=self.win_size//2)[:, 0, :T]
        return x_integ

    @torch.no_grad()
    def _fft_hr(self, x_integ):
        B, T = x_integ.shape
        x_n = x_integ - x_integ.mean(dim=-1, keepdim=True)
        n_fft = 2 ** math.ceil(math.log2(2 * T - 1))
        X = torch.fft.rfft(x_n, n=n_fft)
        acf = torch.fft.irfft(X.real**2 + X.imag**2, n=n_fft)[:, :T]
        acf = acf / (acf[:, 0:1].abs() + 1e-8)
        lag_min = int(self.fs * 60.0 / self.hr_max)
        lag_max = min(int(self.fs * 60.0 / self.hr_min), T - 1)
        peak_lags = acf[:, lag_min:lag_max].argmax(dim=-1) + lag_min
        return (60.0 * self.fs / peak_lags.float()).clamp(self.hr_min, self.hr_max).cpu()

    def extract_dataset(self, ecg_array):
        N = ecg_array.shape[0]
        hr_all = np.zeros(N, dtype=np.float32)
        for start in tqdm(range(0, N, self.batch_size), desc="Extracting HR (GPU)"):
            end = min(start + self.batch_size, N)
            x = torch.from_numpy(ecg_array[start:end]).float().to(self.device)
            hr_all[start:end] = self._fft_hr(self._preprocess(x)).numpy()
        return hr_all


if __name__ == "__main__":
    extractor = GPUBatchedHRExtractor()
    for npz_path in ["data/ptbxl_processed.npz", "data/mitbih_forecasting.npz", "data/chapman_processed.npz"]:
        p = Path(npz_path)
        if not p.exists():
            print(f"Skipping {npz_path} (not found)")
            continue
        data = np.load(npz_path)
        # Handle different key names
        ecg_key = "signals" if "signals" in data else "ecg" if "ecg" in data else list(data.keys())[0]
        ecg = data[ecg_key].astype(np.float32)
        if ecg.ndim == 3:
            ecg = ecg[:, 0, :]  # Lead I only
        print(f"\nProcessing {npz_path}: {ecg.shape}")
        hr = extractor.extract_dataset(ecg)
        print(f"  HR mean={hr.mean():.1f}, std={hr.std():.1f}, range=[{hr.min():.1f},{hr.max():.1f}]")
        # Save new npz with hr labels added
        out = dict(data)
        out["hr_labels"] = hr
        np.savez_compressed(str(p), **out)
        print(f"  Saved with hr_labels to {npz_path}")
```

---

### Step 1: Update `dit_ecg.py` — Replace HRConditioner with HREmbedding + ConditioningProjector

**Key changes**:
1. Replace simple `HRConditioner(Linear 1→d_model)` with research-specified `HREmbedding` (sinusoidal + MLP, normalized to [0,1])  
2. Add `ConditioningProjector` with **three separate MLP heads**: timestep, HR, patient_id  
3. Zero-initialize HR and patient_id MLP last layers — training starts stable  
4. Combine: `cond = t_out + hr_out + pid_out` (additive, not concatenated)

```python
class HREmbedding(nn.Module):
    """Research-confirmed: sinusoidal encoding of scalar HR, then MLP."""
    def __init__(self, hidden_dim=256, hr_min=30.0, hr_max=200.0):
        super().__init__()
        self.hr_min, self.hr_max = hr_min, hr_max
        self.register_buffer("freqs",
            torch.exp(-math.log(10000) * torch.arange(hidden_dim // 2) / (hidden_dim // 2)))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hr):  # hr: (B,) bpm
        hr_norm = ((hr - self.hr_min) / (self.hr_max - self.hr_min)).clamp(0, 1)
        args = hr_norm[:, None] * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)  # (B, hidden_dim)


class ConditioningProjector(nn.Module):
    """3 independent MLP heads → additive combination. HR head zero-initialized."""
    def __init__(self, hidden_dim=256, d_model=768):
        super().__init__()
        def head(out_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        self.t_mlp  = head(d_model)
        self.hr_mlp = head(d_model)
        self.pid_mlp = head(d_model)
        # Zero-init HR and patient heads — training starts from timestep only
        nn.init.zeros_(self.hr_mlp[-1].weight);  nn.init.zeros_(self.hr_mlp[-1].bias)
        nn.init.zeros_(self.pid_mlp[-1].weight); nn.init.zeros_(self.pid_mlp[-1].bias)

    def forward(self, t_emb, hr_emb, pid_emb):  # all (B, hidden_dim)
        return self.t_mlp(t_emb) + self.hr_mlp(hr_emb) + self.pid_mlp(pid_emb)  # (B, d_model)
```

In `DiTECG.__init__`: replace `self.hr_conditioner` with `self.hr_embed = HREmbedding()` and `self.cond_proj = ConditioningProjector()`.

In `DiTECG.forward(x_t, t, hr_bpm, identity_emb)`:
```python
t_emb   = self.t_embed(t)           # (B, 256) — existing sinusoidal
hr_emb  = self.hr_embed(hr_bpm)     # (B, 256) — NEW
pid_emb = self.pid_proj(identity_emb)  # (B, 256) — project from 512→256
cond    = self.cond_proj(t_emb, hr_emb, pid_emb)  # (B, 768) — combined
# then feed cond into each DiT block as before
```

---

### Step 2: Update `losses_v2.py` — Replace rr_interval_loss with DifferentiableHRLoss

**Replace** the existing placeholder `rr_interval_loss` and `hr_variance_regularization_loss` with the research-confirmed class:

```python
class DifferentiableHRLoss(nn.Module):
    """
    Research-confirmed: FFT autocorrelation + soft argmax.
    Fully differentiable, smooth gradients even on noisy signals.
    Weight should be ramped from 0→hr_loss_weight over warmup_steps.
    """
    def __init__(self, fs=500.0, hr_min=30.0, hr_max=200.0, softmax_temp=10.0):
        super().__init__()
        self.fs, self.hr_min, self.hr_max, self.temp = fs, hr_min, hr_max, softmax_temp
        self.lag_min = int(fs * 60.0 / hr_max)   # ~150 samples
        self.lag_max = int(fs * 60.0 / hr_min)   # ~1000 samples

    def forward(self, ecg, hr_target, weight=1.0):
        """ecg: (B,1,T), hr_target: (B,) bpm → (loss, hr_estimated)"""
        x = ecg.squeeze(1) - ecg.squeeze(1).mean(dim=-1, keepdim=True)
        B, T = x.shape
        n_fft = 2 ** math.ceil(math.log2(2 * T - 1))
        X = torch.fft.rfft(x, n=n_fft)
        acf = torch.fft.irfft(X.real**2 + X.imag**2, n=n_fft)[:, :T]
        acf = acf / (acf[:, 0:1] + 1e-8)
        # Soft argmax over physiological lag window
        window = acf[:, self.lag_min:self.lag_max]
        lags = torch.arange(window.shape[-1], device=x.device, dtype=x.dtype) + self.lag_min
        weights = F.softmax(self.temp * window, dim=-1)
        lag = (weights * lags).sum(dim=-1)
        hr_est = (60.0 * self.fs / (lag + 1e-6)).clamp(self.hr_min, self.hr_max)
        loss = F.l1_loss(hr_est, hr_target.float())
        return weight * loss, hr_est.detach()


def hr_variance_loss(hr_estimated, target_std=47.0, weight=1.0):
    """Penalizes when batch HR std drops below target (real HR std = 47.4 bpm)."""
    return weight * torch.relu(target_std - hr_estimated.std())
```

---

### Step 3: Update `train_dit.py` — Wire HR into training loop

**Dataset loading**: Load precomputed `hr_labels` from npz files, concat alongside signals.

**HR-balanced sampling** (replace uniform sampler):
```python
hr_bins = np.digitize(hr_labels, bins=[40, 55, 65, 75, 85, 100, 120, 160])
bin_counts = np.bincount(hr_bins, minlength=9).clip(min=1)
sample_weights = 1.0 / bin_counts[hr_bins]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

**Training loop** — add after `combined_loss` computed:
```python
# HR conditioning loss (ramped weight)
hr_ramp = min(1.0, global_step / 5000)
hr_loss_val, hr_est = hr_loss_fn(x0_pred, hr_batch, weight=0.05 * hr_ramp)
hr_var = hr_variance_loss(hr_est, target_std=47.0, weight=0.5 * hr_ramp)
total_loss = combined_loss + hr_loss_val + hr_var

# Log HR metrics to wandb
log_dict["hr_loss"] = hr_loss_val.item()
log_dict["hr_variance_loss"] = hr_var.item()
log_dict["hr_est_mean"] = hr_est.mean().item()
log_dict["hr_est_std"] = hr_est.std().item()
```

**CFG dropout** for HR (10% probability, use neutral HR 75.0 when dropped):
```python
if cfg_dropout_rate > 0:
    hr_drop_mask = torch.rand(B, device=device) < cfg_dropout_rate
    hr_batch[hr_drop_mask] = 75.0
```

**Instantiate losses after model**:
```python
hr_loss_fn = DifferentiableHRLoss(fs=500.0, hr_min=30.0, hr_max=200.0, softmax_temp=10.0).to(device)
```

---

## Execution Order

```
Day 1:  python src/data/precompute_hr.py          # ~5-10 sec, adds hr_labels to all .npz
Day 1:  Edit dit_ecg.py (HREmbedding + CondProj)  # ~2 hours
Day 2:  Edit losses_v2.py (DifferentiableHRLoss)  # ~1 hour
Day 2:  Edit train_dit.py (wiring + sampler)       # ~2 hours
Day 3:  python run_training.sh                     # Run 5 starts
```

---

## Targets for Run 5

| Metric | Run 4 | Run 5 Target | Why Achievable |
|--------|-------|--------------|----------------|
| FFD ↓ | 21.7 | < 15 | Spectral fix removes noise in FFT space |
| HR MAE ↓ | 31.4 bpm | < 8 bpm | Explicit HR conditioning + HR loss |
| HR Std | 20.5 bpm | > 35 bpm | HR variance loss + balanced sampler |
| ReID Top-1 ↑ | 11.8% | > 15% | Identity stream now isolated from HR |
| spectral (W&B) | 125.19 | < 0.5 | w_spectral = 0.0001 fix |

---

## W&B Things to Watch

| Signal | What to Look For |
|--------|-----------------|
| `component/spectral` | Must stay < 1.0 throughout |
| `hr_est_std` | Should increase from ~20 → ~35+ over training |
| `hr_loss` | Should decrease smoothly (not oscillate) |
| `hr_variance_loss` | Should trend toward 0 |
| `best_val_loss` | Should improve faster than Run 4 |

---

## References

- **Research_report.md** (Claude) — Full PyTorch implementations confirmed  
- **deep-research-report.md** (Gemini) — Architecture comparison + concise verdicts  
- **ECGTwin (2025)** — AdaLN conditioning with zero-initialized auxiliary heads  
- **DiffuSETS (2025)** — Scalar conditioning via independent projection  
- **DiT (Peebles & Xie 2023)** — Class conditioning = separate AdaLN stream, zero-initialized  
