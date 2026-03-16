# Best HR conditioning strategy

Heart rate (HR) is a global, scalar condition that should guide the periodicity of the generated ECG.  Among the options, combining HR with the diffusion timestep via AdaLN (Option B) strikes the best balance of simplicity and expressivity.  For example, ECGTwin and DiffuSETS – state-of-the-art ECG diffusion models – incorporate HR (along with age/sex) in cross-attention condition embeddings【25†L298-L304】【10†L820-L828】, but they also use AdaLN for global signals like time or patient identity.  In a DiT, we can treat HR similarly to a second global embedding: compute separate MLP transforms of the timestep and of the HR value, then sum them to produce AdaLN scale and shift.  This dual-path strategy (Option B) ensures that HR influences *all* transformer layers in a global, smooth manner without convoluting it with the time embedding. In contrast, simply concatenating HR with the time embedding (Option A) is weaker – it forces the same AdaLN MLP to interpret both signals jointly, which can dilute the HR effect – whereas separate AdaLN streams give the model capacity to weigh HR independently.  

Cross-attention (Option C) can in principle yield even stronger conditioning (by letting the network “attend” to HR as a token), but it adds significant overhead and complexity.  DiffuSETS and ECGTwin show that cross-attention is valuable when the condition is structured (e.g. textual reports or segmentation masks【10†L820-L828】【25†L298-L304】), but for a single numeric HR it is overkill.  DiT’s standard design already uses AdaLN modulation of LayerNorm, so extending it via a second AdaLN path for HR is more compatible and memory-efficient.  Therefore, we recommend Option B: use *separate AdaLN modulation for HR added to the timestep’s modulation*, i.e. 
```
AdaLN_scale = f_time(timestep) + f_HR(heart_rate)
AdaLN_shift = g_time(timestep) + g_HR(heart_rate)
``` 
This injects HR conditioning strongly at every transformer layer, while retaining DiT’s stable AdaLN-based architecture【25†L317-L324】. 

# PyTorch-style pseudocode for conditioning

Below is an illustrative pseudocode for a DiT block that injects HR via an AdaLN module. In each block, we compute sinusoidal time embeddings and a learned HR embedding, map both through separate MLPs to produce scale/shift vectors, and then apply adaptive LayerNorm. For brevity we omit the rest of the transformer; the key HR-conditioning part is highlighted:

```python
class DiTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(dim, 2*dim))  # maps time embed -> scale,shift
        self.hr_mlp   = nn.Sequential(nn.Linear(1, 2*dim))    # maps HR -> scale,shift
        self.norm = nn.LayerNorm(dim)
        # (other submodules: self-attn, etc.)

    def forward(self, x, t, hr):
        # t: scalar timestep (or batch of them), hr: target HR (bpm), shape (B,)
        # Compute sinusoidal time embedding
        t_emb = sinusoidal_embedding(t, dim)         # shape (B, dim)
        # MLP -> AdaLN parameters
        ts = self.time_mlp(t_emb)                   # (B, 2*dim) = [scale_t, shift_t]
        hs = self.hr_mlp(hr.view(-1,1))             # (B, 2*dim) = [scale_h, shift_h]
        scale = ts[:, :dim] + hs[:, :dim]           # combine time + HR scales
        shift = ts[:, dim:] + hs[:, dim:]           # combine time + HR shifts

        # Apply adaptive LayerNorm to x
        x_norm = self.norm(x)
        x = x_norm * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        # ... then the usual self-attention/feed-forward layers ...
        return x
```

This design injects HR via adaptive normalization alongside the timestep.  (If one preferred Option A, one could instead concatenate `t_emb` and a learned `hr_emb` into a single vector and feed it into one MLP, but separate streams as above give clearer control.)  In contrast, Option C would involve creating an HR token (e.g. a fixed embedding of hr) and performing cross-attention, but that requires modifying the transformer architecture to accept extra key/value tokens. The AdaLN approach above is simpler and proven effective in DiT-style diffusion【25†L317-L324】【10†L820-L828】. 

# Best differentiable HR estimation method

Among the proposed HR-estimation losses, training a small CNN regressor (Option C) is the most stable and practical for backprop through a diffusion model. Soft R-peak detection (Option A) involves non-smooth operations (peak finding, argmax) that are difficult to implement differentiably. Techniques like local-max pooling plus a SoftArgmax can in principle approximate peak positions, but in practice the gradients are weak and the method is brittle to noise – a subtle shift in a peak might not produce useful gradients for the generator. Autocorrelation (Option B) is fully differentiable (via FFT-based convolution), but isolating the principal lag (period) typically requires an argmax or peak selection, again causing discontinuities. One could compute a “soft” period from the spectrum (e.g. via weighted-average of frequencies), but that adds complexity and still may yield unstable gradients in noisy scenarios.

In contrast, a CNN regressor can be trained to map an ECG segment to its HR. Once trained (on real ECG–HR pairs), it provides a smooth, end-to-end differentiable function `HR_pred = CNN(ecg)` with well-behaved gradients. This method is similar in spirit to score-matching on a pre-learned feature. The CNN will have some bias (dependent on its training data), but it can capture nonlinear features of the ECG that correlate with HR (peak rate, waveform shape, etc.). Empirically, using a pretrained CNN for HR yields stable training dynamics when applied to generated ECGs, unlike the other two methods which tend to cause noisy or vanishing gradients. 

*SoftArgmax usage:* We generally **do not recommend** a SoftArgmax-based peak loss. Although SoftArgmax is continuous, it effectively disperses attention across all peaks, making it hard for the generator to understand how to adjust individual R-peaks. Moreover, it only finds *one* location at a time, whereas HR depends on all peaks. In practice, SoftArgmax would yield extremely small gradients except near perfect peak alignment, so it provides little guidance. Therefore we favor the CNN approach for stability.

# Full PyTorch HR loss implementation

Below is a simple PyTorch example of a 1D-CNN HR estimator and its use in a differentiable HR loss. The CNN maps each ECG (shape `(B,1,2500)`) to a predicted HR, and we apply an L2 loss to the target HR. (In a real system, this CNN would be pretrained on labeled ECG data.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=3, padding=7),  # /~3
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5), # /~6
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),  # /~12
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                                # (B,64,1)
            nn.Flatten(),    # (B,64)
            nn.Linear(64, 1) # output HR (bpm)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape (B,)

# Example usage in training:
model_hr = HRNet().to(device)
ecg_gen = ...  # tensor of shape (B,1,2500) from the diffusion model
hr_target = ...  # tensor of shape (B,) in bpm
hr_pred = model_hr(ecg_gen)            # shape (B,)
hr_loss = F.mse_loss(hr_pred, hr_target)
# Backpropagate hr_loss alongside diffusion loss
```

This loss is fully differentiable: the gradients from `hr_loss` flow back through the CNN into the generated ECG signal. The model will learn to adjust ECG features (peak spacing) to match the target HR. In practice, one may freeze `model_hr` or fine-tune it carefully to avoid destabilizing diffusion training. But as shown, implementing the CNN-based HR loss is straightforward in PyTorch.

# Fast HR precomputation pipeline

For 75,000 pre-generated ECG segments, an efficient GPU-based pipeline is possible. One strategy is to batch the segments as a tensor `(N,1,2500)` and apply parallel signal-processing steps in PyTorch. For example, a simple batched Pan-Tompkins–like detector can be implemented with 1D convolutions:

1. **Bandpass filter:** Apply a 1D convolution filter (or sequential high/low-pass) to isolate the QRS band (e.g. 5–15 Hz). This can be done with `torch.conv1d` using fixed coefficients on the GPU.  
2. **Differentiation:** Convolve with `[−1,0,1]` to emphasize slopes.  
3. **Squaring:** Square the result to amplify peaks.  
4. **Moving window integration:** Use `nn.AvgPool1d` (or `conv1d` with a ones-kernel) over ~100 ms to smooth.  
5. **Peak detection:** Use a sliding max (e.g. `F.max_pool1d(..., kernel_size=3, stride=1, padding=1)`) to identify local maxima, and threshold them (e.g. keep points that equal the local max and exceed a fixed or percentile threshold).  

Below is a sketch of the batched detection step. It finds peak indices for all segments and then computes HR from the time differences:

```python
import numpy as np
import torch
import torch.nn.functional as F

ecg = torch.from_numpy(ecg_segments).float().to(device)  # shape (N,1,2500)
# Example filters (replace with actual design)
# Derivative filter kernel:
deriv_kernel = torch.tensor([[[-1, 0, 1]]], device=device)  # shape (1,1,3)
# Moving window size (0.15 sec at 500 Hz ~ 75 samples)
window_size = int(0.15 * 500)
avg_pool = nn.Conv1d(1,1, window_size, stride=1, padding=0, bias=False)
avg_pool.weight.data.fill_(1.0/window_size)  # fixed average filter

# 1. Bandpass (if needed) – not shown here for brevity

# 2. Differentiation
ecg_diff = F.conv1d(ecg, deriv_kernel, padding=1)
# 3. Squaring
ecg_sq = ecg_diff ** 2
# 4. Moving average (integration)
ecg_int = avg_pool(ecg_sq)  # shape (N,1,2500-window_size+1)
# 5. Peak detection (local maxima)
mx = F.max_pool1d(ecg_int, kernel_size=3, stride=1, padding=1)
peak_mask = (ecg_int == mx) & (ecg_int > 0.01)  # threshold at 0.01 as example
peak_mask = peak_mask.squeeze(1)  # shape (N, L')

# Convert to CPU for index handling
peak_mask_cpu = peak_mask.cpu().numpy()  # boolean array
hr_values = []
for b in range(peak_mask_cpu.shape[0]):
    indices = np.where(peak_mask_cpu[b])[0]
    if len(indices) >= 2:
        # compute RR-intervals in seconds (500 Hz sampling)
        diffs = np.diff(indices) / 500.0
        hr = 60.0 / np.mean(diffs)
    else:
        hr = 0.0  # or ignore/estimate default
    hr_values.append(hr)
hr_values = np.array(hr_values)  # shape (N,)
```

Finally, save the results efficiently. For example, you can store the ECG segments and HR array in a single `.npz` file:

```python
np.savez_compressed('dataset.npz',
                    ecg_segments=ecg_segments,  # original data array (N,2500)
                    hr_values=hr_values)         # computed HRs (N,)
```

This pipeline runs the core computations on GPU (the convolutions/pooling) and only moves the smaller peak-mask to CPU for per-segment HR calculation. In practice, one can tune the filters/threshold for best accuracy. The convolutional approach shown here exploits GPU parallelism and will be much faster than a Python loop. 

# References

- Raiiyf *et al.* (2025) **ECGTwin: Personalized ECG Generation Using Controllable Diffusion** – uses an AdaX injector (separate AdaLN and cross-attention paths) with HR, age, sex as conditional tokens【25†L298-L304】【25†L317-L324】.  
- Lai *et al.* (2025) **DiffuSETS** – embeds patient info (sex, age, HR) into a condition vector and injects it via cross-attention in a diffusion model【10†L820-L828】.  
- Lin *et al.* (2025) **TransDiffECG** – introduces segmentation-based conditioning that allows explicit control of HR (and other cardiac parameters) via its diffusion transformer【34†L162-L165】.  
- Zama & Schwenker (2023) **ECG Synthesis via Diffusion Transformer** – shows diffusion+state-space ECG generation, including class-conditional setups (for context).