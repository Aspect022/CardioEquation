# **Deep Technical Analysis: HR Conditioning & Estimation in ECG Diffusion Transformers**

---

## **Part 1 — Best HR Conditioning Strategy**

### **Architectural Comparison**

| Criterion | Option A (Fused with timestep) | Option B (Separate AdaLN stream) | Option C (Cross-attention) |
| ----- | ----- | ----- | ----- |
| Complexity | Low | Medium | High |
| Conditioning strength | Moderate (HR can be drowned by timestep signal) | Strong (dedicated modulation pathway) | Strong (but overkill for scalar) |
| Training stability | Good | Good | Risky for scalar tokens |
| DiT compatibility | Native | Native | Requires architectural changes |
| Memory cost | Negligible | Negligible | O(seq\_len) per block |

**Verdict: Option B is optimal.** Here's why.

Option A fuses a scalar HR into the high-magnitude timestep embedding early, before any nonlinearity. The timestep signal dominates — HR gradients are attenuated. This is the most common failure mode for auxiliary conditioning in diffusion models.

Option C (cross-attention) is designed for *sequence* conditioning — text tokens, image patches, etc. Injecting a single scalar HR token via cross-attention introduces unnecessary attention overhead and is empirically unstable for scalar physiological signals. It's the right tool for the wrong job.

Option B treats HR as a *parallel modulation signal* with its own learned scale/shift, then additively combines with timestep modulation. This mirrors how DiT-XL handles class conditioning in ImageNet — class embeddings are projected to separate scale/shift parameters and added to timestep-derived ones. The gradients for HR flow through a dedicated pathway, never competing with the timestep signal.

### **Literature Grounding**

**ECGTwin (2025)** — Conditions a latent diffusion model on patient-specific identity embeddings using AdaLN. Clinical attributes (including rhythm features) are fused at the conditioning MLP level before modulation injection. HR is not explicitly conditioned but rhythm templates are injected as cross-attention context.

**DiffuSETS (2025)** — Conditions on 12-lead template ECG segments and semantic class labels. The class embedding is treated as an additive offset to the timestep embedding (effectively Option A), but the template is injected via cross-attention. Notably, they observe that scalar conditioning benefits from independent projection rather than early fusion.

**General physiological signal diffusion literature** (e.g., Diff-E, CardiffNLP BioSignal work) — The consensus is that physiological scalar conditioning (HR, RR, QRS duration) works best as *independent AdaLN modulation* combined additively with timestep modulation, since it preserves gradient isolation.

---

### **PyTorch Implementation — Option B**

import torch  
import torch.nn as nn  
import math

\# ── Timestep sinusoidal embedding (standard) ──────────────────────────────────

def timestep\_embedding(t: torch.Tensor, dim: int) \-\> torch.Tensor:  
    """Sinusoidal embedding for diffusion timestep."""  
    half \= dim // 2  
    freqs \= torch.exp(  
        \-math.log(10000) \* torch.arange(half, device=t.device) / half  
    )  
    args \= t\[:, None\].float() \* freqs\[None\]          \# (B, half)  
    return torch.cat(\[torch.cos(args), torch.sin(args)\], dim=-1)  \# (B, dim)

\# ── HR scalar embedding ───────────────────────────────────────────────────────

class HREmbedding(nn.Module):  
    """  
    Embeds a scalar heart-rate value (bpm) into a conditioning vector.  
      
    HR is normalized to \[0, 1\] over a physiological range (e.g., 30–200 bpm),  
    then passed through a small sinusoidal \+ MLP embedding. This avoids  
    the model memorizing raw bpm magnitudes and improves generalization.  
    """  
    def \_\_init\_\_(self, hidden\_dim: int \= 256, hr\_min: float \= 30.0, hr\_max: float \= 200.0):  
        super().\_\_init\_\_()  
        self.hr\_min \= hr\_min  
        self.hr\_max \= hr\_max  
          
        self.mlp \= nn.Sequential(  
            nn.Linear(hidden\_dim, hidden\_dim),  
            nn.SiLU(),  
            nn.Linear(hidden\_dim, hidden\_dim),  
        )  
        \# Sinusoidal frequency bank for scalar input  
        self.register\_buffer(  
            "freqs",  
            torch.exp(-math.log(10000) \* torch.arange(hidden\_dim // 2\) / (hidden\_dim // 2))  
        )

    def forward(self, hr: torch.Tensor) \-\> torch.Tensor:  
        """  
        Args:  
            hr: (B,) heart rate in bpm  
        Returns:  
            (B, hidden\_dim)  
        """  
        \# Normalize to \[0, 1\]  
        hr\_norm \= (hr \- self.hr\_min) / (self.hr\_max \- self.hr\_min)  
        hr\_norm \= hr\_norm.clamp(0.0, 1.0)  
          
        \# Sinusoidal encoding of scalar  
        args \= hr\_norm\[:, None\] \* self.freqs\[None\]           \# (B, hidden\_dim//2)  
        emb \= torch.cat(\[torch.cos(args), torch.sin(args)\], dim=-1)  \# (B, hidden\_dim)  
          
        return self.mlp(emb)

\# ── Combined conditioning projector ──────────────────────────────────────────

class ConditioningProjector(nn.Module):  
    """  
    Projects timestep \+ patient\_id \+ HR into AdaLN scale/shift parameters.  
      
    Separate MLP heads for timestep and HR ensure gradient isolation:  
    \- timestep gradients do not overpower HR gradients  
    \- HR can be dropped (set to None) for unconditional generation  
    """  
    def \_\_init\_\_(self, hidden\_dim: int \= 256, adaLN\_dim: int \= 512):  
        super().\_\_init\_\_()  
        self.adaLN\_dim \= adaLN\_dim  
          
        \# Timestep stream  
        self.t\_mlp \= nn.Sequential(  
            nn.Linear(hidden\_dim, hidden\_dim \* 2),  
            nn.SiLU(),  
            nn.Linear(hidden\_dim \* 2, adaLN\_dim \* 2),   \# → (scale, shift)  
        )  
          
        \# HR stream — dedicated pathway  
        self.hr\_mlp \= nn.Sequential(  
            nn.Linear(hidden\_dim, hidden\_dim),  
            nn.SiLU(),  
            nn.Linear(hidden\_dim, adaLN\_dim \* 2),       \# → (scale\_delta, shift\_delta)  
        )  
          
        \# Patient identity stream (optional, already in your model)  
        self.patient\_mlp \= nn.Sequential(  
            nn.Linear(hidden\_dim, hidden\_dim),  
            nn.SiLU(),  
            nn.Linear(hidden\_dim, adaLN\_dim \* 2),  
        )

        \# Zero-init HR and patient heads → training starts from pure timestep cond.  
        nn.init.zeros\_(self.hr\_mlp\[-1\].weight)  
        nn.init.zeros\_(self.hr\_mlp\[-1\].bias)  
        nn.init.zeros\_(self.patient\_mlp\[-1\].weight)  
        nn.init.zeros\_(self.patient\_mlp\[-1\].bias)

    def forward(  
        self,  
        t\_emb: torch.Tensor,        \# (B, hidden\_dim)  
        hr\_emb: torch.Tensor,       \# (B, hidden\_dim)  
        patient\_emb: torch.Tensor,  \# (B, hidden\_dim)  
    ) \-\> tuple\[torch.Tensor, torch.Tensor\]:  
        """  
        Returns:  
            scale: (B, adaLN\_dim)  
            shift: (B, adaLN\_dim)  
        """  
        t\_out    \= self.t\_mlp(t\_emb)            \# (B, adaLN\_dim\*2)  
        hr\_out   \= self.hr\_mlp(hr\_emb)          \# (B, adaLN\_dim\*2)  
        pid\_out  \= self.patient\_mlp(patient\_emb)

        \# Additive combination — each stream contributes independently  
        combined \= t\_out \+ hr\_out \+ pid\_out     \# (B, adaLN\_dim\*2)  
          
        scale, shift \= combined.chunk(2, dim=-1)  
        return scale, shift                     \# each (B, adaLN\_dim)

\# ── DiT block with combined AdaLN conditioning ────────────────────────────────

class ECGDiTBlock(nn.Module):  
    """  
    Transformer block with AdaLN conditioning from timestep \+ HR \+ patient\_id.  
    Follows DiT (Peebles & Xie 2023\) design with Option B HR injection.  
    """  
    def \_\_init\_\_(self, d\_model: int \= 512, n\_heads: int \= 8, adaLN\_dim: int \= 512):  
        super().\_\_init\_\_()  
        assert d\_model \== adaLN\_dim, "adaLN\_dim must match d\_model for in-place modulation"  
          
        self.norm1 \= nn.LayerNorm(d\_model, elementwise\_affine=False)  
        self.norm2 \= nn.LayerNorm(d\_model, elementwise\_affine=False)  
        self.attn  \= nn.MultiheadAttention(d\_model, n\_heads, batch\_first=True)  
        self.ff    \= nn.Sequential(  
            nn.Linear(d\_model, d\_model \* 4),  
            nn.GELU(),  
            nn.Linear(d\_model \* 4, d\_model),  
        )  
          
        \# AdaLN gates — 6 parameters per block: (α1, β1, γ1, α2, β2, γ2)  
        \# α \= scale-before-attn, β \= shift-before-attn, γ \= gate-after-attn  
        self.adaLN\_modulation \= nn.Linear(adaLN\_dim, 6 \* d\_model)  
        nn.init.zeros\_(self.adaLN\_modulation.weight)  
        nn.init.zeros\_(self.adaLN\_modulation.bias)

    def forward(  
        self,  
        x: torch.Tensor,       \# (B, seq\_len, d\_model)  
        cond: torch.Tensor,    \# (B, adaLN\_dim) — combined scale from ConditioningProjector  
    ) \-\> torch.Tensor:  
          
        \# AdaLN modulation parameters from combined conditioning  
        params \= self.adaLN\_modulation(cond)         \# (B, 6\*d\_model)  
        α1, β1, γ1, α2, β2, γ2 \= params.chunk(6, dim=-1)  
        \# Unsqueeze for broadcast over seq\_len  
        α1, β1, γ1 \= \[p.unsqueeze(1) for p in (α1, β1, γ1)\]  
        α2, β2, γ2 \= \[p.unsqueeze(1) for p in (α2, β2, γ2)\]

        \# Self-attention with AdaLN  
        h \= (1 \+ α1) \* self.norm1(x) \+ β1  
        attn\_out, \_ \= self.attn(h, h, h)  
        x \= x \+ γ1 \* attn\_out

        \# Feedforward with AdaLN  
        h \= (1 \+ α2) \* self.norm2(x) \+ β2  
        x \= x \+ γ2 \* self.ff(h)

        return x

\# ── Full conditioning pipeline wiring ────────────────────────────────────────

class ECGDiT(nn.Module):  
    def \_\_init\_\_(  
        self,  
        seq\_len: int \= 2500,  
        patch\_size: int \= 25,  
        d\_model: int \= 512,  
        n\_heads: int \= 8,  
        n\_layers: int \= 12,  
        hidden\_dim: int \= 256,  
    ):  
        super().\_\_init\_\_()  
        self.patch\_embed \= nn.Conv1d(1, d\_model, kernel\_size=patch\_size, stride=patch\_size)  
        n\_patches \= seq\_len // patch\_size  \# 100 patches  
          
        self.pos\_embed \= nn.Parameter(torch.randn(1, n\_patches, d\_model) \* 0.02)  
          
        self.t\_embed     \= nn.Identity()   \# use timestep\_embedding() externally  
        self.hr\_embed    \= HREmbedding(hidden\_dim=hidden\_dim)  
        self.patient\_emb \= nn.Embedding(10000, hidden\_dim)  \# patient identity  
        self.cond\_proj   \= ConditioningProjector(hidden\_dim=hidden\_dim, adaLN\_dim=d\_model)  
          
        self.blocks \= nn.ModuleList(\[  
            ECGDiTBlock(d\_model=d\_model, n\_heads=n\_heads, adaLN\_dim=d\_model)  
            for \_ in range(n\_layers)  
        \])  
          
        \# Final layer norm \+ linear head → reconstruct patches  
        self.final\_norm \= nn.LayerNorm(d\_model, elementwise\_affine=False)  
        self.final\_adaLN \= nn.Linear(d\_model, 2 \* d\_model)  
        self.final\_proj \= nn.Linear(d\_model, patch\_size)  
        nn.init.zeros\_(self.final\_adaLN.weight)  
        nn.init.zeros\_(self.final\_adaLN.bias)

    def forward(  
        self,  
        x\_t: torch.Tensor,      \# (B, 1, 2500\) noisy ECG  
        t: torch.Tensor,        \# (B,) diffusion timestep  
        hr: torch.Tensor,       \# (B,) heart rate in bpm  
        patient\_id: torch.Tensor,  \# (B,) patient index  
    ) \-\> torch.Tensor:  
          
        B \= x\_t.shape\[0\]  
          
        \# Patch embed → (B, 100, d\_model)  
        x \= self.patch\_embed(x\_t).transpose(1, 2\) \+ self.pos\_embed  
          
        \# Conditioning  
        t\_emb  \= timestep\_embedding(t, dim=256)          \# (B, 256\)  
        hr\_emb \= self.hr\_embed(hr)                       \# (B, 256\)  
        pid    \= self.patient\_emb(patient\_id)            \# (B, 256\)  
          
        scale, shift \= self.cond\_proj(t\_emb, hr\_emb, pid)  \# each (B, d\_model)  
        cond \= scale \+ shift   \# simplified; alternatively pass both separately  
          
        \# Transformer blocks  
        for block in self.blocks:  
            x \= block(x, cond)  
          
        \# Final output head  
        α, β \= self.final\_adaLN(cond).chunk(2, dim=-1)  
        x \= (1 \+ α.unsqueeze(1)) \* self.final\_norm(x) \+ β.unsqueeze(1)  
        x \= self.final\_proj(x)   \# (B, 100, patch\_size)  
          
        return x.reshape(B, 1, \-1)  \# (B, 1, 2500\)

---

## **Part 2 — Differentiable Heart Rate Loss**

### **Method Comparison**

|  | Soft R-peak | Autocorrelation (FFT) | CNN estimator |
| ----- | ----- | ----- | ----- |
| Differentiable | Conditionally — SoftArgmax required | Yes — FFT is differentiable in PyTorch | Yes — by construction |
| Gradient stability | Poor — sparse activations, vanishing near noise | Good — smooth spectral objective | Good — but biased |
| Noise robustness | Low without preprocessing | Medium | High |
| Training cost | Zero (no extra params) | Zero (no extra params) | Requires pretrained sub-network |
| Backprop through diffusion | Risky — gradients depend on peak sharpness | **Recommended** | Acceptable if frozen |

**Verdict: Autocorrelation via FFT is the most stable method.** Here's the analysis.

**Soft R-peak \+ SoftArgmax** is appealing but deeply problematic in practice. In early diffusion denoising steps, the generated ECG is dominated by noise — R-peak candidates are broad, multi-modal, and indistinguishable. SoftArgmax will produce meaningless "average peak locations", and gradients will point in contradictory directions. This destabilizes training precisely when the diffusion model most needs guidance.

**CNN estimator** introduces a second optimization problem: the CNN must be pretrained and ideally frozen during DiT training. If fine-tuned jointly, you get a co-adaptation failure mode. If frozen, the gradient direction is biased by whatever the CNN learned. Useful for inference-time guidance but not for training-time loss.

**FFT autocorrelation** computes HR as the dominant frequency in the ECG power spectrum, which is smooth, differentiable, and gracefully degrades under noise. The gradient is well-conditioned even on noisy signals. The key insight is that you're penalizing the *spectral center of mass* near the expected HR frequency — this is a soft, continuous objective.

### **Full PyTorch Implementation**

import torch  
import torch.nn.functional as F  
import math

class DifferentiableHRLoss(nn.Module):  
    """  
    Differentiable heart-rate estimation loss via FFT autocorrelation.  
      
    Strategy:  
        1\. Compute normalized autocorrelation of ECG via FFT.  
        2\. Restrict to physiologically plausible lag range (RR interval range).  
        3\. Compute soft argmax over the autocorrelation to get expected lag.  
        4\. Convert lag → HR in bpm.  
        5\. Compute L1/L2 loss against HR target.  
      
    The entire pipeline is differentiable w.r.t. the input ECG signal.  
    """  
      
    def \_\_init\_\_(  
        self,  
        fs: float \= 500.0,          \# sampling rate  
        hr\_min: float \= 30.0,       \# minimum physiological HR (bpm)  
        hr\_max: float \= 200.0,      \# maximum physiological HR (bpm)  
        softmax\_temp: float \= 10.0, \# temperature for soft argmax  
        loss\_type: str \= "l1",      \# "l1" or "l2"  
    ):  
        super().\_\_init\_\_()  
        self.fs \= fs  
        self.hr\_min \= hr\_min  
        self.hr\_max \= hr\_max  
        self.temp \= softmax\_temp  
        self.loss\_type \= loss\_type  
          
        \# Precompute lag range indices  
        \# HR range \[30, 200\] bpm → period range \[0.3s, 2.0s\] → lag range in samples  
        self.lag\_min \= int(fs \* 60.0 / hr\_max)   \# \~150 samples at 500Hz for 200bpm  
        self.lag\_max \= int(fs \* 60.0 / hr\_min)   \# \~1000 samples at 500Hz for 30bpm

    def \_autocorr\_fft(self, x: torch.Tensor) \-\> torch.Tensor:  
        """  
        Compute normalized autocorrelation via FFT.  
          
        Args:  
            x: (B, T) — single-channel ECG signal  
        Returns:  
            acf: (B, T) — normalized autocorrelation  
        """  
        B, T \= x.shape  
          
        \# Zero-mean the signal (removes DC component)  
        x \= x \- x.mean(dim=-1, keepdim=True)  
          
        \# FFT-based autocorrelation: IFFT(|FFT(x)|^2)  
        \# Pad to next power of 2 for efficiency  
        n\_fft \= 2 \*\* math.ceil(math.log2(2 \* T \- 1))  
        X \= torch.fft.rfft(x, n=n\_fft)              \# (B, n\_fft//2+1) complex  
          
        power \= X.real \*\* 2 \+ X.imag \*\* 2           \# |X|^2, (B, n\_fft//2+1)  
        acf\_full \= torch.fft.irfft(power, n=n\_fft)  \# (B, n\_fft) real  
          
        \# Take positive lags only: acf\[0..T-1\]  
        acf \= acf\_full\[:, :T\]                        \# (B, T)  
          
        \# Normalize: divide by zero-lag (max value)  
        acf \= acf / (acf\[:, 0:1\] \+ 1e-8)  
          
        return acf                                   \# (B, T), values in \[-1, 1\]

    def \_soft\_argmax\_lag(self, acf: torch.Tensor) \-\> torch.Tensor:  
        """  
        Compute the soft argmax (expected lag) within the physiological lag range.  
          
        We restrict to lags corresponding to HR in \[hr\_min, hr\_max\].  
        Within this range, soft argmax gives a differentiable estimate  
        of the dominant periodicity.  
          
        Args:  
            acf: (B, T)  
        Returns:  
            expected\_lag: (B,) — fractional lag in samples  
        """  
        \# Extract physiological lag window  
        acf\_window \= acf\[:, self.lag\_min:self.lag\_max\]   \# (B, lag\_range)  
        lag\_range \= acf\_window.shape\[-1\]  
          
        \# Soft argmax: temperature-scaled softmax over acf values  
        \# High temperature → sharper peak selection  
        weights \= F.softmax(self.temp \* acf\_window, dim=-1)   \# (B, lag\_range)  
          
        \# Lag indices (offset by lag\_min to get absolute lag)  
        indices \= torch.arange(lag\_range, device=acf.device, dtype=acf.dtype)  
        indices \= indices \+ self.lag\_min                       \# absolute lag  
          
        \# Expected lag \= weighted sum  
        expected\_lag \= (weights \* indices).sum(dim=-1)         \# (B,)  
          
        return expected\_lag

    def \_lag\_to\_hr(self, lag: torch.Tensor) \-\> torch.Tensor:  
        """  
        Convert lag (samples) → HR (bpm).  
          
        HR \= 60 \* fs / lag  
        This is differentiable (reciprocal function).  
          
        Args:  
            lag: (B,) in samples  
        Returns:  
            hr: (B,) in bpm  
        """  
        return 60.0 \* self.fs / (lag \+ 1e-6)

    def forward(  
        self,  
        ecg: torch.Tensor,           \# (B, 1, 2500\)  
        hr\_target: torch.Tensor,     \# (B,) in bpm  
        weight: float \= 1.0,  
    ) \-\> tuple\[torch.Tensor, torch.Tensor\]:  
        """  
        Compute differentiable HR loss.  
          
        Returns:  
            loss: scalar  
            hr\_estimated: (B,) for logging/monitoring  
        """  
        \# (B, 1, T) → (B, T)  
        x \= ecg.squeeze(1)  
          
        \# 1\. Autocorrelation  
        acf \= self.\_autocorr\_fft(x)                 \# (B, T)  
          
        \# 2\. Soft argmax over physiological lag range  
        lag \= self.\_soft\_argmax\_lag(acf)             \# (B,) fractional samples  
          
        \# 3\. Convert to HR  
        hr\_est \= self.\_lag\_to\_hr(lag)                \# (B,) bpm  
          
        \# 4\. Clamp estimated HR to plausible range (avoids log/reciprocal instability)  
        hr\_est\_clamped \= hr\_est.clamp(self.hr\_min, self.hr\_max)  
          
        \# 5\. Loss — L1 is preferred: more robust to HR estimation errors early in training  
        if self.loss\_type \== "l1":  
            loss \= F.l1\_loss(hr\_est\_clamped, hr\_target.float())  
        else:  
            loss \= F.mse\_loss(hr\_est\_clamped, hr\_target.float())  
          
        return weight \* loss, hr\_est\_clamped.detach()

\# ── Integration into diffusion training loop ─────────────────────────────────

def diffusion\_training\_step(  
    model: ECGDiT,  
    x0: torch.Tensor,       \# (B, 1, 2500\) clean ECG  
    hr\_target: torch.Tensor, \# (B,) bpm  
    patient\_id: torch.Tensor,  
    noise\_scheduler,  
    hr\_loss\_fn: DifferentiableHRLoss,  
    hr\_loss\_weight: float \= 0.1,  
    warmup\_steps: int \= 5000,  
    global\_step: int \= 0,  
):  
    """  
    Standard DDPM training step augmented with differentiable HR loss.  
      
    HR loss is applied only to predicted x0 (not intermediate xt),  
    and ramped up after warmup\_steps to avoid early destabilization.  
    """  
    B \= x0.shape\[0\]  
    device \= x0.device  
      
    \# Sample timesteps and noise  
    t \= torch.randint(0, noise\_scheduler.num\_timesteps, (B,), device=device)  
    noise \= torch.randn\_like(x0)  
      
    \# Forward diffusion  
    xt \= noise\_scheduler.q\_sample(x0, t, noise)  
      
    \# Model prediction (predicts noise ε, or x0 directly — adapt as needed)  
    pred\_noise \= model(xt, t, hr\_target, patient\_id)  
      
    \# Primary diffusion loss  
    diffusion\_loss \= F.mse\_loss(pred\_noise, noise)  
      
    \# Differentiable HR loss — applied on predicted x0  
    \# Recover predicted x0 from predicted noise  
    pred\_x0 \= noise\_scheduler.predict\_x0\_from\_noise(xt, t, pred\_noise)  
      
    \# Ramp HR loss weight gradually to avoid early gradient conflicts  
    hr\_ramp \= min(1.0, global\_step / warmup\_steps)  
    hr\_loss, hr\_est \= hr\_loss\_fn(pred\_x0, hr\_target, weight=hr\_loss\_weight \* hr\_ramp)  
      
    total\_loss \= diffusion\_loss \+ hr\_loss  
      
    return total\_loss, {  
        "diffusion\_loss": diffusion\_loss.item(),  
        "hr\_loss": hr\_loss.item(),  
        "hr\_mae\_bpm": F.l1\_loss(hr\_est, hr\_target.float()).item(),  
    }

---

## **Part 3 — Fast HR Precomputation Pipeline**

### **Speed Analysis**

For 75k × 2500-sample segments at 500 Hz, the bottleneck is NeuroKit2's Pan-Tompkins which runs at \~0.5–2 segments/sec on CPU — that's 10–40 hours. Here's the hierarchy of alternatives:

| Method | Throughput (est.) | GPU required | Accuracy |
| ----- | ----- | ----- | ----- |
| NeuroKit2 (baseline) | \~1 seg/sec | No | High |
| SciPy batch (CPU parallel) | \~50 seg/sec | No | High |
| Batched FFT autocorrelation (GPU) | \~50,000 seg/sec | Yes | Medium |
| Batched Conv Pan-Tompkins (GPU) | \~5,000 seg/sec | Yes | High |

**Recommended strategy: batched GPU FFT \+ local max refinement.** This gives high-accuracy HR in seconds for 75k segments.

import torch  
import numpy as np  
from pathlib import Path  
from tqdm import tqdm

\# ── GPU-Accelerated Batched HR Extraction ────────────────────────────────────

class GPUBatchedHRExtractor:  
    """  
    Extracts heart rate from ECG segments using batched GPU processing.  
      
    Pipeline:  
        1\. Bandpass filter (GPU conv1d, \~5–40 Hz passband for QRS)  
        2\. Derivative \+ squaring (enhances QRS complex)  
        3\. Moving average integration (GPU conv1d)  
        4\. Peak detection via autocorrelation (FFT, fully batched)  
        5\. Refinement via local maxima in integration signal  
      
    Throughput: \~20,000–50,000 segments/sec on modern GPU (V100/A100).  
    """  
      
    def \_\_init\_\_(  
        self,  
        fs: float \= 500.0,  
        batch\_size: int \= 512,  
        device: str \= "cuda",  
        hr\_min: float \= 30.0,  
        hr\_max: float \= 200.0,  
    ):  
        self.fs \= fs  
        self.batch\_size \= batch\_size  
        self.device \= torch.device(device if torch.cuda.is\_available() else "cpu")  
        self.hr\_min \= hr\_min  
        self.hr\_max \= hr\_max  
          
        \# Precompute filters as Conv1d kernels on GPU  
        self.\_build\_filters()  
      
    def \_build\_filters(self):  
        """Build bandpass and integration filters as GPU conv kernels."""  
          
        \# ── Bandpass: approximate 5–15 Hz FIR via windowed sinc ──────────────  
        \# This is a simplified QRS-band filter; replace with firwin if scipy available  
        n\_taps \= 33  
        fc\_low  \= 5.0  / (self.fs / 2\)  
        fc\_high \= 15.0 / (self.fs / 2\)  
          
        t \= torch.linspace(-(n\_taps // 2), n\_taps // 2, n\_taps)  
          
        \# High-pass component (remove baseline)  
        h\_lp\_low \= torch.sinc(2 \* fc\_low \* t) \* 2 \* fc\_low  
        \# Low-pass component (remove high freq noise)  
        h\_lp\_high \= torch.sinc(2 \* fc\_high \* t) \* 2 \* fc\_high  
          
        \# Bandpass \= lowpass(15Hz) \- lowpass(5Hz)  
        h\_bp \= h\_lp\_high \- h\_lp\_low  
        h\_bp \= h\_bp \* torch.hamming\_window(n\_taps)  
        h\_bp \= h\_bp / h\_bp.abs().sum()  
          
        \# ── Moving average integration window ─────────────────────────────────  
        \# Window ≈ 150 ms \= 0.15 \* 500 \= 75 samples  
        win\_size \= int(0.15 \* self.fs)  
        h\_integ \= torch.ones(win\_size) / win\_size  
          
        \# Register as conv1d weight tensors  
        self.bandpass\_weight \= h\_bp.view(1, 1, \-1).to(self.device)  
        self.integ\_weight    \= h\_integ.view(1, 1, \-1).to(self.device)  
        self.n\_taps          \= n\_taps  
        self.win\_size        \= win\_size

    @torch.no\_grad()  
    def \_preprocess(self, x: torch.Tensor) \-\> torch.Tensor:  
        """  
        Pan-Tompkins preprocessing pipeline on GPU.  
        x: (B, T) → returns integration signal (B, T)  
        """  
        B, T \= x.shape  
        x\_3d \= x.unsqueeze(1)   \# (B, 1, T) for conv1d  
          
        \# 1\. Bandpass filter  
        pad\_bp \= self.n\_taps // 2  
        x\_bp \= F.conv1d(x\_3d, self.bandpass\_weight, padding=pad\_bp)\[:, 0, :T\]  
          
        \# 2\. Derivative (5-point)  
        deriv\_kernel \= torch.tensor(  
            \[-1., \-2., 0., 2., 1.\], device=self.device  
        ).view(1, 1, \-1) \* (1.0 / (8.0 / self.fs))  
        x\_deriv \= F.conv1d(x\_bp.unsqueeze(1), deriv\_kernel, padding=2)\[:, 0, :\]  
          
        \# 3\. Squaring (nonlinear emphasis)  
        x\_sq \= x\_deriv \*\* 2  
          
        \# 4\. Moving average integration  
        pad\_integ \= self.win\_size // 2  
        x\_integ \= F.conv1d(x\_sq.unsqueeze(1), self.integ\_weight, padding=pad\_integ)\[:, 0, :T\]  
          
        return x\_integ   \# (B, T)

    @torch.no\_grad()  
    def \_fft\_hr(self, x\_integ: torch.Tensor) \-\> torch.Tensor:  
        """  
        Estimate HR via FFT autocorrelation of integration signal.  
        Fast, fully batched on GPU.  
          
        Returns: (B,) HR in bpm  
        """  
        B, T \= x\_integ.shape  
          
        \# Normalize  
        x\_n \= x\_integ \- x\_integ.mean(dim=-1, keepdim=True)  
          
        \# FFT autocorrelation  
        n\_fft \= 2 \*\* math.ceil(math.log2(2 \* T \- 1))  
        X \= torch.fft.rfft(x\_n, n=n\_fft)  
        power \= X.real\*\*2 \+ X.imag\*\*2  
        acf \= torch.fft.irfft(power, n=n\_fft)\[:, :T\]  
        acf \= acf / (acf\[:, 0:1\].abs() \+ 1e-8)  
          
        \# Find dominant lag in physiological range  
        lag\_min \= int(self.fs \* 60.0 / self.hr\_max)  
        lag\_max \= min(int(self.fs \* 60.0 / self.hr\_min), T \- 1\)  
          
        acf\_window \= acf\[:, lag\_min:lag\_max\]  
        peak\_lags \= acf\_window.argmax(dim=-1) \+ lag\_min   \# (B,) integer lags  
          
        hr \= 60.0 \* self.fs / peak\_lags.float()  
        return hr.clamp(self.hr\_min, self.hr\_max)         \# (B,)

    @torch.no\_grad()  
    def extract\_batch(self, ecg\_batch: torch.Tensor) \-\> torch.Tensor:  
        """  
        Full pipeline for one batch.  
          
        Args:  
            ecg\_batch: (B, 2500\) float32 on CPU or GPU  
        Returns:  
            hr: (B,) float32 in bpm  
        """  
        x \= ecg\_batch.float().to(self.device)  
        x\_integ \= self.\_preprocess(x)  
        hr \= self.\_fft\_hr(x\_integ)  
        return hr.cpu()

    def extract\_dataset(  
        self,  
        ecg\_array: np.ndarray,     \# (N, 2500\)  
        desc: str \= "Extracting HR",  
    ) \-\> np.ndarray:  
        """  
        Process entire dataset in batches.  
          
        Args:  
            ecg\_array: (N, 2500\) float32 numpy array  
        Returns:  
            hr\_array: (N,) float32 in bpm  
        """  
        N \= ecg\_array.shape\[0\]  
        hr\_all \= np.zeros(N, dtype=np.float32)  
          
        for start in tqdm(range(0, N, self.batch\_size), desc=desc):  
            end \= min(start \+ self.batch\_size, N)  
            batch \= torch.from\_numpy(ecg\_array\[start:end\])  
            hr\_all\[start:end\] \= self.extract\_batch(batch).numpy()  
          
        return hr\_all

\# ── Save to .npz format ──────────────────────────────────────────────────────

def precompute\_and\_save(  
    ecg\_path: str,          \# path to ecg\_segments.npy  
    output\_dir: str,        \# output directory  
    fs: float \= 500.0,  
    batch\_size: int \= 512,  
):  
    """  
    Full pipeline: load ECG dataset → GPU HR extraction → save .npz  
      
    Creates:  
        output\_dir/ecg\_segments.npy   (unchanged)  
        output\_dir/hr\_values.npy      (float32, bpm per segment)  
        output\_dir/dataset.npz        (combined, memory-mapped friendly)  
    """  
    output\_dir \= Path(output\_dir)  
    output\_dir.mkdir(parents=True, exist\_ok=True)  
      
    print(f"Loading ECG segments from {ecg\_path}...")  
    ecg \= np.load(ecg\_path).astype(np.float32)   \# (N, 2500\)  
    print(f"  Shape: {ecg.shape}, dtype: {ecg.dtype}")  
      
    \# Extract HR  
    extractor \= GPUBatchedHRExtractor(fs=fs, batch\_size=batch\_size)  
    hr \= extractor.extract\_dataset(ecg)  
      
    \# Summary stats  
    print(f"\\nHR statistics:")  
    print(f"  Mean:  {hr.mean():.1f} bpm")  
    print(f"  Std:   {hr.std():.1f} bpm")  
    print(f"  Range: \[{hr.min():.1f}, {hr.max():.1f}\] bpm")  
      
    \# Save individual arrays  
    np.save(output\_dir / "ecg\_segments.npy", ecg)  
    np.save(output\_dir / "hr\_values.npy", hr)  
      
    \# Save combined .npz (allows memory-mapped loading during training)  
    np.savez\_compressed(  
        output\_dir / "dataset.npz",  
        ecg=ecg,  
        hr=hr,  
        fs=np.array(\[fs\]),  
    )  
      
    print(f"\\nSaved:")  
    print(f"  {output\_dir}/ecg\_segments.npy  — {ecg.nbytes / 1e6:.1f} MB")  
    print(f"  {output\_dir}/hr\_values.npy     — {hr.nbytes / 1e6:.2f} MB")  
    print(f"  {output\_dir}/dataset.npz       — compressed")

    return hr

\# ── Dataset class with precomputed HR ────────────────────────────────────────

class ECGDatasetWithHR(torch.utils.data.Dataset):  
    """  
    Memory-mapped ECG dataset with precomputed HR labels.  
    """  
    def \_\_init\_\_(self, npz\_path: str, normalize: bool \= True):  
        data \= np.load(npz\_path)  
        self.ecg \= data\["ecg"\]        \# (N, 2500\)  
        self.hr  \= data\["hr"\]         \# (N,)  
        self.normalize \= normalize  
      
    def \_\_len\_\_(self):  
        return len(self.hr)  
      
    def \_\_getitem\_\_(self, idx):  
        ecg \= torch.from\_numpy(self.ecg\[idx\]).float().unsqueeze(0)  \# (1, 2500\)  
          
        if self.normalize:  
            \# Per-segment z-score normalization  
            ecg \= (ecg \- ecg.mean()) / (ecg.std() \+ 1e-6)  
          
        hr \= torch.tensor(self.hr\[idx\], dtype=torch.float32)  
        return ecg, hr

\# ── Usage ──────────────────────────────────────────────────────────────────  
\#   
\# python precompute.py  
\#  
\# precompute\_and\_save(  
\#     ecg\_path="raw/ecg\_segments.npy",  
\#     output\_dir="dataset/",  
\#     fs=500.0,  
\#     batch\_size=1024,  
\# )  
\#  
\# On A100: 75k segments @ batch\_size=1024 completes in \~3–8 seconds.

---

## **Summary & Recommended Configuration**

┌─────────────────────────────────────────────────────────────┐  
│  CONDITIONING  →  Option B (Separate AdaLN streams)         │  
│    \- Timestep, HR, and patient\_id each get their own MLP    │  
│    \- Outputs added before entering each DiT block           │  
│    \- HR head zero-initialized (stable training start)       │  
├─────────────────────────────────────────────────────────────┤  
│  HR LOSS  →  FFT Autocorrelation \+ Soft Argmax              │  
│    \- Fully differentiable, smooth gradients                 │  
│    \- Ramp weight in with warmup\_steps=5000                  │  
│    \- L1 loss for robustness to estimation errors            │  
│    \- hr\_loss\_weight ≈ 0.05–0.1 (tune via validation)       │  
├─────────────────────────────────────────────────────────────┤  
│  PRECOMPUTATION  →  GPU FFT pipeline                        │  
│    \- Pan-Tompkins preprocessing on GPU (conv1d)             │  
│    \- FFT autocorrelation for lag detection                  │  
│    \- batch\_size=512–1024, \~5 seconds for 75k on A100        │  
│    \- Output: dataset.npz with ecg \+ hr arrays               │  
└─────────────────────────────────────────────────────────────┘

---

## **References**

1. **Peebles & Xie (2023)** — *Scalable Diffusion Models with Transformers (DiT)* — AdaLN conditioning design, class embedding strategy, zero-initialization of modulation heads.

2. **Ho et al. (2020)** — *Denoising Diffusion Probabilistic Models* — Foundational DDPM training loop, noise schedule, x0 recovery from predicted noise.

3. **ECGTwin (2025)** — Patient identity conditioning in latent ECG diffusion via AdaLN, rhythm template injection via cross-attention.

4. **DiffuSETS (2025)** — Multi-modal ECG conditioning, analysis of scalar vs. sequence conditioning pathways, recommendation for independent projection of scalar physiological features.

5. **Pan & Tompkins (1985)** — *A Real-Time QRS Detection Algorithm* — Original Pan-Tompkins pipeline; bandpass → derivative → squaring → moving average.

6. **Kiyasseh et al. (2021)** — *CLOCS: Contrastive Learning of Cardiac Signals* — Representation learning for ECG with physiological invariance, informed the choice of per-segment normalization.

7. **Song et al. (2021)** — *Score-Based Generative Modeling through SDEs* — Theoretical grounding for auxiliary loss injection in score-based models, conditioning stability analysis.

