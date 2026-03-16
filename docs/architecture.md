# CardioEquation — System Architecture
**Patient-Specific ECG Generation using Diffusion Transformers**

> A comprehensive technical overview for academic review.

---

## 1. What Is CardioEquation?

CardioEquation is a **deep learning system that generates personalized, realistic ECG signals** for a specific patient. Given a short segment of a real patient's ECG as input, the system generates a new synthetic ECG that:

- Looks like **that patient's heart** (preserving morphology — P-wave shape, QRS complex, T-wave)
- Has **realistic rhythm** (heart rate, HRV)
- Is **clinically useful** for training diagnostic AI models, data augmentation, and privacy-preserving medical data sharing

---

## 2. High-Level System Overview

```mermaid
flowchart LR
    A["🏥 Real Patient ECG\n(Hospital PDF / Direct Recording)"] --> B["📷 ECG Digitizer\n(PDF → 2500 samples)"]
    B --> C["Stage 0\nContrastive Pre-training\n(Identity Encoder)"]
    C --> D["Stage 1\nDiT Diffusion Training\n(ECG Generator)"]
    D --> E["⚡ Inference\nDDIM Sampling\n(50 steps, eta=0.75)"]
    E --> F["🫀 Synthetic ECG\n(Personalized)\n2500 samples @ 500Hz"]
    
    style A fill:#e8f4fd,stroke:#2196F3
    style F fill:#e8f5e9,stroke:#4CAF50
    style C fill:#fff3e0,stroke:#FF9800
    style D fill:#fce4ec,stroke:#E91E63
```

---

## 3. The Two-Stage Training Pipeline

### Overview

```mermaid
flowchart TD
    subgraph DS["📊 Datasets (75,540 ECG segments)"]
        D1["MIT-BIH\n8,592 segments\n360Hz → 500Hz"]
        D2["PTB-XL\n56,300 segments\n100Hz → 500Hz"]
        D3["Chapman-Shaoxing\n10,648 segments\n500Hz"]
    end
    
    DS --> S0

    subgraph S0["Stage 0 — Identity Learning (200 epochs)"]
        direction LR
        E1["ECG Segment A\n(same patient)"] --> FE["ContrastiveFeatureExtractor\n9.4M params\nCNN + Projection Head"]
        E2["ECG Segment B\n(same patient)"] --> FE
        FE --> CL["InfoNCE\nContrastive Loss\nPulls same-patient apart,\npushes different-patient together"]
        CL --> FE
    end

    S0 -->|"Freeze encoder\nSave weights"| S1

    subgraph S1["Stage 1 — DiT Diffusion Training (474 epochs)"]
        direction LR
        ECG["Clean ECG x₀"] --> FWD["Forward Diffusion\nAdd Gaussian noise\nCosine schedule\n1000 timesteps"]
        FWD --> XN["Noisy ECG x_t"]
        XN --> DIT["DiT-ECG-B\n265M params"]
        FROZEN["❄️ Frozen\nIdentity Encoder"] --> IDE["Identity\nEmbedding\n512-dim"]
        IDE --> DIT
        T["Timestep t"] --> TEM["Sinusoidal\nEmbedding"] --> DIT
        DIT --> NP["Predicted Noise ε̂"]
        NP --> LOSS["Multi-Component Loss\n↓ see Section 5"]
        LOSS --> DIT
    end

    style S0 fill:#fff8e1
    style S1 fill:#fce4ec
    style DS fill:#e3f2fd
```

---

## 4. Core Model: DiT-ECG-B Architecture

The generator is a **Diffusion Transformer** adapted from DiT (Peebles & Xie, 2022) for 1D ECG signals.

### 4.1 Input Representation

```mermaid
flowchart LR
    RAW["Noisy ECG x_t\n(B, 1, 2500)\n2500 samples = 5s @ 500Hz"] --> PATCH["PatchEmbed1D\nConv1D, kernel=10, stride=10\n2500 → 250 patches"]
    PATCH --> TOKENS["Patch Tokens\n(B, 250, 768)\n250 tokens × 768-dim"]
    TOKENS --> PE["+ Positional Encoding\n(learnable)"]
    PE --> BLOCKS["24 × DiT Block"]
```

**Why patches?** Processing 2500 raw samples through self-attention would require 2500² = 6.25M attention pairs per layer. With patch size 10, this drops to 250² = 62,500 — **100× reduction**.

### 4.2 Conditioning Mechanism (AdaLN-Zero)

```mermaid
flowchart LR
    T["Timestep t\n(scalar 0→1)"] --> SE["Sinusoidal\nEmbedding\n768-dim"]
    ID["Patient Identity\n(512-dim from\nfrozen encoder)"] --> LP["Linear\nProjection\n768-dim"]
    SE --> ADD((+))
    LP --> ADD
    ADD --> MLP["MLP\nSiLU activation"]
    MLP --> COND["Conditioning Vector c\n(768-dim)"]
    COND --> ADA["AdaLN\nScale γ and Shift β\n per block"]
```

**AdaLN-Zero**: Instead of fixed LayerNorm, the model learns to modulate each layer differently based on the conditioning vector. Starts at zero (no modulation) and learns the right amount during training.

### 4.3 Single DiT Block

```mermaid
flowchart TD
    IN["Input: x\n(B, 250, 768)"] --> LN1["LayerNorm\n(AdaLN-Zero)\nmodulated by c"]
    LN1 --> ATTN["Multi-Head Self-Attention\n12 heads, dim=64 each\nPatches attend to each other"]
    ATTN --> G1["× gate α₁\n(learned from c)"]
    G1 --> ADD1((+))
    IN --> ADD1

    ADD1 --> LN2["LayerNorm\n(AdaLN-Zero)"]
    LN2 --> FFN["Feed-Forward Network\n768 → 3072 → 768\nGELU activation"]
    FFN --> G2["× gate α₂\n(learned from c)"]
    G2 --> ADD2((+))
    ADD1 --> ADD2
    ADD2 --> OUT["Output: x'\n(B, 250, 768)"]
```

### 4.4 Output Head

```mermaid
flowchart LR
    TOKENS["Final Patch Tokens\n(B, 250, 768)"] --> LN["LayerNorm"] --> UNPATCH["UnPatch1D\nLinear: 768 → 10\nReshape to signal"]
    UNPATCH --> PRED["Predicted Noise ε̂\n(B, 1, 2500)"]
```

**Full DiT-ECG-B specs:**

| Parameter | Value |
|-----------|-------|
| Transformer blocks | 24 |
| Model dimension `d_model` | 768 |
| Attention heads | 12 |
| FFN expansion | 4× (3072) |
| Patch size | 10 (250 tokens) |
| Total parameters | **265M** |
| Precision | BF16 mixed |

---

## 5. Loss Function — Multi-Component Training Signal

```mermaid
flowchart TD
    PRED["Predicted Noise ε̂\n+ Estimated Clean Signal x̂₀"] --> L1["Noise MSE\nL₁ = MSE(ε̂, ε)\nweight: 1.0 always"]
    PRED --> L2["Signal MSE\nL₂ = MSE(x̂₀, x₀)\nweight: 1.0 × SNR"]
    PRED --> L3["Identity Loss\nL₃ = 1 - cosine_sim(\nencoder(x̂₀), encoder(x₀))\nweight: 0.5 × SNR"]
    PRED --> L4["Spectral Loss\nL₄ = MSE(FFT(x̂₀), FFT(x₀))\nweight: 0.0001 × SNR"]
    PRED --> L5["Correlation Loss\nL₅ = 1 - Pearson(x̂₀, x₀)\nweight: 0.2 × SNR"]
    PRED --> L6["Morphology Gradient Loss\nL₆ = MSE(∇x̂₀, ∇x₀) + 0.1×MSE(∇²x̂₀,∇²x₀)\nweight: 0.3 × SNR"]

    L1 & L2 & L3 & L4 & L5 & L6 --> TOTAL["Total Loss L\n= L₁ + SNR_weight × (L₂+L₃+L₄+L₅+L₆)"]
```

**SNR Weighting**: Auxiliary losses (L₂–L₆) are weighted by `α̅_t` (signal-to-noise ratio at timestep `t`). At high noise levels (large `t`), `α̅_t ≈ 0` so auxiliary losses are suppressed — only the denoising loss L₁ matters. At low noise (small `t`), `α̅_t ≈ 1` so all losses contribute equally. This ensures losses are applied only when the reconstructed signal `x̂₀` is reliable.

> **Run 4 Bug Found**: Spectral loss was producing values ~125 while all others were ~0.1. The weight was changed from 0.1 → 0.0001 to fix this.

---

## 6. Forward & Reverse Diffusion

### Forward Process (Training)

```mermaid
flowchart LR
    X0["Clean ECG x₀"] -->|"t=0\nNo noise"| X200["x₂₀₀\nSlight noise"]
    X200 --> X500["x₅₀₀\nHalf noise"]
    X500 --> X900["x₉₀₀\nMostly noise"]
    X900 -->|"t=1000\nPure noise"| XN["xₙ ~ N(0,I)"]

    style X0 fill:#e8f5e9
    style XN fill:#ffebee
```

**Formula**: `x_t = √α̅_t · x₀ + √(1-α̅_t) · ε` where `ε ~ N(0,I)`

Uses **cosine noise schedule** (Nichol & Dhariwal, 2021) — smoother than linear, better for ECG morphology at fine scales.

### Reverse Process (Inference — DDIM with eta)

```mermaid
flowchart RL
    XN["Pure Noise xₙ\n~ N(0,I)"] --> S1["Step 1\nDiT predicts ε̂\nDDIM update + σₜ·z"]
    S1 --> S2["Step 2\n..."] --> DOTS["..."] --> S50["Step 50\nFinal denoising"]
    S50 --> X0["Generated ECG\nx̂₀ (personalized)"]

    style XN fill:#ffebee
    style X0 fill:#e8f5e9
```

**DDIM with η=0.75**: Standard DDIM uses η=0 (deterministic). We use η=0.75 to re-inject stochastic noise at each step:

`σₜ = η × √((1-α̅ₜ₋₁)/(1-α̅ₜ) × (1-α̅ₜ/α̅ₜ₋₁))`

This increases **HR diversity** by preventing the model from always taking the same deterministic path.

---

## 7. Identity Encoder: ContrastiveFeatureExtractor

```mermaid
flowchart LR
    ECG["ECG Segment\n(B, 1, 2500)"] --> CNN["3-layer CNN\nChannels: 1→64→128→256\nKernel: 15,11,7\nBatchNorm + ReLU"]
    CNN --> GAP["Global Average Pool\n256-dim"]
    GAP --> PROJ["Projection Head\nMLP: 256→512\nL2 Normalize"]
    PROJ --> EMB["Identity Embedding\n512-dim unit sphere"]
```

**Training**: InfoNCE contrastive loss. Two random crops from the **same patient** are pulled together in embedding space; crops from **different patients** are pushed apart.

**At inference**: Encoder is **frozen** — its weights do not change during DiT training. This ensures the identity space is stable and meaningful.

---

## 8. Inference Pipeline

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant PDF as 📄 ECG PDF
    participant DIG as ECG Digitizer
    participant FE as Identity Encoder (frozen)
    participant DIT as DiT-ECG-B
    participant OUT as Output Signal

    U->>PDF: Provide real patient ECG PDF
    PDF->>DIG: Convert PDF to 1D signal
    DIG->>FE: context signal (2500 samples)
    FE->>DIT: identity embedding z (512-dim)
    Note over DIT: Start from pure Gaussian noise
    loop 50 DDIM steps (η=0.75)
        DIT->>DIT: Predict noise ε̂<br/>Compute x₀ estimate<br/>Apply stochastic DDIM step
    end
    DIT->>OUT: Generated ECG (2500 samples)
    OUT->>U: Personalized synthetic ECG
```

---

## 9. Data Flow & Preprocessing

```mermaid
flowchart TD
    subgraph RAW["Raw Data Sources"]
        MB["MIT-BIH\nPhysioNet\n48 records, 360Hz\n2-lead, 30 min each"]
        PX["PTB-XL\nPhysioNet\n21,837 records, 500Hz\n12-lead, 10s each"]
        CH["Chapman-Shaoxing\nPhysioNet\n10,646 records, 500Hz\n12-lead, 10s each"]
    end

    RAW --> PROC["🔧 Processing Pipeline"]
    
    subgraph PROC["Processing"]
        direction LR
        P1["Load Lead I only\n(column 0)"]
        P2["Resample → 500Hz\nall sources unified"]
        P3["Take first 5s\n→ 2500 samples"]
        P4["Normalize: z-score\n(μ=0, σ=1)"]
    end

    PROC --> NPZ["📦 .npz Archives\nmitbih_forecasting.npz\nptbxl_processed.npz\nchapman_processed.npz"]
    NPZ --> COMBINED["Combined Dataset\n75,540 segments total\nTrain: 67,986 (90%)\nVal: 7,554 (10%)"]
```

---

## 10. Training Configuration

| Setting | Value | Why |
|---------|-------|-----|
| Batch size | 16 (micro) | GPU memory constraint (A100 80GB shared) |
| Gradient accumulation | 16 steps | Effective batch = 256 |
| Learning rate | 1e-4 | AdamW, cosine decay |
| Warmup steps | 5,000 | Stable LR ramp-up |
| Max epochs | 500 | Early stopping at patience=30 |
| EMA decay | 0.9999 | Smoothed weights for inference |
| CFG dropout | 10% | Classifier-Free Guidance training |
| Diffusion timesteps | 1,000 | Standard DDPM/DDIM |
| DDIM steps (inference) | 50 | Speed/quality tradeoff |
| DDIM eta | 0.75 | HR diversity (new in Run 4) |
| Precision | BF16 | ~2× memory saving, minimal quality loss |

---

## 11. Evaluation Metrics

| Metric | What It Measures | Run 1 | Run 2 | Run 4 |
|--------|-----------------|-------|-------|-------|
| **FFD** ↓ | Overall signal quality (like FID for images) | 1032 | 42.0 | **21.7** |
| **MMD** ↓ | Distribution mismatch (real vs generated) | 0.661 | 0.383 | **0.432** |
| **HR MAE** ↓ | Mean absolute heart rate error (bpm) | 17.2 | 9.3 | 31.4* |
| **HR Std** | Diversity of generated heart rates | — | 21.1 | 20.5 |
| **Real HR Std** | Target diversity | — | — | **47.4** |
| **ReID Top-1** ↑ | Patient identity preserved (correct ID) | 11.8% | 5.9% | **11.8%** |
| **ReID Top-5** ↑ | Patient in top-5 predictions | 35.3% | 23.5% | **35.3%** |

*HR MAE worsened due to spectral loss imbalance bug (now fixed for Run 5)

---

## 12. Full System File Map

```
CardioEquation/
├── src/
│   ├── models/
│   │   ├── dit_ecg.py              ← Main generator (265M params)
│   │   └── feature_extractor_pt.py ← Identity encoder (9.4M params)
│   ├── training/
│   │   ├── train_contrastive.py    ← Stage 0: Identity pre-training
│   │   ├── train_dit.py            ← Stage 1: Diffusion training
│   │   ├── losses_v2.py            ← 6-component loss functions
│   │   ├── noise_scheduler.py      ← Cosine schedule + DDIM sampler
│   │   └── ema.py                  ← Exponential Moving Average
│   ├── inference/
│   │   └── pipeline_v2.py          ← End-to-end generation pipeline
│   └── evaluation/
│       └── clinical_validation.py  ← Hospital ECG evaluation
├── data/                           ← Downloaded datasets (.npz)
├── checkpoints/                    ← Saved model weights
├── docs/                           ← This document + future techniques
└── run_training.sh                 ← One-command full pipeline
```

---

## 13. Key Design Decisions & Rationale

| Decision | Alternative Considered | Why We Chose This |
|----------|----------------------|------------------|
| **DiT backbone** | U-Net (standard DDPM) | DiT scales better: bigger model = better quality. State-of-the-art for image generation (DALL-E 3, Stable Diffusion 3) |
| **Contrastive identity encoder** | Siamese network, AE | InfoNCE produces better-separated patient embeddings. Used in ECGTwin (2025) |
| **Cosine noise schedule** | Linear | Cosine is smoother at low noise levels — better for ECG fine structure (P-waves) |
| **DDIM sampling** | Full DDPM | 50 DDIM steps ≈ quality of 1000 DDPM steps. 20× faster inference |
| **η=0.75 stochastic DDIM** | η=0 deterministic | Deterministic DDIM collapses HR diversity. Stochasticity forces varied outputs |
| **SNR-weighted auxiliary losses** | Fixed weights | Prevents auxiliary instability at high noise timesteps. Proven in iDDPM (Nichol 2021) |
| **Frozen encoder during DiT training** | Joint fine-tuning | Fine-tuning corrupts carefully learned identity representations |

---

## 14. References

- **DiT**: Peebles & Xie (2022). *Scalable Diffusion Models with Transformers.* ICCV 2023.
- **DDIM**: Song et al. (2020). *Denoising Diffusion Implicit Models.* ICLR 2021.
- **iDDPM**: Nichol & Dhariwal (2021). *Improved DDPM.* ICML 2021.
- **ECGTwin**: (2025). *Personalized ECG generation using controllable diffusion.* arXiv:2508.
- **DiffuSETS**: Lai et al. (2025). *12-lead ECG generation conditioned on clinical text.* Patterns.
- **InfoNCE**: Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding.*
- **Cosine schedule**: Nichol & Dhariwal (2021). *Improved DDPM.*
- **CFG**: Ho & Salimans (2022). *Classifier-Free Diffusion Guidance.*
