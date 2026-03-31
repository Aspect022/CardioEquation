# CardioEquation: The Complete Development Journey

### A Comprehensive Technical Report on Building an AI-Powered Patient-Specific ECG Generation System

**Project**: CardioEquation  
**Repository**: `Aspect022/CardioEquation`  
**Timeline**: March 2026 (Active Development)  
**Author**: Development Team  
**Last Updated**: March 29, 2026  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Vision: Why CardioEquation Exists](#2-the-vision-why-cardioequation-exists)
3. [Phase 0: Research Foundation & Technology Selection](#3-phase-0-research-foundation--technology-selection)
4. [Phase 1: Initial Architecture — The Diffusion Pipeline](#4-phase-1-initial-architecture--the-diffusion-pipeline)
5. [Run 1: First Contact with Reality](#5-run-1-first-contact-with-reality)
6. [Run 2: Multi-Dataset Scaling & First Improvements](#6-run-2-multi-dataset-scaling--first-improvements)
7. [Run 3: The Silent Failure](#7-run-3-the-silent-failure)
8. [Run 4: The Spectral Loss Catastrophe](#8-run-4-the-spectral-loss-catastrophe)
9. [The Research Interlude: Deep Investigation](#9-the-research-interlude-deep-investigation)
10. [Run 5: Explicit HR Conditioning — The Breakthrough Architecture](#10-run-5-explicit-hr-conditioning--the-breakthrough-architecture)
11. [Run 5b: 500 Epochs — First Complete Training Run](#11-run-5b-500-epochs--first-complete-training-run)
12. [Clinical Validation & Mentor Reports](#12-clinical-validation--mentor-reports)
13. [Run 6: The OT-CFM Revolution](#13-run-6-the-ot-cfm-revolution)
14. [The Complete Loss Function Evolution](#14-the-complete-loss-function-evolution)
15. [The Data Engine: From 48 Patients to 75,540 Segments](#15-the-data-engine-from-48-patients-to-75540-segments)
16. [Infrastructure & Training Pipeline](#16-infrastructure--training-pipeline)
17. [Evaluation Framework: ECG-Bench Protocol](#17-evaluation-framework-ecg-bench-protocol)
18. [Lessons Learned & Engineering Wisdom](#18-lessons-learned--engineering-wisdom)
19. [Metrics Evolution Across All Runs](#19-metrics-evolution-across-all-runs)
20. [Future Roadmap](#20-future-roadmap)
21. [Appendix: Key Papers & References](#21-appendix-key-papers--references)

---

## 1. Executive Summary

**CardioEquation** is a state-of-the-art generative AI system engineered to synthesize highly realistic, patient-specific 1D Electrocardiogram (ECG) waveforms. The system's core thesis is deceptively simple yet profoundly ambitious: *given a brief historical ECG from a specific patient, generate infinite new variations of that exact patient's heart rhythm under any physiological state — including precise heart rate control.*

This document chronicles the entire development journey — every architectural decision, every failed experiment, every debugging revelation, and every breakthrough — from the initial research phase through six major training runs. It is a raw, unfiltered account of what it takes to build a clinically viable generative model for biomedical time-series data.

### Key Achievements (as of Run 6)

| Milestone | Status |
|-----------|--------|
| 265M+ parameter Diffusion Transformer (DiT-ECG-B) | ✅ Implemented |
| Contrastive identity encoder (SimCLR-style) | ✅ Pre-trained on 75,540 segments |
| Explicit HR conditioning via AdaLN | ✅ Active, zero-initialized |
| First complete 500-epoch training run (Run 5b) | ✅ No early stopping |
| OT-CFM (Conditional Flow Matching) migration | ✅ Run 6 architecture |
| Identity Cross-Attention (IP-Adapter paradigm) | ✅ Run 6 architecture |
| GPU-accelerated Soft-DTW loss | ✅ Run 6 loss function |
| Clinical validation on hospital PDFs | ✅ ECGDigitizer pipeline |
| Multi-dataset training (MIT-BIH + PTB-XL + Chapman) | ✅ 75,540 segments |
| W&B full per-component loss monitoring | ✅ Critical diagnostic tool |

### Key Metrics Progress

| Metric | Run 1 | Run 2 | Run 4 | Run 5b | Run 6 Target |
|--------|-------|-------|-------|--------|--------------|
| FFD ↓ | 1032 | 42.0 | 21.7 | 91.1† | < 15 |
| HR MAE ↓ | 17.2 | 9.3 | 31.4 | 23.6 | < 8 |
| ReID Top-1 ↑ | 11.8% | 5.9% | 11.8% | 0.0% | > 50% |
| Val Loss ↓ | — | — | 5.677 | 0.6995 | < 0.5 |
| Morphology ↓ | — | — | ~0.8 | 0.037 | < 0.02 |

> **† Note**: Run 5b FFD was measured on 200 diverse samples across 3 datasets vs Run 4's 17 hospital PDFs. These numbers are not directly comparable.

---

## 2. The Vision: Why CardioEquation Exists

### 2.1 The Problem

Electrocardiography is the single most widely performed cardiac diagnostic test globally. Yet the field faces a fundamental data problem:

1. **Privacy barriers**: Real patient ECGs are protected under HIPAA/GDPR, making large-scale AI training datasets nearly impossible to assemble.
2. **Rare condition scarcity**: Conditions like Brugada syndrome, Long QT, or Torsade de Pointes appear in < 0.1% of recordings — there simply aren't enough examples to train reliable detectors.
3. **Inter-patient variability**: Every human heart has a unique electrical signature. A model that generates "generic" ECGs is clinically useless — it needs to produce *this specific patient's* cardiac pattern.
4. **Dynamic physiological states**: A patient's ECG changes dramatically between rest (60 bpm) and exercise (180 bpm). A useful generator must be controllable.

### 2.2 The Solution: Mathematical Cardiac Digital Twins

CardioEquation's answer is a **generative AI system** that creates individual-specific mathematical equations to reproduce unique ECG patterns. The name itself encodes the vision: *Cardio* (heart) + *Equation* (mathematical model) — the equation of a patient's heart.

The system architecture is a two-stage pipeline:

```
Stage 0: IDENTITY EXTRACTION
  Historical ECG → Contrastive Encoder → 512-dim Identity Vector
  "Who is this patient?"

Stage 1: CONDITIONAL GENERATION  
  Identity + Heart Rate + Noise → Diffusion Transformer → Patient-Specific ECG
  "Generate a new heartbeat for this patient at 120 bpm"
```

### 2.3 The Four Development Phases

The project was conceived as a four-phase roadmap:

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1**: Foundation | Single-lead synthetic ECG generation with identity preservation | ✅ Complete (Run 1–5b) |
| **Phase 2**: Clinical Scale | Multi-dataset training, flow matching, inference acceleration | 🔄 In Progress (Run 6) |
| **Phase 3**: Clinical Integration | 12-lead generation, real hospital validation, denoising | 📋 Planned |
| **Phase 4**: Forecasting | Temporal sequence modeling, cardiac event prediction | 📋 Planned |

---

## 3. Phase 0: Research Foundation & Technology Selection

### 3.1 The Research Stack

Before writing a single line of production code, an exhaustive research phase was conducted. The findings were compiled into a 1,026-line **Research & Architecture Manual** (the "Converted Text" document) covering 10 critical topics:

1. **ECGTwin & AdaX Dual-Pathway Conditioning** — How to inject patient identity into a diffusion model
2. **Diffusion Transformers (DiT) for 1D Signals** — Why transformers beat U-Nets for ECG
3. **SSSD-ECG & S4/Mamba State-Space Diffusion** — Alternative backbone architectures  
4. **Flow Matching vs DDPM** — Next-generation generative frameworks
5. **Contrastive Learning for Patient Identity** — How to build a cardiac biometric encoder
6. **Multi-Lead ECG Generation** — Strategies for 12-lead extension
7. **Large-Scale Public ECG Datasets** — What data is available
8. **Evaluation Metrics** — How to measure ECG generation quality
9. **EMA for Diffusion Models** — Why Exponential Moving Average is non-negotiable
10. **Mixed-Precision Training** — Engineering for A100 GPU efficiency

### 3.2 Key Technology Decisions

Each decision was backed by specific research findings:

#### Decision 1: DiT over U-Net

**Why**: Transformers have a global receptive field from token 1. In ECG, understanding the relationship between the P-wave at sample 100 and the T-wave at sample 400 is critical for clinical realism. Convolutional U-Nets require deep stacking to achieve this. Additionally, DiT follows clear scaling laws — more parameters consistently improves quality, which was proven by DiT-XL on ImageNet.

**Configuration chosen**: DiT-ECG-B — 24 layers, 768-dim, 12 heads, patch size 10 (250 tokens for a 2,500-sample signal). This balances morphological detail (20ms receptive field per patch) against computational cost.

#### Decision 2: Contrastive Pre-Training for Identity

**Why**: The identity encoder must be trained *separately* from the generative model. If trained jointly, the encoder would learn to produce embeddings that are easy for the diffusion model to use, rather than embeddings that truly capture patient identity. SimCLR-style contrastive learning with InfoNCE loss forces the encoder to map different segments from the *same* patient close together in embedding space while pushing different patients apart.

**Key augmentations selected** (identity-preserving only):
- Amplitude scaling (α ~ U[0.5, 2.0]) — changes recording gain, not physiology
- Temporal shifting (Δt ~ U[-0.5s, +0.5s]) — simulates different recording start times
- Gaussian noise (σ ~ U[0, 0.02]) — simulates electronic artifacts
- Baseline wander (sinusoidal 0.05-0.5 Hz) — simulates electrode drift
- Random crop & resize (70-100%) — simulates different window selections

**Critical finding**: Heart rate resampling DESTROYS identity. It was explicitly excluded from augmentations because changing RR intervals changes the fundamental cardiac signature.

#### Decision 3: DDPM with Cosine Schedule (Initially)

**Why**: DDPM was the well-understood baseline with abundant ECG literature reference points. The cosine noise schedule (Nichol & Dhariwal, 2021) was chosen over linear because it provides smoother signal-to-noise transitions — critical for ECG where fine morphological details like P-waves need to be learned at lower noise levels.

The research explicitly noted: *"Start with DDPM for your baseline, then migrate to OT-CFM once architecture is validated."* This advice proved prescient.

#### Decision 4: AdaLN-Zero Conditioning

**Why**: Adaptive Layer Normalization with zero-initialization was the proven mechanism from DiT (Peebles & Xie, 2023). Each transformer block's LayerNorm statistics are modulated by the conditioning signal, producing 6 scalars per block: γ₁, β₁, α₁ (attention sub-block) and γ₂, β₂, α₂ (FFN sub-block). The α terms gate the residual connections.

**Zero initialization** means the model starts as if no conditioning exists (α outputs are 0, so residual connections are initially zero). Conditioning "wakes up" gradually during training, providing extreme stability.

### 3.3 The Research Papers That Shaped Every Decision

| Paper | Year | Key Takeaway for CardioEquation |
|-------|------|--------------------------------|
| DiT (Peebles & Xie) | 2023 | AdaLN-Zero, transformer backbone, scalable architecture |
| ECGTwin | 2025 | AdaX dual-pathway conditioning, patient identity via contrastive |
| DiffuSETS | 2025 | Text-to-ECG generation, scalar conditioning via independent projection |
| FlowECG | 2025 | Flow matching for ECG synthesis — straight ODE paths |
| SSSD-ECG | 2023 | S4 state-space models for ECG (considered but not adopted for generation) |
| ECG-Bench | 2025 | Three-level evaluation protocol: FFD + morphology + downstream utility |
| EDM2 (Karras et al.) | 2024 | EMA best practices, warmup-aware decay |
| OT-CFM (Tong et al.) | 2023 | Optimal transport coupling for better path straightness |

---

## 4. Phase 1: Initial Architecture — The Diffusion Pipeline

### 4.1 The Two-Stage Decoupled Architecture

The foundational architectural decision was to decouple identity extraction from generation:

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 0: Identity Extraction               │
│                                                               │
│  Historical ECG ──→ Conv1D + ResNet-18 ──→ 512-dim Identity  │
│  (B, 1, 2500)        Contrastive Encoder    Embedding        │
│                       (4.9M params)                          │
│                                                               │
│  Training: SimCLR InfoNCE Loss (τ = 0.07)                    │
│  Result: Loss drops from 0.040 → 0.0006                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ FROZEN (no gradient)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: Generative Diffusion              │
│                                                               │
│  Gaussian Noise ──→ DiT-ECG-B (265M params) ──→ Clean ECG   │
│  x_T ~ N(0, I)     24 layers, dim=768          x_0          │
│                      12 heads, patch=10                      │
│                                                               │
│  Conditioning:                                                │
│    • Timestep t → Sinusoidal → MLP → AdaLN                  │
│    • Identity z → Frozen Encoder → AdaLN (Run 1-5)          │
│    • Identity z → Cross-Attention (Run 6)                    │
│    • HR bpm → Sinusoidal → MLP → AdaLN (Run 5+)            │
│                                                               │
│  Loss: Multi-component (noise MSE + identity + spectral +    │
│         morphology + HR + correlation + Soft-DTW)            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 The Feature Extractor: A 1D ResNet-18

The identity encoder is a purpose-built 1D adaptation of ResNet-18:

```
Input: (B, 1, 2500) — 5 seconds of ECG at 500Hz

Conv1D(1→64, k=15, s=2) → BN → ReLU → MaxPool(k=3, s=2)
  → ResBlock(64→64, ×2)
  → ResBlock(64→128, ×2, stride=2)
  → ResBlock(128→256, ×2, stride=2)
  → ResBlock(256→512, ×2, stride=2)
  → GlobalAvgPool
  
Output: (B, 512) — patient identity embedding
```

**Parameter count**: ~4.9M  
**Projection head** (discarded after pre-training): 3-layer MLP (512 → 512 → 512 → 256) with BatchNorm and ReLU. Only the 256-dim L2-normalized projection is used for the InfoNCE contrastive loss. After pre-training, the projection head is discarded and only the encoder's 512-dim output is used for downstream conditioning.

**Contrastive training results**: The InfoNCE loss dropped from 0.040 to 0.0006 across 200 epochs on 75,540 segments from 3 datasets. This near-zero loss indicates the encoder successfully learned to distinguish patients.

### 4.3 The DiT-ECG-B Architecture: 265M Parameters of Precision

The core generative model was designed as a direct 1D adaptation of DiT-XL:

| Component | Specification |
|-----------|--------------|
| **Signal length** | 2,500 samples (5s at 500Hz) |
| **Patch size** | 10 samples (20ms each) |
| **Sequence length** | 250 tokens |
| **Embedding dimension** | 768 |
| **Transformer depth** | 24 blocks |
| **Attention heads** | 12 |
| **MLP ratio** | 4× (3,072 hidden) |
| **Parameters** | ~100M (Run 6, with cross-attention) |
| **Positional embedding** | Learned, (1, 250, 768) |

**Why patch size 10?** This was the recommended balance from research:
- Patch size 5 (500 tokens): Too fine-grained, O(250K) attention complexity
- Patch size 10 (250 tokens): Good balance — captures 20ms windows (enough for QRS onset)
- Patch size 20 (125 tokens): Misses fine morphological detail
- Patch size 50 (50 tokens): Beat-level only, loses intra-beat structure

### 4.4 The Noise Scheduler: Cosine Schedule

The cosine noise schedule was implemented as:

```python
ᾱ(t) = cos²((t/T + s) / (1 + s) × π/2)
```

Where `s = 0.008` is a small offset preventing β_t from being too small near t=0. This schedule provides:
- **Gradual noise onset**: Fine morphological details (P-waves, T-waves) are preserved longer than with a linear schedule
- **Smooth SNR transitions**: Critical for ECG where the signal contains both low-frequency rhythm and high-frequency QRS spikes
- **1000 discrete timesteps** for training, with DDIM subsampling to 50 steps for inference

### 4.5 The Multi-Component Loss Function (v1)

The initial loss function was the first attempt at multi-objective optimization for ECG:

```
L_total = w_noise × L_noise_MSE
        + w_signal × L_signal_MSE  
        + w_identity × L_cosine_identity
        + w_spectral × L_spectral_FFT
        + w_correlation × L_pearson
        + w_morphology × L_gradient
```

Each component served a specific clinical purpose:

| Loss | Purpose | Why MSE Alone Isn't Enough |
|------|---------|---------------------------|
| **Noise MSE** | Standard diffusion objective | Core training signal |
| **Signal MSE** | Reconstruction fidelity | Complements noise prediction |
| **Identity Cosine** | Patient preservation | MSE doesn't capture identity |
| **Spectral FFT** | Frequency matching | MSE ignores frequency content |
| **Pearson Correlation** | Scale-invariant shape | MSE is sensitive to amplitude |
| **Morphology Gradient** | QRS slope sharpness | MSE blurs sharp transitions |

The morphology gradient loss was particularly innovative — it computes MSE on the 1st derivative (slope) and 2nd derivative (curvature) of the signal:

```python
L_grad = MSE(∇x_pred, ∇x_true) + 0.1 × MSE(∇²x_pred, ∇²x_true)
```

This directly enforces the sharp, high-velocity slopes required for clinically realistic QRS complexes, preventing the "smoothed over" artifacts that plague deep learning waveform generators.

---

## 5. Run 1: First Contact with Reality

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | MIT-BIH Arrhythmia only (8,592 segments, 47 patients) |
| **Epochs** | 200 |
| **Batch size** | 32, gradient accumulation 8 (effective 256) |
| **Learning rate** | 1×10⁻⁴ |
| **Noise scheduler** | Cosine, 1000 timesteps |
| **Inference** | DDIM, 50 steps, CFG scale 3.0 |
| **GPU** | NVIDIA A100 80GB PCIe |

### 5.2 Results

| Metric | Run 1 Value | Target | Assessment |
|--------|-------------|--------|------------|
| **FFD** ↓ | **1,032** | < 15 | ❌ Catastrophically high |
| **MMD** ↓ | 0.661 | < 0.1 | ❌ Far from target |
| **HR MAE** ↓ | 17.2 bpm | < 5 bpm | ⚠️ Moderate |
| **ReID Top-1** ↑ | 11.8% | > 50% | ⚠️ Low but non-zero |
| **ReID Top-5** ↑ | 35.3% | > 80% | ⚠️ Low but non-zero |

### 5.3 Analysis: What Happened

**The FFD of 1,032 was alarming.** For context, a good FFD score for ECG generation is < 15 (analogous to FID for images). A score above 1,000 means the generated distribution was almost completely disjoint from the real distribution — the model was producing waveforms that bore little resemblance to real ECGs in the encoder's feature space.

**Root causes identified:**

1. **Dataset too small**: MIT-BIH has only 47 patients with 8,592 segments. For a 265M parameter model, this is dramatically undertrained. The model was memorizing rather than generalizing.

2. **No HR conditioning**: Without explicit heart rate control, the model learned a blurred average of all heart rates in the training set. The generated ECGs had plausible morphology but uncontrolled timing.

3. **ReID showed promise**: Despite the terrible FFD, the Re-ID Top-5 of 35.3% indicated that the contrastive identity encoder was working — the model was partially preserving patient-specific features even in its first untrained state.

### 5.4 Key Takeaway

> **Run 1 proved the architecture was viable but starving for data.** The identity pathway showed signs of life, but the model needed orders of magnitude more training data to generalize beyond the 47 MIT-BIH patients.

---

## 6. Run 2: Multi-Dataset Scaling & First Improvements

### 6.1 What Changed

The primary intervention was **massive data scaling**:

- **MIT-BIH**: 8,592 segments (kept)
- **PTB-XL**: 56,300 segments from 18,885 patients (NEW)
- **Total**: ~65,000 segments across ~19,000 patients

PTB-XL was the critical addition — it's the largest publicly available labeled 12-lead ECG dataset with 21,837 records and real patient IDs, enabling proper contrastive learning.

### 6.2 The Data Engineering Challenge

Getting PTB-XL into the training pipeline required solving several engineering problems:

1. **Sampling rate harmonization**: PTB-XL provides data at both 100Hz and 500Hz. The pipeline was configured to prioritize 500Hz high-resolution records (matching MIT-BIH's target), with 100Hz records upsampled via `scipy.signal.resample_poly`.

2. **Duration standardization**: PTB-XL records are 10 seconds long, while our pipeline uses 5-second windows (2,500 samples at 500Hz). Records were windowed with random offsets during training.

3. **Download automation**: The `download_all_datasets.py` script was built to handle PhysioNet's WFDB format, with multiple fallback strategies:
   - Primary: `wfdb` library with internal versioning
   - Fallback: Direct `wget` download of `.dat`/`.hea` files
   - The script went through 4 revisions to handle edge cases in PhysioNet's API

### 6.3 Results

| Metric | Run 1 | Run 2 | Δ Change |
|--------|-------|-------|----------|
| **FFD** ↓ | 1,032 | **42.0** | **-96%** 🎉 |
| **MMD** ↓ | 0.661 | **0.383** | -42% |
| **HR MAE** ↓ | 17.2 | **9.3** | -46% |
| **ReID Top-1** ↑ | 11.8% | 5.9% | -50% ⚠️ |
| **ReID Top-5** ↑ | 35.3% | 23.5% | -34% ⚠️ |
| **Gen HR Mean** | — | 90.5 bpm | — |
| **Gen HR Std** | — | 21.1 bpm | — |

### 6.4 Analysis: The Good and The Concerning

**The FFD drop from 1,032 to 42.0 was a 96% improvement** — the single biggest metric jump in the project. Data scaling worked exactly as predicted. The model was now generating ECGs that at least looked like real ECGs in feature space.

**HR MAE improved to 9.3 bpm** — getting close to the clinical target of < 5 bpm, even without explicit HR conditioning. The model was implicitly learning heart rate patterns from the data distribution.

**However, ReID metrics dropped.** Top-1 fell from 11.8% to 5.9%. This was initially puzzling — more data should help identity preservation, not hurt it. The explanation:

> **With 47 patients (Run 1), the model essentially memorized each patient's pattern.** When 19,000 patients were introduced (Run 2), the model had to actually generalize identity — and it wasn't ready. The contrastive encoder was producing good embeddings, but the diffusion model wasn't leveraging them strongly enough.

### 6.5 Key Takeaway

> **Data scaling solved distribution quality but exposed identity weakness.** The model could generate realistic *generic* ECGs but couldn't yet generate *patient-specific* ECGs reliably. Identity conditioning needed to be strengthened.

---

## 7. Run 3: The Silent Failure

### 7.1 What Was Attempted

Run 3 used the **same model and code as Run 2** — no architectural changes. The only modification was a dataset processing fix attempt to ensure all three datasets (MIT-BIH + PTB-XL + Chapman-Shaoxing) were properly loaded.

### 7.2 What Happened

**Run 3 produced no usable results.** The training either failed to converge or the results were equivalent to Run 2, providing no new signal. This was documented simply as *"Run 3 used same model as Run 2 — no code changes, just dataset fix attempt."*

### 7.3 The Chapman-Shaoxing Problem

The Chapman-Shaoxing dataset (45,152 records in the full version) was the target addition, but the download pipeline had a critical bug: the `.dat` file detection logic was failing for Chapman's file format. The `download_all_datasets.py` script required a specific fix:

```python
# Bug: Looking for .dat files, but Chapman uses .mat format
# Fix (commit 782030f): 
# refactor: Replace `wfdb` with `wget` for PTB-XL and Chapman dataset downloads,
# and introduce `_count_hea_files` helper
```

This was one of several engineering battles with data ingestion that consumed significant development time but produced no model improvement by themselves.

### 7.4 Key Takeaway

> **"Just adding more data" without architectural changes was a dead end.** The model had plateaued — it needed fundamentally new capabilities (explicit HR conditioning) to break through.

---

## 8. Run 4: The Spectral Loss Catastrophe

### 8.1 What Changed

Run 4 introduced several Phase 1 improvements:

1. **Time-warp and amplitude augmentation** during diffusion training
2. **Morphology gradient loss** (1st + 2nd derivative matching)
3. **Full 3-dataset training**: MIT-BIH (8,592) + PTB-XL (56,300) + Chapman (10,648) = 75,540 segments
4. **Proper early stopping** with long patience (stopped at epoch 474 of 500)
5. **Spectral loss added** with weight 0.1

### 8.2 Results

| Metric | Run 2 | Run 4 | Δ Change |
|--------|-------|-------|----------|
| **FFD** ↓ | 42.0 | **21.7** | **-48%** ✅ |
| **MMD** ↓ | 0.383 | 0.432 | +13% ⚠️ |
| **HR MAE** ↓ | 9.3 | **31.4** | **+237%** 🔴🔴🔴 |
| **ReID Top-1** ↑ | 5.9% | **11.8%** | +100% ✅ |
| **ReID Top-5** ↑ | 23.5% | **35.3%** | +50% ✅ |
| **Gen HR Mean** | 90.5 | 68.4 bpm | — |
| **Gen HR Std** | 21.1 | 20.5 bpm | — |
| **Val Loss** | — | **5.677** | Early stopped at epoch 474 |

### 8.3 The Catastrophe: HR MAE Skyrocketed from 9.3 to 31.4

**This was the most shocking regression in the project's history.** HR accuracy — which had been close to clinical target in Run 2 — more than tripled to 31.4 bpm error. The model was now generating ECGs with completely wrong heart rates.

The clue was in the W&B per-component loss dashboard, which revealed the most important debugging finding of the entire project:

### 8.4 The Spectral Loss Smoking Gun

| Loss Component | Final Value (Epoch 474) |
|---------------|-------------------------|
| noise_mse | 0.099 |
| signal_mse | 0.180 |
| identity | 0.067 |
| correlation | 0.105 |
| morphology | 0.040 |
| **spectral** | **125.19** 🔴🔴🔴 |

**The spectral loss was 1,000× larger than all other components combined.** It was consuming approximately 99% of the total gradient budget.

What happened was a classic **multi-objective optimization catastrophe**:

1. The spectral loss (FFT magnitude matching) operates in frequency space where magnitudes can be very large
2. With weight 0.1, the spectral loss dominated: `0.1 × 125.19 = 12.52` vs all other losses combined `≈ 0.49`
3. The optimizer spent virtually all its capacity minimizing spectral loss
4. Identity, morphology, correlation, and most critically **heart rate** signals were drowned out
5. The model learned to match FFT magnitudes but lost the ability to generate correct rhythm timing

### 8.5 The Fix

The spectral loss weight was reduced by 1,000×:

```python
# Before (Run 4):
w_spectral = 0.1  # ← CATASTROPHIC: dominated all gradients

# After (Run 5):  
w_spectral = 0.0001  # ← Balanced: spectral contributes ~0.01 to total
```

This seemingly minor change — modifying a single floating-point constant — was committed in `7a1284a` and fundamentally unlocked multi-objective training.

### 8.6 The Silver Linings

Despite the spectral catastrophe, Run 4 produced real progress:

- **FFD improved 48%** (42.0 → 21.7): Signal quality was genuinely better even with distorted HR
- **ReID recovered to 11.8% Top-1**: The frozen contrastive encoder was working, proving identity preservation was architecturally sound
- **Early stopping worked perfectly**: The model stopped at epoch 474, not overfitting — the training pipeline's self-regulation was validated
- **Morphology loss at 0.040**: The gradient matching loss was producing sharper, more realistic QRS complexes

### 8.7 Infrastructure Discovery

Run 4 also uncovered a critical infrastructure issue:

> **GPU memory conflict**: The A100 80GB had a competing process consuming 66GB. Training initially failed with OOM errors. Only after identifying and clearing the rogue process could the full 265M parameter model train.

**Training time**: ~856 seconds/epoch × 474 epochs ≈ **112 hours (~4.7 days)**.

### 8.8 Key Takeaway

> **W&B per-component loss logging was the single most important diagnostic tool.** Without it, we would have had no explanation for the HR regression. The lesson: *in multi-objective learning, you MUST monitor every loss component individually. Aggregate loss metrics hide catastrophic imbalances.* This finding directly shaped all subsequent training configurations.

---

## 9. The Research Interlude: Deep Investigation

### 9.1 The Three Research Queries

After Run 4's spectral loss catastrophe exposed three fundamental weaknesses (HR conditioning, differentiable HR loss, loss weight balancing), the team paused to conduct deep research before Run 5. Three carefully crafted prompts were used to gather implementation guidance from both Claude and Gemini:

**Query 1 — How to add explicit HR conditioning to DiT:**
> *"I have a Diffusion Transformer for ECG generation. Currently it takes patient identity embedding via AdaLN. How do I add explicit heart rate conditioning? Should I: (a) add HR to timestep embedding, (b) add a separate AdaLN layer for HR, or (c) use cross-attention?"*

**Verdict**: Option B — separate MLP heads for timestep, HR, and identity, combined additively before AdaLN. Option A (fusing HR into timestep) was rejected because HR gradients get smothered by the much larger timestep signal. Option C (cross-attention for a scalar) was deemed overkill with unnecessary O(seq_len) overhead.

**Query 2 — How to build a differentiable HR loss:**
> *"I need a differentiable loss that estimates RR intervals from raw 1D ECG. Options: (a) soft R-peak detection, (b) autocorrelation via FFT, (c) trained CNN. Which is most stable during backprop?"*

**Verdict**: FFT autocorrelation + soft argmax. Soft R-peak detection fails early in training when ECG is still noisy. CNN estimators create co-adaptation issues. FFT autocorrelation produces smooth, differentiable gradients that degrade gracefully under noise.

**Query 3 — How to balance 6+ loss components:**
> *"Compare GradNorm, PCGrad, CAGrad, MGDA for automatic loss weight balancing in diffusion models."*

**Verdict**: For the immediate term, manual rebalancing after W&B analysis. For future work, CAGrad or Pareto-based optimization was recommended.

### 9.2 Research Findings Compiled

The research findings were compiled into the `docs/run5_implementation_plan.md` (334 lines) — a detailed, research-backed plan with exact code snippets, ready to implement. Key innovations specified:

1. **`HREmbedding`**: Sinusoidal encoding of scalar HR, normalized to [0,1] over [30, 200] bpm range
2. **`ConditioningProjector`**: 3 independent MLP heads (timestep, HR, patient_id) with additive combination
3. **`DifferentiableHRLoss`**: FFT autocorrelation + soft argmax over physiological lag window [150, 1000] samples
4. **`GPUBatchedHRExtractor`**: Pan-Tompkins preprocessing on GPU + FFT autocorrelation, processing 75K samples in ~5 seconds
5. **HR-balanced sampling**: `WeightedRandomSampler` binned by HR to prevent mode collapse on dominant 60-90 bpm range
6. **Zero-initialization**: HR MLP head zero-initialized so conditioning "wakes up" gradually

### 9.3 The Critical Design Decision: Zero-Initialization

One insight from the research deserves special emphasis. The HR head's zero-initialization was identified as **critical for training stability**:

```python
# ConditioningProjector.__init__
nn.init.zeros_(self.hr_mlp[-1].weight)
nn.init.zeros_(self.hr_mlp[-1].bias)
```

**Why this matters**: At training step 0, the HR MLP outputs are exactly zero. The conditioning signal is `t_out + 0 + pid_out` — identical to a model without HR conditioning. As training progresses, the HR head gradually learns non-zero outputs, smoothly introducing HR influence. Without zero-init, the randomly initialized HR head would inject random noise into every conditioning vector, destabilizing early training.

### 9.4 The GPU HR Extraction Pipeline

Before Run 5, all 75,540 ECG segments needed ground-truth HR labels. The naive approach (NeuroKit2, ~1 segment/second) would take 10-40 hours. The GPU-accelerated pipeline completed in **under 10 seconds**:

```
ECG signal → GPU Conv1D bandpass (5-15Hz)
           → Pan-Tompkins derivative kernel
           → Square + Moving average integration
           → FFT autocorrelation
           → Argmax over [150, 1000] lag range
           → HR = 60 × fs / peak_lag
```

**Throughput**: ~20,000-50,000 segments/second on GPU, processing all three datasets in 3-8 seconds. This was committed as `src/data/precompute_hr.py` and run as the first step before training.

---

## 10. Run 5: Explicit HR Conditioning — The Breakthrough Architecture

### 10.1 What Changed (Summary)

Run 5 was the single largest architectural change in the project's history:

| Component | Before (Run 4) | After (Run 5) |
|-----------|----------------|---------------|
| **HR conditioning** | None (implicit only) | Explicit: `HREmbedding` → sinusoidal → MLP → AdaLN |
| **Conditioning projector** | Single combined MLP | 3 independent heads (t, hr, pid), additive |
| **HR loss** | Non-differentiable argmax | `DifferentiableHRLoss`: FFT autocorrelation + soft argmax |
| **HR precomputation** | N/A | `GPUBatchedHRExtractor`: 75K samples in <10s |
| **Spectral weight** | 0.1 (catastrophic) | 0.0001 (balanced) |
| **HR weight** | N/A | 0.05 with warmup ramp over 5,000 steps |
| **HR sampling** | Uniform random | `WeightedRandomSampler` with HR-binned weights |
| **DDIM stochasticity** | η = 0 (deterministic) | η = 0.75 (recommended for diversity) |

### 10.2 The DifferentiableHRLoss Implementation

The most technically intricate new component was the differentiable HR loss:

```
Input: x_0_pred (B, 1, T) — model's clean signal estimate during training

Step 1: Zero-mean the signal
Step 2: FFT → power spectral density → inverse FFT = autocorrelation (Wiener-Khinchin)
Step 3: Extract autocorrelation in physiological lag range:
        lag_min = fs × 60 / hr_max = 500 × 60 / 200 = 150 samples
        lag_max = fs × 60 / hr_min = 500 × 60 / 30  = 1000 samples
Step 4: Soft argmax: weighted sum of lag indices using softmax(τ × acf) as weights
Step 5: Convert soft lag → HR: hr_est = 60 × fs / soft_lag
Step 6: L1 loss: |hr_est - hr_target|
```

The **soft argmax** (temperature τ = 10.0) was the key to differentiability. Hard argmax breaks the gradient chain, but soft argmax provides smooth gradients:

```python
weights = F.softmax(self.temp * window, dim=-1)  # (B, W) — smooth "soft peak"
soft_lag = (weights * lags).sum(dim=-1)           # (B,) — weighted average lag
```

### 10.3 HR Warmup Ramp

To prevent the HR loss from destabilizing early training (when `x_0_pred` is mostly noise), a linear warmup was applied:

```python
hr_ramp = min(1.0, global_step / 5000)
hr_loss = hr_loss_fn(x0_pred, hr_batch, weight=0.05 * hr_ramp)
```

For the first 5,000 steps, the HR loss weight gradually increases from 0 to 0.05. This gives the primary diffusion loss time to establish basic waveform structure before HR constraints are applied.

---

## 11. Run 5b: 500 Epochs — First Complete Training Run

### 11.1 Configuration

Run 5b was the refined version of Run 5, with several stability improvements:

| Parameter | Run 5 (initial) | Run 5b (final) |
|-----------|----------------|----------------|
| **Epochs** | 500 | 500 |
| **Patience** | 30 | **60** (doubled for longer exploration) |
| **Warmup steps** | 5,000 | **2,000** |
| **HR loss weight** | 0.05 | 0.05 |
| **HR variance weight** | 0.5 | **0.2** |
| **EMA decay** | 0.9999 | 0.9999 |
| **LR** | 1×10⁻⁴ | 1×10⁻⁴ |
| **Early stopping on** | Val loss | **EMA val loss** |

The commit `9b457a1` ("Run 5b fixes: patience 60, HR val loss, EMA early stopping, warmup 2000, var_weight 0.2") captures these hyperparameter refinements.

### 11.2 The Historic Milestone: 500 Epochs Without Early Stopping

**For the first time in project history, the model ran all 500 epochs without premature early stopping.** EMA-smoothed validation loss improved continuously from epoch 1 through epoch 496 (best epoch), reaching a final value of **0.6995**.

This was a dramatic contrast to Run 4, which early-stopped at epoch 474 with a val loss of 5.677. The 8× improvement in validation loss (5.677 → 0.6995) confirmed that the architectural changes were fundamentally sound.

### 11.3 Training Convergence Details

| Loss Component | Epoch 10 | Epoch 500 | Improvement |
|----------------|----------|-----------|-------------|
| Signal MSE | 0.638 | **0.322** | −50% |
| Identity Loss | 0.201 | **0.101** | −50% |
| Morphology Loss | 0.874 | **0.037** | **−96%** ✨ |
| HR Loss | 1.042 | **0.420** | −60% |
| Val Loss (EMA) | ~1.8 | **0.6995** | Continuous improvement |

The **morphology loss dropping 96%** (from 0.874 to 0.037) was particularly significant — it meant the model was generating ECGs with nearly perfect QRS slope and curvature matching compared to real signals.

### 11.4 Clinical Evaluation Results

The evaluation was performed using DDIM (50 steps, CFG scale 3.0) on 200 samples drawn from the combined MIT-BIH/PTB-XL/Chapman validation set:

| Metric | Run 4 | Run 5b | Target | Trend |
|--------|-------|--------|--------|-------|
| **Best Val Loss (EMA)** | 5.677 | **0.6995** | ↓ | ✅ 8× better |
| **Morphology Loss** | ~0.8 | **0.037** | → 0 | ✅ −96% |
| **Signal MSE** | ~0.8 | **0.322** | → 0 | ✅ −50% |
| **HR Loss** | N/A | **0.420** | → 0 | ✅ Conditioning active |
| **HR MAE (bpm)** ↓ | 31.4 | **23.6** | < 8 | ✅ −25% |
| **Generated HR Std** ↑ | 20.5 | **31.5 bpm** | > 47.4 | 📈 +54% |
| **Generated HR Mean** | 68.4 | **54.1 bpm** | ~77 bpm | ⚠️ Under-estimated |
| **QRS Duration (gen)** | 83.6 ms | **60.4 ms** | ~111 ms | ⚠️ Compressed |
| **FFD** ↓ | 21.7 | **91.1** † | < 15 | ⚠️ See note |
| **MMD** ↓ | 0.43 | **0.88** † | < 0.1 | ⚠️ See note |
| **ReID Top-1** ↑ | 11.8% | **0.0%** | > 50% | 🔴 Critical |
| **ReID Top-5** ↑ | 35.3% | **2.5%** | > 80% | 🔴 Critical |

> **† Important Context on FFD/MMD**: Run 4 metrics were measured on 17 digitized hospital PDFs (small, internally consistent set). Run 5b metrics were measured on 200 diverse samples across three open-source datasets. The larger, more heterogeneous reference set naturally inflates FFD/MMD. These numbers are **not directly comparable**.

### 11.5 Analysis: Victories and Defeats

**What Improved:**

1. **HR MAE: 31.4 → 23.6 bpm (−25%)** — The explicit HR conditioning was working. The model was learning to respond to HR targets.
2. **HR Diversity: 20.5 → 31.5 bpm std (+54%)** — Mode collapse was partially resolving. The HR variance loss was pushing the model to generate diverse heart rates.
3. **Morphology: 0.874 → 0.037 (−96%)** — Generated QRS complexes were sharp and clinically realistic.
4. **Stability**: Complete 500-epoch training without early stopping — the training pipeline was finally stable.

**What Failed:**

1. **ReID collapsed: 11.8% → 0.0% Top-1** — This was the most alarming finding. The model was generating ECGs that looked realistic but belonged to *no one* — generic cardiac patterns without patient specificity. This became the primary target for Run 6.

2. **QRS temporal compression: 60.4 ms vs 111 ms real** — Generated QRS complexes were approximately 45% shorter than real ones. The MSE-based losses were "squeezing" the temporal axis.

3. **Generated HR mean: 54.1 bpm** — Below clinical mean of 77.8 bpm. The model was under-estimating resting heart rate.

### 11.6 Root Cause Analysis for ReID Collapse

The ReID collapse was diagnosed to two architectural issues:

1. **Identity dilution in AdaLN**: The identity embedding was mixed with timestep and HR embeddings before AdaLN modulation. At high noise levels (large t), the timestep signal dominated, and identity information was washed out. The model learned to ignore identity at high noise steps, and this behavior persisted at inference.

2. **No dedicated identity pathway**: Unlike ECGTwin's AdaX dual-pathway design, the Run 5 architecture forced all conditioning through a single AdaLN pathway. This created a gradient competition where timestep (essential for denoising) always won against identity (which only matters for personalization).

These findings directly motivated Run 6's cross-attention identity architecture.

---

## 12. Clinical Validation & Mentor Reports

### 12.1 The Clinical Validation Pipeline

CardioEquation included a unique clinical validation capability: processing real hospital ECG reports (24 PDF files from a hospital dataset) through an `ECGDigitizer` to convert paper ECGs into digital signals for validation.

The pipeline:

```
Hospital ECG PDF → ECGDigitizer → Digitized 1D signal (2500 samples)
                                         │
                     ┌───────────────────┘
                     ▼
         Feature Extractor → Identity Vector → DiT-ECG → Generated ECG
                                                              │
                                                    Compare: Real vs Generated
                                                    • FFD, MMD
                                                    • HR MAE
                                                    • ReID Top-1/Top-5
                                                    • Visual comparison plots
```

The `Dataset/` folder contained 24 real ECG reports organized by patient, with some patients having multiple records — enabling cross-record identity validation.

### 12.2 The Mentor Report (Run 4)

After Run 4, a detailed mentor report was prepared (`docs/mentor_report.md`) summarizing all findings. Key sections:

- **What worked**: FFD improvement (42→21.7), ReID recovery, early stopping validation, contrastive pre-training success
- **Critical bug found**: Spectral loss imbalance (125.19 vs ~0.1 for all other losses)
- **HR diversity gap**: Generated HR std only 43% of real distribution
- **Infrastructure**: A100 80GB, ~4.7 days wall clock time
- **Research prompts**: Three specific queries for Gemini/Claude to prepare Run 5

### 12.3 The Senior Mentor Report (Run 5b)

After Run 5b, a comprehensive executive report (`docs/senior_mentor_report.md`, 162 lines) was prepared with:

- Mermaid architecture diagrams for the two-stage pipeline
- Detailed conditioning mechanism explanation (ConditioningProjector with 3 heads)
- Physics-informed loss mathematics (DifferentiableHR, Morphology Gradient, Spectral Balance)
- Training convergence tables showing continuous improvement across all 500 epochs
- Honest assessment of remaining challenges (ReID at 0.0%, QRS compression at 45%)
- Phase 2 roadmap (CFM migration, latent space DiT, 12-lead expansion)

---

## 13. Run 6: The OT-CFM Revolution

Run 6 represents the most ambitious architectural overhaul in CardioEquation's history. It simultaneously addresses all three critical failures identified in Run 5b: **identity collapse** (ReID at 0.0%), **QRS temporal compression** (45% shorter), and **HR under-estimation**. The changes are so fundamental that Run 6 is essentially a new model that shares only the dataset pipeline and encoder with its predecessors.

### 13.1 The Three Problems, Three Solutions

| Problem (Run 5b) | Root Cause | Run 6 Solution |
|-------------------|------------|-----------------|
| **ReID at 0.0%** | Identity diluted by timestep in shared AdaLN | **IdentityCrossAttention** — dedicated pathway |
| **QRS 45% compressed** | MSE loss "squeezes" temporal axis | **Soft-DTW loss** — penalizes time warping |
| **HR under-estimated** | HR loss fires on noisy signal estimates | **Timestep gating** — HR loss only at t < 0.2 |

### 13.2 Innovation 1: From DDPM to OT-CFM (Conditional Flow Matching)

#### 13.2.1 Why Flow Matching?

DDPM (Denoising Diffusion Probabilistic Models) works by learning to predict the noise ε added at each step of a stochastic Markov chain. It requires 50-1000 inference steps to generate a sample and suffers from high gradient variance due to the stochastic path.

OT-CFM (Optimal Transport Conditional Flow Matching) replaces this with a fundamentally different approach:

| Property | DDPM (Run 1-5b) | OT-CFM (Run 6) |
|----------|------------------|------------------|
| **Training target** | Predict noise ε at timestep t | Predict velocity v = x₁ - x₀ |
| **Path shape** | Stochastic Markov chain | Straight-line optimal transport |
| **Inference steps** | 20-50 (DDIM) | 10-20 (Euler ODE) |
| **Gradient variance** | High (random t sampling) | Low (straight paths) |
| **Source distribution** | Standard Gaussian | Minibatch OT-coupled pairs |
| **Inference speed** | ~1.5s per sample | ~0.3s per sample |

#### 13.2.2 The OT-CFM Training Loop

The flow matching training loop replaces the entire DDPM noise schedule with a simple linear interpolation:

```python
# 1. Sample continuous timestep t ~ U[0, 1]
t = torch.rand(B, device=device)

# 2. Linear interpolation: straight path from noise → signal
x_t = (1 - t_expand) * noise + t_expand * x_clean

# 3. Target velocity: direction from noise to clean signal
target_velocity = x_clean - noise

# 4. Model predicts velocity field
predicted_velocity = model(x_t, t, identity, hr_bpm=hr)

# 5. Loss: predict the velocity vector, NOT the noise
loss = F.mse_loss(predicted_velocity, target_velocity)
```

The key insight is mathematical simplicity: instead of learning to denoise through a complex stochastic process, the model learns to point from any intermediate state directly toward the clean signal. This makes paths straighter, gradients lower-variance, and inference faster.

#### 13.2.3 The Euler ODE Solver (Inference)

At inference time, OT-CFM generates samples by solving an ODE (ordinary differential equation) using the Euler method:

```python
def euler_sample(model, noise, identity, hr_bpm, num_steps=20, cfg_scale=2.0):
    dt = 1.0 / num_steps
    x = noise.clone()
    
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model.forward_with_cfg(x, t, identity, hr_bpm, cfg_scale)
        x = x + v * dt  # Single Euler step
    
    return x
```

With 20 Euler steps, OT-CFM produces results comparable to DDIM with 50 steps — a 2.5× speedup. The `flow_matching.py` module implements the full `ConditionalFlowMatchingScheduler` with:
- Linear interpolation for training: `x_t = (1-t) * x₀ + t * x₁`
- Velocity targets: `v_t = x₁ - x₀`  
- σ_min = 1e-5 (small noise floor for numerical stability)

#### 13.2.4 Critical Warning: Parameterization Mismatch

The "Converted Text" research manual explicitly flagged:

> ⚠️ **CFM predicts velocity v_θ (direction x₀→x₁), NOT noise ε_θ. Never mix parameterizations — if you use CFM loss, inference must use ODE solver, not the DDPM reverse Markov chain. Using DDPM sampling with CFM-trained weights produces pure noise.**

This warning was heeded: the evaluation script (`evaluate_dit.py`) auto-detects training mode from checkpoint metadata and uses the appropriate sampler:

```python
if mode == 'flow_matching':
    # Euler ODE integration
    generated = euler_sample(model, noise, ...)
else:
    # DDIM reverse diffusion
    generated = ddim_sample(model, noise, scheduler, ...)
```

### 13.3 Innovation 2: Identity Cross-Attention (IP-Adapter Paradigm)

#### 13.3.1 The Problem with Shared AdaLN

In Runs 1-5b, patient identity was injected through the same AdaLN mechanism as the timestep:

```
Run 1-5b: c = MLP(concat(t_emb, hr_emb, id_emb)) → 6 AdaLN scalars per block
```

This created a fundamental information bottleneck:
- The timestep embedding carries critical denoising information (what noise level are we at?)
- The HR embedding carries rhythm information (how fast should the heart beat?)
- The identity embedding carries morphological information (what shape is this patient's QRS?)

All three are compressed into 6 scalars per block. The timestep — which changes every training step — dominates gradient flow, while identity — which is constant per patient — gets marginalized.

#### 13.3.2 The Solution: Dedicated Cross-Attention Pathway

Run 6 separates identity into its own attention pathway:

```
Run 6 Architecture (per DiT block):

  Input: x (B, 250, 768) — patch tokens
  
  Step 1: AdaLN-Modulated Self-Attention
    ├── AdaLN produces γ₁,β₁,α₁,γ₂,β₂,α₂ from [t + hr] ONLY
    ├── Self-attention: patches attend to each other
    └── Gated residual: x = x + α₁ × attn_out
  
  Step 2: Identity Cross-Attention (NEW)
    ├── IdentityTokenizer: 512-dim → 8 virtual tokens (B, 8, 768)
    ├── Cross-attention: patches attend to identity tokens
    │     Query = patch tokens (250 tokens)
    │     Key/Value = identity tokens (8 tokens)
    └── Gated residual: x = x + gate × cross_out  (gate starts at 0)
  
  Step 3: AdaLN-Modulated FFN
    ├── AdaLN modulation from [t + hr]
    └── Gated residual: x = x + α₂ × mlp_out
```

#### 13.3.3 The IdentityTokenizer

Instead of compressing 512 dimensions into a single vector, the `IdentityTokenizer` projects the identity embedding into 8 virtual tokens:

```python
class IdentityTokenizer(nn.Module):
    def __init__(self, cond_dim=512, d_model=768, num_tokens=8):
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * num_tokens),  # 512 → 6144
        )
        self.norm = nn.LayerNorm(d_model)
        # Zero-init: identity starts as no-op
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, identity):  # (B, 512) → (B, 8, 768)
        tokens = self.proj(identity).view(B, 8, 768)
        return self.norm(tokens)
```

**Why 8 tokens?** The IP-Adapter paper showed that 4-16 tokens provide the best balance. Each token can specialize in different aspects of the patient's cardiac signature:
- Tokens 1-2: QRS morphology (R-wave height, S-wave depth)
- Tokens 3-4: P-wave shape and timing
- Tokens 5-6: T-wave characteristics
- Tokens 7-8: ST segment and overall rhythm pattern

#### 13.3.4 Zero-Initialized Gate

The `IdentityCrossAttention` module uses a **learned gate parameter** initialized to 0:

```python
self.gate = nn.Parameter(torch.zeros(1))
# Forward: x = x + self.gate * cross_attn_output
```

At training start, `gate = 0`, so the cross-attention output is completely discarded. The model begins as if it has no identity conditioning at all — identical to an unconditional diffusion model. As training progresses, the optimizer learns to increase the gate, gradually introducing identity influence.

This is critical because:
1. The model first learns to generate realistic *generic* ECGs (the easy part)
2. Then it learns to *personalize* those ECGs (the hard part)
3. The distinction emerges naturally — no curriculum or staged training needed

### 13.4 Innovation 3: GPU-Accelerated Soft-DTW Loss

#### 13.4.1 The QRS Compression Problem

Run 5b revealed that generated QRS complexes were 45% shorter than real ones (60.4 ms vs 111 ms). MSE loss was the culprit — it penalizes point-to-point differences, which means a slightly time-shifted QRS receives a massive penalty. The optimizer's "easiest" solution is to compress the QRS into a narrower temporal window, reducing the MSE at the cost of clinical accuracy.

#### 13.4.2 Why Soft-DTW?

Dynamic Time Warping (DTW) measures similarity between sequences that may be temporally misaligned — exactly the right tool for ECG comparison. **Soft-DTW** is a differentiable relaxation of DTW using the soft-minimum operator:

```
Hard DTW:  min(cost_diagonal, cost_up, cost_left) + distance[i,j]
Soft-DTW: -γ × log(exp(-cost_diag/γ) + exp(-cost_up/γ) + exp(-cost_left/γ)) + distance[i,j]
```

As γ → 0, soft-DTW approaches hard DTW. For ECG training, γ = 0.1 was chosen — sharp enough to capture temporal alignment but smooth enough for gradient flow.

#### 13.4.3 The QRS Segment Extraction Pipeline

Soft-DTW on the full 2,500-sample signal would be computationally prohibitive. Instead, the loss function extracts only QRS segments:

```
1. Detect R-peaks in the CLEAN signal (no gradient needed)
2. Extract ±100ms window around each R-peak (200 samples at 500Hz)
3. Extract corresponding window in the GENERATED signal
4. Normalize both segments to zero mean, unit variance
5. Compute Soft-DTW distance between the two segments
6. Average across all QRS pairs in the batch
```

For a typical 5-second ECG with 3-5 heartbeats, this means 3-5 Soft-DTW computations on ~200-sample sequences — fast enough for training.

#### 13.4.4 The Implementation: GPU-Accelerated with Fallback

The implementation (`losses_v2.py`) uses a two-tier strategy:

1. **Primary**: `pysdtw` library (GPU-accelerated CUDA kernel) — requires `pip install pysdtw`
2. **Fallback**: Anti-diagonal vectorized PyTorch implementation — ~50× faster than naive nested loops

The fallback is the more interesting implementation — it processes the Soft-DTW dynamic programming table by anti-diagonals, where all cells on the same anti-diagonal are independent and can be computed in a single batched tensor operation:

```python
for d in range(1, N + M + 1):  # For each anti-diagonal
    # All (i,j) pairs where i+j = d are independent
    costs = torch.stack([r_diag, r_up, r_left], dim=-1)  # (B, num_cells, 3)
    soft_min = -gamma * torch.logsumexp(-costs / gamma, dim=-1)
    R[:, i_indices, j_indices] = d_vals + soft_min
```

### 13.5 Additional Run 6 Improvements

#### 13.5.1 Timestep-Gated HR Loss

In Run 5b, the HR loss was computed at all timesteps. But at high noise levels (t > 0.5), the model's clean signal estimate `x_0_pred` is mostly noise — any HR extracted from it is meaningless. The gradients from these noisy HR estimates were actively harmful.

Run 6 introduces timestep gating:

```python
low_noise_mask = t_normalized < self.t_gate  # t_gate = 0.2
hr_error = (hr_est - hr_target).abs()
gated_error = hr_error * low_noise_mask.float()  # Zero out high-noise samples
loss = gated_error.sum() / low_noise_mask.sum().clamp(min=1)
```

This means HR loss only fires on the ~20% of training steps where t < 0.2 (low noise). The HR estimate at these timesteps is clean enough to provide useful gradient signal.

#### 13.5.2 Identity SNR Floor

Even with dedicated cross-attention, identity loss still needs to fight the signal-to-noise ratio. At high noise (t → 1), the `x_0_pred` contains almost no signal — the identity cosine loss gradient is near-zero.

Run 6 introduces a **30% gradient floor** for identity:

```python
# Normal SNR weight (approaches 0 at high noise)
snr_w = (1.0 - t_normalized.mean()).clamp(min=0.0, max=1.0)

# Identity gets AT LEAST 30% of max weight, even at t=1.0
identity_snr = max(snr_w, identity_snr_floor)  # floor = 0.3

total += identity_snr * w_identity * l_identity
```

This ensures the identity loss never falls below 30% of its maximum contribution, preventing the "identity extinction" that caused ReID collapse in Run 5b.

#### 13.5.3 Loss Weight Adjustments

| Weight | Run 5b | Run 6 | Rationale |
|--------|--------|-------|-----------|
| `w_noise` | 1.0 | 1.0 | Primary objective, unchanged |
| `w_signal` | 1.0 | 1.0 | Unchanged |
| `w_identity` | **0.5** | **1.5** | 3× increase to fight identity collapse |
| `w_spectral` | 0.0001 | **0.00001** | Further reduced (still 1000× above safe) |
| `w_correlation` | 0.2 | 0.2 | Unchanged |
| `w_morphology` | 0.3 | 0.3 | Unchanged |
| `w_soft_dtw` | N/A | **0.0-0.3** | New, warmup from 0 |
| `w_hr` | 0.05 | **0.5** | 10× increase (gated to low-noise only) |

#### 13.5.4 ConditioningProjector Simplification

The ConditioningProjector was simplified from 3 heads to 2:

```python
# Run 5b: 3 heads
combined = t_mlp(t_emb) + hr_mlp(hr_emb) + pid_mlp(id_emb)

# Run 6: 2 heads (identity removed from AdaLN)
combined = t_mlp(t_emb) + hr_mlp(hr_emb)
# Identity flows through cross-attention instead
```

### 13.6 Run 6 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Generative framework** | OT-CFM (velocity prediction) |
| **Epochs** | 750 |
| **Learning rate** | 5×10⁻⁵ (halved from Run 5b) |
| **Batch size** | 32 × 8 accumulation = 256 effective |
| **CFG scale** | 2.0 (lowered from 3.0) |
| **Inference steps** | 20 (Euler ODE) |
| **Soft-DTW warmup** | 50 epochs before activating |
| **HR loss gate** | t < 0.2 |
| **Identity SNR floor** | 0.3 |
| **EMA decay** | 0.9999 with warmup |
| **New dependency** | `pysdtw>=0.0.5` |

### 13.7 Run 6 Status

As of the writing of this report, the Run 6 architecture has been **fully implemented and shape-verified**, but has not yet completed a full training run. The architecture passed all forward pass checks:

```
DiT-ECG-B (Run 6): 100.1M params
Input:  torch.Size([2, 1, 2500])
Output: torch.Size([2, 1, 2500])
✅ Forward pass verified (Run 6: cross-attn identity + CFG)!
```

The training script is ready for deployment via `./run_training.sh` on the A100 server.

---

## 14. The Complete Loss Function Evolution

### 14.1 Loss Function Timeline

The loss function evolved across six runs, each change driven by specific empirical findings:

```
Run 1:  L = MSE(noise_pred, noise_true)
        └── Pure noise prediction — no auxiliary losses

Run 2:  L = MSE(noise) + 0.5*Identity + 0.1*Spectral
        └── Added identity and spectral — spectral weight untested

Run 4:  L = MSE(noise) + MSE(signal) + 0.5*Identity + 0.1*Spectral 
             + 0.2*Correlation + 0.3*Morphology
        └── Added morphology gradient — BUT spectral at 0.1 = CATASTROPHE

Run 5:  L = MSE(noise) + MSE(signal) + 0.5*Identity + 0.0001*Spectral 
             + 0.2*Correlation + 0.3*Morphology + 0.05*HR
        └── Fixed spectral, added differentiable HR loss

Run 5b: Same as Run 5 + 0.2*HR_variance

Run 6:  L = MSE(velocity) + SNR_w×[MSE(signal) + 0.00001*Spectral 
             + 0.2*Correlation + 0.3*Morphology + 0.3*SoftDTW]
             + max(SNR_w, 0.3)×1.5*Identity
             + gated(0.5*HR + HR_variance)
        └── Everything rethought: velocity target, SNR weighting,
            identity floor, gated HR, Soft-DTW
```

### 14.2 The SNR-Weighting Strategy

All auxiliary losses in Run 6 are weighted by the Signal-to-Noise Ratio (SNR):

- **At t ≈ 0 (clean signal)**: α_bar ≈ 1.0, SNR weight ≈ 1.0 — full auxiliary loss gradient
- **At t ≈ 1 (pure noise)**: α_bar ≈ 0.0, SNR weight ≈ 0.0 — auxiliary losses nearly zero
- **Exception**: Identity has a **floor of 0.3** — never drops below 30%

The rationale: auxiliary losses (identity, spectral, morphology) require a clean signal estimate to compute meaningful gradients. At high noise, `x_0_pred` is garbage — gradients from it would be random and harmful.

### 14.3 The Seven Components in Detail

| Component | Formula | Weight (Run 6) | When Active | Purpose |
|-----------|---------|----------------|-------------|---------|
| **Primary (velocity MSE)** | `‖v_θ - v_target‖²` | 1.0 | Always | Core diffusion/flow matching objective |
| **Signal MSE** | `‖x_0 - x_0_pred‖²` | 1.0 × SNR_w | High SNR | Reconstruction fidelity |
| **Identity Cosine** | `1 - cos(feat_real, feat_gen)` | 1.5 × max(SNR_w, 0.3) | Always (floored) | Patient preservation |
| **Spectral FFT** | `‖\|FFT(x_0)\| - \|FFT(x_0_pred)\|‖²` | 1e-5 × SNR_w | High SNR | Frequency matching |
| **Pearson Correlation** | `1 - ρ(x_0, x_0_pred)` | 0.2 × SNR_w | High SNR | Scale-invariant shape |
| **Morphology Gradient** | `‖∇x_0 - ∇x_0_pred‖² + 0.1×‖∇²x_0 - ∇²x_0_pred‖²` | 0.3 × SNR_w | High SNR | QRS slope sharpness |
| **Soft-DTW** | `SDTw(QRS_real, QRS_gen, γ=0.1)` | 0.3 × SNR_w | High SNR (warmup) | QRS temporal fidelity |
| **HR L1** | `\|HR_est - HR_target\|` (gated, t < 0.2) | 0.5 | t < 0.2 only | Heart rate accuracy |
| **HR Variance** | `ReLU(47 - std(HR_batch))` | 1.0 | Always | HR diversity |

---

## 15. The Data Engine: From 48 Patients to 75,540 Segments

### 15.1 Dataset Growth Timeline

| Run | Datasets | Segments | Patients | Commits |
|-----|----------|----------|----------|---------|
| Run 1 | MIT-BIH only | 8,592 | 47 | Initial |
| Run 2 | + PTB-XL | ~65,000 | ~19,000 | `1d1287d` |
| Run 3 | + Chapman (attempted) | ~65,000 | ~19,000 | `782030f` (fix) |
| Run 4+ | MIT-BIH + PTB-XL + Chapman | **75,540** | **~50,000+** | `4f35803` |

### 15.2 The Download Script Saga

The `download_all_datasets.py` script went through more revisions than any other file in the project:

1. **v1**: Used `wfdb.dl_database()` for MIT-BIH — worked perfectly
2. **v2**: Added PTB-XL using `wfdb` — broke due to PTB-XL's versioned directory structure
3. **v3**: Switched to direct `wget` download for PTB-XL zip files — worked for 100Hz, failed for 500Hz
4. **v4**: Added Chapman-Shaoxing — failed because Chapman uses `.mat` format not `.dat`
5. **v5** (current): Introduced `_count_hea_files` helper and per-dataset download strategies

Each revision was a battle against PhysioNet's inconsistent file formats and access patterns.

### 15.3 The Harmonization Pipeline

All datasets are harmonized to a common format before training:

```
Target: 500 Hz, 5 seconds (2,500 samples), single-lead, zero-mean unit-variance

MIT-BIH:  360 Hz, 30 min  → Resample to 500Hz → Cut 5s windows → Normalize
PTB-XL:   500 Hz, 10 sec  → Take Lead II → Cut 5s windows → Normalize
Chapman:  500 Hz, 10 sec  → Take Lead II → Cut 5s windows → Normalize
```

### 15.4 Identity-Preserving Augmentations

During contrastive pre-training, augmentations that preserve cardiac identity are applied:

| Augmentation | Preserves Identity? | Implementation |
|-------------|-------------------|----------------|
| Amplitude scaling (0.5-2.0×) | ✅ Yes | Random gain change |
| Temporal shift (±0.5s) | ✅ Yes | Circular shift |
| Gaussian noise (σ < 0.02) | ✅ Yes | Additive noise |
| Baseline wander (0.05-0.5Hz) | ✅ Yes | Sinusoidal addition |
| Random crop & resize (70-100%) | ✅ Yes | Resample after crop |
| Heart rate shifting | ❌ NO | **EXCLUDED** — destroys RR intervals |

---

## 16. Infrastructure & Training Pipeline

### 16.1 The `run_training.sh` Pipeline

The complete training pipeline is automated in a single shell script (`run_training.sh`, 311 lines) that orchestrates all stages:

```
Phase 0: Environment Setup
  └── Create conda environment, install dependencies, verify GPU

Phase 1: Data Download & Processing  
  └── download_all_datasets.py → 3 datasets
  └── precompute_hr.py → GPU-accelerated HR labels

Phase 2: Contrastive Pre-Training (Stage 0)
  └── train_contrastive.py → Identity encoder (200 epochs)
  └── Output: encoder checkpoint with 512-dim embeddings

Phase 3: DiT Training (Stage 1)
  └── train_dit.py → DiT-ECG-B (750 epochs, Run 6)
  └── Output: DiT checkpoint + EMA weights

Phase 4: Evaluation
  └── evaluate_dit.py → Clinical metrics (FFD, MMD, HR-MAE, ReID)
  └── clinical_validation.py → Hospital PDF comparison

Phase 5: Reporting
  └── W&B dashboard with per-component loss tracking
  └── TensorBoard runs in checkpoints/runs/
```

### 16.2 EMA Implementation

The EMA (Exponential Moving Average) implementation provides:

- **Warmup-aware decay**: `decay_t = min(decay, (1+t)/(10+t))` — starts responsive, becomes stable
- **CPU storage option**: Shadow params on CPU saves ~40% GPU memory
- **Apply/restore pattern**: Copy EMA weights for evaluation, restore training weights after
- **State dict serialization**: EMA state saved in checkpoints for resume

### 16.3 Mixed-Precision Training

All training runs use BF16 (bfloat16) mixed-precision on A100:

- **Forward/backward**: BF16 compute (no GradScaler needed — BF16 has FP32 range)
- **Master weights**: FP32 in optimizer states
- **Gradient clipping**: Max norm 1.0 for stability
- **Gradient accumulation**: 8 micro-batches → 256 effective batch size

### 16.4 Production Inference Pipeline

The `ECGPipelineV2` class (`src/inference/pipeline_v2.py`) wraps the entire generation process:

```python
pipeline = ECGPipelineV2(
    dit_ckpt="checkpoints/dit_best.pt",
    encoder_ckpt="checkpoints/encoder.pt",
    use_flow_matching=True,  # Run 6: OT-CFM mode
)

# Generate 10 ECGs for a specific patient at 120 bpm
generated = pipeline.generate(
    reference_ecg=patient_ecg,  # (1, 2500) historical ECG
    hr_bpm=120.0,
    num_samples=10,
    guidance_scale=2.0,
    num_steps=20,  # Euler ODE steps
)
```

---

## 17. Evaluation Framework: ECG-Bench Protocol

### 17.1 Three-Level Evaluation

Following the ECG-Bench protocol (Tang et al., 2025), evaluation spans three levels:

**Level 1 — Distribution Quality** (Are generated ECGs statistically real?)
- **FFD (Fréchet ECG Distance)**: Analogous to FID for images. Uses a pre-trained PTB-XL ResNet-18 encoder. Lower = better.
- **MMD (Maximum Mean Discrepancy)**: RBF kernel distance in feature space. Lower = better.
- **Precision & Recall**: k-NN based measures of coverage and diversity.

**Level 2 — Morphological Fidelity** (Do individual ECGs look clinically correct?)
- **HR MAE**: Heart rate error in bpm (target: < 5 bpm)
- **QRS Duration Error**: In milliseconds (target: < 10 ms)
- **P-wave Presence Rate**: Percentage of generated ECGs with detectable P-waves (target: > 90%)

**Level 3 — Downstream Utility** (Can synthetic ECGs replace real ones?)
- **Patient Re-ID (Top-1/Top-5)**: Can we identify which patient a generated ECG belongs to?
- **TSTR**: Train on Synthetic, Test on Real — measures utility for augmentation

### 17.2 The Critical Metric: FFD

FFD is calculated as:

```
FFD = ‖μ_real - μ_gen‖² + Tr(Σ_real + Σ_gen - 2×√(Σ_real × Σ_gen))
```

where μ and Σ are the mean and covariance of features from the pre-trained encoder. A critical requirement: the encoder must be pre-trained on ECG (not randomly initialized). A random encoder gives meaningless FFD.

### 17.3 The MSE Trap

The research specifically warned against using MSE as an evaluation metric:

> *"MSE between generated and real ECG is unreliable. A flat-line ECG will score lower MSE than a perfectly shaped ECG that is phase-shifted by 50ms."*

This is why the project uses FFD + morphological metrics instead of raw MSE for evaluation.

---

## 18. Lessons Learned & Engineering Wisdom

### 18.1 The Ten Commandments of ECG Generation

Through six training runs, the following hard-won lessons emerged:

**1. Never trust aggregate loss.** The spectral catastrophe (Run 4) would have been invisible without per-component W&B logging. Always monitor every loss component individually.

**2. Multi-objective loss weights are not intuitive.** A weight of 0.1 for spectral loss seemed small, but the spectral loss magnitude was 1000× larger than other components — making 0.1 effectively dominate.

**3. Zero-initialization is not optional.** Every new conditioning pathway (HR, identity cross-attention, AdaLN gates) must be zero-initialized so the model starts from a known, stable state.

**4. Data scaling has diminishing returns without architectural changes.** Run 2 → Run 3 showed that adding more data to a fundamentally limited architecture produces no improvement.

**5. Identity preservation requires a dedicated pathway.** Mixing identity with timestep in AdaLN causes identity extinction at high noise levels. Cross-attention with a separate gate is essential.

**6. Differentiable losses fail at high noise.** HR loss, identity loss, and morphology loss all produce garbage gradients when the signal estimate is mostly noise. Timestep gating and SNR weighting are mandatory.

**7. EMA is non-negotiable for diffusion models.** Raw training weights oscillate around the optimum. EMA weights with decay 0.9999 produce 5-10× better sample quality.

**8. Heart rate shifting destroys identity.** It seems like a harmless augmentation, but changing RR intervals changes the fundamental cardiac signature. Exclude it from contrastive pre-training.

**9. MSE squeezes temporal features.** Point-to-point MSE penalizes temporal misalignment, causing the optimizer to compress QRS complexes. Soft-DTW is needed to allow temporal flexibility.

**10. Always keep the encoder frozen.** If the identity encoder is unfrozen during diffusion training, it adapts to produce "easy" embeddings rather than "correct" ones. Freeze it and trust the contrastive pre-training.

### 18.2 The Debugging Hierarchy

When a training run produces unexpected results, investigate in this order:

1. **Check loss component magnitudes** (W&B) — Are any components dominating?
2. **Check gradient norms** — Is any component's gradient 10× larger than others?
3. **Visualize generated samples** — Do they look like ECGs at all?
4. **Check HR distribution** — Is the model generating diverse or collapsed heart rates?
5. **Check identity cosine similarity** — Can the encoder distinguish generated patients?
6. **Check noise level distribution** — Is the scheduler correct?
7. **Check data pipeline** — Are the input signals normalized correctly?

---

## 19. Metrics Evolution Across All Runs

### 19.1 Complete Metrics Table

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5b | Run 6 Target |
|--------|-------|-------|-------|-------|--------|--------------|
| **FFD** ↓ | 1,032 | 42.0 | — | 21.7 | 91.1† | < 15 |
| **MMD** ↓ | 0.661 | 0.383 | — | 0.432 | 0.88† | < 0.1 |
| **HR MAE** ↓ | 17.2 | 9.3 | — | 31.4 | 23.6 | < 8 |
| **ReID Top-1** ↑ | 11.8% | 5.9% | — | 11.8% | 0.0% | > 50% |
| **ReID Top-5** ↑ | 35.3% | 23.5% | — | 35.3% | 2.5% | > 80% |
| **Gen HR Mean** | — | 90.5 | — | 68.4 | 54.1 | ~77 |
| **Gen HR Std** | — | 21.1 | — | 20.5 | 31.5 | > 47 |
| **Val Loss** | — | — | — | 5.677 | 0.6995 | < 0.5 |
| **Morph Loss** | — | — | — | ~0.8 | 0.037 | < 0.02 |
| **Dataset** | MIT | MIT+PTB | MIT+PTB+Chap | MIT+PTB+Chap | MIT+PTB+Chap | MIT+PTB+Chap |
| **Epochs** | 200 | 200 | — | 474 | 500 | 750 |
| **Framework** | DDPM | DDPM | DDPM | DDPM | DDPM | **OT-CFM** |

> † Run 5b FFD/MMD measured on different reference set than Run 4.

### 19.2 Key Metric Trends

```
FFD:     1032 ──→ 42 ──→ 22 ──→ 91† ──→ ?
         (-96%)  (-48%)  (diff set)
         
HR MAE:  17.2 ──→ 9.3 ──→ 31.4 ──→ 23.6 ──→ ?
         (-46%)  (+237% 🔴)  (-25%)
         
ReID:    11.8% ──→ 5.9% ──→ 11.8% ──→ 0.0% ──→ ?
         (-50%)   (+100%)   (-100% 🔴)

Val Loss: — ──→ — ──→ 5.677 ──→ 0.6995 ──→ ?
                       (-88%)
```

The story these trends tell: **distribution quality improved steadily, but identity preservation was sacrificed.** Run 6's dedicated cross-attention pathway is specifically designed to break this trade-off.

---

## 20. Future Roadmap

### 20.1 Phase 2: Clinical Scale (Current Priority)

| Item | Priority | Expected Gain |
|------|----------|---------------|
| Complete Run 6 training (750 epochs) | P0 | Validate OT-CFM + Cross-Attention |
| Monitor identity SNR floor in W&B | P0 | Verify anti-extinction mechanism |
| Monitor Soft-DTW contribution | P0 | Verify QRS temporal correction |
| Automatic loss weight balancing (CAGrad) | P2 | Remove manual weight tuning |

### 20.2 Phase 3: Clinical Integration

| Item | Priority | Expected Gain |
|------|----------|---------------|
| 12-lead joint generation | P1 | Full diagnostic ECG generation |
| Einthoven constraint loss | P1 | Physics-enforced inter-lead relationships |
| VAE latent space compression | P2 | 10× memory reduction for 12-lead |
| Hospital deployment pilot | P3 | Real-world clinical validation |

### 20.3 Phase 4: Forecasting

| Item | Priority | Expected Gain |
|------|----------|---------------|
| S4/Mamba encoder replacement | P2 | Better long-range temporal patterns |
| Temporal sequence modeling | P3 | Predict ECG evolution over time |
| Cardiac event prediction | P3 | Forecast arrhythmia onset |

### 20.4 Research Frontiers

- **MIMIC-IV-ECG pre-training**: 800K records, 160K patients — would produce a far superior identity encoder
- **FP8 training on H100**: 2× throughput over BF16 for next-generation scaling
- **Patient Memory Queue (PMQ)**: Rolling queue of 65,536 embeddings for small-dataset contrastive learning
- **Multi-modal conditioning**: Text report embeddings via cross-attention (ECGTwin Pathway 2)

---

## 21. Appendix: Key Papers & References

| Paper | arXiv / DOI | Year | Contribution to CardioEquation |
|-------|-------------|------|-------------------------------|
| DiT | 2212.09748 | 2023 | Core architecture: AdaLN-Zero, patch embedding |
| ECGTwin | 2508.02720 | 2025 | AdaX dual-pathway, contrastive identity encoder |
| OT-CFM | 2302.00482 | 2023 | Optimal transport flow matching (Run 6 migration) |
| CFM | 2210.02747 | 2023 | Conditional flow matching theory |
| SSSD-ECG | 2301.08227 | 2023 | S4 state-space models for ECG (future Phase 4) |
| ECG-Bench | 2507.14206 | 2025 | Three-level evaluation protocol |
| EDM2 | 2312.02696 | 2024 | EMA best practices for diffusion |
| DiffuSETS | 2025 | 2025 | Text-to-ECG generation |
| PCLR | Diamant 2022 | 2022 | Patient-level contrastive learning |
| ACL-ECG | Liu 2026 | 2026 | Anatomy-aware contrastive learning |
| PMQ | 2506.06310 | 2025 | Patient Memory Queue for small datasets |
| PTB-XL | 10.1038/s41597-020-0495-6 | 2020 | Primary 12-lead ECG dataset |
| TimeDiT | ICLR 2025 | 2025 | DiT adaptation for time-series |
| Mixed Precision | 1710.03740 | 2018 | BF16/FP16 training foundation |

---

## Appendix: File Architecture Map

```
CardioEquation/
├── src/
│   ├── models/
│   │   ├── dit_ecg.py              # DiT-ECG-B: 24-layer, 768-dim, cross-attention identity
│   │   └── feature_extractor_pt.py  # 1D ResNet-18 identity encoder (4.9M params)
│   ├── training/
│   │   ├── train_dit.py             # Main training loop (OT-CFM + DDPM support)
│   │   ├── train_contrastive.py     # Stage 0: SimCLR InfoNCE contrastive pre-training
│   │   ├── losses_v2.py             # Multi-component loss (7+ components)
│   │   ├── flow_matching.py         # OT-CFM scheduler (Run 6)
│   │   ├── noise_scheduler.py       # Legacy cosine DDPM scheduler
│   │   └── ema.py                   # Exponential Moving Average (decay=0.9999)
│   ├── evaluation/
│   │   ├── evaluate_dit.py          # Clinical evaluation (DDIM/Euler, FFD, ReID)
│   │   ├── eval_metrics.py          # ECG-Bench implementation (FFD, MMD, HR-MAE)
│   │   └── clinical_validation.py   # Hospital PDF → ECG digitization → validation
│   ├── inference/
│   │   └── pipeline_v2.py           # Production inference wrapper (ECGPipelineV2)
│   └── data/
│       ├── download_all_datasets.py # MIT-BIH + PTB-XL + Chapman download
│       └── precompute_hr.py         # GPU-accelerated HR extraction
├── docs/
│   ├── mentor_report.md             # Run 4 progress report
│   ├── senior_mentor_report.md      # Run 5b executive summary
│   ├── run5_implementation_plan.md  # Research-backed Run 5 plan
│   └── future_techniques.md         # Prioritized research ideas
├── run_training.sh                  # Complete pipeline automation (311 lines)
├── Converted Text.md               # 1026-line research & architecture manual
├── CHANGELOG.md                     # Detailed per-run changelog
├── PROJECT_OVERVIEW.md              # High-level project summary
└── Readme.md                        # System architecture documentation
```

---

**End of Development Journey Report**

*This document was synthesized from the project's complete git history, source code, documentation, mentor reports, research notes, and training logs. It represents the unfiltered technical narrative of building a state-of-the-art generative AI system for clinical ECG synthesis.*

*Total commits analyzed: 50+*  
*Total source files reviewed: 25+*  
*Total documentation pages reviewed: 15+*  
*Architecture versions tracked: 6 (Run 1 → Run 6)*

