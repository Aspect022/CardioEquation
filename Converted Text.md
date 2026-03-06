CardioEquation

Research & Architecture Manual

State-of-the-Art ECG Digital Twin via Conditional Diffusion

10 Research Topics Covered

1\. ECGTwin & AdaX Dual-Pathway Conditioning

2\. Diffusion Transformers (DiT) for 1D ECG Signals

3\. SSSD-ECG & S4/Mamba State-Space Diffusion

4\. Flow Matching vs DDPM for Biomedical Signals

5\. Contrastive Learning for ECG Patient Identity

6\. Multi-Lead ECG Generation Strategies

7\. Large-Scale Public ECG Datasets

8\. Evaluation Metrics for Generated ECG

9\. EMA (Exponential Moving Average) for Diffusion

10\. Mixed-Precision Training & Gradient Accumulation

Compiled March 2026  •  Unlimited A100/H100 Compute Target

 

1\. ECGTwin & AdaX Condition Injector

Paper: ECGTwin: Personalized ECG Digital Twin via Conditional Diffusion Models (2025)  |  arXiv:2508.02720

1.1  The AdaX Dual-Pathway Architecture

AdaX (Adaptive Cross-Attention) replaces the single-pathway conditioning used in standard FiLM or AdaLN with two independent pathways that handle fundamentally different types of conditioning signals:

Pathway Input Signal Mechanism Output

Pathway 1 Identity vector (512-dim) + timestep t Adaptive LayerNorm (adaLN) via MLP 6 modulation scalars per block: γ₁,β₁,α₁,γ₂,β₂,α₂

Pathway 2 Clinical report / label embeddings Cross-attention (nomic-embed-text-v1.5) Key/Value pairs injected into self-attn

1.1.1  AdaLN Modulation (Pathway 1)

The MLP accepts a concatenation of the 512-dim identity vector and the sinusoidal timestep embedding, producing 6 scalars per transformer block:

\# AdaLN modulation - Pathway 1

class AdaLNPathway(nn.Module):

    def \_\_init\_\_(self, d\_identity=512, d\_model=768):

        super().\_\_init\_\_()

        self.mlp = nn.Sequential(

            nn.Linear(d\_identity + d\_model, d\_model \* 4),

            nn.SiLU(),

            nn.Linear(d\_model \* 4, 6 \* d\_model),  # 6 scalars × d\_model

        )

        nn.init.zeros\_(self.mlp\[-1].weight)

        nn.init.zeros\_(self.mlp\[-1].bias)  # zero-init for training stability

    def forward(self, z\_id, t\_emb):

        cond = torch.cat(\[z\_id, t\_emb], dim=-1)

        shifts\_scales = self.mlp(cond)

        # returns (gamma1, beta1, alpha1, gamma2, beta2, alpha2)

        return shifts\_scales.chunk(6, dim=-1)

1.1.2  Cross-Attention (Pathway 2)

Clinical text reports or diagnostic label embeddings are tokenized via nomic-embed-text-v1.5 (768-dim) and injected as Key/Value pairs into each transformer block's cross-attention layer:

\# Cross-Attention conditioning - Pathway 2

class CrossAttentionPathway(nn.Module):

    def \_\_init\_\_(self, d\_model=768, d\_context=768, n\_heads=8):

        super().\_\_init\_\_()

        self.cross\_attn = nn.MultiheadAttention(d\_model, n\_heads, batch\_first=True)

        self.kv\_proj = nn.Linear(d\_context, 2 \* d\_model)

    def forward(self, x, context):  # x: \[B,T,D], context: \[B,L,D\_ctx]

        kv = self.kv\_proj(context)  # \[B,L,2D]

        k, v = kv.chunk(2, dim=-1)

        out, \_ = self.cross\_attn(x, k, v)

        return out

1.2  Comparison: Conditioning Mechanisms

Method How it works Pros Cons

FiLM γ·x + β (channel-wise scale/shift) Simple, fast No residual gating

AdaLN Scale/shift after LayerNorm Stable, DiT standard Single signal only

AdaLN-Zero + zero-init gate α on residual Better training stability Still single signal

Cross-Attn Q from x, K/V from context Handles sequential context Cannot modulate norm stats

AdaX (ECGTwin) adaLN for scalars + XAttn for context — dual path Best of both worlds; handles identity+text jointly Higher param count

⚠️  Key Pitfall

NEVER concatenate identity vector + text embeddings into a single adaLN pathway. The different

semantic natures of scalar identity codes vs sequential text tokens require separate mechanisms.

Mixing them collapses the sequential structure of the text embeddings.

1.3  Stage 1: Patient Identity Encoder

ECGTwin uses a SimCLR-style patient contrastive pre-training to build the 512-dim identity vector:

\# InfoNCE loss for patient identity pre-training

def info\_nce\_loss(z\_i, z\_j, temperature=0.07):

    z\_i = F.normalize(z\_i, dim=-1)

    z\_j = F.normalize(z\_j, dim=-1)

    logits = torch.matmul(z\_i, z\_j.T) / temperature  # \[B,B]

    labels = torch.arange(len(z\_i), device=z\_i.device)

    loss\_ij = F.cross\_entropy(logits, labels)

    loss\_ji = F.cross\_entropy(logits.T, labels)

    return (loss\_ij + loss\_ji) / 2

1.4  Application to CardioEquation

Action items for upgrade: Replace your current single MLP conditioning in the 1D U-Net with the AdaX dual-pathway. Your existing 512-dim identity vector feeds Pathway 1 (adaLN). Diagnostic labels can feed Pathway 2 (cross-attention). Initialize α gate outputs to 0 at training start.

 

2\. Diffusion Transformers (DiT) for 1D ECG Signals

Key Papers: DiT (Peebles & Xie, arXiv:2212.09748, 2023); TimeDiT (ICLR 2025); Diffusion-TS (arXiv:2403.01742, 2024)

2.1  Patch Embedding for 1D ECG

DiT processes signals by splitting them into non-overlapping patches. For a 2500-sample ECG at 500Hz (5 seconds), patch size P determines the sequence length fed to the transformer:

Patch Size P Seq Length (2500/P) Receptive Field Best For Complexity

5 500 tokens 10ms Fine morphology (P-wave) O(250K) — HIGH

10 ✅ Recommended 250 tokens 20ms Balance: detail + context O(62K) — GOOD

20 125 tokens 40ms Rhythm, morphology level O(15K) — FAST

50 50 tokens 100ms Beat-level only O(2.5K) — VERY FAST

2.2  Model Configurations

Variant Params Layers d\_model Heads Target Dataset

DiT-ECG-S 25M 12 384 6 MIT-BIH, small experiments

DiT-ECG-B ✅ 85M 24 768 12 PTB-XL + CPSC2018 (recommended)

DiT-ECG-L 340M 24 1024 16 Multi-dataset (60K+ records)

DiT-ECG-XL 670M 28 1152 16 MIMIC-IV-ECG (800K records)

2.3  DiT Block Implementation (ECG-Adapted)

class ECGDiTBlock(nn.Module):

    def \_\_init\_\_(self, d\_model=768, n\_heads=12, mlp\_ratio=4):

        super().\_\_init\_\_()

        self.norm1 = nn.LayerNorm(d\_model, elementwise\_affine=False)

        self.norm2 = nn.LayerNorm(d\_model, elementwise\_affine=False)

        self.attn = nn.MultiheadAttention(d\_model, n\_heads, batch\_first=True)

        self.mlp = nn.Sequential(

            nn.Linear(d\_model, d\_model \* mlp\_ratio),

            nn.GELU(), nn.Linear(d\_model \* mlp\_ratio, d\_model)

        )

        # adaLN-Zero: MLP produces 6 scalars, zero-initialized

        self.adaLN\_mlp = nn.Sequential(

            nn.SiLU(), nn.Linear(d\_model, 6 \* d\_model)

        )

        nn.init.zeros\_(self.adaLN\_mlp\[-1].weight)

        nn.init.zeros\_(self.adaLN\_mlp\[-1].bias)

    def forward(self, x, cond):

        g1, b1, a1, g2, b2, a2 = self.adaLN\_mlp(cond).chunk(6, dim=-1)

        # Attention sub-block

        h = self.norm1(x)

        h = h \* (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)

        attn\_out, \_ = self.attn(h, h, h)

        x = x + a1.unsqueeze(1) \* attn\_out

        # FFN sub-block

        h = self.norm2(x)

        h = h \* (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)

        x = x + a2.unsqueeze(1) \* self.mlp(h)

        return x

💡  Latent vs Direct Diffusion

Direct diffusion (operating on raw 1D signal) is preferred for single-lead ECG where signal fidelity

is critical. Use latent diffusion (VAE compression \~10×) only for 12-lead joint generation to manage

memory. CRITICAL: If using a VAE, set KL weight β < 0.01 to avoid over-smoothing QRS complexes.

 

3\. SSSD-ECG & S4-Based Diffusion Models

Key Papers: SSSD-ECG (Alcaraz & Strodthoff, arXiv:2301.08227, 2023); SSSD-ECG-nle (arXiv:2407.11108, 2024); SSSDS4 (arXiv:2208.09399, 2022)

3.1  Architecture Specifications

Hyperparameter SSSD-ECG Value Notes

Residual layers 36 3 groups of 12 each

Channel width 256 Main residual channel dim

Diffusion embedding dims 128 / 256 / 256 Per group, 3-level pyramid

S4 layers per block 2 Bidirectional S4

Diffusion timesteps 200 Linear schedule

Noise schedule β 0.0001 → 0.02 Linear, same as original DDPM

Conditioning Concatenate condition to input Simple concat, no cross-attn

3.2  Why S4 for ECG?

• O(N log N) via FFT convolution vs O(N²) attention — critical for long ECG sequences (5000+ samples)

• HiPPO initialization captures long-range recurrent patterns (essential for PQ/QT intervals)

• Naturally handles variable-length sequences without positional embedding issues

• State space formulation mirrors the underlying physiological dynamical system of the heart

3.3  S4 vs Mamba for ECG Generation

Property S4 / S5 Mamba

State spaces Fixed (LTI) Input-selective (selective SSM)

Arrhythmia detection Good Better (selective gates)

Generation stability High — production recommended Less stable for generation

Long sequences Excellent (FFT conv) Good (parallel scan)

Recommendation Use for production generation Use for discriminative tasks

3.4  SSSD-ECG-nle: Improved Conditioning

The nle (non-label-exclusive) variant uses separate embeddings for positive vs neutral diagnostic labels instead of a collapsed binary:

\# SSSD-ECG-nle conditioning: separate pos/neutral embeddings

c\_pos = label\_embedder\_positive(positive\_labels)   # present diagnoses

c\_neut = label\_embedder\_neutral(neutral\_labels)    # absent / normal

condition = torch.cat(\[c\_pos, c\_neut], dim=-1)

\# Then concat with noisy signal for diffusion

3.5  Hybrid Recommendation for CardioEquation

🔬  Architecture Recommendation

Stage 1 (Encoder): S4 or S5 encoder for patient identity extraction from raw ECG.

  S4 captures long-range temporal structure (PQ, QT, RR intervals) better than ResNet-18.

Stage 2 (Diffusion): DiT-ECG-B with AdaX conditioning for generation.

  DiT scales better to large datasets than SSSD; AdaX handles identity+label conditioning jointly.

This hybrid gives you the best of both: S4 identity capture + DiT generation quality.

 

4\. Flow Matching vs DDPM for Biomedical Signals

Key Papers: CFM (Lipman et al., arXiv:2210.02747, 2023); OT-CFM (Tong et al., arXiv:2302.00482, 2023); CFM-TS (ICML 2024)

4.1  Core Differences

Property DDPM/DDIM OT-CFM

Training target Predict noise ε at timestep t Predict velocity field v = x₁ - x₀

Path shape Stochastic (Markov chain) Straight-line optimal transport

Inference steps 20-1000 (DDIM: 20-50) 10-20 ODE steps

Gradient variance High (random t sampling) Low (straight paths)

Source distribution Standard Gaussian Minibatch OT coupled pairs

Training complexity Simple loss: ||ε\_θ - ε||² Requires Sinkhorn OT: O(B² log B)

4.2  OT-CFM Training Loop

from ot import emd2  # POT library: pip install POT

def sample\_ot\_coupling(x0, x1):

    """Minibatch optimal transport coupling."""

    B = x0.shape\[0]

    a = torch.ones(B) / B  # uniform source

    b = torch.ones(B) / B  # uniform target

    # Compute cost matrix (L2 distance between samples)

    M = torch.cdist(x0.view(B,-1), x1.view(B,-1)) \*\* 2

    # Solve OT (returns transport plan)

    T = emd2(a.numpy(), b.numpy(), M.numpy())

    T = torch.from\_numpy(T) \* B

    # Sample pairs from transport plan

    idx = torch.multinomial(T.flatten(), B, replacement=True)

    i, j = idx // B, idx % B

    return x0\[i], x1\[j]

def cfm\_loss(model, x1, condition):

    x0 = torch.randn\_like(x1)        # source noise

    x0\_coupled, x1\_coupled = sample\_ot\_coupling(x0, x1)

    t = torch.rand(x1.shape\[0], 1, device=x1.device)

    xt = (1 - t) \* x0\_coupled + t \* x1\_coupled  # linear interpolation

    ut = x1\_coupled - x0\_coupled                # target velocity

    vt = model(xt, t.squeeze(-1), condition)     # predicted velocity

    return F.mse\_loss(vt, ut)

⚠️  Critical Pitfall: Parameterization

CFM predicts velocity v\_θ (direction x₀→x₁), NOT noise ε\_θ.

Never mix parameterizations — if you use CFM loss, inference must use ODE solver,

not the DDPM reverse Markov chain. Using DDPM sampling with CFM-trained weights

produces pure noise.

4.3  When to Use Each

• DDPM/DDIM: Best for initial prototyping — well-understood, stable, abundant ECG literature reference points

• OT-CFM: Best for production — 5-10× fewer inference steps, lower memory, better for real-time applications

• Recommendation: Start with DDPM for your CardioEquation baseline, then migrate to OT-CFM once architecture is validated

 

5\. Contrastive Learning for ECG Patient Identity

Key Papers: ACL-ECG (Liu et al., Sensors 26(3):1080, 2026); PCLR (Diamant et al., PLOS Comp Bio, 2022); PMQ (arXiv:2506.06310, 2025); TA-PCLR (ICLR 2025)

5.1  Best Augmentations for Patient Identity Preservation

Identity-preserving augmentations must change recording artifacts without changing patient-specific physiology:

Augmentation Preserves Identity? Parameters Rating

Neighboring segment sampling Yes ✅ Adjacent 5s windows same patient ⭐⭐⭐⭐⭐ Best

Amplitude scaling Yes ✅ α \~ U(0.5, 2.0) ⭐⭐⭐⭐

Temporal shifting Yes ✅ Δt \~ U(-0.5s, +0.5s) ⭐⭐⭐⭐

Random scale cropping Yes ✅ Crop 70-100% then resize ⭐⭐⭐⭐

Cardiac-cycle masking Partial ⚠️ Mask 10-30% of signal ⭐⭐⭐

Gaussian noise Yes ✅ σ \~ U(0.0, 0.02) ⭐⭐

Baseline wander Yes ✅ Sinusoidal 0.05-0.5Hz ⭐⭐

Heart rate shift NO ❌ Resampling changes RR interval AVOID — destroys identity

5.2  Temperature τ Selection

τ Value Effect Recommendation

0.05 Very sharp distribution — high-gradient on hard negatives Risk: instability if negatives too similar

0.07 (standard) SimCLR/MoCo default — proven stable Default for MIT-BIH and PTB-XL

0.1 Softer, more tolerant of label noise Use when data is noisy or inter-patient similarity is high

0.2+ Too soft — poor identity separation Avoid for ECG identity tasks

5.3  Projection Head & Encoder Setup

class ECGContrastiveEncoder(nn.Module):

    def \_\_init\_\_(self, backbone, d\_backbone=512, d\_proj=256):

        super().\_\_init\_\_()

        self.encoder = backbone  # Your ResNet-18 or S4 encoder

        # 2-layer projection head: discard after pre-training

        self.projector = nn.Sequential(

            nn.Linear(d\_backbone, 512), nn.BatchNorm1d(512), nn.ReLU(),

            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),

            nn.Linear(512, d\_proj)  # final projection dim

        )

    def forward(self, x, return\_proj=True):

        z = self.encoder(x)  # \[B, 512] identity vector

        if return\_proj:

            return F.normalize(self.projector(z), dim=-1)  # for contrastive loss

        return z  # for downstream diffusion conditioning

5.4  Patient Memory Queue (PMQ) — Critical for MIT-BIH

🔑  PMQ for Small Datasets

MIT-BIH has only 48 patients — a batch of size 32 may contain only 3-4 patients,

making InfoNCE loss degenerate (too few negatives). PMQ maintains a rolling queue

of 65,536 embeddings across all batches, providing dense negatives regardless of batch size.

This is essential for any ECG dataset with fewer than 1000 patients.

Use: github.com (search: patient memory queue ECG contrastive)

5.5  ACL-ECG: Anatomy-Aware Contrastive Learning

For 12-lead ECG, ACL-ECG groups leads into anatomical regions and applies region-level contrastive objectives:

Region Leads Anatomical View

Anterior V1, V2, V3, V4 Anterior left ventricle wall

Inferior II, III, aVF Inferior left ventricle

Lateral I, aVL, V5, V6 Lateral left ventricle

Septal V1, V2 Interventricular septum

 

6\. Multi-Lead ECG Generation Strategies

Key Papers: DiffuSETS (2025); ECG-JEPA (Kim et al., arXiv:2410.08559, 2024); P2Es (arXiv:2509.25480, 2025)

6.1  Strategy Comparison

Strategy Inter-lead Correlation Complexity Recommendation

Per-lead independent ❌ Violated — leads generated independently Low Avoid for 12-lead ECG

Sequential conditioning ⚠️ Error accumulates lead-to-lead Medium Acceptable for 2-3 leads only

Joint single-model ✅ ✅ All correlations preserved High Best for 12-lead generation

Hierarchical limb→precordial ✅ Physics-informed Medium-High Best for clinical validity

6.2  Joint Multi-Channel Architecture

\# Treat 12 leads as channels — patch across time, attend across leads

class MultiLeadECGDiT(nn.Module):

    def \_\_init\_\_(self, n\_leads=12, patch\_size=10, d\_model=768, n\_blocks=24):

        super().\_\_init\_\_()

        self.patch\_embed = nn.Conv1d(

            n\_leads, d\_model, kernel\_size=patch\_size, stride=patch\_size

        )

        self.blocks = nn.ModuleList(\[ECGDiTBlock(d\_model) for \_ in range(n\_blocks)])

        self.unpatch = nn.ConvTranspose1d(

            d\_model, n\_leads, kernel\_size=patch\_size, stride=patch\_size

        )

    def forward(self, x, t, cond):  # x: \[B, 12, T]

        x = self.patch\_embed(x).transpose(1, 2)  # \[B, T/P, D]

        for block in self.blocks:

            x = block(x, cond)

        x = self.unpatch(x.transpose(1, 2))  # \[B, 12, T]

        return x

6.3  Einthoven Constraint Auxiliary Loss

Enforce the physical laws of lead relationships as an auxiliary training loss:

def einthoven\_loss(leads):

    """Enforce Einthoven's triangle and augmented lead relationships."""

    I, II, III = leads\[:,0], leads\[:,1], leads\[:,2]

    aVR, aVL, aVF = leads\[:,3], leads\[:,4], leads\[:,5]

    # Einthoven's triangle: Lead II = Lead I + Lead III

    loss\_triangle = F.mse\_loss(II, I + III)

    # Augmented leads

    loss\_aVR = F.mse\_loss(aVR, -(I + II) / 2)

    loss\_aVL = F.mse\_loss(aVL, (I - III) / 2)

    loss\_aVF = F.mse\_loss(aVF, (II + III) / 2)

    return loss\_triangle + loss\_aVR + loss\_aVL + loss\_aVF

\# Add to total training loss with small weight

total\_loss = diffusion\_loss + 0.1 \* einthoven\_loss(generated\_leads)

 

7\. Large-Scale Public ECG Datasets

7.1  Dataset Inventory

Dataset Records Patients Leads Duration Hz Labels Best Use

MIT-BIH 48 47 2 30 min 360 Arrhythmia Evaluation baseline

PTB-XL ✅ 21,837 18,885 12 10s 100/500 71 SCP-ECG Fine-tuning, conditional gen

PTB 549 290 15 \~2 min 1000 MI High-res fine-tuning

CPSC2018 6,877 6,877 12 10s 500 9 arrhythmia Arrhythmia augmentation

Georgia 10,344 10,344 12 10s 500 27 diagnoses Diverse label augmentation

Chapman-Shaoxing 45,152 45,152 12 10s 500 11 rhythm Large-scale rhythm data

INCART 75 32 12 30 min 257 Arrhythmia Long-duration ECG

MIMIC-IV-ECG ✅ 800K+ 160K+ 12 10s 500 Minimal Identity encoder pre-training

CODE-15% 345,779 233,770 12 7-10s 400 6 conditions Large-scale pre-training

7.2  Recommended Training Stack

• Stage 1 — Identity Encoder Pre-training: MIMIC-IV-ECG (800K records, 160K+ patients). Use patient\_id for InfoNCE contrastive loss. No labels needed.

• Stage 2 — Conditional Generation Fine-tuning: PTB-XL (21,837 records, 71 SCP-ECG labels) + PTB. Rich label schema enables strong clinical conditioning.

• Stage 3 — Augmentation: CPSC2018 + Georgia + Chapman-Shaoxing (\~60K records). Diverse arrhythmia coverage.

• Evaluation Only: MIT-BIH. Use for evaluating morphological metrics and arrhythmia generation quality.

7.3  Harmonization Pipeline

Different datasets have incompatible sampling rates and durations. Harmonize before combining:

import scipy.signal

import numpy as np

class HarmonizedECGDataset:

    TARGET\_HZ = 500

    TARGET\_LEN = 5000  # 10 seconds at 500Hz

    def harmonize(self, ecg, src\_hz):

        # 1. Resample to 500Hz

        if src\_hz != self.TARGET\_HZ:

            ecg = scipy.signal.resample\_poly(

                ecg, self.TARGET\_HZ, src\_hz, axis=-1

            )

        # 2. Standardize length: pad or trim to 5000 samples

        L = ecg.shape\[-1]

        if L < self.TARGET\_LEN:

            pad = self.TARGET\_LEN - L

            ecg = np.pad(ecg, ((0,0),(0,pad)), mode='edge')

        elif L > self.TARGET\_LEN:

            ecg = ecg\[..., :self.TARGET\_LEN]

        # 3. Normalize per-lead to zero mean, unit variance

        ecg = (ecg - ecg.mean(-1, keepdims=True)) / (ecg.std(-1, keepdims=True) + 1e-6)

        return ecg

 

8\. Evaluation Metrics for Generated ECG

Key Papers: ECG-Bench (Tang et al., arXiv:2507.14206, 2025); ECGTwin three-level evaluation protocol (arXiv:2508.02720, 2025)

8.1  Three-Level Evaluation Protocol

Level Category Metric Tool / Method Target

1 Distribution Quality FFD (Fréchet ECG Distance) PTB-XL ResNet18 encoder + sqrtm Lower = better

1 Distribution Quality Precision & Recall k-NN in feature space, k=5 Both near 1.0

1 Distribution Quality MMD RBF kernel on features Lower = better

2 Morphological Fidelity HR-MAE (heart rate error) neurokit2.ecg\_rate() < 5 bpm

2 Morphological Fidelity QRS Duration Error neurokit2.ecg\_delineate() < 10ms

2 Morphological Fidelity QT Interval Error neurokit2 + Pan-Tompkins < 20ms

2 Morphological Fidelity P-wave Presence Rate neurokit2.ecg\_findpeaks() > 90%

3 Downstream Utility TSTR accuracy Train on Synthetic, Test Real Near real-data baseline

3 Downstream Utility TRTS accuracy Train Real, Test Synthetic Near real-data baseline

3 Downstream Utility Patient Re-ID (top-1/top-5) Cosine similarity of embeddings High for conditioned gen

8.2  FFD Implementation

import torch, numpy as np

from scipy.linalg import sqrtm

def compute\_ffd(real\_ecgs, fake\_ecgs, encoder):

    """Fréchet ECG Distance using pre-trained ECG encoder."""

    encoder.eval()

    with torch.no\_grad():

        mu\_r = encoder(real\_ecgs).cpu().numpy()   # \[N, D]

        mu\_f = encoder(fake\_ecgs).cpu().numpy()   # \[N, D]

    m\_r, C\_r = mu\_r.mean(0), np.cov(mu\_r.T)   # real mean, cov

    m\_f, C\_f = mu\_f.mean(0), np.cov(mu\_f.T)   # fake mean, cov

    # Fréchet distance: ||μ\_r - μ\_f||² + Tr(C\_r + C\_f - 2\*sqrt(C\_r·C\_f))

    diff = m\_r - m\_f

    covmean = sqrtm(C\_r @ C\_f)

    if np.iscomplexobj(covmean):

        covmean = covmean.real  # numerical noise

    ffd = diff @ diff + np.trace(C\_r + C\_f - 2 \* covmean)

    return float(ffd)

\# CRITICAL: encoder must be pre-trained on ECG (e.g. PTB-XL ResNet18)

\# NOT a randomly initialized encoder — that gives meaningless FFD

⚠️  Common Metric Pitfall

MSE between generated and real ECG is unreliable as an evaluation metric.

A flat-line ECG will score lower MSE than a perfectly shaped ECG that is phase-shifted by 50ms.

Always use FFD + morphological metrics (HR-MAE, QRS duration) for evaluation.

Also: if using a random encoder for FFD, the metric is meaningless — always use a pre-trained ECG encoder.

 

9\. Exponential Moving Average (EMA) for Diffusion Models

Key Papers: EDM2 (Karras et al., arXiv:2312.02696, NeurIPS 2024, FID 1.81 on ImageNet-512); EMA Dynamics (arXiv:2411.18704, 2024)

9.1  Why EMA is Non-Negotiable

Diffusion training draws random timestep t and noise ε at each step. This makes the loss surface highly stochastic — raw model weights oscillate around the optimum rather than converging to it. EMA smooths the weight trajectory to produce a stable estimate of the true optimum.

📊  Impact on Sample Quality

Using raw training weights θ\_train for inference gives FID 5-10× WORSE than EMA weights θ\_ema.

This effect is even stronger for ECG because morphological details (P-wave shape, QRS slope) are

highly sensitive to weight noise. Always use θ\_ema for all evaluations and inference.

9.2  Decay Rate Selection

Training Steps Recommended Decay Half-life (steps) Notes

< 100K 0.999 693 Responsive to early weight changes

100K – 500K ✅ 0.9999 6,931 Standard CardioEquation target

500K – 2M 0.9999 6,931 Use with EMA warmup (see below)

\> 2M 0.99995 – 0.99999 13K – 69K Large-scale (MIMIC-IV-ECG scale)

9.3  EMA with Warmup (Hugging Face Approach)

from diffusers import EMAModel

\# Create EMA model with warmup — decay adapts during early training

ema\_model = EMAModel(

    parameters=model.parameters(),

    decay=0.9999,

    use\_ema\_warmup=True,   # decay\_t = min(0.9999, (1+t)/(10+t))

    power=2/3,             # warmup schedule exponent

    model\_cls=type(model),

    model\_config=model.config,

)

\# Training loop

for step, batch in enumerate(dataloader):

    optimizer.zero\_grad()

    loss = diffusion\_loss(model, batch)

    loss.backward()

    torch.nn.utils.clip\_grad\_norm\_(model.parameters(), 1.0)

    optimizer.step()

    ema\_model.step(model.parameters())  # update EMA after optimizer step

\# For evaluation: temporarily copy EMA weights to model

ema\_model.copy\_to(model.parameters())

evaluate(model)  # uses EMA weights

\# Then restore training weights if needed

9.4  Key EMA Rules

• Start EMA from iteration 0 with warmup for small datasets (MIT-BIH / PTB-XL)

• Store EMA model on CPU — saves 40% GPU memory during training; copy to GPU only for evaluation

• Do NOT apply EMA to BatchNorm statistics — EMA of BN stats causes distribution mismatch. Use GroupNorm or LayerNorm instead (DiT uses LayerNorm)

• At each evaluation checkpoint, compare EMA model vs raw model quality — EMA should always win

 

10\. Mixed-Precision Training & Gradient Accumulation

Key Papers: Mixed Precision Training (Micikevicius et al., arXiv:1710.03740, 2018); ZeRO (Rajbhandari et al., arXiv:1910.02054, 2020)

10.1  Precision Format Comparison

Format Use For Pros Cons / Notes

FP32 Master weights, optimizer states Full precision, no overflow 2× memory of BF16

BF16 ✅ A100/H100 Forward/backward compute Same range as FP32; no GradScaler needed Slightly less mantissa precision than FP16

FP16 Forward/backward (older GPUs) Half memory vs FP32 Overflow risk; requires GradScaler

FP8 (H100 only) Experimental — some linear layers 2× faster than BF16 on H100 Requires transformer\_engine library

10.2  Batch Size & Gradient Accumulation Table

Model GPUs Micro Batch Accum Steps Effective Batch Peak LR Notes

DiT-ECG-S (25M) 1×A100 64 4 256 3×10⁻⁴ Fine for MIT-BIH

DiT-ECG-B (85M) ✅ 1×A100 32 8 256 1×10⁻⁴ Recommended baseline

DiT-ECG-L (340M) 4×A100 16 8 512 1×10⁻⁴ DDP across 4 GPUs

DiT-ECG-XL (670M) 8×A100 16 4 512 5×10⁻⁵ FSDP/ZeRO-3 required

DiT-ECG-XL (670M) 8×H100 32 4 1024 5×10⁻⁵ BF16 + Flash Attn 2

10.3  Full Training Loop Template

from torch.cuda.amp import autocast

from diffusers import EMAModel

import torch

scaler = None  # Not needed for BF16 (no GradScaler)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight\_decay=0.01

)

scheduler = torch.optim.lr\_scheduler.CosineAnnealingLR(

    optimizer, T\_max=total\_steps, eta\_min=1e-5

)

ACCUM = 8  # gradient accumulation steps

optimizer.zero\_grad()

for global\_step, batch in enumerate(dataloader):

    with autocast(dtype=torch.bfloat16):  # BF16 for A100/H100

        x0 = batch\['ecg']           # clean ECG \[B, 1, T]

        cond = batch\['identity']    # 512-dim identity vector

        # Forward diffusion: sample t and add noise

        t = torch.randint(0, T, (x0.shape\[0],), device=x0.device)

        noise = torch.randn\_like(x0)

        xt = q\_sample(x0, t, noise)  # x\_t = sqrt(alpha\_bar)\*x0 + sqrt(1-alpha\_bar)\*noise

        # Predict noise (DDPM) or velocity (CFM)

        pred = model(xt, t, cond)

        loss = F.mse\_loss(pred, noise) / ACCUM  # scale for accumulation

    loss.backward()  # BF16 backward — no scaler needed

    if (global\_step + 1) % ACCUM == 0:

        torch.nn.utils.clip\_grad\_norm\_(model.parameters(), 1.0)

        optimizer.step()

        ema\_model.step(model.parameters())

        scheduler.step()

        optimizer.zero\_grad()

10.4  Common Failure Modes & Fixes

Problem Likely Cause Fix

Loss NaN / Inf FP16 overflow or LR too high Switch to BF16; add grad clipping; reduce LR 10×

Loss plateaus early LR too low or batch too small Increase LR; use gradient accumulation to reach eff batch ≥ 256

Generated ECG looks noisy Using raw weights at inference Enable EMA; increase DDIM steps to 50+

OOM on A100 80GB Micro batch too large Enable FSDP ZeRO-3; enable gradient checkpointing; halve micro batch

Gradient explosion Missing grad clipping or LR warmup Clip at 1.0; add 5-10K linear warmup steps; verify AdaLN zero-init

Mode collapse (all ECGs similar) Guidance scale too high Reduce classifier-free guidance to 1.5-3.0; add noise to conditioning

 

Appendix: CardioEquation Upgrade Roadmap

Prioritized action plan to upgrade from current U-Net+ResNet18 system to state-of-the-art:

Priority Action Expected Gain Effort Prerequisite

P1 ⭐ Replace U-Net with DiT-ECG-B (P=10, 24 layers, 768 dim) Large quality + scalability improvement Medium None

P1 ⭐ Implement AdaX dual-pathway (adaLN for identity, XAttn for labels) Large: better conditioning fidelity Medium DiT upgrade

P1 ⭐ Add EMA (decay=0.9999, warmup) to all training runs Large: 5-10× sample quality improvement Low None

P2 🔵 Upgrade dataset: PTB-XL + CPSC2018 + Chapman (>60K records) Large: diversity and label quality Medium Data pipeline

P2 🔵 Pre-train identity encoder with PMQ contrastive on MIMIC-IV-ECG Large: patient identity capture from 160K patients High MIMIC access

P2 🔵 Switch to BF16 mixed precision + grad accumulation to eff batch 256+ Medium: training stability, 2× throughput Low None

P3 🟢 Evaluate OT-CFM as DDPM replacement for inference speed Medium: 5-10× faster inference Medium P1 done

P3 🟢 Add Einthoven constraint loss for 12-lead upgrade Medium: clinical validity Low Multi-lead data

P3 🟢 Implement FFD + morphological metrics (HR-MAE, QRS duration) Critical for publication-quality evaluation Medium neurokit2

P4 🔬 Explore S4/Mamba encoder as ResNet18 alternative Potential: better long-range patterns High P1, P2 done

Key Papers Quick Reference

Paper arXiv / DOI Year Key Contribution

ECGTwin 2508.02720 2025 AdaX dual-pathway conditioning for personalized ECG generation

DiT 2212.09748 2023 Diffusion Transformer: adaLN-Zero, scalable architecture

TimeDiT ICLR 2025 2025 DiT adapted for time-series with temporal patch embedding

Diffusion-TS 2403.01742 2024 Transformer-based diffusion for multivariate time series

SSSD-ECG 2301.08227 2023 S4 structured state space models for conditional ECG generation

SSSD-ECG-nle 2407.11108 2024 Improved label conditioning with non-label-exclusive embeddings

CFM 2210.02747 2023 Conditional flow matching: straight-line paths, lower variance

OT-CFM 2302.00482 2023 Optimal transport coupling for faster, more stable flow matching

PCLR 10.1371/pcbi.1009862 2022 Patient-level contrastive learning for ECG representation

ACL-ECG 10.3390/s26031080 2026 Anatomy-aware contrastive learning with lead region grouping

PMQ 2506.06310 2025 Patient Memory Queue for small-dataset contrastive ECG learning

ECG-JEPA 2410.08559 2024 Joint ECG prediction architecture for multi-lead representation

ECG-Bench 2507.14206 2025 Comprehensive benchmark for ECG generation evaluation metrics

EDM2 2312.02696 2024 EMA best practices; FID 1.81 ImageNet-512; NeurIPS 2024

Mixed Precision 1710.03740 2018 BF16/FP16 training with loss scaling — foundational reference

PTB-XL 10.1038/s41597-020-0495-6 2020 Largest labeled 12-lead ECG dataset: 21,837 records, 71 SCP codes
