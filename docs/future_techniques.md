# Future Techniques We Can Try

> Ideas for improving CardioEquation beyond the current Phase 1 improvements.  
> Prioritized by expected impact and effort.

---

## 🔴 High Priority (Phase 2 — Next Sprint)

### 1. Explicit Heart Rate Conditioning
**What**: Inject HR as a first-class conditioning signal via AdaLN (alongside timestep + patient identity).  
**Why**: All 3 research sources agree this is the **single most impactful fix** for HR diversity collapse.  
**How**: Extract HR for each training sample → small MLP projects HR → add to timestep embedding.  
**Effort**: 1-2 days  
**Refs**: ECGTwin (2025), DiffuSETS (2025), Claude research §5.2

### 2. RR-Interval Consistency Loss
**What**: Differentiable loss that penalizes RR-interval mismatch between generated and real ECGs.  
**Why**: Directly targets HR diversity at the loss level.  
**How**: Soft R-peak detection via local max pooling + autocorrelation for mean RR estimation.  
**Effort**: 1 day  
**Refs**: Claude research §4.2, Gemini research §3

### 3. Spectral Coherence Loss (Upgraded)
**What**: Replace current FFT magnitude-only loss with full spectral coherence (phase + magnitude).  
**Why**: Phase consistency preserves HRV patterns that magnitude alone misses.  
**Effort**: 0.5 days

### 4. Pareto Optimization for Loss Weights
**What**: Multi-objective optimization to find optimal balance across 6+ loss components.  
**Why**: Fixed weights (1.0, 0.5, 0.3, etc.) are guesswork. Pareto frontier finds the true optimal tradeoffs.  
**How**: After Run 4 W&B analysis, run a Pareto sweep across weight combinations. Or use gradient-based methods like CAGrad/MGDA.  
**Effort**: 2-3 days  
**Note**: Mentor-suggested technique.

### 5. CFG Scale Sweep at Inference
**What**: Generate samples at CFG scales [1.0, 2.0, 3.0, 5.0, 7.5] and compare FFD/ReID/HR-std.  
**Why**: High CFG concentrates distribution → kills diversity.  
**Effort**: 0.5 days

---

## 🟡 Medium Priority (Phase 3 — Architecture Changes)

### 6. Conditional Flow Matching (CFM)
**What**: Replace DDPM noise prediction with velocity field prediction. Linear interpolation instead of forward diffusion.  
**Why**: 
- 10× faster inference (4-8 steps vs 50 DDIM steps)
- Diversity by construction (no DDIM determinism issue)
- Straight ODE paths = fewer morphological artifacts  
**How**: Change training loss from ε-prediction to v-prediction. Replace DDIM sampler with Euler ODE solver. DiT backbone stays identical.  
**Effort**: 1-2 weeks  
**Refs**: FlowECG (2025), Rectified Flow (Liu et al., 2023), all 3 research sources

### 7. Latent Space DiT (VAE Compression)
**What**: Compress ECG from 2500 → ~312 samples via a VAE, then run DiT in latent space.  
**Why**: 8× faster training, enables bigger models.  
**Refs**: PPGFlowECG (2025) — latent rectified flow outperformed data-space rectified flow.  
**Effort**: 1-2 weeks

### 8. Mel-Spectrogram Informed Loss (MIDT-ECG)
**What**: Compute mel-spectrogram of ECG, use as auxiliary training signal.  
**Why**: Enforces P-QRS-T morphological structure without explicit beat detection. Reduced interlead correlation error by 74% in MIDT-ECG.  
**Refs**: MIDT-ECG (2025, arXiv:2510)  
**Effort**: 2-3 days

---

## 🟢 Lower Priority (Long-Term / Research)

### 9. ESRGAN-Style Adversarial Discriminator
**What**: Add a GAN discriminator that classifies real vs generated ECGs.  
**Why**: Could improve fine-grained realism. ME-GAN (ICML 2022) showed this works for multi-lead ECG.  
**Risk**: Training instability, mode collapse. Not recommended until flow matching is stable.  
**Note**: Mentor-suggested technique. Better suited after CFM migration stabilizes.

### 10. Multi-Lead Generation (12-Lead)
**What**: Extend from single-lead to 12-lead ECG generation.  
**Why**: Clinical utility — most hospital ECGs are 12-lead.  
**Refs**: UniCardio (Nature MI, 2025) — DiT for multi-modal cardiovascular signals.  
**Effort**: 2-4 weeks

### 11. Clinical Text Conditioning
**What**: Condition generation on clinical text reports (e.g., "sinus tachycardia with ST elevation").  
**Why**: Enables targeted synthetic data generation for specific conditions.  
**Refs**: DiffuSETS (2025) — text-to-ECG generation, >90% accuracy vs real patterns.  
**Effort**: 2-3 weeks

### 12. Rectified Flow with 1-Step Distillation
**What**: Distill CFM model into a 1-2 step generator via ReFlow.  
**Why**: Real-time ECG generation for deployment.  
**Refs**: Rectified Flow (Liu et al., 2023), PPGFlowECG (2025)  
**Effort**: 1 week (after CFM is working)

### 13. RR-Sequence Conditioning
**What**: Instead of just mean HR, condition on the full RR interval sequence (4-8 intervals per 5s window).  
**Why**: Enables generation of ECGs with specific HRV patterns (e.g., atrial fibrillation with irregular RR).  
**Refs**: Claude research §5.2  
**Effort**: 2-3 days

### 14. Soft-DTW Waveform Loss
**What**: Differentiable Dynamic Time Warping as a loss function.  
**Why**: Unlike MSE, DTW correctly handles minor timing shifts without over-penalizing. Better for beat alignment.  
**Risk**: Computationally expensive (O(T²) per sample). Use with downsampling.  
**Effort**: 1-2 days

### 15. HR Variance Regularization Loss
**What**: Batch-level loss that penalizes when generated HR standard deviation falls below a target (47 bpm).  
**Why**: Directly addresses the HR diversity gap at the loss level.  
**Refs**: Claude research §5.4, Gemini research §4  
**Effort**: 0.5 days
