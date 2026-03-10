# Overview

The *current model* is a diffusion transformer (DiT) trained to generate 1D ECG waveforms, but it produces too-narrow heart-rate (HR) variability. We first compare diffusion models (DDPM/DDIM) against newer continuous-flow methods (Flow Matching, Rectified Flow, Conditional Flow Matching) in terms of convergence, stability, sample quality, and speed – especially for 1D biosignals. Next we survey recent ECG/EEG generative models (2024–2025) using diffusion or flow approaches (e.g. **ECGTwin**, **DiffECG**, **FlowECG**, **PPGFlowECG**, **DiffuSETS**). We then discuss ECG-specific loss functions (e.g. RR/QT interval losses, waveform morphology losses) and provide PyTorch pseudocode implementations. Finally, we analyze why diffusion models may collapse HR diversity and propose strategies (conditioning, augmentation, losses, model tweaks) to increase HR variance. We conclude with actionable recommendations: which model to use, which losses to add first, how to diversify HR, and a training pipeline outline. Throughout we cite recent research results to support our analysis【17†L35-L40】【22†L51-L60】.

# Diffusion vs Flow Matching Comparison

| **Approach**             | **Convergence (training)**             | **Stability**                               | **Sample Quality**                         | **Inference Speed**                  | **1D Suitability**                    |
|--------------------------|----------------------------------------|---------------------------------------------|--------------------------------------------|--------------------------------------|---------------------------------------|
| **DDPM (stochastic)**    | Moderate; robust but can be slow       | Very stable (simple MSE loss)               | High (SOTA ECG quality using DDPM)         | Slow: ~100–1000 steps (DDIM can reduce to tens)【3†L313-L321】【17†L35-L40】 | Proven for ECG (SSSD-ECG, DiffECG)    |
| **DDIM (deterministic)** | Same as DDPM (same model/training)     | Very stable (same)                          | High (nearly same as DDPM)                 | Faster than DDPM (~10–50 steps)      | Same as DDPM                           |
| **Flow Matching (FM)**   | Fast: “simulation-free” training【24†L51-L60】| High (regression to velocity field)         | Comparable or better【24†L61-L68】【17†L35-L40】| Very fast: few steps (≲10–25)【17†L35-L40】   | Good: applied to ECG in FlowECG【17†L35-L40】 |
| **Rectified Flow (RF)**  | Fast (like FM with linear path)        | High (OT objective encourages straight path)【22†L51-L60】| High quality even with 1–few steps【22†L69-L72】| Extremely fast: often one-step possible【22†L69-L72】 | Suitable: effectively a special FM    |
| **Cond. Flow Matching**  | Same as FM (adds conditioning)         | High (similarly stable)                    | High (direct conditional control)         | Fast (like FM + condition)           | Very suitable: allows patient/HR control |

- **Architecture (1D signals):**  Both diffusion and flow methods can use 1D architectures (convolutions, SSMs, or transformers). A *Diffusion Transformer (DiT)* can in principle be reused for flow matching by outputting the velocity field instead of noise. Time embeddings and patient/condition inputs carry over between approaches. For example, FlowECG replaces the diffusion denoiser U-Net with a velocity predictor (same backbone) and trains it with an MSE on vector fields【45†L7-L10】.  
- **Compute differences:**  Training cost per iteration is similar (each processes one time step or condition sample). The big difference is **sampling:** diffusion models need many steps (hundreds) to produce high-quality signals【3†L313-L321】【17†L35-L40】, whereas flow methods (FM/RF) often yield good samples in 10× fewer evaluations【17†L35-L40】【49†L197-L205】. Rectified flows can even generate high-fidelity outputs with a single Euler step【22†L69-L72】. The experimental latency of Flow Matching is reported ~10 ms for 10 steps vs 10 ms of pure noise for diffusion at 10 steps【3†L313-L321】, and ~50 ms vs 1000 ms for near-reference results【3†L313-L321】. In practice, replacing DDPM/DDIM with Flow Matching or Rectified Flow can drastically cut sampling time while maintaining quality【17†L35-L40】【49†L197-L205】.  
- **Compatibility with Diffusion Transformers:**  The DiT backbone can accept either SDE noise or ODE velocity targets. For FM, one modifies the output head to predict a vector field and uses the FM loss (MSE to target velocity)【45†L7-L10】. Conditioning (e.g. on patient ID or HR) can be added by concatenating embeddings, much as in conditional diffusion. Thus the DiT architecture is largely compatible with flow-based training; only the training objective and solver change.  

In summary, flow-based methods (Flow Matching/Rectified Flow) have *similar or faster convergence*, *stable training*, *state-of-art quality*, and *much faster inference* than DDPM. They are well-suited to 1D ECG (as shown by FlowECG and PPGFlowECG【17†L35-L40】【7†L60-L69】). **Recommendation:** switching from DDPM to a Flow Matching or Rectified Flow approach is promising for faster sampling without sacrificing quality【17†L35-L40】【22†L69-L72】.

【50†embed_image】 *Figure: Trajectory straightness (curvature $c$) for Flow Matching vs Diffusion. Flow Matching produces near-straight ($c\approx1.02$) transport paths (blue) while Diffusion paths are highly curved ($c\approx1.05$), confirming Flow Matching’s efficiency【49†L197-L205】.* 

【51†embed_image】 *Figure: Projected velocity field learned by Flow Matching (blue arrows). The field pulls noise (red) directly to data (green) along nearly straight lines, explaining why a simple Euler solver suffices【49†L197-L205】【49†L218-L220】.*

# Paper Survey (2024–2025)

We highlight recent (2024–2025) generative models for ECG/EEG using diffusion or flows:

- **ECGTwin (Lai et al. 2025)** – *Diffusion (DDPM)*. Two-stage personalized ECG generation【30†L59-L67】. Stage 1: *Individual Base Extractor* (contrastive learning) captures patient-specific ECG features. Stage 2: a diffusion model incorporates these features plus a target cardiac condition via an *AdaX Condition Injector*. Losses: diffusion denoising loss plus contrastive loss for the extractor. Key result: generates highly realistic, diverse ECGs that preserve individual-specific morphology and allow fine-grained control【30†L59-L67】. (Links: arXiv 2508.02720)  
- **DiffECG (Neifar et al. 2023/2024)** – *Diffusion (DDPM)*. A *versatile diffusion* framework for ECG tasks【32†L68-L73】. It handles (i) heartbeat synthesis, (ii) partial-signal imputation, (iii) full-beat forecasting, all in one conditional model. Architecture: 1D U-Net with conditional input (e.g. lead info, masking). Loss: standard diffusion MSE. Key result: outperforms previous ECG generators (GANs, waveform models) on synthetic data quality and improves ECG classifier performance【32†L68-L73】. Accepted at SERA 2024. (arXiv 2306.01875)  
- **ME-GAN (Chen et al. ICML 2022)** – *GAN-based* (included for context). Multi-lead ECG synthesis *conditioned on heart disease labels*【35†L20-L29】. Innovations: a “mixup normalization” injects disease info into specific waveform locations, and a “view discriminator” ensures correct lead ordering【35†L30-L37】. Losses: adversarial + disease-label supervision. They report realistic multi-lead ECGs with disease-specific morphology; a new metric rFID is proposed. (ICML 2022)  
- **FlowECG (Bondar et al. 2025)** – *Flow Matching* for ECG (12-lead)【17†L35-L40】. Replaces iterative diffusion in SSSD-ECG with continuous flow. Architecture: uses the same SSM/diffusion U-Net backbone but trained via flow matching loss. Loss: MSE between predicted and target vector fields【45†L7-L10】. Key result: matches SSSD-ECG quality at *∼10× fewer steps* (10–25 vs ~200) on PTB-XL, with comparable DTW, Wasserstein, MMD metrics【17†L35-L40】. Demonstrates flow methods can achieve high ECG quality with much faster sampling. (Sensors 2025)  
- **PPGFlowECG (Fang et al. 2025)** – *Rectified Flow* in latent space. Generates ECG from PPG (photoplethysmogram) signals【7†L60-L69】. Core idea: map PPG and ECG to a shared latent space via encoders, then train a *latent rectified flow* (optimal transport) that transforms Gaussian noise into ECG latents. Loss: flow matching (MSE on velocity) in latent domain. Key result: high-fidelity ECGs *one order faster* than diffusion baselines, with cardiologist-validated realism【7†L60-L69】. The latent flow avoids noisy waveform space issues; direct data-space rectified flow was much worse【8†L73-L82】.  
- **DiffuSETS (Chen et al. 2024)** – *Conditional Diffusion*. 12-lead ECG generation *conditioned on clinical text and patient info*【43†L1-L4】. Uses the new MIMIC-IV-ECG dataset with text reports. Architecture: diffusion model with text encoder for clinical reports. Loss: diffusion reconstruction loss. They propose a 3-level evaluation (signal/feature/diagnostic) and a clinical Turing test. Key result: generates semantically aligned, diverse ECGs that boost downstream diagnosis; outperforms previous text-to-ECG methods【42†L273-L282】【43†L1-L4】.  
- **EEG Diffusion (various)** – Example: *EEGDM (Hu et al. 2025)* uses DDPM+SSM to learn EEG representations【37†L57-L66】; *DESAM (Chen et al. 2024)* uses diffusion+mixup for motor-EEG augmentation【40†L19-L27】. These apply diffusion to EEG data (augmenting scarce EEG epochs), demonstrating the broad potential of diffusion in physiological signals. (2024–2025)

*Loss functions:*  Notably, these papers primarily use standard diffusion losses (MSE on noise or velocity). Few explicitly include *ECG-specific* terms. One exception is ECGTwin’s contrastive extractor loss for patient features【30†L59-L67】. Otherwise, specialized ECG losses (RR/QT preservation, morphological matching) are not yet mainstream, highlighting an opportunity (see next section).

# ECG-Specific Loss Functions

To improve physiological realism, one can add losses that enforce ECG-specific features:

- **RR Interval Consistency Loss:** Penalize differences in RR (beat-to-beat) intervals between generated and reference signals. For each ECG in the batch, detect R-peaks and compute RR sequences. For paired generation, match each generated RR to the target RR (e.g. by index or optimal alignment) and take MSE or an Earth-Mover’s distance.  This enforces realistic heart rate and variability.

- **QT Interval Preservation Loss:** The QT interval (Q wave onset to T wave end) shortens as HR increases (Bazett’s law). We can enforce this by detecting Q and T points (e.g. using gradient or threshold heuristics) and penalizing deviations from expected QT for the given RR. For training pairs, simply MSE-match the QT durations of gen vs target. More generally, one can regularize the (QT,RR) relationship to follow a known curve.

- **P-QRS-T Morphology Loss:** Compare the shapes of individual waves. For each beat, segment the signal into windows: P-wave, QRS complex, and T-wave (based on R-peak timing). Compute e.g. L2 loss on each segment between generated and target beats (after aligning R-peaks). This encourages wave amplitudes and widths to match. Optionally weight errors by segment importance (e.g. emphasize QRS shape).

- **Wave Boundary Detection Losses:** Encourage the generator to produce identifiable P-onsets, QRS onsets, T-endings. For example, pass the ECG through a differentiable “peak detector” network or heuristic to locate P, QRS, T times in both signals; then penalize location errors. (In practice, one can supervise a small boundary-detection network on real ECGs and use its output on generated signals.)

- **Beat Template Alignment Loss:** Encourage each beat’s waveform to align with a template. Compute an average beat (via gating on R-peak) in the real data. Then enforce that each generated beat can be (time-)warped to match the template (e.g. using DTW as loss). This ensures beats have realistic morphology.

- **Dynamic Time Warping (DTW) Loss:** Compute DTW distance between the generated and target ECG (or between their beat sequences). Unlike naive L2, DTW allows elastic alignment of waveforms. A differentiable approximation of DTW can be used as a loss, penalizing overall shape differences under time shifts.  

- **Spectral (Amplitude) Coherence Loss:** Encourage the power spectrum of the ECG to match the target. Compute the Fourier transform of gen and target, then define a coherence metric: e.g. $$C(f)=\frac{|\mathcal{F}(G)\overline{\mathcal{F}(T)}|^2}{|\mathcal{F}(G)|^2|\mathcal{F}(T)|^2}$$. Penalize deviations from $C(f)=1$ across frequencies. Alternatively, simply match log-spectra or spectral envelope.

- **Physiological Constraint Losses:** Impose hard constraints, e.g. R-peak amplitudes, ECG axis, or minimal slope constraints. For example, penalize generated signals whose maximum R amplitude is too low/high or whose baseline wander exceeds realistic bounds.

**PyTorch Implementation (pseudocode):** Suppose `pred` and `target` are tensors of shape `(batch, 2500)` with ECG signals. Below are illustrative implementations:

```python
import torch
import torch.nn.functional as F

def rr_interval_loss(pred, target):
    # Simple peak detection by local maximum (for illustration only).
    B, L = pred.shape
    loss = 0.0
    for i in range(B):
        # Detect R-peaks (naive threshold + local max).
        signal_p = pred[i]
        signal_t = target[i]
        # (In practice use a more robust detector)
        peaks_p = (signal_p[1:-1] > signal_p[:-2]) & (signal_p[1:-1] > signal_p[2:]) & (signal_p[1:-1] > 0.5*signal_p.max())
        peaks_t = (signal_t[1:-1] > signal_t[:-2]) & (signal_t[1:-1] > signal_t[2:]) & (signal_t[1:-1] > 0.5*signal_t.max())
        idx_p = torch.where(peaks_p)[0] + 1
        idx_t = torch.where(peaks_t)[0] + 1
        # Compute RR intervals
        rr_p = (idx_p[1:] - idx_p[:-1]).float()
        rr_t = (idx_t[1:] - idx_t[:-1]).float()
        # Match lengths (truncate to min length)
        n = min(rr_p.size(0), rr_t.size(0))
        if n > 0:
            loss += F.mse_loss(rr_p[:n], rr_t[:n])
    return loss / B

def qt_interval_loss(pred, target):
    # Detect Q (slight drop before R) and T (peak after R).
    # This is highly simplified; real implementation needs signal processing.
    B, L = pred.shape
    loss = 0.0
    for i in range(B):
        sig_p, sig_t = pred[i], target[i]
        # Reuse R-peaks from above
        peaks = (sig_p[1:-1] > sig_p[:-2]) & (sig_p[1:-1] > sig_p[2:]) & (sig_p[1:-1] > 0.5*sig_p.max())
        idx_p = torch.where(peaks)[0] + 1
        for r in idx_p:
            # Define Q as last minimum before R within window
            q = r - torch.argmin(sig_p[r-50:r])  # assume P-wave within 50 samples
            # Define T as first max after R within window
            segment = sig_p[r:r+100]  # assume T-wave within 100 samples
            if segment.numel() == 0: continue
            t_idx = r + torch.argmax(segment)
            qt_p = t_idx - q
            # Do same for target
            peaks_t = (sig_t[1:-1] > sig_t[:-2]) & (sig_t[1:-1] > sig_t[2:]) & (sig_t[1:-1] > 0.5*sig_t.max())
            idx_t = torch.where(peaks_t)[0] + 1
            if idx_t.numel() == 0: continue
            # For simplicity, assume same beat index
            rt = idx_t.min()  # use first R of target
            q_t = rt - torch.argmin(sig_t[rt-50:rt])
            segment_t = sig_t[rt:rt+100]
            if segment_t.numel() == 0: continue
            t_idx_t = rt + torch.argmax(segment_t)
            qt_t = t_idx_t - q_t
            loss += (qt_p - qt_t).pow(2)
    return loss / (B + 1e-6)

def morphology_loss(pred, target):
    # Compute weighted MSE on P-wave, QRS, T-wave segments.
    B, L = pred.shape
    loss = 0.0
    for i in range(B):
        sig_p, sig_t = pred[i], target[i]
        # Find R-peak index
        peaks = (sig_p[1:-1] > sig_p[:-2]) & (sig_p[1:-1] > sig_p[2:]) & (sig_p[1:-1] > 0.5*sig_p.max())
        if peaks.sum() == 0: 
            loss += F.mse_loss(sig_p, sig_t)
            continue
        r = (torch.where(peaks)[0] + 1)[0]
        # Define segments (example widths)
        P_seg_p = sig_p[r-100:r-50]   # P-wave region
        QRS_seg_p = sig_p[r-20:r+20]  # QRS region
        T_seg_p = sig_p[r+50:r+150]   # T-wave region
        P_seg_t = sig_t[r-100:r-50]
        QRS_seg_t = sig_t[r-20:r+20]
        T_seg_t = sig_t[r+50:r+150]
        loss += F.mse_loss(P_seg_p, P_seg_t)
        loss += F.mse_loss(QRS_seg_p, QRS_seg_t)
        loss += F.mse_loss(T_seg_p, T_seg_t)
    return loss / B

def dtw_loss(pred, target):
    # PyTorch-friendly DTW (naive version for one pair)
    # In practice, use a GPU-based DTW or approximate.
    B, L = pred.shape
    loss = 0.0
    for i in range(B):
        x = pred[i]
        y = target[i]
        # Compute pairwise distance matrix
        # (Using small L here; for L=2500 this is expensive)
        D = torch.cdist(x.unsqueeze(0).T, y.unsqueeze(0).T, p=2).squeeze()  # [L,L] matrix
        # Dynamic programming for DTW
        dp = torch.full((L+1, L+1), float('inf'))
        dp[0,0] = 0.0
        for ii in range(L):
            for jj in range(L):
                cost = D[ii,jj]
                dp[ii+1,jj+1] = cost + torch.min(dp[ii,jj+1], dp[ii+1,jj], dp[ii,jj])
        loss += dp[L, L]
    return loss / B

def spectral_coherence_loss(pred, target):
    # Mean inverse coherence as loss
    # Compute FFT along time axis
    B, L = pred.shape
    X = torch.fft.rfft(pred, dim=-1)
    Y = torch.fft.rfft(target, dim=-1)
    S_xy = (X * torch.conj(Y)).abs().pow(2).mean(dim=0)  # cross power
    S_xx = (X.abs().pow(2)).mean(dim=0)
    S_yy = (Y.abs().pow(2)).mean(dim=0)
    coherence = S_xy / (S_xx * S_yy + 1e-8)
    # Loss = 1 - coherence (want coherence≈1 at all f)
    return (1 - coherence).mean()
```

These losses are illustrative; in practice one would use robust ECG peak detectors and possibly differentiate through them or use soft approximations. For example, one can compute R-peaks using a convolutional network trained on real ECGs, and then backpropagate through the detected peak times to compute RR/QT losses. The key idea is to penalize deviations in intervals and waveform shapes (RR, QT, P/QRS/T features) rather than only per-sample MSE.

# Increasing HR Diversity

**Why diffusion collapses HR variance:** A diffusion model trained with MSE-style losses tends to average out stochastic variations if they are not strongly constrained. Without explicit pressure, the model may learn the *mean* heart rate for a given patient/condition. Also, using classifier-free guidance or other conditioning may bias outputs toward “typical” HRs. This leads to low σ(HR) in samples. Essentially, the model prioritizes waveform fidelity (PQRST shapes) over beat-to-beat timing diversity, resulting in uniform RR intervals.

**Techniques to increase HR diversity:**

- **Conditioning strategies:**  
  - *Explicit HR Conditioning:* Provide the model with a target HR or a parameter that controls beat rate. For example, append a normalized HR scalar or one-hot HR-bin vector to the condition input. During sampling, sample different HR values to generate diverse rates.  
  - *RR-Interval Conditioning:* Instead of or in addition to HR, condition on a sequence of RR intervals (e.g. as a small time series or embedding). The diffusion model then must match that RR profile.  
  - *Patient Embedding:* Include a learned patient-specific latent or demographic features that capture typical HR range for that individual. Ensuring the model knows patient ID or age/sex can induce more realistic patient-specific HR spread.  

- **Data augmentation:**  
  - *Time Warping:* Randomly stretch or compress the ECG in time (tempo variations) during training. This simulates different heart rates. For example, warp each beat by ±10%. PyTorch code:
    ```python
    import torch.nn.functional as F
    def time_warp(ecg, scale):
        # ecg: (batch, L); scale ~ 0.9-1.1
        B, L = ecg.shape
        new_len = int(L * scale)
        return F.interpolate(ecg.unsqueeze(1), size=new_len, mode='linear').squeeze(1)
    ```
  - *Beat Resampling:* Change the spacing between beats by randomly duplicating/skipping samples within beats.  
  - *Stochastic HR Perturbation:* During generation, add Gaussian noise to RR intervals: e.g. if using ancestral DDPM sampling, perturb the timestep schedule to jitter timing.

- **Model modifications:**  
  - *Classifier-Free Guidance Tuning:* If using classifier-free guidance with a “null” HR condition, adjusting the guidance scale can trade off fidelity vs diversity. A higher scale enforces conditioning (less diversity), while a lower scale yields more random variation. One can tune this specifically for HR.  
  - *Latent HR Variable:* Introduce an auxiliary latent variable representing HR. For example, have the diffusion model generate both an ECG and a scalar HR. Or use a VAE-like branch that samples an HR from a learned distribution, then conditions the diffusion on it.  
  - *Noise Schedule Changes:* Use a noise schedule that emphasizes global rhythm features. For example, place more diffusion timesteps (less noise) when the model should capture slow variations.  
  - *Stochastic Sampling:* Introduce randomness in sampling beyond DDIM; e.g. use probabilistic skip sampling (like DDPM) for some runs to get variant outcomes.

- **Loss modifications:**  
  - *HR Variance Regularization:* During training, compute the sample HR (mean of RR) across a batch and regularize its variance toward the training-set HR variance (47 bpm). For instance, add a loss term $|\mathrm{Var}_{\text{batch}}(\mathrm{HR}) - \sigma_{\text{target}}^2|$.  
  - *RR Distribution Matching:* Compute the distribution of RR intervals in each generated ECG (or batch) and use a metric like Maximum Mean Discrepancy (MMD) to match it to the real RR distribution.  
  - *Adversarial Morphology Loss:* Train a discriminator on RR sequences or beat intervals. The diffusion generator would try to fool this discriminator, encouraging realistic RR variability.  

**PyTorch pseudocode examples:**

```python
# HR variance regularization (batch-wise)
def hr_variance_loss(pred_signals, real_std=47.0):
    # pred_signals: (batch, L)
    B, L = pred_signals.shape
    hr_batch = []
    for i in range(B):
        sig = pred_signals[i]
        peaks = (sig[1:-1]>sig[:-2])&(sig[1:-1]>sig[2:])&(sig[1:-1]>0.5*sig.max())
        idx = torch.where(peaks)[0]+1
        if idx.numel()>1:
            rr = (idx[1:]-idx[:-1]).float()
            hr = 60.0 * 250.0 / rr.mean()  # BPM (if fs=250Hz)
            hr_batch.append(hr)
    if len(hr_batch)<2:
        return torch.tensor(0.0)
    hr_batch = torch.stack(hr_batch)
    var = hr_batch.var()
    return (var - (real_std**2))**2

# Example of classifier-free guidance sampling control:
# (conceptual; actual implementation depends on your pipeline)
alpha = 1.5  # guidance scale >1 biases to condition
x_noise = torch.randn_like(x0)
for t in range(T, 0, -1):
    eps_uncond = model(x_t, t)          # no condition
    eps_cond = model(x_t, t, cond=hr)   # conditioned on HR
    eps = eps_uncond + alpha*(eps_cond - eps_uncond)
    x_t = p_sample(eps, x_t, t)
```

The overall strategy is to **explicitly incentivize heart-rate diversity** either by conditioning the model on a randomly sampled HR (or RR series) or by penalizing lack of variability. By combining conditioning, data warping, and specialized losses, the model can learn to cover the full HR distribution (mean ~100 bpm, std ~47 bpm) rather than collapsing to a narrow range (as currently ~90±21 bpm).

# Practical Recommendations

1. **Model choice:** Adopt a flow-based generative model (Flow Matching or Rectified Flow) instead of plain DDPM. Flow Matching (with an optimal transport path) is a strong choice: it is compatible with a transformer backbone and yields sharp ECGs in ≲25 steps【17†L35-L40】【22†L69-L72】. Given our Diffusion Transformer, we can keep the same architecture and switch the objective to flow matching (constant-velocity/rectified path) to cut sampling time ~10×.  
2. **Losses to implement first:** 
   - **Morphology/interval losses**: Start by adding RR-interval and QT-interval losses to the existing MSE. These directly target rhythm and repolarization realism. For example, add an RR MSE loss between predicted and target peaks (as in code above). Then implement a P/QRS/T segment loss (as above) to enforce wave shapes.  
   - **Spectral coherence loss:** Also add a spectral loss (FFT-domain) to capture overall rhythm content (e.g. power peaks at HR frequency).  
   These should be added gradually and weighted to preserve core waveform fidelity.  
3. **HR diversity fixes:** 
   - Use **explicit HR conditioning**. For instance, augment each training ECG with its true HR as a condition to the model, and train classifier-free. During generation, sample HR from a wider range (100±47) to force variety.  
   - **Data augmentation:** Apply random tempo warping to training ECGs so the model learns beats at varying rates.  
   - If diversity is still low, add a **HR variance loss** (matching the variance of generated HRs to the real data’s 47 bpm) and possibly an adversarial RR-discriminator.  
   The combination of conditioning and loss regularizers should broaden the output HR distribution.  
4. **Training pipeline:** 
   - Precompute R-peaks for all training ECGs (e.g. using Pan–Tompkins) to feed into the RR/QT losses.  
   - Modify the diffusion transformer to take an extra input (HR or patient ID embedding). Train the flow matching objective (if switching) or continue DDPM with new losses.  
   - Start training with baseline diffusion loss + one new loss (e.g. RR-loss). Monitor ECG statistics (mean/std of HR).  
   - Gradually introduce additional losses (QT, morphology) once basic rhythm is captured.  
   - Use early stopping and periodic sampling to check diversity. Adjust loss weights as needed.  
   - For inference, use DDIM or an ODE solver: if Flow Matching is used, simply run a small number of Euler steps (10–25) to sample an ECG at the conditioned HR.  
   - Finally, validate that the generated ECGs not only “look” realistic but have HR statistics matching the dataset.  

By following this roadmap – moving to a flow-based model, adding physiologically informed losses, and explicitly controlling HR variability – the model should generate patient-specific ECGs with both high fidelity and the correct broad HR distribution【17†L35-L40】【42†L273-L282】.

# References

- Liu *et al.*, **“Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,”** ICLR 2023.【22†L51-L60】【22†L69-L72】  
- Lipman *et al.*, **“Flow Matching for Generative Modeling,”** ICLR 2023【24†L51-L60】【24†L61-L68】.  
- Bondar *et al.*, **“FlowECG: Using Flow Matching to Create a More Efficient ECG Signal Generator,”** (preprint 2025)【17†L35-L40】【45†L7-L10】.  
- Fang *et al.*, **“PPGFlowECG: Latent Rectified Flow for PPG-to-ECG Translation,”** (preprint 2025)【7†L60-L69】【8†L73-L82】.  
- Lai *et al.*, **“ECGTwin: Personalized ECG Generation Using Controllable Diffusion Model,”** (preprint 2025)【30†L59-L67】.  
- Neifar *et al.*, **“DiffECG: A Versatile Probabilistic Diffusion Model for ECG Signals Synthesis,”** arXiv 2306.01875 (2024)【32†L68-L73】.  
- Chen *et al.*, **“ME-GAN: Multi-view ECG Synthesis Conditioned on Diseases,”** ICML 2022【35†L20-L29】【35†L30-L37】.  
- Chen *et al.*, **“DiffuSETS: 12-Lead ECG generation conditioned on clinical text and patient info,”** (PMCID: PMC12546759, 2024)【42†L273-L282】【43†L1-L4】.  
- Gupta *et al.*, **“Efficiency vs. Fidelity: Diffusion vs Flow Matching on Low-Resource Hardware,”** arXiv:2511.19379 (2025)【49†L197-L205】【49†L218-L220】.  
- (Others: SSSD-ECG [17†L78-L86], Diffusion models [24†L51-L60], Standard ECG physiology [46†L11-L18].)