# **Technical Analysis and Architectural Optimization of Diffusion Transformers for High-Fidelity Personalized ECG Synthesis**

The synthesis of high-fidelity, patient-specific 1D physiological signals represents a critical frontier in digital health, enabling the creation of "digital twins" for longitudinal monitoring and rare disease simulation. While the Diffusion Transformer (DiT) architecture has demonstrated remarkable scalability and morphological quality in early experimental iterations, the transition from general waveform generation to precise, controllable personalization remains technically fraught. Analysis of recent experimental failures, specifically in Run 5b, reveals a systemic misalignment between auxiliary conditioning signals—such as heart rate (HR) and patient identity—and the underlying denoising process. This report provides an exhaustive investigation into the mechanisms of heart rate conditioning, identity preservation, and temporal fidelity within the diffusion paradigm, drawing on state-of-the-art research in biomedical signal generation, audio diffusion, and biometric image personalization.

## **Analysis of Experimental Failures in Baseline Run 5b**

The current results from Run 5b underscore a fundamental challenge in multi-condition diffusion: the model exhibits high morphological fidelity but significant semantic drift. The reported heart rate (HR) Mean Absolute Error (MAE) of 23.6 bpm indicates that the model has effectively decoupled the HR conditioning input from the generated output. When conditioned on 82 bpm, the model produces a mean HR of 54.1 bpm, which closely aligns with the population mean of many resting ECG datasets, suggesting that the DiT has regressed to the data distribution's mode rather than learning the conditional mapping.

Furthermore, the 0% Top-1 Patient Re-ID score confirms that the contrastive identity embedding is not exerting sufficient influence on the reverse diffusion process. This "identity collapse" is compounded by a 45% reduction in QRS duration (60.4 ms vs. 110.7 ms clinical reference), a phenomenon termed "temporal compression." This distortion occurs when the optimization objective—typically Mean Squared Error (MSE)—prioritizes local waveform smoothness and R-peak alignment over the global temporal semantics of the cardiac cycle.

## **Heart Rate Conditioning Strength and Auxiliary Loss Dynamics**

In conditional diffusion models, the weight assigned to auxiliary losses ($w\_{aux}$) is the primary mechanism for enforcing adherence to non-structural metadata. In Run 5b, the hr\_loss\_weight of 0.05 is likely an order of magnitude too low to influence the high-dimensional gradient of a 265M-parameter Transformer.

### **Comparative Auxiliary Loss Weighting in Literature**

Research into physiological signal generators highlights a range of auxiliary loss weights required for stable conditioning. Models such as ArterialNet, which reconstructs arterial blood pressure (ABP) waveforms, utilize a hybrid objective function $\\Lambda$ that balances multiple components.

### **Table 1: Loss Weight Hyperparameters in Physiological Generative Frameworks**

| Framework | Primary Task | Auxiliary Term | Weight (λ or Φ) | Rationale |
| :---- | :---- | :---- | :---- | :---- |
| ArterialNet | ABP Reconstruction | Waveform Correlation ($\\lambda\_r$) | 10.0 | Enforces phase alignment |
| ArterialNet | ABP Reconstruction | Soft-DTW Alignment ($\\lambda\_a$) | 0.01 | Corrects temporal warping |
| PhysDiff-LBM | ECG Synthesis | Topology (QRS Mask) | 0.1 \- 0.5 | Ensures rhythmic coherence |
| CardiacGen | ECG/PPG Generation | HRV Consistency | Hierarchical | Decoupled HR/Morphology |
| Experimental Run 5b | ECG Synthesis | Differentiable HR | 0.05 | Insufficient gradient signal |

The ECGTwin framework (arXiv:2508.02720) addresses this by utilizing a two-stage approach where individual features are extracted via a contrastive "Individual Base Extractor" before being injected into the latent diffusion process via the "AdaX Condition Injector". This architecture suggests that simple MLP heads on AdaLN-Zero may be insufficient for high-dimensional Transformers; instead, dedicated pathways for different condition types—such as rhythmic (HR) vs. morphological (Identity)—are necessary to prevent "condition interference".

For Run 6, an increase of hr\_loss\_weight to the range of 0.5 to 1.0 is recommended. This adjustment aligns with findings that structural adherence in time-series diffusion requires auxiliary gradients that are commensurate in magnitude with the denoising MSE. If the model continues to ignore HR conditioning, a hierarchical strategy similar to CardiacGen—where heart rate variability (HRV) and morphology are modeled by separate specialized modules—may be required to manage the conflicting objectives of rhythmic precision and morphological diversity.

## **Identity Preservation in State-of-the-Art Diffusion Models**

The preservation of patient-specific identity in 1D signals mirrors the challenges of "person-driven" image generation. The current failure (0% Re-ID) indicates that the identity embedding is being treated as a global style hint rather than a restrictive morphological constraint.

### **Decoupled Cross-Attention and IdentityNet Paradigms**

State-of-the-art face and biometric models like IP-Adapter and InstantID move beyond simple additive conditioning (like AdaLN) toward decoupled cross-attention mechanisms. In these architectures, text features and image/identity features are processed through separate cross-attention pathways.

* **IP-Adapter Mechanism**: The adapter uses a parallel cross-attention pathway for image tokens, which are linearly combined with text-conditioned outputs. This prevents the identity information from being "overwhelmed" by the primary prompt or timestep embedding.  
* **InstantID and IdentityNet**: InstantID introduces a "IdentityNet" that imposes strong semantic conditions and weak spatial conditions. By using a dedicated face encoder (rather than a general CLIP encoder), it captures the "nuanced semantics" required for identity preservation.

### **SNR-Weighted Identity Loss**

The application of a Signal-to-Noise Ratio (SNR) weighted identity loss is a critical technique for preserving fine-grained details during the denoising process. Research into "SNR-weighted sampling" indicates that at high noise levels (low SNR), the model should focus on global structure, while at low noise levels (high SNR), the loss should emphasize identity-related details.

In the current DiT setup, the SNR-weighted identity loss ($w\_{identity} \= 0.5$) is active, but the injection method (AdaLN-Zero) might be the bottleneck. DiTs often benefit from "identity-specific" CFG channels or dedicated reference networks (ReferenceNet) that process the patient's baseline ECG to provide feature-level guidance.

### **Table 2: Identity Conditioning Comparison in Diffusion Models**

| Method | Conditioning Type | Guidance Scale (w) | Weighting | Key Mechanism |
| :---- | :---- | :---- | :---- | :---- |
| IP-Adapter | Visual Prompt | 0.5 \- 1.0 | Constant | Decoupled Cross-Attention |
| InstantID | Face ID Embedding | 0.8 \- 1.2 | Constant/Adjustable | IdentityNet \+ Spatial Control |
| TextBoost | Text Encoder | Adaptive | SNR-Weighted | Augmentation Tokens |
| Run 5b | MLP Head (AdaLN) | 3.0 | SNR-Weighted | Global Modulation |

The guidance scale of 3.0 in Run 5b may be excessive for 1D signals. While image models often use scales of 7.0+, excessive guidance in physiological domains can lead to "prototypical" waveform generation, where the model produces a "generic healthy ECG" at the expense of patient-specific variations.

## **Mitigating QRS Temporal Compression: Advanced Loss Functions**

The 45% narrowing of the QRS complex is a classic indicator of a model failing to capture the temporal "elasticity" of physiological signals. Standard MSE losses treat a 2ms shift in an R-peak as a catastrophic error, causing the model to generate unnaturally narrow peaks to minimize the average squared distance across slightly misaligned samples.

### **Soft-DTW: A Differentiable Alignment Loss**

ECG literature and general time-series research strongly advocate for **Soft-DTW (Soft Dynamic Time Warping)** as a differentiable alternative to Euclidean distances. Unlike MSE, Soft-DTW is robust to shifts and dilatations across the time dimension.

The Soft-DTW objective computes a soft-minimum of all possible alignment costs, providing a smooth gradient for backpropagation. For ECG generation, this allows the model to be penalized for the *shape* and *duration* of the QRS complex even if the global timing is slightly off. ArterialNet specifically uses a Soft-DTW alignment loss ($\\lambda\_a \= 0.01$) to stabilize the reconstruction of pulsatile waveforms.

### **Table 3: Comparison of Time-Series Alignment Losses**

| Loss Function | Complexity | Differentiable | Robustness to Warping | Use Case |
| :---- | :---- | :---- | :---- | :---- |
| MSE / L1 | $O(N)$ | Yes | Low | Pixel-wise/Sample-wise accuracy |
| Soft-DTW | $O(N^2)$ | Yes | High | Morphological duration/shape |
| Correlation | $O(N)$ | Yes | Moderate | Phase and rhythm alignment |
| CTC Loss | $O(N \\cdot M)$ | Yes | High | Unaligned sequence mapping |

While Connectionist Temporal Classification (CTC) losses are standard in speech for mapping unaligned sequences, they are less common in waveform generation compared to Soft-DTW, which preserves the "volumetric" shape of the signal more effectively.

## **Optimal CFG Scales for 1D Physiological Signal Generation**

Classifier-Free Guidance (CFG) is a powerful tool for controlling the trade-off between sample diversity and condition adherence. However, the "sweet spot" for 1D signals—particularly audio and physiological data—is typically lower than for images.

### **Audio and Music Diffusion Benchmarks**

Research into models like AudioLDM, MusicGen, and MusicLDM provides valuable context for CFG scaling.

* **MusicLDM**: Uses a guidance scale $w=2.0$ for text-to-music generation.  
* **AudioLDM**: Emphasizes that high CFG scales (e.g., \> 5.0) can significantly reduce sample diversity and introduce artifacts in the acoustic environment.  
* **Diffusion Waveform Generators (DiffWave, WaveGrad)**: Often utilize very low guidance or rely on strong auxiliary conditioners (like Mel-spectrograms) without the need for high CFG extrapolation.

For Run 6, a CFG scale in the range of **1.5 to 2.5** is recommended. The current scale of 3.0 may be pushing the model toward a "generic mode" that ignores the subtle nuances of patient identity and HR variability.

## **Differentiable Heart Rate Estimation: Beyond FFT Autocorrelation**

The current approach (FFT autocorrelation \+ soft-argmax) is theoretically sound but practically fragile in a training loop. FFT-based methods are sensitive to signal windowing and the periodicity assumptions of the Fourier transform, which can be violated by arrhythmias or noise artifacts.

### **Learned Heart Rate Predictor Networks**

A more robust alternative is the use of a "Perceptual Loss" derived from a pre-trained HR predictor network. Frameworks like **DeepBeat** and **DeepPhys** utilize multi-task convolutional networks to extract vital signs from noisy signals.

Integrating a frozen, pre-trained HR estimator as part of the loss function offers several advantages:

1. **Noise Robustness**: Learned models are typically more resilient to the artifacts generated during the early stages of the diffusion reverse process.  
2. **Clinical Relevance**: Predictors like DeepBeat are trained on real-world datasets (e.g., UK Biobank) to identify QRS complexes and R-R intervals, providing a gradient that aligns more closely with cardiological reality.  
3. **Holistic Evaluation**: Instead of just BPM, these networks can provide feedback on QRS duration and ST-segment morphology, addressing the "temporal compression" issue indirectly.

## **The Generative Paradigm Shift: Conditional Flow Matching (OT-CFM)**

A critical decision for Run 6 is whether to move from the Denoising Diffusion Probabilistic Model (DDPM) framework to **Conditional Flow Matching (CFM)**, specifically Optimal Transport Flow Matching (OT-CFM).

### **Efficiency and Path Rectification**

CFM models the generation process as a Continuous Normalizing Flow (CNF), regressing a velocity field that maps the source (noise) distribution to the target (data) distribution.

* **Geometric Curvature**: DDPM trajectories are inherently stochastic and tortuous (Curvature $\\mathcal{C} \\approx 3.45$). In contrast, OT-CFM learns a "highly rectified" transport path ($\\mathcal{C} \\approx 1.02$), which is near-optimal.  
* **Conditioning Adherence**: Straight-line paths in OT-CFM lead to better adherence to conditioning signals. For time-series forecasting and generation, Flow Matching has been shown to outperform DDPM in "bridging the last mile" of prediction accuracy.  
* **Sampling Speed**: CFM reaches the "Efficiency Frontier" at approximately 10 function evaluations (NFE), whereas DDPM often collapses at such low step counts.

### **Suitability for Physiological Signals**

Recent implementations of Flow Matching for time-series, such as **TSFlow**, demonstrate superior generative capabilities by aligning the prior distribution with the temporal structure of the data using Gaussian Processes. For ECG synthesis, the deterministic nature of CFM's ODE paths provides a more stable mapping between the 512-dim identity embedding and the final waveform, potentially solving the identity preservation issue encountered in Run 5b.

## **Comprehensive Strategy for Run 6 Optimization**

The failure of Run 5b to achieve target HR accuracy and identity preservation necessitates a multi-faceted architectural update. The transition to Run 6 should prioritize the "decoupling" of conditioning signals and the implementation of alignment-aware loss functions.

### **Table 4: Proposed Configuration for Experimental Run 6**

| Component | Run 5b (Baseline) | Run 6 (Proposed) | Mechanism for Improvement |
| :---- | :---- | :---- | :---- |
| **Generative Framework** | DDPM / DDIM | **OT-CFM (Flow Matching)** | Straighter paths for better condition adherence |
| **Identity Injection** | AdaLN-Zero (MLP) | **Decoupled Cross-Attention** | Prevents identity signals from being diluted |
| **HR Loss Weight** | 0.05 | **0.5 \- 1.0** | Forces the DiT backbone to respect rhythmic constraints |
| **Temporal Loss** | MSE (Sample-wise) | **Soft-DTW ($\\lambda \= 0.1$)** | Corrects QRS temporal compression and warping |
| **CFG Scale** | 3.0 | **1.5 \- 2.0** | Promotes diversity and prevents mode collapse |
| **HR Estimator** | FFT Autocorrelation | **DeepBeat (Pre-trained CNN)** | Robust, perceptually-aware HR gradients |
| **Conditioning Source** | Contrastive Embedding | **ID Tokens \+ Reference ECG** | Provides multi-focal context for personalization |

### **Implementing the "Topology Branch"**

To further ensure clinical validity, Run 6 should adopt a **Region-Disentangled U-Net/DiT** approach similar to PhysDiff-LBM. By adding an auxiliary "Topology Branch" that predicts a probability mask for the QRS complexes, the model is forced to learn the internal semantics of the cardiac cycle. This topological grounding, when combined with a Binary Cross-Entropy loss against the ground-truth R-peak locations, significantly reduces "hallucinations"—non-physiological waveforms that appear visually realistic but violate cardiovascular physics.

### **Identity Preservation via Multi-Focal Conditioning**

The current identity embedding (512-dim) should be augmented with "pose-invariant" features of the patient's heart rhythm. In person-image synthesis, this is achieved by cropping sensitive areas (like the face) and treating them as independent focal conditions. In the ECG domain, this translates to treating a single "template" heartbeat from the patient's history as a visual prompt injected via cross-attention, while using the 512-dim embedding for global style and metadata.

The final success of the DiT model in generating patient-specific waveforms will depend on the synergy between the **rectified paths** of Flow Matching and the **alignment-robustness** of Soft-DTW. By shifting the objective from "predicting noise" to "matching a transport flow," the model can more effectively navigate the complex manifold of human electrophysiology, producing "digital twins" that are both mathematically accurate and clinically actionable.

