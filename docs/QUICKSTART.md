# 🚀 CardioEquation Quick Start Guide

## 🎯 What is CardioEquation?

CardioEquation is an **AI-driven system** that generates **personalized mathematical equations** (Digital Twins) to reproduce individual ECG waveform patterns. We move from traditional pattern recognition to **generative forecasting**.

## ⚡ 30-Second Demo

```bash
# 1. Install dependencies
pip install numpy scipy matplotlib tensorflow scikit-learn opencv-python pillow

# 2. Run the Denoising Pipeline (Phase 3)
python src/main_process_pdf.py

# 3. Train Forecasting Model (Phase 4, optional)
python src/train_forecasting.py
```

## 🧮 The Core Idea

Each person's ECG is modeled as a patient-specific equation:

```python
ECG(t; θ) = Σ A_i · exp(-((t - μ_i)²)/(2σ_i²))
```

Where `θ` is extracted via a **1D ResNet-18** identity learner.

## 🏗️ System Components

| Component | File | Purpose |
|-----------|------|---------|
| **ECG Digitizer** | `src/ecg_digitizer.py` | Extracts signals from clinical PDFs |
| **Identity Learner** | `src/models/feature_extractor.py` | Extracts 512-dim "cardiac DNA" |
| **Diffusion U-Net** | `src/models/diffusion_unet.py` | Generates/Denoises conditioned on identity |
| **Master Pipeline** | `src/main_process_pdf.py` | End-to-end PDF-to-Signal workflow |

## 📊 Phase Roadmap

- ✅ **Phase 1**: Synthetic Bootstrap (Diffusion Denoising)
- ✅ **Phase 2**: Realistic Artifacts (Grid/Texture Robustness)
- ✅ **Phase 3**: Clinical Integration (PDF Digitization)
- 🔄 **Phase 4**: Personalized Forecasting (Digital Twin)

## 🏆 Success Criteria

- ✅ ECG reconstruction accuracy ≥ 97%
- ✅ Reliable digitization of clinical PDF reports
- ✅ Identity preservation during forecasting
- ✅ Optimized inference (<10ms on CPU)

---

**🫀 Ready to explore? Run `python src/main_process_pdf.py` to process your first clinical ECG!**
