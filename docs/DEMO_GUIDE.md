# CardioEquation Digital Twin Demo Guide

## Overview
This guide provides step-by-step instructions for demonstrating the CardioEquation digital twin system. The demo showcases how AI can create personalized mathematical equations that model individual ECG patterns.

## Prerequisites
- Python 3.8+
- Dependencies: `numpy, tensorflow, scikit-learn, matplotlib, opencv-python, pillow`

## Demo Steps

### Step 1: Project Introduction
**What to say:**
"Today I'll be demonstrating the CardioEquation digital twin system. We use Conditional Diffusion models to transform noisy clinical ECG scans into clean, personalized digital twins."

### Step 2: Show Project Structure
**Actions:**
1. Open the root directory.
2. Highlight:
- `src/`: Core logic (Digitizer, Models, Training).
- `outputs/`: Multi-phase verification proofs (v1_phase1, etc.).
- `Dataset/`: Clinical PDFs being processed.

### Step 3: Demonstrate PDF Processing (Live Demo)
**Actions:**
1. Run the master pipeline:
   ```bash
   python src/main_process_pdf.py
   ```

**What to say:**
"This script runs our Phase 3 Clinical Pipeline:
1. **Digitization**: Extracting Lead II from the PDF scan.
2. **Diffusion Denoising**: Using our score-based U-Net to remove paper artifacts and noise.
3. **Visualization**: Comparing the original scan vs. the clean digital output."

### Step 4: Explain the 4-Phase Roadmap
**Actions:**
1. Open `PROJECT_OVERVIEW.md`.
2. Navigate to "Development Lifecycle".

**What to say:**
"The project has evolved through 4 phases:
- **Phase 1-2**: Perfected diffusion on synthetic data.
- **Phase 3**: Integrated real clinical PDFs.
- **Phase 4**: Currently implementing 'Identity Preservation' for forecasting future beats."

### Step 5: Future Vision
**What to say:**
"In 2026, we are moving towards 12-lead support and pathological arrhythmia modeling to help clinicians compare current ECGs against a patient's personalized baseline."

---

*Last Updated: January 2026 | Version: 2.1*