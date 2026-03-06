#!/bin/bash
# ============================================================
# CardioEquation V2: Full Training Pipeline
# ============================================================
# Run this script on the GPU server after cloning the repo.
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh           # Full pipeline (venv + train)
#   ./run_training.sh smoke     # Quick smoke test (2 epochs)
#   ./run_training.sh contrastive  # Only contrastive pre-training
#   ./run_training.sh dit       # Only DiT training
#   ./run_training.sh validate  # Only clinical validation
# ============================================================

set -e  # Exit on error

SMOKE_MODE=false
RUN_CONTRASTIVE=true
RUN_DIT=true
RUN_VALIDATION=true

if [ "$1" == "smoke" ]; then
    echo "🔥 SMOKE TEST MODE"
    SMOKE_MODE=true
elif [ "$1" == "contrastive" ]; then
    RUN_DIT=false
    RUN_VALIDATION=false
    echo "🔬 Running only contrastive pre-training"
elif [ "$1" == "dit" ]; then
    RUN_CONTRASTIVE=false
    RUN_VALIDATION=false
    echo "🚀 Running only DiT training"
elif [ "$1" == "validate" ]; then
    RUN_CONTRASTIVE=false
    RUN_DIT=false
    echo "🏥 Running only clinical validation"
fi

echo "============================================================"
echo "  CardioEquation V2: Training Pipeline"
echo "============================================================"

# ── 0. Virtual Environment Setup ─────────────────────────
VENV_DIR="venv_v2"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Step 0: Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "   Created: $VENV_DIR/"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "   Activated: $(which python)"
echo "   Python: $(python --version)"

# Install/upgrade pip
pip install --upgrade pip -q

# Install PyTorch (CUDA 12.1 — adjust for your GPU server)
echo ""
echo "📦 Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
pip install torch torchvision -q  # Fallback to default CUDA

# Install project dependencies
echo "📦 Installing project dependencies..."
pip install -r requirements_pytorch.txt -q

# ── 1. GPU Check ──────────────────────────────────────────
echo ""
echo "🖥️  GPU Verification:"
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    print(f'   ✅ GPU: {name} ({mem:.1f} GB)')
    print(f'   ✅ BF16 Support: {bf16}')
    print(f'   ✅ CUDA Version: {torch.version.cuda}')
else:
    print('   ⚠️  No GPU detected! Training will be very slow.')
    print('   Check: nvidia-smi and CUDA installation')
"

# ── 2. Download Datasets ──────────────────────────────────
echo ""
echo "============================================================"
echo "📥 Step 1: Downloading Datasets"
echo "============================================================"

if [ "$SMOKE_MODE" = true ]; then
    echo "   Smoke mode: using synthetic data, skipping downloads"
else
    python download_all_datasets.py
fi

# ── 3. Verify Datasets ───────────────────────────────────
echo ""
echo "============================================================"
echo "🔍 Step 2: Verifying Datasets"
echo "============================================================"
python verify_datasets.py || {
    if [ "$SMOKE_MODE" = false ]; then
        echo "⚠️  Dataset verification has warnings/errors. Check above."
        echo "   Continuing anyway (non-critical items may be missing)..."
    fi
}

# ── 4. Model Shape Check ─────────────────────────────────
echo ""
echo "🧪 Step 3: Verifying model shapes..."
python -c "
import torch, sys
sys.path.insert(0, '.')
from src.models.dit_ecg import dit_ecg_b
from src.models.feature_extractor_pt import FeatureExtractorPT

model = dit_ecg_b()
fe = FeatureExtractorPT()
x = torch.randn(2, 1, 2500)
t = torch.rand(2)
c = torch.randn(2, 512)

out = model(x, t, c)
z = fe(x)

dit_p = sum(p.numel() for p in model.parameters()) / 1e6
fe_p = sum(p.numel() for p in fe.parameters()) / 1e6

assert out.shape == (2, 1, 2500), f'DiT shape error: {out.shape}'
assert z.shape == (2, 512), f'FE shape error: {z.shape}'
print(f'   ✅ DiT-ECG-B: {dit_p:.1f}M params, output {out.shape}')
print(f'   ✅ FeatureExtractor: {fe_p:.1f}M params, output {z.shape}')
print(f'   ✅ All shapes verified!')
"

# Create directories
mkdir -p checkpoints
mkdir -p data
mkdir -p outputs/clinical_validation

# ── 3. Contrastive Pre-Training (Stage 0) ────────────────
if [ "$RUN_CONTRASTIVE" = true ]; then
    echo ""
    echo "============================================================"
    echo "🔬 Step 2: Contrastive Pre-Training (Stage 0)"
    echo "============================================================"

    if [ "$SMOKE_MODE" = true ]; then
        python src/training/train_contrastive.py \
            --epochs 5 \
            --batch_size 32 \
            --lr 3e-4 \
            --output_dir checkpoints
    else
        python src/training/train_contrastive.py \
            --epochs 100 \
            --batch_size 64 \
            --lr 3e-4 \
            --output_dir checkpoints
    fi
fi

# ── 4. DiT-ECG Training (Main) ───────────────────────────
if [ "$RUN_DIT" = true ]; then
    echo ""
    echo "============================================================"
    echo "🚀 Step 3: DiT-ECG Diffusion Training"
    echo "============================================================"

    if [ "$SMOKE_MODE" = true ]; then
        python src/training/train_dit.py \
            --epochs 2 \
            --batch_size 4 \
            --accum_steps 1 \
            --model_size B \
            --dataset synthetic \
            --output_dir checkpoints \
            --save_every 1 \
            --warmup_steps 10
    else
        python src/training/train_dit.py \
            --epochs 200 \
            --batch_size 32 \
            --accum_steps 8 \
            --model_size B \
            --dataset mitbih \
            --output_dir checkpoints \
            --save_every 10 \
            --lr 1e-4 \
            --warmup_steps 5000
    fi
fi

# ── 5. Clinical Validation ────────────────────────────────
if [ "$RUN_VALIDATION" = true ]; then
    echo ""
    echo "============================================================"
    echo "🏥 Step 4: Clinical Validation (Hospital ECG PDFs)"
    echo "============================================================"

    if [ -d "Dataset" ]; then
        python src/evaluation/clinical_validation.py \
            --model_path checkpoints/dit_ecg_ema_final.pt \
            --fe_path checkpoints/feature_extractor_contrastive.pt \
            --dataset_dir Dataset \
            --output_dir outputs/clinical_validation
    else
        echo "   ⚠️  Dataset/ folder not found — skipping clinical validation"
        echo "   Copy hospital ECG PDFs to Dataset/ and re-run with: ./run_training.sh validate"
    fi
fi

# ── Done ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ Pipeline Complete!"
echo "  Checkpoints:  checkpoints/"
echo "  Validation:   outputs/clinical_validation/"
echo "  Deactivate:   deactivate"
echo "============================================================"
