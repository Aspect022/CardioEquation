#!/bin/bash
# ============================================================
# CardioEquation V2: Full Training Pipeline — Run 6
# ============================================================
# Run this script on the GPU server after cloning the repo.
#
# Run 6 Changes:
# - OT-CFM (Conditional Flow Matching) replaces DDPM
# - Identity cross-attention (IP-Adapter paradigm)
# - HR loss with timestep gating + 10× weight increase
# - Soft-DTW loss for QRS temporal fidelity
# - 750 epochs, LR=5e-5, CFG=2.0
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
echo "  CardioEquation V2: Training Pipeline — Run 6"
echo "  OT-CFM + Cross-Attention Identity + Soft-DTW"
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

# ── Weights & Biases Setup ────────────────────────────────
export WANDB_API_KEY="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"
echo "📊 W&B: Logging in..."
python -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" 2>/dev/null && echo "   ✅ W&B authenticated" || echo "   ⚠️  W&B login failed (will try during training)"

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

# ── 2. Download & Process ALL Datasets ────────────────────
echo ""
echo "============================================================"
echo "📥 Step 1: Downloading & Processing ALL Datasets"
echo "============================================================"

if [ "$SMOKE_MODE" = true ]; then
    echo "   Smoke mode: using synthetic data, skipping downloads"
else
    # Download and process ALL datasets (MIT-BIH + PTB-XL + Chapman-Shaoxing)
    # Skip download if processed .npz already exists (saves 30+ min)

    if [ -f "data/mitbih_forecasting.npz" ]; then
        echo "   ✅ MIT-BIH already processed, skipping download"
    else
        echo "📦 Downloading MIT-BIH..."
        python download_all_datasets.py --mitbih
    fi

    if [ -f "data/ptbxl_processed.npz" ]; then
        echo "   ✅ PTB-XL already processed, skipping download"
    else
        echo "📦 Downloading PTB-XL..."
        python download_all_datasets.py --ptbxl
    fi

    # Chapman-Shaoxing is optional and currently causing slow down/redundant downloads.
    # Skipping for Run 6 to start training immediately.
    # if [ -f "data/chapman_processed.npz" ]; then
    #     echo "   ✅ Chapman-Shaoxing already processed, skipping download"
    # else
    #     echo "📦 Downloading Chapman-Shaoxing..."
    #     python download_all_datasets.py --chapman
    # fi

    # Process any unprocessed datasets (each function skips internally if .npz exists)
    echo ""
    echo "⚙️  Processing datasets (skips already-processed)..."
    python download_all_datasets.py --process
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
echo "🧪 Step 3: Verifying model shapes (Run 6 cross-attention)..."
python -c "
import torch, sys
sys.path.insert(0, '.')
from src.models.dit_ecg import dit_ecg_b
from src.models.feature_extractor_pt import FeatureExtractorPT
from src.training.flow_matching import FlowMatchingScheduler

model = dit_ecg_b()
fe = FeatureExtractorPT()
x = torch.randn(2, 1, 2500)
t = torch.rand(2)
c = torch.randn(2, 512)
hr = torch.tensor([72.0, 85.0])

out = model(x, t, c, hr_bpm=hr)
z = fe(x)

dit_p = sum(p.numel() for p in model.parameters()) / 1e6
fe_p = sum(p.numel() for p in fe.parameters()) / 1e6

assert out.shape == (2, 1, 2500), f'DiT shape error: {out.shape}'
assert z.shape == (2, 512), f'FE shape error: {z.shape}'
print(f'   ✅ DiT-ECG-B (Run 6): {dit_p:.1f}M params, output {out.shape}')
print(f'   ✅ FeatureExtractor: {fe_p:.1f}M params, output {z.shape}')

# Test flow matching scheduler
fm = FlowMatchingScheduler()
x0 = torch.randn(2, 1, 2500)
x1 = torch.randn(2, 1, 2500)
t_fm = torch.tensor([0.0, 0.5])
x_t, v = fm.interpolate(x0, x1, t_fm)
print(f'   ✅ FlowMatching: x_t={x_t.shape}, v={v.shape}')
print(f'   ✅ All shapes verified (Run 6)!')
"

# Create directories
mkdir -p checkpoints
mkdir -p data
mkdir -p outputs/clinical_validation_run6

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
            --epochs 200 \
            --batch_size 64 \
            --lr 3e-4 \
            --output_dir checkpoints
    fi
fi

# ── 4. DiT-ECG Training (Main — Run 6) ──────────────────
if [ "$RUN_DIT" = true ]; then
    echo ""
    echo "============================================================"
    echo "🚀 Step 3: DiT-ECG Training (Run 6: OT-CFM + CrossAttn)"
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
            --warmup_steps 10 \
            --guidance_scale 2.0 \
            --no_wandb
    else
        python src/training/train_dit.py \
            --epochs 750 \
            --batch_size 32 \
            --accum_steps 8 \
            --model_size B \
            --dataset mitbih \
            --output_dir checkpoints \
            --save_every 10 \
            --lr 5e-5 \
            --warmup_steps 3000 \
            --patience 60 \
            --val_split 0.1 \
            --guidance_scale 2.0 \
            --wandb_run "Run6-OT-CFM-CrossAttn"
    fi
fi

# ── 5. Clinical Validation ────────────────────────────────
if [ "$RUN_VALIDATION" = true ]; then
    echo ""
    echo "============================================================"
    echo "🏥 Step 4: Clinical Validation (Run 6)"
    echo "============================================================"

    set +e  # Don't exit on error for validation
    python src/evaluation/evaluate_dit.py \
        --checkpoint checkpoints/dit_ecg_best.pt \
        --fe_path checkpoints/feature_extractor_contrastive.pt \
        --output_dir outputs/clinical_validation_run6 \
        --n_samples 200 \
        --guidance 2.0
    VALIDATION_EXIT=$?
    set -e

    if [ $VALIDATION_EXIT -ne 0 ]; then
        echo "   ⚠️  Clinical validation had errors but training is complete."
        echo "   Re-run later with: ./run_training.sh validate"
    fi
fi

# ── 6. Auto Git Push Results ──────────────────────────────
echo ""
echo "============================================================"
echo "📤 Step 5: Pushing Results to GitHub"
echo "============================================================"

# Save a training summary log
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "CardioEquation Run 6 Training" > checkpoints/training_log.txt
echo "==============================" >> checkpoints/training_log.txt
echo "Timestamp: $TIMESTAMP" >> checkpoints/training_log.txt
echo "Run: Run 6 (OT-CFM + CrossAttn Identity + Soft-DTW)" >> checkpoints/training_log.txt
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')" >> checkpoints/training_log.txt
echo "Smoke Mode: $SMOKE_MODE" >> checkpoints/training_log.txt
echo "" >> checkpoints/training_log.txt
echo "Files in checkpoints/:" >> checkpoints/training_log.txt
ls -lh checkpoints/ >> checkpoints/training_log.txt 2>/dev/null
echo "" >> checkpoints/training_log.txt
echo "Files in outputs/:" >> checkpoints/training_log.txt
ls -lhR outputs/ >> checkpoints/training_log.txt 2>/dev/null

# Git add, commit, and push
set +e  # Don't fail if git push has issues
git add checkpoints/train_config.json checkpoints/training_log.txt 2>/dev/null
git add outputs/clinical_validation_run6/ 2>/dev/null
git add -u  # Stage any modified tracked files

COMMIT_MSG="🏁 Run 6 complete ($TIMESTAMP) — OT-CFM + CrossAttn — $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'GPU')"
git commit -m "$COMMIT_MSG" && {
    echo "   📤 Pushing to GitHub..."
    git push origin main
    if [ $? -eq 0 ]; then
        echo "   ✅ Results pushed to GitHub!"
    else
        echo "   ⚠️  Git push failed. You can push manually with: git push origin main"
    fi
} || echo "   ℹ️  Nothing new to commit."
set -e

# ── Done ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ Run 6 Pipeline Complete!"
echo "  Timestamp: $TIMESTAMP"
echo "  Checkpoints:  checkpoints/"
echo "  Validation:   outputs/clinical_validation_run6/"
echo "  Training Log: checkpoints/training_log.txt"
echo "  GitHub:       Check your repo for pushed results!"
echo "  Deactivate:   deactivate"
echo "============================================================"
