"""
DiT-ECG Training Script — Run 6
==================================
Main training entry point for CardioEquation DiT-ECG-B.

Run 6 Features:
- OT-CFM (Conditional Flow Matching) — velocity prediction instead of noise
- Identity via cross-attention (no longer in AdaLN)
- HR loss with timestep gating (only fires at t < 0.2)
- Soft-DTW loss for QRS temporal fidelity
- Identity SNR floor (≥30% gradient at all noise levels)
- Updated weights: hr=0.5, identity=1.5, CFG=2.0
- 750 epochs, LR=5e-5, warmup=3000

Backward compatible with legacy DDPM mode via --no_flow_matching flag.
"""

import os
import sys
import time
import json
import argparse
import math

import torch
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.dit_ecg import dit_ecg_s, dit_ecg_b
from src.models.feature_extractor_pt import FeatureExtractorPT
from src.training.ema import EMAModel
from src.training.noise_scheduler import CosineNoiseScheduler
from src.training.flow_matching import FlowMatchingScheduler
from src.training.losses_v2 import combined_diffusion_loss, DifferentiableHRLoss, hr_variance_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train DiT-ECG (Run 6)')
    # ── Core Training ──
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--batch_size', type=int, default=32, help='Micro batch size per GPU')
    parser.add_argument('--accum_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=3000)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--cfg_dropout', type=float, default=0.1, help='CFG conditioning dropout rate')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='CFG guidance scale for eval')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--model_size', type=str, default='B', choices=['S', 'B'])
    # ── Data ──
    parser.add_argument('--dataset', type=str, default='mitbih', choices=['mitbih', 'ptbxl', 'synthetic'])
    parser.add_argument('--data_path', type=str, default='data/mitbih_forecasting.npz')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--use_identity_loss', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_bf16', action='store_true', default=True)
    # ── Flow Matching (Run 6) ──
    parser.add_argument('--no_flow_matching', action='store_true', default=False,
                        help='Disable flow matching, use legacy DDPM')
    parser.add_argument('--fm_num_steps', type=int, default=20,
                        help='Euler ODE steps for flow matching sampling')
    # ── Experiment Tracking ──
    parser.add_argument('--wandb_project', type=str, default='CardioEquation', help='W&B project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable W&B logging')
    parser.add_argument('--no_tensorboard', action='store_true', default=False, help='Disable TensorBoard')
    # ── Validation + Early Stopping ──
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--patience', type=int, default=60, help='Early stopping patience (epochs)')
    # ── Augmentation ──
    parser.add_argument('--no_augment', action='store_true', default=False, help='Disable DiT augmentation')
    return parser.parse_args()


# Cosine LR schedule with linear warmup.
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── ECG Augmenter for DiT Training ─────────────────────────────────
class ECGDiTAugmenter:
    """
    Identity-preserving augmentations for DiT diffusion training.

    Applied to both context and future signals with INDEPENDENT random seeds
    to prevent the model from simply copying augmentation patterns.

    Augmentations:
      - Amplitude scaling (0.8–1.2×): simulates sensor gain variation
      - Gaussian noise (σ=0.01): simulates sensor noise
      - Baseline wander (low-freq sinusoid): simulates electrode drift
      - Time warp (±10% resample): KEY for HR diversity
    """
    def __init__(self, p=0.5):
        self.p = p  # probability of applying each augmentation

    def __call__(self, x):
        """
        Apply random augmentations.
        x: (B, 1, T) tensor
        """
        B, C, T = x.shape

        # Amplitude scaling (0.8–1.2×)
        if torch.rand(1).item() < self.p:
            scale = 0.8 + 0.4 * torch.rand(B, 1, 1, device=x.device)
            x = x * scale

        # Gaussian noise (σ=0.01)
        if torch.rand(1).item() < self.p:
            noise = 0.01 * torch.randn_like(x)
            x = x + noise

        # Baseline wander (low-freq sinusoid, amplitude 0.05)
        if torch.rand(1).item() < self.p:
            t_axis = torch.linspace(0, 2 * math.pi, T, device=x.device)
            freq = 0.1 + 0.3 * torch.rand(B, 1, 1, device=x.device)
            phase = 2 * math.pi * torch.rand(B, 1, 1, device=x.device)
            wander = 0.05 * torch.sin(freq * t_axis.unsqueeze(0).unsqueeze(0) + phase)
            x = x + wander

        # Time warp (±10% resample) — KEY for HR diversity
        if torch.rand(1).item() < self.p:
            scale = 0.9 + 0.2 * torch.rand(1).item()  # 0.9 to 1.1
            new_len = int(T * scale)
            if new_len > 100 and new_len < T * 3:
                x = torch.nn.functional.interpolate(x, size=new_len, mode='linear', align_corners=True)
                if x.shape[-1] > T:
                    x = x[:, :, :T]
                elif x.shape[-1] < T:
                    pad = T - x.shape[-1]
                    x = torch.nn.functional.pad(x, (0, pad), mode='replicate')

        return x


def create_synthetic_dataset(num_samples=2000, signal_length=2500):
    """Create a simple synthetic dataset for smoke testing."""
    import numpy as np

    print(f"🔧 Creating synthetic dataset ({num_samples} samples)...")

    segments = []
    hr_labels = []
    for i in range(num_samples):
        t = np.linspace(0, 5, signal_length)
        hr = np.random.uniform(60, 100)
        freq = hr / 60.0

        # Simplified ECG: P-wave + QRS + T-wave
        p_wave = 0.15 * np.sin(2 * np.pi * freq * t)
        qrs = 1.0 * np.exp(-50 * (np.mod(t * freq, 1.0) - 0.4) ** 2)
        t_wave = 0.3 * np.exp(-10 * (np.mod(t * freq, 1.0) - 0.7) ** 2)
        noise = 0.02 * np.random.randn(signal_length)

        ecg = p_wave + qrs + t_wave + noise
        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
        segments.append(ecg)
        hr_labels.append(hr)

    segments = np.array(segments)[:, np.newaxis, :]
    hr_labels = np.array(hr_labels, dtype=np.float32)

    # context = future for synthetic (self-reconstruction)
    context = torch.from_numpy(segments).float()
    future = torch.from_numpy(segments).float()
    hr_tensor = torch.from_numpy(hr_labels).float()

    dataset = torch.utils.data.TensorDataset(context, future, hr_tensor)
    print(f"   ✅ Synthetic dataset: {num_samples} samples, shape {context.shape}")
    return dataset


def load_dataset(args):
    """Load training dataset based on args."""
    import numpy as np

    if args.dataset == 'synthetic':
        print("📊 Using synthetic dataset (smoke test)")
        return create_synthetic_dataset(num_samples=500)

    elif args.dataset == 'mitbih':
        data_path = args.data_path
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found at {data_path}")
            print("   Run: python download_all_datasets.py --mitbih to generate it")
            print("   Falling back to synthetic dataset...")
            return create_synthetic_dataset(num_samples=500)

        print(f"📊 Loading MIT-BIH dataset from {data_path}")
        data = np.load(data_path)
        # Expected shape: (N, T, 1) → convert to (N, 1, T) for PyTorch
        context = data['context']
        future = data['future']
        # Load HR labels if available
        hr_labels = data['hr_labels'] if 'hr_labels' in data else None

        if context.shape[-1] == 1:
            context = context.transpose(0, 2, 1)  # (N, 1, T)
            future = future.transpose(0, 2, 1)

        # ── Ensure signal length matches model (2500 samples = 5s at 500Hz) ──
        signal_len = context.shape[-1]
        target_len = 2500
        if signal_len != target_len:
            print(f"   ⚠️  Signal length is {signal_len}, model expects {target_len}")
            if signal_len > target_len:
                context = context[:, :, :target_len]
                future = future[:, :, :target_len]
                print(f"   ✂️  Truncated to {target_len} samples (first 5s)")
            else:
                pad = target_len - signal_len
                context = np.pad(context, ((0, 0), (0, 0), (0, pad)), mode='constant')
                future = np.pad(future, ((0, 0), (0, 0), (0, pad)), mode='constant')
                print(f"   📏 Padded to {target_len} samples")

        # ── Combine with PTB-XL if available (subsampled for epoch speed) ──
        ptbxl_path = 'data/ptbxl_processed.npz'
        if os.path.exists(ptbxl_path):
            ptbxl = np.load(ptbxl_path)
            ptbxl_signals = ptbxl['signals']  # (N, 1, 2500)
            # Subsample to cap epoch inflation — keep diversity, not volume
            max_ptbxl = min(4000, len(ptbxl_signals))
            rng = np.random.RandomState(42)
            idx = rng.choice(len(ptbxl_signals), max_ptbxl, replace=False)
            ptbxl_signals = ptbxl_signals[idx]
            context = np.concatenate([context, ptbxl_signals], axis=0)
            future = np.concatenate([future, ptbxl_signals], axis=0)
            print(f"   + PTB-XL: {max_ptbxl}/{len(ptbxl['signals'])} samples (subsampled)")
            if hr_labels is not None and 'hr_labels' in ptbxl:
                hr_labels = np.concatenate([hr_labels, ptbxl['hr_labels'][idx]], axis=0)

        # ── Combine with Chapman-Shaoxing if available (subsampled) ──
        chapman_path = 'data/chapman_processed.npz'
        if os.path.exists(chapman_path):
            chapman = np.load(chapman_path)
            chapman_signals = chapman['signals']  # (N, 1, 2500)
            max_chapman = min(4000, len(chapman_signals))
            rng = np.random.RandomState(43)
            idx = rng.choice(len(chapman_signals), max_chapman, replace=False)
            chapman_signals = chapman_signals[idx]
            context = np.concatenate([context, chapman_signals], axis=0)
            future = np.concatenate([future, chapman_signals], axis=0)
            print(f"   + Chapman: {max_chapman}/{len(chapman['signals'])} samples (subsampled)")
            if hr_labels is not None and 'hr_labels' in chapman:
                hr_labels = np.concatenate([hr_labels, chapman['hr_labels'][idx]], axis=0)

        n_mitbih = len(data['context'])
        print(f"   = Total: {len(context)} samples (MIT-BIH: {n_mitbih}, +PTB-XL/Chapman subsampled)")

        # Build HR labels tensor (default to 75 bpm if not available)
        if hr_labels is not None and len(hr_labels) == len(context):
            print(f"   ❤️  HR labels loaded: mean={hr_labels.mean():.1f}, "
                  f"std={hr_labels.std():.1f} bpm")
        else:
            print(f"   ⚠️  HR labels not found — using default 75 bpm")
            print(f"   ⚠️  Run: python src/data/precompute_hr.py first!")
            hr_labels = np.full(len(context), 75.0, dtype=np.float32)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(future).float(),
            torch.from_numpy(hr_labels).float(),
        )
        return dataset

    elif args.dataset == 'ptbxl':
        print("📊 PTB-XL dataset — use harmonized_dataset.py for full pipeline")
        print("   Falling back to synthetic dataset...")
        return create_synthetic_dataset(num_samples=500)

    else:
        return create_synthetic_dataset()


def train(args):
    """Main training loop — Run 6 with Flow Matching + Cross-Attention."""
    import numpy as np  # Used for HR-balanced sampling
    # ── Setup ─────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_flow_matching = not args.no_flow_matching

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  No GPU detected — training will be very slow")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Experiment Tracking ───────────────────────────────
    wandb_run = None
    tb_writer = None

    if not args.no_wandb:
        try:
            import wandb
            run_name = args.wandb_run or f"Run6-{'CFM' if use_flow_matching else 'DDPM'}-DiT-{args.model_size}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                reinit=True,
            )
            print(f"📊 W&B initialized: {wandb_run.url}")
        except Exception as e:
            print(f"   ⚠️  W&B init failed: {e} — continuing without W&B")

    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.output_dir, 'runs')
            tb_writer = SummaryWriter(log_dir=tb_dir)
            print(f"📊 TensorBoard initialized: {tb_dir}")
        except Exception as e:
            print(f"   ⚠️  TensorBoard init failed: {e} — continuing without TB")

    # ── Models ────────────────────────────────────────────────
    print(f"🏗️  Building DiT-ECG-{args.model_size} (Run 6: CrossAttn Identity)...")
    if args.model_size == 'S':
        model = dit_ecg_s().to(device)
    else:
        model = dit_ecg_b().to(device)

    feature_extractor = None
    if args.use_identity_loss:
        feature_extractor = FeatureExtractorPT().to(device)
        # ── Load pre-trained contrastive weights ──
        contrastive_path = os.path.join(args.output_dir, 'feature_extractor_contrastive.pt')
        if os.path.exists(contrastive_path):
            print(f"   📥 Loading pre-trained feature extractor from {contrastive_path}")
            state_dict = torch.load(contrastive_path, map_location=device, weights_only=True)
            feature_extractor.load_state_dict(state_dict)
            print(f"   ✅ Pre-trained weights loaded successfully!")
        else:
            print(f"   ⚠️  No pre-trained encoder found at {contrastive_path}")
            print(f"   ⚠️  Using random initialization (not recommended!)")
        # ── Freeze the encoder — do NOT train it alongside DiT ──
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        print(f"   ❄️  Feature extractor frozen for diffusion training.")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   DiT Parameters: {total_params:.1f}M")

    if feature_extractor:
        fe_params = sum(p.numel() for p in feature_extractor.parameters()) / 1e6
        print(f"   FeatureExtractor Parameters: {fe_params:.1f}M")

    # ── Scheduler ─────────────────────────────────────────────
    if use_flow_matching:
        fm_scheduler = FlowMatchingScheduler()
        print("   🌊 Training Mode: OT-CFM (Conditional Flow Matching)")
        print(f"   🌊 Sampling: Euler ODE, {args.fm_num_steps} steps")
    else:
        scheduler = CosineNoiseScheduler(num_train_timesteps=1000)
        print("   Training Mode: DDPM (legacy, cosine schedule)")
        print("   Sampling: DDIM, 50 steps")

    # ── EMA ────────────────────────────────────────────────────
    ema = EMAModel(model, decay=args.ema_decay, use_warmup=True)
    print(f"   EMA: decay={args.ema_decay}, warmup=True")

    # ── Augmenter ─────────────────────────────────────────────
    augmenter = None
    if not args.no_augment:
        augmenter = ECGDiTAugmenter(p=0.5)
        print("   Augmentation: ON (amplitude, noise, baseline wander, time-warp)")
    else:
        print("   Augmentation: OFF")

    # ── Dataset ───────────────────────────────────────────────
    full_dataset = load_dataset(args)

    # ── Train/Val Split ──
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"   Train: {train_size} samples | Val: {val_size} samples")

    # ── HR-balanced sampling ──
    try:
        all_hr = full_dataset.tensors[2].numpy()  # 3rd tensor = HR
        train_indices = train_dataset.indices
        train_hr = all_hr[train_indices]

        hr_bins = np.digitize(train_hr, bins=[40, 55, 65, 75, 85, 100, 120, 160])
        bin_counts = np.bincount(hr_bins, minlength=9).clip(min=1)
        sample_weights = 1.0 / bin_counts[hr_bins]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        print(f"   ❤️  HR-balanced sampling enabled (bins: {bin_counts})")
        use_hr_sampler = True
    except Exception as e:
        print(f"   ⚠️  HR-balanced sampling failed ({e}), using random shuffle")
        sampler = None
        use_hr_sampler = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(not use_hr_sampler),
        sampler=sampler if use_hr_sampler else None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"   Batch={args.batch_size}, accum={args.accum_steps}, "
          f"effective={args.batch_size * args.accum_steps}")

    # ── HR Loss (Run 6: timestep-gated) ────────────────────
    hr_loss_fn = DifferentiableHRLoss(
        fs=500.0, hr_min=30.0, hr_max=200.0, softmax_temp=10.0,
        t_gate=0.2,  # Only compute HR loss when t < 0.2 (low noise)
    ).to(device)
    hr_warmup_steps = 2000
    hr_loss_weight = 0.5    # Run 6: 10× increase from 0.05
    hr_var_weight = 0.5     # Run 6: matched weight for HR diversity
    print(f"   ❤️  HR Loss: DifferentiableHRLoss (gated t<0.2, warmup={hr_warmup_steps})")
    print(f"   ❤️  HR Loss weight: {hr_loss_weight}, HR variance weight: {hr_var_weight}")

    # ── Soft-DTW warmup schedule ──
    soft_dtw_target_weight = 0.3
    soft_dtw_start_epoch = 20      # Don't compute DTW at all before this
    soft_dtw_warmup_epochs = 100   # Ramp from 0→target over 100 epochs after start
    soft_dtw_batch_prob = 0.1      # Only compute DTW on 10% of batches (amortize CUDA kernel)
    print(f"   📐 Soft-DTW: target={soft_dtw_target_weight}, start=epoch {soft_dtw_start_epoch}, "
          f"warmup={soft_dtw_warmup_epochs}ep, batch_prob={soft_dtw_batch_prob}")

    # ── Optimizer & Scheduler ─────────────────────────────────
    all_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    total_steps = args.epochs * len(train_loader) // args.accum_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    print(f"   Optimizer: AdamW (lr={args.lr}, wd=0.01)")
    print(f"   LR Schedule: Cosine with {args.warmup_steps}-step warmup")
    print(f"   Total training steps: {total_steps}")
    print(f"   CFG dropout: {args.cfg_dropout}")
    print(f"   CFG guidance scale (eval): {args.guidance_scale}")
    print(f"   Early stopping: patience={args.patience} epochs")
    if args.use_bf16:
        print("   Precision: BF16 mixed precision")

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"   Resumed at epoch {start_epoch}, step {global_step}")

    # ── Training Loop ─────────────────────────────────────────
    mode_str = "OT-CFM" if use_flow_matching else "DDPM"
    print(f"\n🚀 Starting Run 6 training ({mode_str}) for {args.epochs} epochs...")
    print("=" * 70)

    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    val_loss_ema = None
    val_loss_ema_decay = 0.8

    for epoch in range(start_epoch, args.epochs):
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        # Per-component accumulators for epoch logging
        epoch_components = {}

        # Soft-DTW weight warmup (deferred start + stochastic batches)
        if epoch < soft_dtw_start_epoch:
            soft_dtw_weight = 0.0  # Completely skip DTW computation
        else:
            ramp = min(1.0, (epoch - soft_dtw_start_epoch) / max(1, soft_dtw_warmup_epochs))
            soft_dtw_weight = ramp * soft_dtw_target_weight

        for batch_idx, batch_data in enumerate(train_loader):
            context = batch_data[0].to(device)
            future = batch_data[1].to(device)
            hr_batch = batch_data[2].to(device) if len(batch_data) > 2 else \
                torch.full((batch_data[0].shape[0],), 75.0, device=device)

            # ── Apply augmentations ──
            if augmenter is not None:
                context = augmenter(context)
                future = augmenter(future)

            # ── Forward process ──────────────────────────
            with autocast(dtype=torch.bfloat16 if args.use_bf16 else torch.float32):

                if use_flow_matching:
                    # ── OT-CFM: velocity prediction ──
                    t = fm_scheduler.sample_t(future.shape[0], device)  # U[0,1]
                    noise = torch.randn_like(future)
                    x_t, v_target = fm_scheduler.interpolate(future, noise, t)

                    # Extract identity from context
                    if feature_extractor:
                        identity = feature_extractor(context)
                    else:
                        identity = torch.zeros(future.shape[0], 512, device=device)

                    # CFG: randomly drop conditioning during training
                    if args.cfg_dropout > 0:
                        drop_mask = torch.rand(identity.shape[0], device=device) < args.cfg_dropout
                        identity[drop_mask] = 0.0
                        hr_batch[drop_mask] = 75.0

                    # Predict velocity
                    v_pred = model(x_t, t, identity, hr_bpm=hr_batch)

                    # Reconstruct x_0 estimate for auxiliary losses
                    x_0_pred = fm_scheduler.predict_x0(x_t, v_pred, t)

                    # Compute multi-component loss
                    # Stochastic DTW: only compute on batch_prob fraction of batches
                    dtw_w_this_batch = soft_dtw_weight if (
                        soft_dtw_weight > 0 and torch.rand(1).item() < soft_dtw_batch_prob
                    ) else 0.0
                    loss, loss_dict = combined_diffusion_loss(
                        v_target, v_pred, future, x_0_pred,
                        feature_extractor=feature_extractor if args.use_identity_loss else None,
                        t_normalized=t,
                        w_soft_dtw=dtw_w_this_batch,
                        use_flow_matching=True,
                    )

                    t_for_hr = t  # Already normalized [0,1]

                else:
                    # ── Legacy DDPM: noise prediction ──
                    t = scheduler.sample_timesteps(future.shape[0], device)
                    x_t, noise = scheduler.q_sample(future, t)

                    if feature_extractor:
                        identity = feature_extractor(context)
                    else:
                        identity = torch.zeros(future.shape[0], 512, device=device)

                    if args.cfg_dropout > 0:
                        drop_mask = torch.rand(identity.shape[0], device=device) < args.cfg_dropout
                        identity[drop_mask] = 0.0
                        hr_batch[drop_mask] = 75.0

                    t_normalized = t.float() / scheduler.num_train_timesteps
                    noise_pred = model(x_t, t_normalized, identity, hr_bpm=hr_batch)

                    alpha_bar_t = scheduler.alpha_bar_t.to(device)[t].view(-1, 1, 1)
                    x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)

                    dtw_w_this_batch = soft_dtw_weight if (
                        soft_dtw_weight > 0 and torch.rand(1).item() < soft_dtw_batch_prob
                    ) else 0.0
                    loss, loss_dict = combined_diffusion_loss(
                        noise, noise_pred, future, x_0_pred,
                        feature_extractor=feature_extractor if args.use_identity_loss else None,
                        alpha_bar_t=alpha_bar_t.squeeze(),
                        w_soft_dtw=dtw_w_this_batch,
                        use_flow_matching=False,
                    )

                    t_for_hr = t_normalized

                # ── HR losses (timestep-gated, Run 6) ──
                hr_ramp = min(1.0, global_step / hr_warmup_steps)
                hr_loss_val, hr_est = hr_loss_fn(
                    x_0_pred, hr_batch, t_for_hr,
                    weight=hr_loss_weight * hr_ramp
                )
                hr_var_val = hr_variance_loss(
                    hr_est, target_std=47.0, weight=hr_var_weight * hr_ramp
                )
                loss = loss + hr_loss_val + hr_var_val

                # Add HR metrics to loss_dict
                loss_dict['hr_loss'] = hr_loss_val.detach()
                loss_dict['hr_variance_loss'] = hr_var_val.detach()
                loss_dict['hr_est_mean'] = hr_est.mean()
                loss_dict['hr_est_std'] = hr_est.std()
                loss_dict['hr_ramp'] = hr_ramp
                loss_dict['soft_dtw_weight'] = soft_dtw_weight

                loss = loss / args.accum_steps  # Scale for gradient accumulation

            # Backward
            loss.backward()

            # Accumulate per-component losses
            for k, v in loss_dict.items():
                if k not in epoch_components:
                    epoch_components[k] = 0.0
                val = v.item() if torch.is_tensor(v) else v
                epoch_components[k] += val

            # Gradient accumulation step
            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                optimizer.step()
                ema.update(model, step=global_step)
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']

                # Step-level logging (every 50 steps)
                if global_step % 50 == 0:
                    step_loss = loss.item() * args.accum_steps
                    log_data = {
                        'loss_step': step_loss,
                        'lr': current_lr,
                        'step': global_step,
                    }
                    for k, v in loss_dict.items():
                        val = v.item() if torch.is_tensor(v) else v
                        log_data[f'component/{k}'] = val

                    if tb_writer:
                        for k, v in log_data.items():
                            tb_writer.add_scalar(f'train/{k}', v, global_step)
                    if wandb_run:
                        import wandb
                        wandb.log(log_data)

            epoch_loss += loss.item() * args.accum_steps
            num_batches += 1

        # ── Epoch Summary ─────────────────────────────────
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Average per-component losses
        avg_components = {k: v / max(num_batches, 1) for k, v in epoch_components.items()}

        # ── Validation Loss ──────────────────────────────
        val_loss = 0.0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                context_v = val_batch[0].to(device)
                future_v = val_batch[1].to(device)
                hr_v = val_batch[2].to(device) if len(val_batch) > 2 else \
                    torch.full((val_batch[0].shape[0],), 75.0, device=device)

                with autocast(dtype=torch.bfloat16 if args.use_bf16 else torch.float32):
                    if use_flow_matching:
                        t = fm_scheduler.sample_t(future_v.shape[0], device)
                        noise = torch.randn_like(future_v)
                        x_t, v_target = fm_scheduler.interpolate(future_v, noise, t)

                        if feature_extractor:
                            identity = feature_extractor(context_v)
                        else:
                            identity = torch.zeros(future_v.shape[0], 512, device=device)

                        v_pred = model(x_t, t, identity, hr_bpm=hr_v)
                        x_0_pred = fm_scheduler.predict_x0(x_t, v_pred, t)

                        v_loss, _ = combined_diffusion_loss(
                            v_target, v_pred, future_v, x_0_pred,
                            feature_extractor=feature_extractor if args.use_identity_loss else None,
                            t_normalized=t,
                            use_flow_matching=True,
                        )
                    else:
                        t = scheduler.sample_timesteps(future_v.shape[0], device)
                        x_t, noise = scheduler.q_sample(future_v, t)

                        if feature_extractor:
                            identity = feature_extractor(context_v)
                        else:
                            identity = torch.zeros(future_v.shape[0], 512, device=device)

                        t_normalized = t.float() / scheduler.num_train_timesteps
                        noise_pred = model(x_t, t_normalized, identity, hr_bpm=hr_v)

                        alpha_bar_t = scheduler.alpha_bar_t.to(device)[t].view(-1, 1, 1)
                        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)

                        v_loss, _ = combined_diffusion_loss(
                            noise, noise_pred, future_v, x_0_pred,
                            feature_extractor=feature_extractor if args.use_identity_loss else None,
                            alpha_bar_t=alpha_bar_t.squeeze(),
                            use_flow_matching=False,
                        )

                    # Add HR loss to validation
                    t_for_hr_val = t if use_flow_matching else t_normalized
                    v_hr_loss, _ = hr_loss_fn(x_0_pred, hr_v, t_for_hr_val, weight=hr_loss_weight)
                    v_loss = v_loss + v_hr_loss

                val_loss += v_loss.item()
                val_batches += 1
        model.train()

        avg_val_loss = val_loss / max(val_batches, 1)

        # EMA-smoothed val loss
        if val_loss_ema is None:
            val_loss_ema = avg_val_loss
        else:
            val_loss_ema = val_loss_ema_decay * val_loss_ema + (1 - val_loss_ema_decay) * avg_val_loss

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"Val(EMA): {val_loss_ema:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"DTW_w: {soft_dtw_weight:.3f} | "
              f"Time: {elapsed:.1f}s | "
              f"Step: {global_step}")

        # Print component breakdown every 10 epochs
        if (epoch + 1) % 10 == 0:
            components_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_components.items()])
            print(f"   Components: {components_str}")

        # ── Epoch-level logging ──
        epoch_log = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'val_loss': avg_val_loss,
            'val_loss_ema': val_loss_ema,
            'lr': current_lr,
            'epoch_time_s': elapsed,
            'best_loss': best_loss if avg_loss >= best_loss else avg_loss,
            'best_val_loss': best_val_loss if avg_val_loss >= best_val_loss else avg_val_loss,
            'patience_counter': patience_counter,
            'soft_dtw_weight': soft_dtw_weight,
        }
        for k, v in avg_components.items():
            epoch_log[f'component/{k}'] = v

        if tb_writer:
            for k, v in epoch_log.items():
                tb_writer.add_scalar(f'train/{k}', v, epoch + 1)
        if wandb_run:
            import wandb
            wandb.log(epoch_log)

        # ── Checkpointing ─────────────────────────────────
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"dit_ecg_epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict(),
                'args': vars(args),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'use_flow_matching': use_flow_matching,
            }, ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path}")

        # Best model (by EMA-smoothed validation loss)
        if val_loss_ema < best_val_loss:
            best_val_loss = val_loss_ema
            patience_counter = 0
            best_path = os.path.join(args.output_dir, "dit_ecg_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'val_loss_ema': val_loss_ema,
                'use_flow_matching': use_flow_matching,
            }, best_path)
            print(f"   🏆 New best val_loss(EMA): {val_loss_ema:.6f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping triggered! No val improvement for {args.patience} epochs.")
                print(f"   Best val_loss(EMA): {best_val_loss:.6f}")
                break

        if avg_loss < best_loss:
            best_loss = avg_loss

    # ── Save Final EMA Model ──────────────────────────────
    print("\n" + "=" * 70)
    print(f"✅ Training complete! Best val_loss: {best_val_loss:.6f} | Best train_loss: {best_loss:.6f}")

    # Save EMA-only weights for inference
    ema.apply_to(model)
    ema_path = os.path.join(args.output_dir, "dit_ecg_ema_final.pt")
    torch.save(model.state_dict(), ema_path)
    print(f"💾 EMA model saved: {ema_path}")
    ema.restore(model)

    # Save training config
    config = vars(args)
    config['use_flow_matching'] = use_flow_matching
    config['run'] = 'Run6'
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📋 Config saved: {config_path}")

    # ── Finalize Experiment Tracking ──
    if tb_writer:
        tb_writer.close()
        print("📊 TensorBoard logs saved.")
    if wandb_run:
        import wandb
        wandb.log({'final_loss': best_loss, 'final_val_loss': best_val_loss})
        wandb.finish()
        print("📊 W&B run finished.")


if __name__ == '__main__':
    args = parse_args()
    train(args)
