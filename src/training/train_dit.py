"""
DiT-ECG Training Script
=========================
Main training entry point for CardioEquation DiT-ECG-B.

Features:
- Cosine noise schedule (DDPM)
- EMA (decay=0.9999 with warmup)
- BF16 mixed precision
- Gradient accumulation (effective batch size = micro_batch × accum_steps)
- Gradient clipping (max_norm=1.0)
- Classifier-Free Guidance training (10% conditioning dropout)
- Cosine LR schedule with linear warmup
- Checkpoint saving
- Weights & Biases + TensorBoard logging

Usage:
    python src/training/train_dit.py --epochs 200 --batch_size 32 --accum_steps 8
    python src/training/train_dit.py --epochs 2 --batch_size 4 --dataset synthetic  # smoke test
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import sys
import argparse
import time
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.dit_ecg import dit_ecg_b, dit_ecg_s
from src.models.feature_extractor_pt import FeatureExtractorPT
from src.training.ema import EMAModel
from src.training.noise_scheduler import CosineNoiseScheduler
from src.training.losses_v2 import combined_diffusion_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train DiT-ECG')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32, help='Micro batch size per GPU')
    parser.add_argument('--accum_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--cfg_dropout', type=float, default=0.1, help='CFG conditioning dropout rate')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale for eval')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--model_size', type=str, default='B', choices=['S', 'B'])
    parser.add_argument('--dataset', type=str, default='mitbih', choices=['mitbih', 'ptbxl', 'synthetic'])
    parser.add_argument('--data_path', type=str, default='data/mitbih_forecasting.npz')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--use_identity_loss', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_bf16', action='store_true', default=True)
    # ── Experiment Tracking ──
    parser.add_argument('--wandb_project', type=str, default='CardioEquation', help='W&B project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='W&B run name (auto-generated if None)')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable W&B logging')
    parser.add_argument('--no_tensorboard', action='store_true', default=False, help='Disable TensorBoard')
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_synthetic_dataset(num_samples=2000, signal_length=2500):
    """Create a simple synthetic dataset for smoke testing."""
    import numpy as np

    contexts = []
    futures = []

    for i in range(num_samples):
        # Generate random ECG-like signal (5 Gaussian bumps)
        t = np.linspace(0, 5, signal_length)
        hr = np.random.uniform(60, 100)
        beat_duration = 60.0 / hr

        signal = np.zeros(signal_length)
        for beat_start in np.arange(0, 5, beat_duration):
            # P wave
            signal += 0.2 * np.exp(-((t - beat_start - 0.1) ** 2) / (2 * 0.02 ** 2))
            # QRS complex
            signal += -0.15 * np.exp(-((t - beat_start - 0.2) ** 2) / (2 * 0.01 ** 2))
            signal += 1.0 * np.exp(-((t - beat_start - 0.22) ** 2) / (2 * 0.008 ** 2))
            signal += -0.2 * np.exp(-((t - beat_start - 0.24) ** 2) / (2 * 0.01 ** 2))
            # T wave
            signal += 0.3 * np.exp(-((t - beat_start - 0.4) ** 2) / (2 * 0.04 ** 2))

        # Add small noise for context
        noisy = signal + np.random.normal(0, 0.05, signal_length)

        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        noisy = (noisy - noisy.mean()) / (noisy.std() + 1e-8)

        contexts.append(noisy.astype(np.float32))
        futures.append(signal.astype(np.float32))

    contexts = np.array(contexts)[:, np.newaxis, :]  # (N, 1, T)
    futures = np.array(futures)[:, np.newaxis, :]     # (N, 1, T)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(contexts),
        torch.from_numpy(futures)
    )
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

        # ── Combine with PTB-XL if available ──
        ptbxl_path = 'data/ptbxl_processed.npz'
        if os.path.exists(ptbxl_path):
            print(f"   + Loading PTB-XL from {ptbxl_path}")
            ptbxl = np.load(ptbxl_path)
            ptbxl_signals = ptbxl['signals']  # (N, 1, 2500)
            context = np.concatenate([context, ptbxl_signals], axis=0)
            future = np.concatenate([future, ptbxl_signals], axis=0)

        # ── Combine with Chapman-Shaoxing if available ──
        chapman_path = 'data/chapman_processed.npz'
        if os.path.exists(chapman_path):
            print(f"   + Loading Chapman-Shaoxing from {chapman_path}")
            chapman = np.load(chapman_path)
            chapman_signals = chapman['signals']  # (N, 1, 2500)
            context = np.concatenate([context, chapman_signals], axis=0)
            future = np.concatenate([future, chapman_signals], axis=0)

        n_mitbih = len(data['context'])
        print(f"   = Combined dataset: {len(context)} samples (MIT-BIH: {n_mitbih})")

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(future).float()
        )
        return dataset

    elif args.dataset == 'ptbxl':
        # PTB-XL loading will be handled by the harmonized dataset
        print("📊 PTB-XL dataset — use harmonized_dataset.py for full pipeline")
        print("   Falling back to synthetic dataset...")
        return create_synthetic_dataset(num_samples=500)

    else:
        return create_synthetic_dataset()


def train(args):
    """Main training loop."""
    # ── Setup ─────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"DiT-ECG-{args.model_size}_e{args.epochs}",
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
    print(f"🏗️  Building DiT-ECG-{args.model_size}...")
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

    # ── EMA ────────────────────────────────────────────────────
    ema = EMAModel(model, decay=args.ema_decay, use_warmup=True)
    print(f"   EMA: decay={args.ema_decay}, warmup=True")

    # ── Noise Scheduler ───────────────────────────────────────
    scheduler = CosineNoiseScheduler(num_train_timesteps=1000)
    print("   Noise Schedule: Cosine (1000 discrete steps)")

    # ── Dataset ───────────────────────────────────────────────
    dataset = load_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"   Dataset: {len(dataset)} samples, batch={args.batch_size}, "
          f"accum={args.accum_steps}, effective={args.batch_size * args.accum_steps}")

    # ── Optimizer & Scheduler ─────────────────────────────────
    # Only train DiT parameters — feature extractor is frozen
    all_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    total_steps = args.epochs * len(dataloader) // args.accum_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    print(f"   Optimizer: AdamW (lr={args.lr}, wd=0.01)")
    print(f"   LR Schedule: Cosine with {args.warmup_steps}-step warmup")
    print(f"   Total training steps: {total_steps}")
    print(f"   CFG dropout: {args.cfg_dropout}")
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
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 70)

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # feature_extractor stays in eval() mode — frozen

        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, (context, future) in enumerate(dataloader):
            context = context.to(device)
            future = future.to(device)

            # ── Forward diffusion ──────────────────────────
            with autocast(dtype=torch.bfloat16 if args.use_bf16 else torch.float32):
                # Sample timesteps
                t = scheduler.sample_timesteps(future.shape[0], device)

                # Add noise to future (clean target)
                x_t, noise = scheduler.q_sample(future, t)

                # Extract identity from context
                if feature_extractor:
                    identity = feature_extractor(context)
                else:
                    identity = torch.zeros(future.shape[0], 512, device=device)

                # CFG: randomly drop conditioning during training
                if args.cfg_dropout > 0:
                    drop_mask = torch.rand(identity.shape[0], device=device) < args.cfg_dropout
                    identity[drop_mask] = 0.0

                # Predict noise
                t_normalized = t.float() / scheduler.num_train_timesteps
                noise_pred = model(x_t, t_normalized, identity)

                # Reconstruct x_0 estimate (for auxiliary losses)
                alpha_bar_t = scheduler.alpha_bar_t.to(device)[t].view(-1, 1, 1)
                x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)

                # Compute loss (with SNR weighting for auxiliary losses)
                loss = combined_diffusion_loss(
                    noise, noise_pred, future, x_0_pred,
                    feature_extractor=feature_extractor if args.use_identity_loss else None,
                    alpha_bar_t=alpha_bar_t.squeeze(),
                )
                loss = loss / args.accum_steps  # Scale for gradient accumulation

            # Backward
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                optimizer.step()
                ema.update(model, step=global_step)
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Step-level logging (every 50 steps) ──
                if global_step % 50 == 0:
                    step_loss = loss.item() * args.accum_steps
                    if tb_writer:
                        tb_writer.add_scalar('train/loss_step', step_loss, global_step)
                        tb_writer.add_scalar('train/lr', current_lr, global_step)
                    if wandb_run:
                        import wandb
                        wandb.log({'loss_step': step_loss, 'lr': current_lr, 'step': global_step})

            epoch_loss += loss.item() * args.accum_steps
            num_batches += 1

        # ── Epoch Summary ─────────────────────────────────
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed:.1f}s | "
              f"Step: {global_step}")

        # ── Epoch-level logging ──
        if tb_writer:
            tb_writer.add_scalar('train/loss_epoch', avg_loss, epoch + 1)
            tb_writer.add_scalar('train/lr_epoch', current_lr, epoch + 1)
            tb_writer.add_scalar('train/epoch_time_s', elapsed, epoch + 1)
        if wandb_run:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': current_lr,
                'epoch_time_s': elapsed,
                'best_loss': best_loss if avg_loss >= best_loss else avg_loss,
            })

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
            }, ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path}")

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "dit_ecg_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'loss': best_loss,
            }, best_path)

    # ── Save Final EMA Model ──────────────────────────────
    print("\n" + "=" * 70)
    print(f"✅ Training complete! Best loss: {best_loss:.6f}")

    # Save EMA-only weights for inference
    ema.apply_to(model)
    ema_path = os.path.join(args.output_dir, "dit_ecg_ema_final.pt")
    torch.save(model.state_dict(), ema_path)
    print(f"💾 EMA model saved: {ema_path}")
    ema.restore(model)

    # Save training config
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"📋 Config saved: {config_path}")

    # ── Finalize Experiment Tracking ──
    if tb_writer:
        tb_writer.close()
        print("📊 TensorBoard logs saved.")
    if wandb_run:
        import wandb
        wandb.log({'final_loss': best_loss})
        wandb.finish()
        print("📊 W&B run finished.")


if __name__ == '__main__':
    args = parse_args()
    train(args)
