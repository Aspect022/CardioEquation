"""
Contrastive Pre-Training for ECG Identity Encoder
====================================================
Pre-trains the FeatureExtractor using SimCLR-style InfoNCE loss
to learn a patient-specific identity embedding space.

This is Stage 0 — run BEFORE diffusion training.

Usage:
    python src/training/train_contrastive.py --data_path data/mitbih_forecasting.npz --epochs 100
"""

import torch
import torch.nn.functional as F
import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.feature_extractor_pt import ContrastiveFeatureExtractor
from src.training.losses_v2 import info_nce_loss


class ECGAugmenter:
    """
    Identity-preserving augmentations for contrastive ECG learning.

    These augmentations change recording artifacts WITHOUT changing
    patient-specific cardiac physiology.
    """

    def __init__(self):
        pass

    def __call__(self, x):
        """Apply random augmentations. x: (1, T) tensor."""
        x = x.clone()
        device = x.device

        # 1. Amplitude scaling (preserves morphology shape)
        if torch.rand(1).item() < 0.8:
            scale = torch.empty(1, device=device).uniform_(0.5, 2.0)
            x = x * scale

        # 2. Temporal shift (small jitter)
        if torch.rand(1).item() < 0.7:
            shift = torch.randint(-250, 250, (1,)).item()  # ±0.5s at 500Hz
            x = torch.roll(x, shift, dims=-1)

        # 3. Gaussian noise (recording artifacts)
        if torch.rand(1).item() < 0.8:
            sigma = torch.empty(1, device=device).uniform_(0.0, 0.02)
            x = x + torch.randn_like(x) * sigma

        # 4. Baseline wander (low-frequency artifact)
        if torch.rand(1).item() < 0.5:
            T = x.shape[-1]
            t = torch.linspace(0, 2 * 3.14159, T, device=device)
            freq = torch.empty(1, device=device).uniform_(0.05, 0.5)
            amplitude = torch.empty(1, device=device).uniform_(0.0, 0.1)
            wander = amplitude * torch.sin(freq * t)
            x = x + wander.unsqueeze(0)

        # 5. Random crop and resize
        if torch.rand(1).item() < 0.5:
            T = x.shape[-1]
            crop_ratio = torch.empty(1).uniform_(0.7, 1.0).item()
            crop_len = int(T * crop_ratio)
            start = torch.randint(0, T - crop_len + 1, (1,)).item()
            x_crop = x[:, start:start + crop_len]
            x = F.interpolate(x_crop.unsqueeze(0), size=T, mode='linear', align_corners=False).squeeze(0)

        # Re-normalize
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x


def create_contrastive_dataset(data_path, signal_length=2500):
    """
    Create pairs for contrastive learning.
    Each sample is a segment; positive pairs are adjacent segments
    from the same recording (same patient).
    """
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, generating synthetic")
        # Synthetic fallback: 200 "patients" × 10 segments each
        all_segments = []
        patient_ids = []
        for pid in range(200):
            hr = np.random.uniform(60, 100)
            amplitude_scale = np.random.uniform(0.5, 1.5)
            t = np.linspace(0, 5, signal_length)
            for seg in range(10):
                signal = np.zeros(signal_length)
                beat_dur = 60.0 / hr
                for bs in np.arange(0, 5, beat_dur):
                    signal += amplitude_scale * np.exp(-((t - bs - 0.22)**2) / (2 * 0.008**2))
                    signal += 0.2 * amplitude_scale * np.exp(-((t - bs - 0.1)**2) / (2 * 0.02**2))
                    signal += 0.3 * amplitude_scale * np.exp(-((t - bs - 0.4)**2) / (2 * 0.04**2))
                signal += np.random.normal(0, 0.01 * seg, signal_length)
                signal = (signal - signal.mean()) / (signal.std() + 1e-8)
                all_segments.append(signal.astype(np.float32))
                patient_ids.append(pid)

        segments = np.array(all_segments)[:, np.newaxis, :]
        patient_ids = np.array(patient_ids)
    else:
        data = np.load(data_path)
        context = data['context']
        if context.shape[-1] == 1:
            context = context.transpose(0, 2, 1)
        segments = context
        # For MIT-BIH: approximate patient IDs based on ordering
        # Each record ~3 segments, so group by record
        patient_ids = np.arange(len(segments)) // 3

    return torch.from_numpy(segments).float(), torch.from_numpy(patient_ids).long()


def train_contrastive(args):
    """Pre-train the identity encoder with contrastive learning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    print("🔬 Stage 0: Contrastive Pre-Training for Identity Encoder")
    print("=" * 60)

    # Model
    model = ContrastiveFeatureExtractor(proj_dim=256).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Model: ContrastiveFeatureExtractor ({params:.1f}M params)")

    # Data
    segments, patient_ids = create_contrastive_dataset(args.data_path)
    print(f"   Dataset: {len(segments)} segments, {len(torch.unique(patient_ids))} patients")

    # Augmenter
    augmenter = ECGAugmenter()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)

    # Training
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        # Shuffle
        perm = torch.randperm(len(segments))
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(segments) - args.batch_size, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = segments[idx].to(device)

            # Create two augmented views
            view1 = torch.stack([augmenter(s) for s in batch])
            view2 = torch.stack([augmenter(s) for s in batch])

            # Forward through encoder + projection
            z1 = model(view1, return_projection=True)
            z2 = model(view2, return_projection=True)

            # InfoNCE loss
            loss = info_nce_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # Save only the encoder weights (discard projection head)
    encoder_state = model.encoder.state_dict()
    save_path = os.path.join(args.output_dir, "feature_extractor_contrastive.pt")
    torch.save(encoder_state, save_path)
    print(f"\n✅ Contrastive pre-training complete!")
    print(f"💾 Encoder saved: {save_path}")
    print(f"   Use this to initialize FeatureExtractorPT in train_dit.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/mitbih_forecasting.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_contrastive(args)
