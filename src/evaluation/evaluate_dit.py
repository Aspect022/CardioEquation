"""
DiT-ECG Clinical Evaluation Script
=====================================
Loads the best DiT checkpoint, generates ECG samples from real patient
context signals, then computes FFD, MMD, HR-MAE, and Re-ID metrics.

This is the primary evaluation entry-point for a completed training run.

Usage:
    python src/evaluation/evaluate_dit.py \\
        --checkpoint checkpoints/dit_ecg_best.pt \\
        --output_dir outputs/clinical_validation_run5b \\
        --n_samples 200
"""

import os
import sys
import json
import argparse
import numpy as np

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(n_samples: int):
    """
    Load real ECG signals from the pre-processed .npz files (all 3 sources).

    Returns:
        signals : (N, 2500) float32 numpy array  — one lead per sample
        patient_ids : (N,) int array — patient identity labels
    """
    sources = [
        ('data/mitbih_forecasting.npz',  'context'),
        ('data/ptbxl_processed.npz',     'signals'),
        ('data/chapman_processed.npz',   'signals'),
    ]

    all_signals = []
    all_pids    = []
    pid_offset  = 0

    for path, key in sources:
        if not os.path.exists(path):
            print(f"  ⚠️  {path} not found — skipping")
            continue
        d = np.load(path, allow_pickle=True)
        sigs = d[key]                         # (N, T) or (N, 1, T)
        if sigs.ndim == 3:
            sigs = sigs[:, 0, :]              # (N, T)

        # Truncate / pad to 2500
        if sigs.shape[1] > 2500:
            sigs = sigs[:, :2500]
        elif sigs.shape[1] < 2500:
            sigs = np.pad(sigs, ((0, 0), (0, 2500 - sigs.shape[1])))

        # Patient IDs — use provided ones or synthesise sequential IDs
        if 'patient_ids' in d:
            pids = d['patient_ids'].astype(np.int64)
        else:
            pids = np.arange(len(sigs), dtype=np.int64)

        all_signals.append(sigs.astype(np.float32))
        all_pids.append(pids + pid_offset)
        pid_offset += int(pids.max()) + 1
        print(f"  ✅ Loaded {len(sigs):,} samples from {path}")

    if not all_signals:
        raise RuntimeError("No dataset files found. Run from the project root.")

    signals    = np.concatenate(all_signals, axis=0)
    patient_ids = np.concatenate(all_pids,   axis=0)

    # Normalise each signal
    mu  = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1,  keepdims=True) + 1e-8
    signals = (signals - mu) / std

    # Random subsample
    idx = np.random.choice(len(signals), size=min(n_samples, len(signals)), replace=False)
    return signals[idx], patient_ids[idx]


@torch.no_grad()
def generate_samples(
    dit_path: str,
    fe_path:  str,
    real_signals: np.ndarray,      # (N, 2500)
    device: torch.device,
    ddim_steps: int = 50,
    guidance_scale: float = 3.0,
    target_hr: float = 82.0,       # bpm — mid-range physiological value
) -> np.ndarray:
    """
    Generate one synthetic ECG for every real signal in real_signals.

    Returns:
        generated : (N, 2500) float32 numpy array
    """
    from src.models.dit_ecg import dit_ecg_b
    from src.models.feature_extractor_pt import FeatureExtractorPT
    from src.training.noise_scheduler import CosineNoiseScheduler

    # ── Load DiT ──────────────────────────────────────────────────────────────
    print("🔧 Loading DiT-ECG-B checkpoint...")
    model = dit_ecg_b()
    ckpt = torch.load(dit_path, map_location=device, weights_only=True)

    # Handle both raw state-dict and wrapped checkpoint formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'ema_state_dict' in ckpt:
        state = ckpt['ema_state_dict']
    else:
        state = ckpt

    # Strip 'module.' prefix if saved from DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print(f"  DiT params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Load Feature Extractor ─────────────────────────────────────────────
    fe = FeatureExtractorPT().to(device).eval()
    if os.path.exists(fe_path):
        fe_ckpt = torch.load(fe_path, map_location=device, weights_only=True)
        if isinstance(fe_ckpt, dict) and 'model_state_dict' in fe_ckpt:
            fe_ckpt = fe_ckpt['model_state_dict']
        fe.load_state_dict(fe_ckpt, strict=False)
        print(f"  ✅ Feature extractor loaded from {fe_path}")
    else:
        print(f"  ⚠️  FE weights not found at {fe_path} — using random encoder")

    scheduler = CosineNoiseScheduler(num_train_timesteps=1000)
    # Move alpha schedule to device once
    alpha_bar = scheduler.alpha_bar_t.to(device)   # (1000,)

    # ── DDIM timestep schedule ─────────────────────────────────────────────
    total_T = scheduler.num_train_timesteps
    step_ratio = total_T // ddim_steps
    timesteps = list(reversed(range(0, total_T, step_ratio)))[:ddim_steps]

    # ── Generate in mini-batches of 16 ────────────────────────────────────
    generated = []
    batch_size = 16
    N = len(real_signals)


    print(f"🎨 Generating {N} samples ({ddim_steps} DDIM steps, cfg={guidance_scale})...")

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        ctx  = torch.from_numpy(real_signals[start:end]).unsqueeze(1).to(device)  # (B,1,T)
        B    = ctx.shape[0]

        # Identity embedding from real context signal
        identity = fe(ctx)                            # (B, 512)

        # HR conditioning — one value per sample in batch
        hr_cond = torch.full((B,), target_hr, dtype=torch.float32, device=device)

        # Start from pure noise
        x = torch.randn(B, 1, 2500, device=device)

        for t_val in timesteps:
            # t as continuous float [0,1] — matches model.forward signature
            t_batch = torch.full((B,), t_val / total_T, dtype=torch.float32, device=device)

            # CFG: forward_with_cfg handles cond + null internally
            noise_pred = model.forward_with_cfg(
                x, t_batch, identity, hr_bpm=hr_cond, guidance_scale=guidance_scale
            )

            # DDIM step
            alpha_t  = alpha_bar[t_val]
            alpha_t1 = alpha_bar[max(t_val - step_ratio, 0)]

            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-4, 4)

            dir_xt = (1 - alpha_t1).clamp(min=0).sqrt() * noise_pred
            x      = alpha_t1.sqrt() * x0_pred + dir_xt

        gen_np = x.squeeze(1).cpu().numpy()   # (B, 2500)

        # Per-sample normalise
        mu  = gen_np.mean(axis=1, keepdims=True)
        std = gen_np.std(axis=1,  keepdims=True) + 1e-8
        gen_np = (gen_np - mu) / std

        generated.append(gen_np)

        print(f"  {min(end, N)}/{N} samples generated", end='\r')

    print()
    return np.concatenate(generated, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DiT-ECG Clinical Evaluation")
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/dit_ecg_best.pt',
                        help='Path to trained DiT checkpoint')
    parser.add_argument('--fe_path',     type=str, default='checkpoints/feature_extractor_contrastive.pt',
                        help='Path to frozen feature extractor weights')
    parser.add_argument('--output_dir',  type=str, default='outputs/clinical_validation_run5b',
                        help='Directory to save results')
    parser.add_argument('--n_samples',   type=int, default=200,
                        help='Number of real/fake pairs to evaluate')
    parser.add_argument('--ddim_steps',  type=int, default=50,
                        help='DDIM denoising steps (fewer = faster)')
    parser.add_argument('--guidance',    type=float, default=3.0,
                        help='Classifier-Free Guidance scale')
    parser.add_argument('--target_hr',   type=float, default=82.0,
                        help='Target HR (bpm) to condition generation on')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📂 Output: {args.output_dir}")
    print()

    # ── 1. Load real data ──────────────────────────────────────────────────
    print("📊 Loading real ECG data...")
    real_signals, patient_ids = load_dataset(args.n_samples)
    print(f"  Total real samples: {len(real_signals)}")
    print()

    # ── 2. Generate ────────────────────────────────────────────────────────
    generated_signals = generate_samples(
        dit_path       = args.checkpoint,
        fe_path        = args.fe_path,
        real_signals   = real_signals,
        device         = device,
        ddim_steps     = args.ddim_steps,
        guidance_scale = args.guidance,
        target_hr      = args.target_hr,
    )
    print()

    # Save raw arrays for later inspection
    np.savez_compressed(
        os.path.join(args.output_dir, 'generated_samples.npz'),
        real=real_signals,
        generated=generated_signals,
    )

    # ── 3. Evaluate ────────────────────────────────────────────────────────
    print("📏 Computing evaluation metrics...")
    try:
        import neurokit2  # noqa: F401
    except ImportError:
        print("  ⚠️  neurokit2 not installed — HR/QRS metrics will be N/A")
        print("      Install with:  pip install neurokit2")

    from src.evaluation.eval_metrics import ECGEvaluator
    from src.models.feature_extractor_pt import FeatureExtractorPT

    encoder = FeatureExtractorPT()
    if os.path.exists(args.fe_path):
        fe_ckpt = torch.load(args.fe_path, map_location=device, weights_only=True)
        if isinstance(fe_ckpt, dict) and 'model_state_dict' in fe_ckpt:
            fe_ckpt = fe_ckpt['model_state_dict']
        encoder.load_state_dict(fe_ckpt, strict=False)
    encoder = encoder.to(device)

    evaluator = ECGEvaluator(encoder=encoder, device=device)
    results   = evaluator.evaluate_all(real_signals, generated_signals, fs=500)

    real_morph = results.get('real_morphology', {})
    fake_morph = results.get('fake_morphology', {})

    def _fmt(v, fmt='.4f', suffix=''):
        """Format a metric value safely, returning N/A if unavailable."""
        if isinstance(v, (int, float)):
            return f"{v:{fmt}}{suffix}"
        return 'N/A (install neurokit2)'

    # ── 5. Print results ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("📊 RUN 5b EVALUATION RESULTS")
    print("=" * 60)
    print(f"  FFD  (↓ better, Run4=21.7): {_fmt(results.get('FFD'), '.4f')}")
    print(f"  MMD  (↓ better, Run4=0.43): {_fmt(results.get('MMD'), '.6f')}")
    print(f"  HR MAE bpm (↓ better, Run4=31.4): {_fmt(results.get('HR_MAE'), '.2f')}")
    reid1 = results.get('ReID_Top1', 0)
    reid5 = results.get('ReID_Top5', 0)
    print(f"  ReID Top-1 (↑ better, Run4=11.8%): {reid1*100:.1f}%")
    print(f"  ReID Top-5 (↑ better, Run4=35.3%): {reid5*100:.1f}%")
    print()
    print("  Real ECG morphology:")
    print(f"    HR mean: {_fmt(real_morph.get('hr_mean'), '.1f', ' bpm')}")
    print(f"    HR std : {_fmt(real_morph.get('hr_std'),  '.1f', ' bpm')}  ← target >47.4")
    print(f"    QRS dur: {_fmt(real_morph.get('qrs_duration_mean_ms'), '.1f', ' ms')}")
    print()
    print("  Generated ECG morphology:")
    print(f"    HR mean: {_fmt(fake_morph.get('hr_mean'), '.1f', ' bpm')}")
    print(f"    HR std : {_fmt(fake_morph.get('hr_std'),  '.1f', ' bpm')}  ← target >47.4")
    print(f"    QRS dur: {_fmt(fake_morph.get('qrs_duration_mean_ms'), '.1f', ' ms')}")
    print("=" * 60)
    if not real_morph:
        print("  ⚠️  Re-run after: pip install neurokit2  (for HR/QRS metrics)")
        print("=" * 60)

    # ── 6. Save JSON ───────────────────────────────────────────────────────
    results_serialisable = {}
    for k, v in results.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            results_serialisable[k] = float(v)
        elif isinstance(v, dict):
            results_serialisable[k] = {kk: float(vv) for kk, vv in v.items()
                                       if isinstance(vv, (int, float, np.floating, np.integer))}
        else:
            results_serialisable[k] = str(v)

    results_serialisable['run']      = 'Run5b'
    results_serialisable['n_samples'] = len(real_signals)
    results_serialisable['checkpoint'] = args.checkpoint

    out_json = os.path.join(args.output_dir, 'clinical_results.json')
    with open(out_json, 'w') as f:
        json.dump(results_serialisable, f, indent=2)

    print(f"\n💾 Results saved → {out_json}")
    print("✅ Evaluation complete.")


if __name__ == '__main__':
    main()
