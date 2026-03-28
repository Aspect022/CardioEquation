"""
V2 Inference Pipeline for DiT-ECG — Run 6
============================================
Wraps the DiT-ECG model for easy inference with EMA weights and
supports both OT-CFM (Euler) and legacy DDPM (DDIM) sampling.

Usage:
    from src.inference.pipeline_v2 import ECGPipelineV2
    pipe = ECGPipelineV2("checkpoints/dit_ecg_ema_final.pt")
    generated = pipe.generate(context_ecg, num_steps=20, guidance_scale=2.0)
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.dit_ecg import dit_ecg_b, dit_ecg_s
from src.models.feature_extractor_pt import FeatureExtractorPT
from src.training.noise_scheduler import CosineNoiseScheduler
from src.training.flow_matching import FlowMatchingScheduler


class ECGPipelineV2:
    """
    Production inference pipeline for DiT-ECG (Run 6).

    Supports:
    - EMA model loading
    - OT-CFM Euler sampling (Run 6 default, 20 steps)
    - DDIM deterministic sampling (legacy, 50 steps)
    - Classifier-Free Guidance (scale=2.0)
    - HR conditioning
    - Batch generation
    """

    def __init__(
        self,
        dit_weights_path,
        fe_weights_path=None,
        model_size='B',
        use_flow_matching=True,
        device=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_flow_matching = use_flow_matching

        # Load models
        print(f"🔧 Loading DiT-ECG-{model_size} (Run 6)...")
        if model_size == 'S':
            self.model = dit_ecg_s()
        else:
            self.model = dit_ecg_b()

        # Handle different checkpoint formats
        ckpt = torch.load(dit_weights_path, map_location=self.device, weights_only=True)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        elif isinstance(ckpt, dict) and 'ema' in ckpt:
            state = ckpt['ema']
        else:
            state = ckpt
        state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Feature extractor
        self.fe = FeatureExtractorPT()
        if fe_weights_path and os.path.exists(fe_weights_path):
            fe_ckpt = torch.load(fe_weights_path, map_location=self.device, weights_only=True)
            if isinstance(fe_ckpt, dict) and 'model_state_dict' in fe_ckpt:
                fe_ckpt = fe_ckpt['model_state_dict']
            self.fe.load_state_dict(fe_ckpt, strict=False)
        self.fe = self.fe.to(self.device)
        self.fe.eval()

        # Schedulers
        if use_flow_matching:
            self.fm_scheduler = FlowMatchingScheduler()
            print(f"  🌊 Mode: OT-CFM (Euler sampling)")
        else:
            self.ddpm_scheduler = CosineNoiseScheduler(num_train_timesteps=1000)
            print(f"  Mode: DDPM (DDIM sampling)")

        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  ✅ Pipeline ready: {params:.1f}M params on {self.device}")

    @torch.no_grad()
    def extract_identity(self, context_signal):
        """
        Extract patient identity from context ECG.

        Args:
            context_signal: numpy array (T,) or (1, T) or (B, 1, T)
        Returns:
            identity: (B, 512) tensor
        """
        if isinstance(context_signal, np.ndarray):
            context_signal = (context_signal - context_signal.mean()) / (context_signal.std() + 1e-8)
            context_signal = torch.from_numpy(context_signal).float()

        if context_signal.dim() == 1:
            context_signal = context_signal.unsqueeze(0).unsqueeze(0)
        elif context_signal.dim() == 2:
            context_signal = context_signal.unsqueeze(0)

        context_signal = context_signal.to(self.device)
        return self.fe(context_signal)

    @torch.no_grad()
    def generate(
        self,
        context_signal,
        num_steps=20,
        guidance_scale=2.0,
        num_samples=1,
        target_hr=None,
    ):
        """
        Generate personalized ECG conditioned on patient context.

        Args:
            context_signal: (T,) or (1, T) numpy array — patient's ECG context
            num_steps: sampling steps (20 for CFM, 50 for DDIM)
            guidance_scale: CFG scale (2.0 recommended for Run 6)
            num_samples: Number of ECG samples to generate
            target_hr: Target HR in bpm (None = 75 bpm default)
        Returns:
            generated: (num_samples, T) numpy array — generated ECG signals
        """
        identity = self.extract_identity(context_signal)
        identity = identity.repeat(num_samples, 1)

        shape = (num_samples, 1, self.model.signal_length)

        hr_bpm = None
        if target_hr is not None:
            hr_bpm = torch.full((num_samples,), target_hr, dtype=torch.float32, device=self.device)

        if self.use_flow_matching:
            generated = self.fm_scheduler.euler_sample(
                self.model, shape, identity,
                hr_bpm=hr_bpm,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=self.device,
            )
        else:
            generated = self.ddpm_scheduler.ddim_sample(
                self.model, shape, identity,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=self.device,
            )

        return generated.squeeze(1).cpu().numpy()

    @torch.no_grad()
    def denoise(self, noisy_signal, num_steps=20, guidance_scale=2.0):
        """
        Denoise an ECG signal (Phase 3 backward compatibility).
        Uses the noisy signal itself as both the conditioning context
        and the starting point for guided denoising.
        """
        identity = self.extract_identity(noisy_signal)
        shape = (1, 1, self.model.signal_length)

        if self.use_flow_matching:
            denoised = self.fm_scheduler.euler_sample(
                self.model, shape, identity,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=self.device,
            )
        else:
            denoised = self.ddpm_scheduler.ddim_sample(
                self.model, shape, identity,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=self.device,
            )

        return denoised.squeeze().cpu().numpy()


if __name__ == '__main__':
    print("ECGPipelineV2 (Run 6) — requires trained weights to run.")
    print("Usage:")
    print("  pipe = ECGPipelineV2('checkpoints/dit_ecg_ema_final.pt')")
    print("  output = pipe.generate(context_ecg, guidance_scale=2.0, target_hr=72.0)")
