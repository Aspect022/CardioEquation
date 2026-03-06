"""
V2 Inference Pipeline for DiT-ECG
====================================
Wraps the DiT-ECG model for easy inference with EMA weights,
Classifier-Free Guidance, and DDIM sampling.

Usage:
    from src.inference.pipeline_v2 import ECGPipelineV2
    pipe = ECGPipelineV2("checkpoints/dit_ecg_ema_final.pt")
    generated = pipe.generate(context_ecg, num_steps=50, guidance_scale=3.0)
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.dit_ecg import dit_ecg_b, dit_ecg_s
from src.models.feature_extractor_pt import FeatureExtractorPT
from src.training.noise_scheduler import CosineNoiseScheduler


class ECGPipelineV2:
    """
    Production inference pipeline for DiT-ECG.

    Supports:
    - EMA model loading
    - DDIM deterministic sampling (50 steps)
    - Classifier-Free Guidance (scale=3.0)
    - Batch generation
    """

    def __init__(
        self,
        dit_weights_path,
        fe_weights_path=None,
        model_size='B',
        device=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models
        print(f"🔧 Loading DiT-ECG-{model_size}...")
        if model_size == 'S':
            self.model = dit_ecg_s()
        else:
            self.model = dit_ecg_b()

        self.model.load_state_dict(
            torch.load(dit_weights_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Feature extractor
        self.fe = FeatureExtractorPT()
        if fe_weights_path and os.path.exists(fe_weights_path):
            self.fe.load_state_dict(
                torch.load(fe_weights_path, map_location=self.device, weights_only=True)
            )
        self.fe = self.fe.to(self.device)
        self.fe.eval()

        # Noise scheduler
        self.scheduler = CosineNoiseScheduler(num_train_timesteps=1000)

        print(f"✅ Pipeline ready on {self.device}")

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
            # Normalize
            context_signal = (context_signal - context_signal.mean()) / (context_signal.std() + 1e-8)
            context_signal = torch.from_numpy(context_signal).float()

        # Ensure shape is (B, 1, T)
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
        num_steps=50,
        guidance_scale=3.0,
        num_samples=1,
    ):
        """
        Generate personalized ECG conditioned on patient context.

        Args:
            context_signal: (T,) or (1, T) numpy array — patient's ECG context
            num_steps: DDIM sampling steps (50 recommended)
            guidance_scale: CFG scale (3.0 recommended; 1.0 = no guidance)
            num_samples: Number of ECG samples to generate
        Returns:
            generated: (num_samples, T) numpy array — generated ECG signals
        """
        # Extract patient identity
        identity = self.extract_identity(context_signal)
        # Repeat for multiple samples
        identity = identity.repeat(num_samples, 1)

        # Generate via DDIM
        shape = (num_samples, 1, self.model.signal_length)
        generated = self.scheduler.ddim_sample(
            self.model, shape, identity,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            device=self.device,
        )

        return generated.squeeze(1).cpu().numpy()  # (num_samples, T)

    @torch.no_grad()
    def denoise(self, noisy_signal, num_steps=50, guidance_scale=3.0):
        """
        Denoise an ECG signal (Phase 3 backward compatibility).

        Uses the noisy signal itself as both the conditioning context
        and the starting point for guided denoising.
        """
        identity = self.extract_identity(noisy_signal)
        shape = (1, 1, self.model.signal_length)

        denoised = self.scheduler.ddim_sample(
            self.model, shape, identity,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            device=self.device,
        )

        return denoised.squeeze().cpu().numpy()


if __name__ == '__main__':
    print("ECGPipelineV2 — requires trained weights to run.")
    print("Usage:")
    print("  pipe = ECGPipelineV2('checkpoints/dit_ecg_ema_final.pt')")
    print("  output = pipe.generate(context_ecg, guidance_scale=3.0)")
