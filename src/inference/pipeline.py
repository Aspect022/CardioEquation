import tensorflow as tf
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.feature_extractor import FeatureExtractor
from src.models.diffusion_unet import ConditionalDiffusionUNet

class ECGDenoisingPipeline:
    """
    Phase 3: Production Inference Pipeline.
    Wraps Model loading and Diffusion Sampling.
    """
    def __init__(self, weights_dir="models"):
        self.weights_dir = weights_dir
        self.fe = FeatureExtractor()
        self.unet = ConditionalDiffusionUNet()
        
        # Build Models
        self._build_models()
        # Load Weights
        self._load_weights()
        
    def _build_models(self):
        dummy = tf.zeros((1, 2500, 1))
        self.fe(dummy)
        self.unet(dummy, tf.zeros((1,)), tf.zeros((1, 512)))
        
    def _load_weights(self):
        # Prefer Phase 2 weights
        fe_path = os.path.join(self.weights_dir, "feature_extractor_epoch_02.weights.h5")
        unet_path = os.path.join(self.weights_dir, "diffusion_unet_epoch_02.weights.h5")
        
        if not os.path.exists(fe_path):
            print(f"⚠️ Phase 2 weights not found at {fe_path}. Trying Epoch 1.")
            fe_path = os.path.join(self.weights_dir, "feature_extractor_epoch_01.weights.h5")
            unet_path = os.path.join(self.weights_dir, "diffusion_unet_epoch_01.weights.h5")
            
        print(f"Loading weights from {fe_path} and {unet_path}...")
        self.fe.load_weights(fe_path)
        self.unet.load_weights(unet_path)
        print("✅ Models Loaded Successfully.")
        
    def process_signal(self, noisy_signal):
        """
        Denoise a single 1D signal (length 2500).
        """
        # 1. Preprocessing (Normalization)
        # Instance Norm
        mean = np.mean(noisy_signal)
        std = np.std(noisy_signal) + 1e-8
        input_norm = (noisy_signal - mean) / std
        
        # Prepare Tensor
        input_tf = tf.convert_to_tensor(input_norm[np.newaxis, ..., np.newaxis], dtype=tf.float32)
        
        # 2. Extract Features
        cond = self.fe(input_tf)
        
        # 3. Diffusion Sampling (50 steps)
        steps = 50
        x_t = tf.random.normal((1, 2500, 1))
        ts = np.linspace(1.0, 0.0, steps)
        
        for i in range(steps-1):
            t_curr = ts[i]
            t_next = ts[i+1]
            
            t_tensor = tf.constant([t_curr], dtype=tf.float32)
            pred_noise = self.unet(x_t, t_tensor, cond)
            
            denom = max(1.0 - t_curr, 1e-3)
            est_x0 = (x_t - t_curr * pred_noise) / denom
            
            x_next = (1.0 - t_next) * est_x0 + t_next * pred_noise
            x_t = x_next
            
        denoised_norm = x_t.numpy()[0, :, 0]
        
        # 4. Denormalize?
        # The output is clean and normalized. 
        # Usually we want to return it on a standard scale (e.g. mV).
        # Since we don't know the original mV scale from the PDF, 
        # we return the normalized clean shape.
        return denoised_norm

if __name__ == '__main__':
    # Simple Test
    pipe = ECGDenoisingPipeline()
    dummy_input = np.random.normal(0, 1, 2500)
    out = pipe.process_signal(dummy_input)
    print(f"Output shape: {out.shape}, Range: {out.min():.2f} to {out.max():.2f}")
