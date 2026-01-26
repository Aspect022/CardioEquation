import tensorflow as tf
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.feature_extractor import FeatureExtractor
from src.models.diffusion_unet import ConditionalDiffusionUNet

def debug_arch():
    print("🩺 Debugging Architecture Shapes")
    
    # 1. Dummy Data
    B = 2
    L = 2500
    noisy = tf.random.normal((B, L, 1))
    clean = tf.random.normal((B, L, 1))
    times = tf.random.uniform((B,), 0, 1)
    
    # 2. Feature Extractor
    print(f"\n[1] Testing FeatureExtractor input: {noisy.shape}")
    fe = FeatureExtractor()
    try:
        feats = fe(noisy)
        print(f"✅ Feature Extractor Output: {feats.shape}")
    except Exception as e:
        print(f"❌ Feature Extractor Failed: {e}")
        return

    # 3. Diffusion UNet
    print(f"\n[2] Testing Diffusion UNet input: {noisy.shape}, t: {times.shape}, cond: {feats.shape}")
    unet = ConditionalDiffusionUNet()
    try:
        noise_pred = unet(noisy, times, feats)
        print(f"✅ Diffusion UNet Output: {noise_pred.shape}")
    except Exception as e:
        print(f"❌ Diffusion UNet Failed: {e}")
        return
        
    print("\n✅ Architecture is valid.")

if __name__ == '__main__':
    debug_arch()
