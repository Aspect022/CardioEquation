import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.feature_extractor import FeatureExtractor
from src.models.diffusion_unet import ConditionalDiffusionUNet
from src.data.ecg_synthesizer import generate_clean_ecg
from src.data.realistic_artifacts import RealisticScanArtifacts

def verify():
    print("🔬 Verifying Phase 2 (Realistic Artifacts)...")
    
    # 1. Models
    fe = FeatureExtractor()
    unet = ConditionalDiffusionUNet()
    
    # Init
    dummy = tf.zeros((1, 2500, 1))
    fe(dummy)
    unet(dummy, tf.zeros((1,)), tf.zeros((1, 512)))
    
    try:
        # Load Phase 2 Weights (Epoch 2)
        fe.load_weights("models/feature_extractor_epoch_02.weights.h5")
        unet.load_weights("models/diffusion_unet_epoch_02.weights.h5")
        print("✅ Models Loaded (Phase 2).")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # 2. Generate Realistic Test Sample
    clean_np = generate_clean_ecg(hr=75)
    
    print("Generating realistic scan artifacts (this may take a moment)...")
    sim = RealisticScanArtifacts(dpi=150)
    noisy_np = sim.add_artifacts(clean_np)
    
    # Normalize inputs
    # Note: RealisticDataset uses Instance Norm. We must match.
    noisy_norm = (noisy_np - np.mean(noisy_np)) / (np.std(noisy_np) + 1e-8)
    clean_norm = (clean_np - np.mean(clean_np)) / (np.std(clean_np) + 1e-8)
    
    noisy_tf = tf.convert_to_tensor(noisy_norm[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    
    # 3. Diffusion Sampling
    print("Sampling...")
    cond = fe(noisy_tf)
    
    steps = 50
    x_t = tf.random.normal((1, 2500, 1))
    ts = np.linspace(1.0, 0.0, steps)
    
    for i in range(steps-1):
        t_curr = ts[i]
        t_next = ts[i+1]
        
        t_tensor = tf.constant([t_curr], dtype=tf.float32)
        pred_noise = unet(x_t, t_tensor, cond)
        
        denom = max(1.0 - t_curr, 1e-3)
        est_x0 = (x_t - t_curr * pred_noise) / denom
        
        x_next = (1.0 - t_next) * est_x0 + t_next * pred_noise
        x_t = x_next
        
    denoised = x_t.numpy()[0, :, 0]
    
    # 4. Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(noisy_norm, color='black', linewidth=0.8)
    plt.title("Realistic Scan Input (Simulated Grid/Blur/Skew)")
    
    plt.subplot(3, 1, 2)
    plt.plot(denoised, color='blue', linewidth=1.5)
    plt.title("Diffusion Reconstruction (Phase 2 Model)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(clean_norm, color='green', linewidth=1.5)
    plt.title("Ground Truth")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_verification.png')
    print("✅ Plot saved to realistic_verification.png")

if __name__ == '__main__':
    verify()
