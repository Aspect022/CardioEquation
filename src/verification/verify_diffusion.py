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

def add_noise_demo(clean):
    noise = np.random.normal(0, 0.1, len(clean))
    # Add heavy baseline wander
    t = np.arange(len(clean))
    noise += 0.5 * np.sin(2*np.pi*0.2*t/500)
    return clean + noise

from src.training.train import DiffusionTrainer

def verify():
    print("🔬 Verifying Diffusion Model...")
    
    # 1. Load Models via Trainer (to match checkpoint structure)
    fe = FeatureExtractor()
    unet = ConditionalDiffusionUNet()
    
    # Build models first
    dummy = tf.zeros((1, 2500, 1))
    fe(dummy)
    unet(dummy, tf.zeros((1,)), tf.zeros((1, 512)))
    
    # Load Weights
    try:
        fe.load_weights("models/feature_extractor_epoch_01.weights.h5")
        unet.load_weights("models/diffusion_unet_epoch_01.weights.h5")
        print("✅ Models Loaded.")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # 2. Generate Test Sample
    clean_np = generate_clean_ecg(hr=70)
    noisy_np = add_noise_demo(clean_np)
    
    # Normalize inputs as per training
    noisy_norm = (noisy_np - np.mean(noisy_np)) / (np.std(noisy_np) + 1e-8)
    clean_norm = (clean_np - np.mean(clean_np)) / (np.std(clean_np) + 1e-8)
    
    noisy_tf = tf.convert_to_tensor(noisy_norm[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    
    # 3. Extract Features (Conditioning)
    print("Extracting Features...")
    cond = fe(noisy_tf)
    
    # 4. Reverse Diffusion Sampling (Linear Schedule)
    # x_t = (1-t)x0 + t*noise
    # We start at t=1 (Pure Noise) -> t=0 (Pure Signal)
    
    print("Sampling...")
    steps = 50
    # Start with random noise
    x_t = tf.random.normal((1, 2500, 1))
    
    ts = np.linspace(1.0, 0.0, steps)
    
    for i in range(steps-1):
        t_curr = ts[i]
        t_next = ts[i+1]
        
        # Predict Noise (epsilon) at current level x_t
        t_tensor = tf.constant([t_curr], dtype=tf.float32)
        pred_noise = unet(x_t, t_tensor, cond)
        
        # Estimate x_0 (Clean Signal)
        # x_t = (1-t)x0 + t*eps
        # x0 = (x_t - t*eps) / (1-t)
        # Handle t=1 singularity approx
        denom = max(1.0 - t_curr, 1e-3)
        est_x0 = (x_t - t_curr * pred_noise) / denom
        
        # Move to Input for next step t_next
        # x_next = (1 - t_next)*x0 + t_next*eps (using projected x0)
        # Or better: x_next = x_t - ... (Euler update)
        # Simple Linear Interpolation:
        x_next = (1.0 - t_next) * est_x0 + t_next * pred_noise
        
        x_t = x_next
        
    denoised = x_t.numpy()[0, :, 0]
    
    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(noisy_norm, color='gray')
    plt.title("Noisy Input (Normalized)")
    
    plt.subplot(3, 1, 2)
    plt.plot(denoised, color='blue')
    plt.title("Diffusion Reconstruction (Phase 1 Model)")
    
    plt.subplot(3, 1, 3)
    plt.plot(clean_norm, color='green')
    plt.title("Ground Truth")
    
    plt.tight_layout()
    plt.savefig('diffusion_verification.png')
    print("✅ Plot saved to diffusion_verification.png")

if __name__ == '__main__':
    verify()
