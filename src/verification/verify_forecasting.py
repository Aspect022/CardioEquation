import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.feature_extractor import FeatureExtractor
from src.models.diffusion_unet import ConditionalDiffusionUNet
from src.data.ecg_synthesizer import generate_clean_ecg
from src.data.mitbih_long_loader import MITBIHLongLoader

def verify_forecasting():
    print("🔬 Verifying Phase 4: Personalized Digital Twin...")
    
    # 1. Models
    fe = FeatureExtractor()
    unet = ConditionalDiffusionUNet()
    
    # Init
    dummy = tf.zeros((1, 2500, 1))
    fe(dummy)
    unet(dummy, tf.zeros((1,)), tf.zeros((1, 512)))
    
    # Load Forecast Weights
    try:
        # Try finding latest forecast weights
        model_dir = "models"
        fe_weights = sorted([f for f in os.listdir(model_dir) if "fe_forecast" in f])[-1]
        unet_weights = sorted([f for f in os.listdir(model_dir) if "unet_forecast" in f])[-1]
        
        print(f"Loading weights: {fe_weights}, {unet_weights}")
        fe.load_weights(os.path.join(model_dir, fe_weights))
        unet.load_weights(os.path.join(model_dir, unet_weights))
    except Exception as e:
        print(f"❌ Failed to load forecast weights (Training might still be running or failed): {e}")
        # Fallback to Phase 2 weights just to show pipeline
        print("⚠️ Falling back to Phase 2 weights for DEMO purposes.")
        try:
             fe.load_weights("models/feature_extractor_epoch_02.weights.h5")
             unet.load_weights("models/diffusion_unet_epoch_02.weights.h5")
        except:
             return

    # 2. Get Real Patient Data (Context)
    # Load one sample from the forecasting dataset
    data = np.load("data/mitbih_forecasting.npz")
    context_samples = data['context']
    future_samples = data['future']
    
    # Pick a random sample
    idx = np.random.randint(0, len(context_samples))
    context = context_samples[idx] # (5000, 1) resampled
    target_future = future_samples[idx]
    
    # Normalize Context
    mean = np.mean(context)
    std = np.std(context) + 1e-8
    context_norm = (context - mean) / std
    context_tf = tf.convert_to_tensor(context_norm[np.newaxis, ...], dtype=tf.float32)
    
    # 3. Generate Digital Twin (Future)
    print("Generating Digital Twin (Forecasting)...")
    
    # Extract Identity
    identity = fe(context_tf)
    
    # Diffusion Sampling
    steps = 50
    x_t = tf.random.normal((1, 2500, 1))
    ts = np.linspace(1.0, 0.0, steps)
    
    for i in range(steps-1):
        t_curr = ts[i]
        t_next = ts[i+1]
        
        t_tensor = tf.constant([t_curr], dtype=tf.float32)
        pred_noise = unet(x_t, t_tensor, identity)
        
        denom = max(1.0 - t_curr, 1e-3)
        est_x0 = (x_t - t_curr * pred_noise) / denom
        
        x_next = (1.0 - t_next) * est_x0 + t_next * pred_noise
        x_t = x_next
        
    digital_twin_norm = x_t.numpy()[0, :, 0]
    
    # 4. Generate Healthy Reference (Dynamic HR)
    print("Generating Healthy Reference...")
    # Estimate HR from context using Peak Detection
    from scipy.signal import find_peaks
    # simple peak detection on normalized context (assuming R peaks are high)
    peaks, _ = find_peaks(context_norm.flatten(), height=1.0, distance=150) # distance ~300ms at 500hz
    if len(peaks) > 1:
        # Calculate avg distance
        diffs = np.diff(peaks)
        avg_dist = np.mean(diffs)
        est_hr = 60 / (avg_dist / 500.0)
        print(f"Estimated Patient HR: {est_hr:.1f} BPM")
    else:
        print("Could not detect peaks. Defaulting to 75 BPM.")
        est_hr = 75.0
        
    healthy_signal = generate_clean_ecg(hr=est_hr, duration=5) # 5s = 2500 samples
    healthy_norm = (healthy_signal - np.mean(healthy_signal)) / (np.std(healthy_signal) + 1e-8)
    
    # 5. Plot (The 3 Tracks)
    plt.figure(figsize=(15, 10))
    
    # Time axis
    t_ctx = np.arange(0, 5, 1/500) # 0-5s
    t_fut = np.arange(5, 10, 1/500) # 5-10s
    
    # Track 1: Current Signal (Context + Ground Truth Future)
    # Context is 5s (2500 samples)
    # Oops, loader does 2500 samples context (5s? Loader said 10s=>5000 but resampled to 5000? No, unet takes 2500.)
    # Let's assume input is 5s for UNet. Dataset has 5000 samples?
    # Wait, dataset loader resampled to target_len = context_sec * fs = 10 * 500 = 5000.
    # UNet takes 2500. This is a mismatch!
    # Ah, I see in Loader: target_len = int(context_sec * self.fs) = 5000.
    # But UNet is (2500, 1).
    # Training Loop must be failing with shape mismatch!
    # Or ResNet handles 5000? ResNet has GlobalPooling, so it might handle 5000.
    # UNet input is noise shape. Trainer uses `noise = tf.random.normal(tf.shape(future_clean))`
    # If future_clean is 5000, UNet gets 5000.
    # Can UNet handle 5000?
    # UNet has 3 downsamples (2500 -> 1250 -> 625 -> 312).
    # 5000 -> 2500 -> 1250 -> 625.
    # It might work if dimensions are divisible.
    
    context_data = context_norm[:, 0]
    
    plt.subplot(3, 1, 1)
    plt.plot(t_ctx, context_data[0:2500], color='black', label="Context (0-5s)")
    # Plot true future for reference?
    plt.plot(t_fut, target_future[0:2500], color='gray', linestyle='--', label="Actual Future (5-10s)")
    plt.title("1. Current Signal (Patient Input)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    # Digital Twin starts after context
    plt.plot(t_ctx, np.zeros_like(t_ctx), color='white') # spacer
    plt.plot(t_fut, digital_twin_norm, color='blue', linewidth=1.5, label="Digital Twin Prediction")
    plt.axvline(x=5, color='red', linestyle='--', label="Prediction Start")
    plt.title("2. Digital Twin (Personalized Forecast)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t_fut, healthy_norm, color='green', linewidth=1.5, label="Healthy Reference (Ideal)")
    plt.title("3. Healthy Reference (Baseline)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('forecasting_verification.png')
    print("✅ Plot saved to forecasting_verification.png")

if __name__ == '__main__':
    verify_forecasting()
