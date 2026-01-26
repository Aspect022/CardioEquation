import numpy as np
import tensorflow as tf
from .ecg_synthesizer import generate_clean_ecg

class SyntheticNoiseDataset(tf.keras.utils.Sequence):
    """
    Phase 1: Synthetic Dataset
    Generates (Noisy, Clean) pairs on the fly.
    """
    def __init__(self, batch_size=32, epoch_length=1000, signal_length=2500):
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.signal_length = signal_length
        
    def __len__(self):
        return self.epoch_length
        
    def __getitem__(self, index):
        # Generate batch
        noisy_batch = []
        clean_batch = []
        cond_batch = [] # If feature extractor is trained separately or integrated?
        # Ideally, Diffusion takes Noisy -> FeatureExtractor -> Cond
        # So inputs are simply Noisy.
        
        # NOTE: To train diffusion, we need:
        # X: Noisy Signal
        # Y: Random Gaussian Noise (The target for diffusion at step t)
        # BUT standard Keras fit expects (x, y). 
        # Our model `train_step` will handle time sampling.
        # So we return (Noisy, Clean) and let train_step do the rest.
        
        for _ in range(self.batch_size):
            # 1. Clean
            # Randomize HR slightly
            hr = np.random.uniform(50, 100)
            clean = generate_clean_ecg(hr=hr) # 2500 samples
            
            # 2. Add Noise
            noisy, scale = self.add_noise(clean)
            
            # Normalize Clean to [-1, 1] for stability (Target)
            # Or should we denoise to original scale?
            # Diffusion usually predicts noise, so input is scaled.
            
            # Let's standardize Clean to [-1, 1] roughly
            clean_std = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
            
            # Noisy is scaled by `scale` in add_noise, but let's normalize it for the network input
            # The network should learn to handle amplitude?
            # User says: "Scale-invariant loss".
            # So we pass Raw Noisy and Raw Clean?
            # Or Instance Norm them?
            # Let's Instance Norm both for training stability, 
            # OR pass Raw and let Loss handle it.
            # Best practice: Normalize input to N(0,1).
            noisy_norm = (noisy - np.mean(noisy)) / (np.std(noisy) + 1e-8)
            
            # Reshape (2500, 1)
            noisy_batch.append(noisy_norm[..., np.newaxis])
            clean_batch.append(clean_std[..., np.newaxis])
            
        return np.array(noisy_batch, dtype=np.float32), np.array(clean_batch, dtype=np.float32)
        
    def add_noise(self, clean_ecg):
        noisy = clean_ecg.copy()
        
        # 1. Gaussian Setup (Noise Floor)
        # SNR 15-25dB
        sigma = 0.05 * np.max(np.abs(clean_ecg)) 
        noisy += np.random.normal(0, sigma, len(noisy))
        
        # 2. Baseline Wander (Low Freq)
        t = np.arange(len(noisy))
        f_wander = np.random.uniform(0.1, 0.5)
        a_wander = np.random.uniform(0.1, 0.5)
        noisy += a_wander * np.sin(2*np.pi*f_wander*t/500)
        
        # 3. Powerline (50 Hz)
        a_line = np.random.uniform(0.0, 0.1)
        noisy += a_line * np.sin(2*np.pi*50*t/500)
        
        # 4. Amplitude Scaling (The User's Bug)
        scale = np.random.uniform(0.5, 2.0)
        noisy *= scale
        
        return noisy, scale
