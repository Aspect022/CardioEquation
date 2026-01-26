import tensorflow as tf
import numpy as np
import os

class RealisticDataset(tf.keras.utils.Sequence):
    """
    Phase 2: Scanned Artifact Dataset Loader.
    Loads pre-generated .npz file.
    """
    def __init__(self, data_path="data/realistic_train.npz", batch_size=32):
        self.batch_size = batch_size
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Realistic dataset not found at {data_path}. Run generate_realistic_data.py first.")
            
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path)
        self.noisy = data['noisy']
        self.clean = data['clean']
        
        # Shuffle indices
        self.indices = np.arange(len(self.noisy))
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return int(np.ceil(len(self.noisy) / self.batch_size))
        
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_noisy = self.noisy[batch_indices]
        batch_clean = self.clean[batch_indices]
        
        # Augmentation?
        # Maybe some random amplitude scaling purely for robustness?
        # For now, raw data from generator.
        
        # Normalize?
        # Generator outputs raw values (roughly mV).
        # We should normalize here for Neural Net stability.
        
        # Instance Normalization per sample
        # (x - mean) / std
        
        batch_noisy_norm = (batch_noisy - np.mean(batch_noisy, axis=1, keepdims=True)) / \
                           (np.std(batch_noisy, axis=1, keepdims=True) + 1e-8)
                           
        batch_clean_norm = (batch_clean - np.mean(batch_clean, axis=1, keepdims=True)) / \
                           (np.std(batch_clean, axis=1, keepdims=True) + 1e-8)
                           
        return batch_noisy_norm, batch_clean_norm
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
