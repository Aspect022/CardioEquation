import tensorflow as tf
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.realistic_artifacts import RealisticScanArtifacts

class PredictiveDataset(tf.keras.utils.Sequence):
    """
    Phase 4: Dataset for Context-Conditioned Forecasting.
    Input: Noisy Context (10s)
    Target: Clean Future (10s)
    """
    def __init__(self, data_path="data/mitbih_forecasting.npz", batch_size=8, augment=True):
        self.batch_size = batch_size
        self.augment = augment
        
        if not os.path.exists(data_path):
             # Create dummy if waiting (to allow code import)
            print(f"Dataset {data_path} not found. Using dummy for scaffolding.")
            self.context = np.zeros((10, 2500, 1), dtype=np.float32)
            self.future = np.zeros((10, 2500, 1), dtype=np.float32)
        else:
            print(f"Loading Forecasting Data from {data_path}...")
            data = np.load(data_path)
            self.context = data['context']
            self.future = data['future']
            
        self.indices = np.arange(len(self.context))
        np.random.shuffle(self.indices)
        
        # Simulator for artifacts
        if self.augment:
            self.simulator = RealisticScanArtifacts(dpi=150)
        
    def __len__(self):
        return int(np.ceil(len(self.context) / self.batch_size))
        
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_context = self.context[batch_indices] # (B, 5000, 1) -> Actually 2500 resampled in loader
        batch_future = self.future[batch_indices]
        
        # Apply Augmentation to Context ONLY
        # The model sees NOISY context and predicts CLEAN future
        
        noisy_contexts = []
        for i in range(len(batch_context)):
            ctx = batch_context[i, :, 0]
            if self.augment:
                 # Add scan artifacts
                 try:
                    # Simulator needs approx -1..1 input?
                    # Data is normalized (0 mean, 1 std).
                    ctx_noisy = self.simulator.add_artifacts(ctx)
                 except:
                    ctx_noisy = ctx + np.random.normal(0, 0.1, len(ctx))
            else:
                ctx_noisy = ctx
                
            # Normalize Noisy
            mean = np.mean(ctx_noisy)
            std = np.std(ctx_noisy) + 1e-8
            ctx_norm = (ctx_noisy - mean) / std
            noisy_contexts.append(ctx_norm)
            
        X_context = np.array(noisy_contexts, dtype=np.float32)[..., np.newaxis]
        Y_future = np.array(batch_future, dtype=np.float32) # Clean Future
        
        return X_context, Y_future
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
