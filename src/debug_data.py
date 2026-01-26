import os
import sys
import numpy as np

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.synthetic_dataset import SyntheticNoiseDataset

def debug_dataloader():
    print("🔍 Debugging Data Loader")
    ds = SyntheticNoiseDataset(batch_size=4, epoch_length=1)
    
    noisy, clean = ds[0]
    print(f"Batch Shapes: Noisy={noisy.shape}, Clean={clean.shape}")
    print(f"Data Types: {noisy.dtype}, {clean.dtype}")
    print(f"Noisy Range: {np.min(noisy):.3f} to {np.max(noisy):.3f}")
    print(f"Clean Range: {np.min(clean):.3f} to {np.max(clean):.3f}")
    
    if np.isnan(noisy).any():
        print("❌ Noisy contains NaNs!")
    if np.isnan(clean).any():
        print("❌ Clean contains NaNs!")
        
    # Check dimensions
    if noisy.ndim != 3 or noisy.shape[-1] != 1:
        print("❌ Incorrect Dimensions (Expected 3, last dim 1)")
        
if __name__ == '__main__':
    debug_dataloader()
