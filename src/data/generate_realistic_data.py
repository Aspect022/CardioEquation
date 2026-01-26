import numpy as np
import os
import sys
import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.ecg_synthesizer import generate_clean_ecg
from src.data.realistic_artifacts import RealisticScanArtifacts

def generate_dataset(num_samples=1000, output_file="data/realistic_train.npz"):
    print(f"🚀 Generating {num_samples} Realistic Synthetic Samples...")
    
    simulator = RealisticScanArtifacts(dpi=150)
    
    noisy_list = []
    clean_list = []
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    for i in tqdm.tqdm(range(num_samples)):
        # 1. Clean
        hr = np.random.uniform(55, 110)
        clean = generate_clean_ecg(hr=hr)
        
        # 2. Noisy (Artifacts)
        try:
            noisy = simulator.add_artifacts(clean)
        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            continue
            
        # 3. Validation
        if len(noisy) != 2500 or len(clean) != 2500:
            print(f"Shape mismatch: {len(noisy)}")
            continue
        
        if np.isnan(noisy).any():
            print("NaN detected")
            continue
            
        noisy_list.append(noisy)
        clean_list.append(clean)
        
    X_noisy = np.array(noisy_list, dtype=np.float32)
    Y_clean = np.array(clean_list, dtype=np.float32)
    
    # Reshape for model (B, 2500, 1)
    X_noisy = X_noisy[..., np.newaxis]
    Y_clean = Y_clean[..., np.newaxis]
    
    print(f"✅ Generated {len(X_noisy)} samples.")
    print(f"Shapes: {X_noisy.shape}, {Y_clean.shape}")
    
    np.savez_compressed(output_file, noisy=X_noisy, clean=Y_clean)
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    # Generate 100 samples for Phase 2 Quick Test
    generate_dataset(num_samples=100)
