"""
Test script to verify the trained ECG model can be loaded and used for inference.
"""

import numpy as np
import joblib
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from src.config import (
    SIGNAL_LENGTH, FS, NUM_PARAMS, PARAM_KEYS, 
    INPUT_SCALER_PATH, OUTPUT_SCALER_PATH, MODEL_WEIGHTS_PATH
)

# Import model building function
from src.ecg_model_trainer import build_autoencoder

def test_model_loading():
    """Test that the trained model can be loaded and used for inference."""
    print("Testing model loading and inference...")
    
    try:
        # Load scalers
        print(f"Loading input scaler from {INPUT_SCALER_PATH}")
        input_scaler = joblib.load(INPUT_SCALER_PATH)
        print(f"Loading output scaler from {OUTPUT_SCALER_PATH}")
        output_scaler = joblib.load(OUTPUT_SCALER_PATH)
        print("[SUCCESS] Scalers loaded successfully")
        
        # Build model
        print("Building autoencoder model...")
        training_model, encoder, decoder = build_autoencoder(output_scaler)
        print("[SUCCESS] Model built successfully")
        
        # Load weights
        print(f"Loading model weights from {MODEL_WEIGHTS_PATH}")
        training_model.load_weights(MODEL_WEIGHTS_PATH)
        print("[SUCCESS] Model weights loaded successfully")
        
        # Test with a sample ECG signal (zeros for simplicity)
        print("Testing inference with sample data...")
        sample_ecg = np.zeros((1, SIGNAL_LENGTH, 1))
        normalized_ecg = input_scaler.transform(sample_ecg.reshape(1, -1)).reshape(1, SIGNAL_LENGTH, 1)
        
        # Predict parameters
        predicted_params = encoder.predict(normalized_ecg)
        print(f"[SUCCESS] Parameter prediction successful. Shape: {predicted_params.shape}")
        
        # Reconstruct ECG
        reconstructed_ecg = decoder.predict(predicted_params)
        print(f"[SUCCESS] ECG reconstruction successful. Shape: {reconstructed_ecg.shape}")
        
        print("\n[SUCCESS] All tests passed! The trained model is ready for use.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)