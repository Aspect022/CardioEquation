"""
Usage example for the trained ECG digital twin model.
Demonstrates how to load the model and use it for inference.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
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

def generate_sample_ecg():
    """Generate a sample ECG for demonstration purposes."""
    # Simple sine wave with some noise as a placeholder
    t = np.linspace(0, 5, SIGNAL_LENGTH)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t) + np.random.normal(0, 0.1, SIGNAL_LENGTH)
    return ecg.reshape(1, SIGNAL_LENGTH, 1)

def plot_ecg_comparison(original_ecg, reconstructed_ecg, title="ECG Comparison"):
    """Plot comparison between original and reconstructed ECG signals."""
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Reshape for plotting
    original_flat = original_ecg.flatten()
    reconstructed_flat = reconstructed_ecg.flatten()
    
    # Time vector
    t = np.arange(len(original_flat)) / FS
    
    ax.plot(t, original_flat, label='Original ECG', alpha=0.8)
    ax.plot(t, reconstructed_flat, label='Reconstructed ECG', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating model usage."""
    print("=== ECG Digital Twin Usage Example ===\n")
    
    # Load scalers
    print("1. Loading trained scalers...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)
    print("   [SUCCESS] Scalers loaded successfully\n")
    
    # Build and load model
    print("2. Building and loading trained model...")
    training_model, encoder, decoder = build_autoencoder(output_scaler)
    training_model.load_weights(MODEL_WEIGHTS_PATH)
    print("   [SUCCESS] Model built and weights loaded successfully\n")
    
    # Generate or load sample ECG data
    print("3. Preparing sample ECG data...")
    sample_ecg = generate_sample_ecg()
    print(f"   Sample ECG shape: {sample_ecg.shape}\n")
    
    # Normalize input ECG
    print("4. Normalizing input ECG...")
    normalized_ecg = input_scaler.transform(sample_ecg.reshape(1, -1)).reshape(1, SIGNAL_LENGTH, 1)
    print("   [SUCCESS] ECG normalized successfully\n")
    
    # Predict parameters
    print("5. Predicting ECG parameters...")
    predicted_params_scaled = encoder.predict(normalized_ecg)
    predicted_params = output_scaler.inverse_transform(predicted_params_scaled)
    print(f"   [SUCCESS] Parameter prediction successful. Shape: {predicted_params.shape}")
    print(f"   Predicted parameters: {predicted_params[0]}\n")
    
    # Reconstruct ECG from predicted parameters
    print("6. Reconstructing ECG from predicted parameters...")
    reconstructed_ecg = decoder.predict(predicted_params_scaled)
    print(f"   [SUCCESS] ECG reconstruction successful. Shape: {reconstructed_ecg.shape}\n")
    
    # Display parameter meanings
    print("7. Interpreted Parameters:")
    param_names = [
        "Heart Rate (BPM)",
        "P-wave Amplitude (mV)", "P-wave Position", "P-wave Width (s)",
        "Q-wave Amplitude (mV)", "Q-wave Position", "Q-wave Width (s)",
        "R-wave Amplitude (mV)", "R-wave Position", "R-wave Width (s)",
        "S-wave Amplitude (mV)", "S-wave Position", "S-wave Width (s)",
        "T-wave Amplitude (mV)", "T-wave Position", "T-wave Width (s)"
    ]
    
    for i, (name, value) in enumerate(zip(param_names, predicted_params[0])):
        print(f"   {name}: {value:.4f}")
    print()
    
    # Show reconstruction quality
    print("8. Calculating reconstruction metrics...")
    mse = np.mean((sample_ecg.flatten() - reconstructed_ecg.flatten())**2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(sample_ecg.flatten(), reconstructed_ecg.flatten())[0, 1]
    print(f"   Reconstruction MSE: {mse:.6f}")
    print(f"   Reconstruction RMSE: {rmse:.6f}")
    print(f"   Pearson Correlation: {corr:.4f}\n")
    
    # Plot comparison
    print("9. Plotting ECG comparison...")
    plot_ecg_comparison(sample_ecg, reconstructed_ecg, "ECG Digital Twin: Original vs Reconstructed")
    print("   [SUCCESS] Plot displayed successfully\n")
    
    print("=== Usage Example Completed Successfully ===")
    print("\nTo reuse the trained model in your own code:")
    print("1. Load the scalers:")
    print("   input_scaler = joblib.load('../models/input_scaler.joblib')")
    print("   output_scaler = joblib.load('../models/output_scaler.joblib')")
    print("2. Re-instantiate the model:")
    print("   training_model, encoder, decoder = build_autoencoder(output_scaler)")
    print("3. Load the saved weights:")
    print("   training_model.load_weights('../models/best_ecg_model.weights.h5')")
    print("4. Use for inference:")
    print("   normalized_ecg = input_scaler.transform(your_ecg.reshape(1, -1)).reshape(1, 2500, 1)")
    print("   predicted_params = encoder.predict(normalized_ecg)")
    print("   real_params = output_scaler.inverse_transform(predicted_params)")

if __name__ == "__main__":
    main()