"""
Configuration file for CardioEquation project.
Contains all constants and hyperparameters used across the project.
"""

# Signal parameters
SIGNAL_LENGTH = 2500
FS = 500  # Sampling frequency (Hz)
NUM_BEATS_DEFAULT = 5

# Model parameters
NUM_PARAMS = 16
PARAM_KEYS = [
    'HR', 'A_p', 'μ_p', 'σ_p', 'A_q', 'μ_q', 'σ_q', 'A_r',
    'μ_r', 'σ_r', 'A_s', 'μ_s', 'σ_s', 'A_t', 'μ_t', 'σ_t'
]

# Training parameters
LEARNING_RATE = 1e-4
PARAM_LOSS_WEIGHT = 0.3
EPOCHS = 40
BATCH_SIZE = 16

# Data generation parameters
NUM_SAMPLES = 2000

# File paths
MODEL_WEIGHTS_PATH = '../models/best_ecg_model.weights.h5'
INPUT_SCALER_PATH = '../models/input_scaler.joblib'
OUTPUT_SCALER_PATH = '../models/output_scaler.joblib'