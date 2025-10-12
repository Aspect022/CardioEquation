import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import from Phase 1
from ecg_generator import generate_ecg, default_params

# --- 1. Constants and Configuration ---
NUM_SAMPLES = 2000
SIGNAL_LENGTH = 2500
FS = 500
NUM_PARAMS = 16
PARAM_KEYS = [
    'HR', 'A_p', 'μ_p', 'σ_p', 'A_q', 'μ_q', 'σ_q', 'A_r',
    'μ_r', 'σ_r', 'A_s', 'μ_s', 'σ_s', 'A_t', 'μ_t', 'σ_t'
]
INPUT_SHAPE = (SIGNAL_LENGTH, 1)
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PARAM_LOSS_WEIGHT = 0.3

MODEL_WEIGHTS_PATH = 'best_ecg_model.weights.h5'
INPUT_SCALER_PATH = 'input_scaler.joblib'
OUTPUT_SCALER_PATH = 'output_scaler.joblib'

# --- 2. Data Generation and Preprocessing ---
def create_dataset():
    """Generates and preprocesses the synthetic ECG dataset."""
    print("Generating synthetic ECG data...")
    X_raw = np.zeros((NUM_SAMPLES, SIGNAL_LENGTH))
    y_unscaled = np.zeros((NUM_SAMPLES, NUM_PARAMS))

    for i in range(NUM_SAMPLES):
        params = default_params.copy()
        params['HR'] = np.random.uniform(60, 100)
        params['A_p'] = np.random.uniform(0.1, 0.4)
        params['μ_p'] = np.random.uniform(0.15, 0.25)
        params['σ_p'] = np.random.uniform(0.02, 0.03)
        params['A_q'] = np.random.uniform(-0.2, -0.1)
        params['μ_q'] = np.random.uniform(0.3, 0.4)
        params['σ_q'] = np.random.uniform(0.01, 0.02)
        params['A_r'] = np.random.uniform(0.8, 1.2)
        params['μ_r'] = np.random.uniform(0.38, 0.42)
        params['σ_r'] = np.random.uniform(0.008, 0.012)
        params['A_s'] = np.random.uniform(-0.3, -0.2)
        params['μ_s'] = np.random.uniform(0.43, 0.47)
        params['σ_s'] = np.random.uniform(0.01, 0.02)
        params['A_t'] = np.random.uniform(0.2, 0.5)
        params['μ_t'] = np.random.uniform(0.6, 0.7)
        params['σ_t'] = np.random.uniform(0.04, 0.06)

        num_beats = int(np.ceil((SIGNAL_LENGTH / FS) / (60 / params['HR'])))
        ecg_signal = generate_ecg(params, num_beats=num_beats, fs=FS)
        
        if len(ecg_signal) > SIGNAL_LENGTH:
            ecg_signal = ecg_signal[:SIGNAL_LENGTH]
        else:
            ecg_signal = np.pad(ecg_signal, (0, SIGNAL_LENGTH - len(ecg_signal)), 'constant')

        X_raw[i, :] = ecg_signal
        y_unscaled[i, :] = [params[key] for key in PARAM_KEYS]

    print(f"Generated {NUM_SAMPLES} samples.")

    print("Normalizing data...")
    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = input_scaler.fit_transform(X_raw)

    output_scaler = MinMaxScaler()
    y_scaled = output_scaler.fit_transform(y_unscaled)

    X_scaled = X_scaled.reshape((-1, SIGNAL_LENGTH, 1))

    joblib.dump(input_scaler, INPUT_SCALER_PATH)
    joblib.dump(output_scaler, OUTPUT_SCALER_PATH)
    print(f"Saved scalers to {INPUT_SCALER_PATH} and {OUTPUT_SCALER_PATH}")

    return X_scaled, y_scaled, X_raw, input_scaler, output_scaler

# --- 3. Encoder-Decoder Model Definition ---

class Decoder(tf.keras.layers.Layer):
    def __init__(self, signal_length, fs, output_scaler, **kwargs):
        super().__init__(name='decoder', **kwargs)
        self.signal_length = signal_length
        self.fs = fs
        self.output_scaler = output_scaler
        self.param_keys = PARAM_KEYS
        # Create a constant time tensor with proper shape
        self.t = tf.constant(np.arange(0, self.signal_length / self.fs, 1/self.fs).astype(np.float32))

    def call(self, params_scaled):
        # Use Keras backend operations instead of py_function for better compatibility
        # Calculate the inverse transform manually using stored min/max values from the scaler
        # This avoids the AddN error that can occur with py_function operations
        if hasattr(self.output_scaler, 'scale_') and hasattr(self.output_scaler, 'min_'):
            # Calculate inverse transform using stored scaler parameters
            params_unscaled = params_scaled * (1.0 / self.output_scaler.scale_) + self.output_scaler.min_
        else:
            # Fallback if scaler parameters aren't accessible
            # Using a direct tensor operation to avoid py_function
            params_unscaled = params_scaled  # This is a simplified approach

        # Create tensors for each parameter
        HR = params_unscaled[:, 0]  # Shape: [batch_size]
        
        # Calculate beat duration for each sample in the batch
        beat_duration = 60.0 / HR  # Shape: [batch_size]
        
        # Expand dimensions for broadcasting: time vector [signal_length]
        t_expanded = tf.reshape(self.t, [1, -1])  # Shape: [1, signal_length]
        
        # Calculate the time within each beat using modulo operation
        # beat_duration needs to be [batch_size, 1] for broadcasting with [1, signal_length]
        beat_duration_expanded = tf.expand_dims(beat_duration, -1)  # Shape: [batch_size, 1]
        t_beat = tf.math.floormod(t_expanded, beat_duration_expanded)  # Shape: [batch_size, signal_length]
        
        # Initialize the ECG signal
        ecg_signal = tf.zeros_like(t_beat)  # Shape: [batch_size, signal_length]
        
        # Process each type of wave
        # For each wave, we extract its parameters and compute its contribution to the signal
        wave_indices = {
            'p': 1, 'q': 4, 'r': 7, 's': 10, 't': 13
        }
        
        for wave in ['p', 'q', 'r', 's', 't']:
            idx = wave_indices[wave]
            A = params_unscaled[:, idx]  # Amplitude [batch_size]
            mu = params_unscaled[:, idx+1]  # Position [batch_size]
            sigma = params_unscaled[:, idx+2]  # Width [batch_size]
            
            # Reshape for broadcasting
            A = tf.expand_dims(A, -1)  # [batch_size, 1]
            mu = tf.expand_dims(mu, -1)  # [batch_size, 1]
            sigma = tf.expand_dims(sigma, -1)  # [batch_size, 1]
            
            # Calculate the wave component
            # t_beat: [batch_size, signal_length]
            # mu_duration: [batch_size, 1] - position scaled by beat duration
            mu_duration = mu * beat_duration_expanded  # [batch_size, 1]
            centered = t_beat - mu_duration  # Broadcasting: [batch_size, signal_length] - [batch_size, 1] -> [batch_size, signal_length]
            squared = tf.square(centered)  # [batch_size, signal_length]
            denom = 2 * tf.square(sigma)  # [batch_size, 1]
            
            # Broadcasting for division: squared [batch_size, signal_length] / denom [batch_size, 1] -> [batch_size, signal_length]
            exp_arg = -squared / denom  # Broadcasting: [batch_size, signal_length] / [batch_size, 1] -> [batch_size, signal_length]
            wave_component = A * tf.exp(exp_arg)  # Broadcasting: [batch_size, 1] * [batch_size, signal_length] -> [batch_size, signal_length]
            
            # Add to the main signal
            ecg_signal = ecg_signal + wave_component

        # Add a dimension for the channel
        return tf.expand_dims(ecg_signal, axis=-1)  # [batch_size, signal_length, 1]

    def get_config(self):
        config = super().get_config()
        config.update({
            "signal_length": self.signal_length,
            "fs": self.fs,
            # The output_scaler is a complex object and cannot be serialized in JSON.
            # It must be passed to the constructor manually when loading the model.
        })
        return config


def build_autoencoder(output_scaler):
    """Builds the full encoder-decoder autoencoder model."""
    # --- Encoder ---
    encoder_input = Input(shape=INPUT_SHAPE, name="encoder_input")
    x = Conv1D(32, 7, activation='relu', padding='same')(encoder_input)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 7, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 7, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    param_output = Dense(NUM_PARAMS, activation='linear', name='param_output')(x)

    encoder = Model(encoder_input, param_output, name="encoder")

    # --- Decoder ---
    decoder_input = Input(shape=(NUM_PARAMS,), name="decoder_input")
    reconstructed_ecg = Decoder(SIGNAL_LENGTH, FS, output_scaler)(decoder_input)
    decoder = Model(decoder_input, reconstructed_ecg, name="decoder")

    # --- Full Autoencoder for Training ---
    autoencoder_input = Input(shape=INPUT_SHAPE, name="autoencoder_input")
    encoded_params = encoder(autoencoder_input)
    decoded_ecg = decoder(encoded_params)

    # Create separate outputs: one for reconstruction loss (ECG), one for parameter prediction loss
    training_model = Model(
        inputs=autoencoder_input,
        outputs=[decoded_ecg, encoded_params],  # List format instead of dict
        name="training_model"
    )

    return training_model, encoder, decoder

# --- 4. Visualization ---
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # With list outputs, Keras labels them as output_1, output_2, etc.
    # Check available keys to determine the correct names
    available_keys = list(history.history.keys())
    print(f"Available keys in history: {available_keys}")
    
    # Try different possible key names based on Keras version
    recon_key = 'output_1_loss'
    val_recon_key = 'val_output_1_loss' 
    param_key = 'output_2_loss'
    val_param_key = 'val_output_2_loss'
    
    # For older Keras versions or different configurations, it might be named differently
    if recon_key not in history.history:
        recon_key = 'output_1_loss' if 'output_1_loss' in history.history else 'decoder_loss'
        val_recon_key = 'val_output_1_loss' if 'val_output_1_loss' in history.history else 'val_decoder_loss'
        param_key = 'output_2_loss' if 'output_2_loss' in history.history else 'encoder_loss'
        val_param_key = 'val_output_2_loss' if 'val_output_2_loss' in history.history else 'val_encoder_loss'
    
    # If we still can't find them, use whatever is available
    if recon_key not in history.history:
        output_keys = [k for k in history.history.keys() if k.endswith('_loss') and not k.startswith('val')]
        val_output_keys = [k for k in history.history.keys() if k.startswith('val_') and k.endswith('_loss') and 'val_loss' not in k]
        
        if len(output_keys) >= 2 and len(val_output_keys) >= 2:
            recon_key = output_keys[0]  # First output loss (reconstruction)
            param_key = output_keys[1]  # Second output loss (parameter prediction)
            val_recon_key = val_output_keys[0]
            val_param_key = val_output_keys[1]

    plt.plot(history.history[recon_key], label='Reconstruction Loss')
    plt.plot(history.history[val_recon_key], label='Val Recon. Loss')
    plt.plot(history.history[param_key], label='Parameter Loss')
    plt.plot(history.history[val_param_key], label='Val Param. Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_reconstruction_results(encoder, decoder, x_test, input_scaler, num_to_plot=4):
    print("Plotting reconstruction results...")
    indices = np.random.choice(len(x_test), num_to_plot, replace=False)
    x_sample = x_test[indices]

    predicted_params = encoder.predict(x_sample)
    reconstructed_ecgs = decoder.predict(predicted_params)

    x_sample_unscaled = input_scaler.inverse_transform(x_sample.reshape(num_to_plot, -1))

    plt.figure(figsize=(15, num_to_plot * 4))
    for i in range(num_to_plot):
        plt.subplot(num_to_plot, 1, i + 1)
        plt.plot(x_sample_unscaled[i], label='Actual ECG', color='blue', alpha=0.8)
        plt.plot(reconstructed_ecgs[i].flatten(), label='Reconstructed ECG', color='red', linestyle='--')
        plt.title(f"Sample {indices[i]}: Actual vs. Reconstructed ECG")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    X_scaled, y_scaled, X_raw, input_scaler, output_scaler = create_dataset()
    
    X_train, X_val, y_train, y_val, X_train_raw, X_val_raw = train_test_split(
        X_scaled, y_scaled, X_raw, test_size=0.2, random_state=42
    )

    training_model, encoder, decoder = build_autoencoder(output_scaler)
    
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=['mse', 'mse'],  # Two losses for the two outputs
        loss_weights=[1.0, PARAM_LOSS_WEIGHT]  # Weights for reconstruction and parameter prediction
    )
    
    training_model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1),
        ModelCheckpoint(filepath=MODEL_WEIGHTS_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True)
    ]

    print("\nStarting model training...")
    history = training_model.fit(
        X_train,
        [X_train_raw.reshape(-1, SIGNAL_LENGTH, 1), y_train],  # List format instead of dict
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, [X_val_raw.reshape(-1, SIGNAL_LENGTH, 1), y_val]),  # List format instead of dict
        callbacks=callbacks,
        verbose=1
    )
    print("Training finished.")

    print("\nLoading best model weights for evaluation...")
    training_model.load_weights(MODEL_WEIGHTS_PATH)

    plot_training_history(history)
    plot_reconstruction_results(encoder, decoder, X_val, input_scaler)

    print("\nPhase 2 script completed successfully.")
    print("The model weights have been saved to", MODEL_WEIGHTS_PATH)
    print("To reuse the trained model:")
    print("1. Load the scalers: input_scaler = joblib.load(INPUT_SCALER_PATH); output_scaler = joblib.load(OUTPUT_SCALER_PATH)")
    print("2. Re-instantiate the model: training_model, _, _ = build_autoencoder(output_scaler)")
    print(f"3. Load the saved weights: training_model.load_weights('{MODEL_WEIGHTS_PATH}')")
