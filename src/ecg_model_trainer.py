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

# Import configuration
from config import (
    SIGNAL_LENGTH, FS, NUM_PARAMS, PARAM_KEYS, LEARNING_RATE, 
    PARAM_LOSS_WEIGHT, EPOCHS, BATCH_SIZE, NUM_SAMPLES,
    MODEL_WEIGHTS_PATH, INPUT_SCALER_PATH, OUTPUT_SCALER_PATH
)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import from Phase 1
from ecg_generator import generate_ecg, default_params

# --- 1. Constants and Configuration ---
INPUT_SHAPE = (SIGNAL_LENGTH, 1)

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

    # Ensure the models directory exists before saving scalers
    os.makedirs(os.path.dirname(INPUT_SCALER_PATH), exist_ok=True)
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
        # Calculate the inverse transform using stored scaler parameters
        if hasattr(self.output_scaler, 'scale_') and hasattr(self.output_scaler, 'min_'):
            # Calculate inverse transform using stored scaler parameters
            params_unscaled = params_scaled * (1.0 / self.output_scaler.scale_) + self.output_scaler.min_
        else:
            # Fallback if scaler parameters aren't accessible
            params_unscaled = params_scaled

        # Get the batch size
        batch_size = tf.shape(params_scaled)[0]
        
        # Create the time vector for the entire signal length
        t = tf.range(0, self.signal_length, dtype=tf.float32) / self.fs  # [signal_length]
        t_expanded = tf.reshape(t, [1, -1])  # [1, signal_length]
        
        # Initialize ECG signal [batch_size, signal_length]
        ecg_signal = tf.zeros((batch_size, self.signal_length), dtype=tf.float32)
        
        # Extract parameters for each sample in the batch
        HR = params_unscaled[:, 0]  # [batch_size]
        A_p, mu_p, sigma_p = params_unscaled[:, 1], params_unscaled[:, 2], params_unscaled[:, 3]
        A_q, mu_q, sigma_q = params_unscaled[:, 4], params_unscaled[:, 5], params_unscaled[:, 6]
        A_r, mu_r, sigma_r = params_unscaled[:, 7], params_unscaled[:, 8], params_unscaled[:, 9]
        A_s, mu_s, sigma_s = params_unscaled[:, 10], params_unscaled[:, 11], params_unscaled[:, 12]
        A_t, mu_t, sigma_t = params_unscaled[:, 13], params_unscaled[:, 14], params_unscaled[:, 15]
        
        # Calculate beat duration for each sample in the batch
        beat_duration = 60.0 / HR  # [batch_size]
        
        # Reshape for broadcasting: [batch_size, 1]
        beat_duration = tf.expand_dims(beat_duration, axis=1)
        A_p = tf.expand_dims(A_p, axis=1)
        mu_p = tf.expand_dims(mu_p, axis=1)
        sigma_p = tf.expand_dims(sigma_p, axis=1)
        A_q = tf.expand_dims(A_q, axis=1)
        mu_q = tf.expand_dims(mu_q, axis=1)
        sigma_q = tf.expand_dims(sigma_q, axis=1)
        A_r = tf.expand_dims(A_r, axis=1)
        mu_r = tf.expand_dims(mu_r, axis=1)
        sigma_r = tf.expand_dims(sigma_r, axis=1)
        A_s = tf.expand_dims(A_s, axis=1)
        mu_s = tf.expand_dims(mu_s, axis=1)
        sigma_s = tf.expand_dims(sigma_s, axis=1)
        A_t = tf.expand_dims(A_t, axis=1)
        mu_t = tf.expand_dims(mu_t, axis=1)
        sigma_t = tf.expand_dims(sigma_t, axis=1)

        # For each wave type, add its contribution to the signal
        waves = [
            ('p', A_p, mu_p, sigma_p),
            ('q', A_q, mu_q, sigma_q), 
            ('r', A_r, mu_r, sigma_r),
            ('s', A_s, mu_s, sigma_s),
            ('t', A_t, mu_t, sigma_t)
        ]
        
        # Calculate wave contributions for each wave type
        for wave_name, A, mu, sigma in waves:
            # Calculate wave positions in continuous time based on HR
            # For each point in time, determine where it falls in the beat cycle
            # Use modulo to handle multiple beats
            
            # The modulo operation should be based on the actual beat duration, not HR*t
            # For each time point, determine where it is within the current beat cycle
            # We'll calculate time within each beat by using floor mod of time by beat duration
            t_beat_position = tf.math.floormod(t_expanded, beat_duration)  # Time within current beat: [batch_size, signal_length]
            
            # Calculate where wave should occur in the beat cycle (based on mu and beat duration)
            expected_wave_time = mu * beat_duration  # [batch_size, 1] - where wave should be in beat
            
            # Calculate the time difference from expected wave position
            time_diff = t_beat_position - expected_wave_time  # [batch_size, signal_length]
            
            # Calculate the Gaussian component
            squared_diff = tf.square(time_diff)  # [batch_size, signal_length]
            denominator = 2.0 * tf.square(sigma)  # [batch_size, 1]
            
            # Calculate wave component: A * exp(-diff^2 / (2*sigma^2))
            exp_term = -squared_diff / denominator  # Broadcasting: [batch_size, signal_length]
            wave_component = A * tf.exp(exp_term)  # [batch_size, signal_length]
            
            # Add this wave's contribution to the signal
            ecg_signal = ecg_signal + wave_component

        # Add channel dimension: [batch_size, signal_length, 1]
        return tf.expand_dims(ecg_signal, axis=-1)

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
    available_keys = list(history.history.keys())
    print(f"Available keys in history: {available_keys}")
    
    # For list outputs, Keras typically labels them as 'output_1_loss', 'output_2_loss', etc.
    # Try to find the correct keys
    recon_key = None
    param_key = None
    val_recon_key = None
    val_param_key = None
    
    # Look for the keys in the history
    for key in history.history.keys():
        if 'output_1_loss' in key and not key.startswith('val_'):
            recon_key = key
        elif 'output_2_loss' in key and not key.startswith('val_'):
            param_key = key
        elif 'output_1_loss' in key and key.startswith('val_'):
            val_recon_key = key
        elif 'output_2_loss' in key and key.startswith('val_'):
            val_param_key = key
    
    # Fallback if we can't find the standard keys
    if not all([recon_key, param_key, val_recon_key, val_param_key]):
        # Try to identify them by pattern matching
        loss_keys = [k for k in available_keys if k.endswith('_loss') and not k.startswith('val_') and k != 'loss']
        val_loss_keys = [k for k in available_keys if k.startswith('val_') and k.endswith('_loss') and k != 'val_loss']
        
        if len(loss_keys) >= 2:
            recon_key = loss_keys[0]  # First output loss (reconstruction)
            param_key = loss_keys[1]  # Second output loss (parameter prediction)
            
        if len(val_loss_keys) >= 2:
            val_recon_key = val_loss_keys[0]  # First validation output loss
            val_param_key = val_loss_keys[1]  # Second validation output loss

    # Plot the losses if we found the keys
    if all([recon_key, param_key, val_recon_key, val_param_key]):
        plt.plot(history.history[recon_key], label='Reconstruction Loss')
        plt.plot(history.history[val_recon_key], label='Val Recon. Loss')
        plt.plot(history.history[param_key], label='Parameter Loss')
        plt.plot(history.history[val_param_key], label='Val Param. Loss')
    else:
        # Fallback to generic plotting if we can't identify the specific keys
        print("Could not identify specific loss keys, plotting available losses:")
        for key in available_keys:
            if key.endswith('_loss') and key not in ['loss']:
                plt.plot(history.history[key], label=key)
                
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
    
    # Split the data properly
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Also split the raw data
    X_train_raw, X_val_raw = train_test_split(
        X_raw, test_size=0.2, random_state=42
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
    print("To reuse the trained model:")
    print("1. Load the scalers: input_scaler = joblib.load(INPUT_SCALER_PATH); output_scaler = joblib.load(OUTPUT_SCALER_PATH)")
    print("2. Re-instantiate the model: training_model, _, _ = build_autoencoder(output_scaler)")
    print(f"3. Load the saved weights: training_model.load_weights('{MODEL_WEIGHTS_PATH}')")
