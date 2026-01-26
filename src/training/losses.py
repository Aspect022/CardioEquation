import tensorflow as tf

def combined_loss(y_true, y_pred, y_clean, y_recon):
    """
    Combined Loss for Training Loop.
    y_true: Noise (Target for Diffusion)
    y_pred: Predicted Noise
    y_clean: Original Clean Signal
    y_recon: Reconstructed Signal (Noisy - Predicted Noise)
    """
    # 1. Reconstruction Loss (Diffusion Objective)
    mse_noise = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 2. Reconstruction Fidelity (Signal Space)
    mse_signal = tf.reduce_mean(tf.square(y_clean - y_recon))
    
    # 3. Spectral Loss (Frequency domain)
    fft_clean = tf.signal.rfft(tf.squeeze(y_clean, -1))
    fft_recon = tf.signal.rfft(tf.squeeze(y_recon, -1))
    
    # Magnitude only
    mag_clean = tf.abs(fft_clean)
    mag_recon = tf.abs(fft_recon)
    loss_freq = tf.reduce_mean(tf.square(mag_clean - mag_recon))
    
    # 4. Scale Invariant Loss (Correlation)
    # Pearson Correlation = cov(x,y) / (std(x)*std(y))
    # We want to Maximize correlation -> Minimize (1 - corr)
    
    # Flatten batches
    flat_clean = tf.reshape(y_clean, [tf.shape(y_clean)[0], -1])
    flat_recon = tf.reshape(y_recon, [tf.shape(y_recon)[0], -1])
    
    mean_clean = tf.reduce_mean(flat_clean, axis=1, keepdims=True)
    mean_recon = tf.reduce_mean(flat_recon, axis=1, keepdims=True)
    
    cen_clean = flat_clean - mean_clean
    cen_recon = flat_recon - mean_recon
    
    cov = tf.reduce_mean(cen_clean * cen_recon, axis=1)
    std_clean = tf.math.reduce_std(flat_clean, axis=1)
    std_recon = tf.math.reduce_std(flat_recon, axis=1)
    
    corr = cov / (std_clean * std_recon + 1e-8)
    loss_scale = tf.reduce_mean(1.0 - corr)

    # Weights
    total_loss = mse_noise + 0.5 * mse_signal + 0.1 * loss_freq + 0.2 * loss_scale
    
    return total_loss
