import tensorflow as tf

def cosine_similarity_loss(y_true, y_pred):
    """
    Maximizes cosine similarity (minimizes 1 - cos)
    Expects (B, D) vectors.
    """
    normalize_a = tf.nn.l2_normalize(y_true, axis=1)        
    normalize_b = tf.nn.l2_normalize(y_pred, axis=1)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
    return tf.reduce_mean(1.0 - cos_similarity)

def forecasting_loss(y_true, y_pred, y_clean_future, y_recon_future, feature_extractor):
    """
    Phase 4 Loss:
    1. MSE Noise (Standard Diffusion)
    2. MSE Signal (Reconstruction)
    3. IDENTITY LOSS (Cosine Sim of Features)
    
    feature_extractor: The ResNet model to extract "DNA" from generated beats.
    """
    # 1. Diffusion Loss (Noise Prediction)
    mse_noise = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 2. Reconstruction Loss (Signal Fidelity)
    mse_signal = tf.reduce_mean(tf.square(y_clean_future - y_recon_future))
    
    # 3. Identity Loss (Personalization)
    # Extract features from Ground Truth Future
    # We detach gradients here? No, we want generator to match it.
    # Actually, we rely on the pre-trained FE to define "Identity".
    # So FE weights should be frozen during this check or we trust current state.
    
    # Note: y_recon_future is (B, 2500, 1)
    # y_clean_future is (B, 2500, 1)
    
    feat_true = feature_extractor(y_clean_future)
    feat_pred = feature_extractor(y_recon_future)
    
    loss_identity = cosine_similarity_loss(feat_true, feat_pred)
    
    # 4. Spectral Loss (Frequency)
    fft_true = tf.signal.rfft(tf.squeeze(y_clean_future, -1))
    fft_pred = tf.signal.rfft(tf.squeeze(y_recon_future, -1))
    loss_freq = tf.reduce_mean(tf.square(tf.abs(fft_true) - tf.abs(fft_pred)))
    
    total_loss = mse_noise + 1.0 * mse_signal + 0.5 * loss_identity + 0.1 * loss_freq
    
    return total_loss
