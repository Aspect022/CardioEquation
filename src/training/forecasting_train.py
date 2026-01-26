import tensorflow as tf
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.training.forecasting_loss import forecasting_loss

class ForecastingTrainer(tf.keras.Model):
    """
    Phase 4: Personalized Trainer.
    Input: Context (10s)
    Target: Future (10s)
    """
    def __init__(self, feature_extractor, denoiser):
        super(ForecastingTrainer, self).__init__()
        self.feature_extractor = feature_extractor
        self.denoiser = denoiser
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        # inputs is (Context)
        # Returns predicted future
        
        # 1. extract Context DNA
        ctx_features = self.feature_extractor(inputs)
        
        # 2. Diffusion Sampling (Predict Future)
        # For training we just return the Conditional Noise Prediction?
        # No, for Call we usually want generation or just dummy for build.
        # Let's return dummy.
        B = tf.shape(inputs)[0]
        t = tf.zeros((B,))
        
        noise = tf.random.normal((B, 2500, 1))
        return self.denoiser(noise, t, ctx_features)

    def train_step(self, data):
        # Data: (Context_Noisy, Future_Clean)
        context_noisy, future_clean = data
        batch_size = tf.shape(context_noisy)[0]
        
        # 1. Extract Identity from Context
        # This Z represents "The Patient"
        identity_features = self.feature_extractor(context_noisy)
        
        # 2. Diffusion Process on FUTURE
        # We want to generate FUTURE conditioned on CONTEXT
        
        # Sample random noise
        noise = tf.random.normal(tf.shape(future_clean))
        
        # Sample timesteps
        timesteps = tf.random.uniform(
            minval=0, maxval=1, shape=(batch_size,), dtype=tf.float32
        )
        
        # Forward diffusion (add noise to Future)
        # x_t = (1-t) * x0 + t * noise
        t_reshaped = tf.reshape(timesteps, [batch_size, 1, 1])
        noisy_future = (1.0 - t_reshaped) * future_clean + t_reshaped * noise
        
        # 3. Predict Noise
        # Condition on Identity Features (Context)
        with tf.GradientTape() as tape:
            pred_noise = self.denoiser(noisy_future, timesteps, identity_features)
            
            # Reconstruct x0 estimate (for loss calc)
            # x0_est = (x_t - t*pred) / (1-t)
            denom = tf.maximum(1.0 - t_reshaped, 1e-3)
            pred_future = (noisy_future - t_reshaped * pred_noise) / denom
            
            loss = forecasting_loss(noise, pred_noise, future_clean, pred_future, self.feature_extractor)
            
        # 4. Gradient Update
        # Train both FE and UNet?
        # YES. We want FE to learn to extract "Predictive Features".
        trainable_vars = self.feature_extractor.trainable_variables + self.denoiser.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
