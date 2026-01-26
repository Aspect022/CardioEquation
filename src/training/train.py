import tensorflow as tf
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.feature_extractor import FeatureExtractor
from src.models.diffusion_unet import ConditionalDiffusionUNet
from src.data.synthetic_dataset import SyntheticNoiseDataset
from src.training.losses import combined_loss

class DiffusionTrainer(tf.keras.Model):
    def __init__(self, feature_extractor, denoiser):
        super(DiffusionTrainer, self).__init__()
        self.feature_extractor = feature_extractor # ResNet
        self.denoiser = denoiser # UNet
        
        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        # Dummy call to build graph for checkpoint loading
        # inputs: noisy
        feats = self.feature_extractor(inputs)
        # Dummy timestep/cond for shape inference
        B = tf.shape(inputs)[0]
        t = tf.zeros((B,))
        return self.denoiser(inputs, t, feats)

    def train_step(self, data):
        noisy_signal, clean_signal = data
        batch_size = tf.shape(noisy_signal)[0]
        
        # 1. Sample Random Timesteps (0 to 1) check distribution?
        # Standard: Uniform [0, 1] or discrete?
        # Let's use continuous t \in [0, 1] for embedding
        timesteps = tf.random.uniform((batch_size,), minval=0.0, maxval=1.0)
        
        # 2. Sample Gaussian Noise for Diffusion Target
        noise = tf.random.normal(tf.shape(clean_signal))
        
        # 3. Forward Diffusion Process: Create "Diffused" Training Input
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        # Linear noise schedule for simplicity
        # alpha_bar = 1 - t  (approx)
        # Signal fades, Noise constant? Or simple interpolation?
        # y_noisy = (1-t) * clean + t * noise
        # Let's use simple interpolation for "Denoising Score Matching" style
        # signal_scale = tf.cos(timesteps * 3.1415 / 2)
        # noise_scale = tf.sin(timesteps * 3.1415 / 2)
        # Using linear interpolation for simplicity in Phase 1
        
        signal_scale = 1.0 - timesteps
        noise_scale = timesteps
        
        # Broadcasting scale
        signal_scale = tf.reshape(signal_scale, [batch_size, 1, 1])
        noise_scale = tf.reshape(noise_scale, [batch_size, 1, 1])
        
        # Target for UNet is NOISE (epsilon)
        diffused_input = signal_scale * clean_signal + noise_scale * noise
        
        with tf.GradientTape() as tape:
            # 4. Extract Features from the OBSERVATION (The Noisy Clinical Input)
            # wait, the "Condition" is the corrupted clinical input? 
            # OR is it just general features?
            # In "Conditional Diffusion", we look at the Noisy Input `noisy_signal`
            # and try to denoise `diffused_input` guided by `noisy_signal`.
            # Wait. `diffused_input` is derived from CLEAN.
            # `noisy_signal` is the real artifact-laden input.
            
            # The Goal: Mapping Noisy_Signal -> Clean_Signal.
            # Standard Diffusion: Input=Diffused(Clean), Cond=Class/Text.
            # Image-to-Image / Restore: Input=Concatenate(Diffused, Noisy)?
            # OR Cond=Noisy.
            
            # Our Architecture:
            # Inputs to UNet: (Diffused_State, Timestep, Condition)
            # Condition = Features(Noisy_Clinical_Input)
            
            features = self.feature_extractor(noisy_signal)
            
            # 5. Predict Noise
            pred_noise = self.denoiser(diffused_input, timesteps, features)
            
            # 6. Reconstruct Signal (for aux losses)
            # clean_recon = (diffused - noise_scale * pred) / signal_scale
            # Handle numerical stability for t=1
            safe_signal_scale = tf.maximum(signal_scale, 1e-4)
            clean_recon = (diffused_input - noise_scale * pred_noise) / safe_signal_scale
            
            # 7. Compute Loss
            loss = combined_loss(noise, pred_noise, clean_signal, clean_recon)

        # Gradients
        trainable_vars = self.feature_extractor.trainable_variables + self.denoiser.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

def main():
    # Detect Hardware
    # Allow memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("🚀 Starting Diffusion Training (Phase 1)...")
    
    # Models
    feature_net = FeatureExtractor()
    unet = ConditionalDiffusionUNet()
    
    # Warmup / Build
    print("Building models...")
    dummy_in = tf.zeros((1, 2500, 1))
    dummy_t = tf.zeros((1,))
    dummy_cond = tf.zeros((1, 512))
    
    _ = feature_net(dummy_in)
    _ = unet(dummy_in, dummy_t, dummy_cond)
    print("Models built.")
    
    # Compile
    # Using slightly lower LR for diffusion stability
    trainer = DiffusionTrainer(feature_net, unet)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), run_eagerly=True) # Lower LR for fine-tuning
    
    # Resume from Phase 1 (if available)
    # Note: Keras Trainer loading might be tricky directly.
    # We load submodels manually if needed, or build and load.
    # Let's try loading submodels manually data is safest.
    try:
        if os.path.exists("models/feature_extractor_epoch_01.weights.h5"):
            print("🔄 Resuming from Phase 1 Weights...")
            feature_net.load_weights("models/feature_extractor_epoch_01.weights.h5")
            unet.load_weights("models/diffusion_unet_epoch_01.weights.h5")
            print("✅ Weights Loaded.")
    except Exception as e:
        print(f"⚠️ Could not load Phase 1 weights: {e}")

    # Dataset
    # Phase 2: Realistic Dataset
    from src.data.realistic_dataset import RealisticDataset
    dataset = RealisticDataset(batch_size=4) 

    
    # Train
    checkpoint_path = "models/diffusion_epoch_{epoch:02d}.weights.h5"
    os.makedirs("models", exist_ok=True)
    
    from src.training.callbacks import SaveSubmodelsCallback
    
    ckpt_callback = SaveSubmodelsCallback(feature_net, unet)
    
    trainer.fit(dataset, epochs=2, callbacks=[ckpt_callback]) # 2 Epochs for Quick Proof
    
    print("✅ Training Complete.")
    
    # Save Final Models
    feature_net.save_weights("src/models/feature_extractor_final.weights.h5")
    unet.save_weights("src/models/diffusion_unet_final.weights.h5")
    print("Saved final weights.")

if __name__ == '__main__':
    main()
