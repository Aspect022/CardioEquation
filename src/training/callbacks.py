import tensorflow as tf
import os

class SaveSubmodelsCallback(tf.keras.callbacks.Callback):
    def __init__(self, feature_extractor, denoiser, output_dir="models"):
        super(SaveSubmodelsCallback, self).__init__()
        self.feature_extractor = feature_extractor
        self.denoiser = denoiser
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        fe_path = os.path.join(self.output_dir, f"feature_extractor_epoch_{epoch+1:02d}.weights.h5")
        unet_path = os.path.join(self.output_dir, f"diffusion_unet_epoch_{epoch+1:02d}.weights.h5")
        
        print(f"\nSaving submodels to {self.output_dir}...")
        try:
            self.feature_extractor.save_weights(fe_path)
            self.denoiser.save_weights(unet_path)
            print("✅ Saved weights.")
        except Exception as e:
            print(f"❌ Failed to save weights: {e}")
