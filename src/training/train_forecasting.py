import numpy as np
import tensorflow as tf
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.feature_extractor import FeatureExtractor
from src.models.diffusion_unet import ConditionalDiffusionUNet
from src.training.forecasting_train import ForecastingTrainer
from src.data.predictive_dataset import PredictiveDataset
from src.training.callbacks import SaveSubmodelsCallback

def train_phase_4():
    print("🚀 Phase 4: Personalized Forecasting Training...")
    
    # 1. Models
    fe = FeatureExtractor()
    unet = ConditionalDiffusionUNet()
    
    # Build
    dummy = tf.zeros((1, 2500, 1))
    fe(dummy)
    unet(dummy, tf.zeros((1,)), tf.zeros((1, 512)))
    
    # 2. Load Phase 2 Weights (Transfer Learning)
    # We transfer knowledge from Denoising (Phase 2) to Forecasting (Phase 4)
    # The FE knows how to extract features from noise.
    # The UNet knows how to generate ECGs.
    print("🔄 Loading Phase 2 Weights...")
    try:
        fe.load_weights("models/feature_extractor_epoch_02.weights.h5")
        unet.load_weights("models/diffusion_unet_epoch_02.weights.h5")
        print("✅ Phase 2 Weights Loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load Phase 2 weights: {e}")
        
    # 3. Dataset
    # Context -> Future
    dataset = PredictiveDataset(batch_size=4)
    
    # 4. Trainer
    trainer = ForecastingTrainer(fe, unet)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), run_eagerly=True)
    
    # 5. Callbacks
    # Save as "forecasting_epoch_XX"
    # We need a custom callback wrapper again or update the existing one.
    # The existing SaveSubmodelsCallback relies on passing fe/unet instances.
    # It should work fine.
    
    # 6. Train (Overnight Configuration)
    print("Training Focus: Minimizing Identity Loss (Personalization)...")
    
    # Save every 10 epochs to avoid disk fill, but keep latest
    class SaveForecastingPeriodic(SaveSubmodelsCallback):
        def on_epoch_end(self, epoch, logs=None):
            # Save if epoch is multiple of 10 or last
            if (epoch + 1) % 10 == 0:
                fe_path = os.path.join(self.output_dir, f"fe_forecast_epoch_{epoch+1:03d}.weights.h5")
                unet_path = os.path.join(self.output_dir, f"unet_forecast_epoch_{epoch+1:03d}.weights.h5")
                print(f"\nSaving Forecasting models to {self.output_dir}...")
                self.feature_extractor.save_weights(fe_path)
                self.denoiser.save_weights(unet_path)
    
    ckpt = SaveForecastingPeriodic(fe, unet)
    
    trainer.fit(dataset, epochs=200, callbacks=[ckpt])
    print("✅ Phase 4 Training Complete.")

if __name__ == '__main__':
    train_phase_4()
