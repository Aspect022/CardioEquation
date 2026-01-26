import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ecg_digitizer import ECGDigitizer
from src.inference.pipeline import ECGDenoisingPipeline

def main():
    print("🚀 CardioEquation 2.0: Processing Real PDF...")
    
    # 1. Find a PDF
    dataset_dir = r"d:\Projects\CardioEquation\Dataset"
    pdfs = [os.path.join(r, f) for r, d, files in os.walk(dataset_dir) for f in files if f.endswith('.pdf')]
    
    if not pdfs:
        print("❌ No PDFs found in Dataset directory.")
        return
        
    target_pdf = pdfs[0] # Pick first one
    print(f"📄 Processing: {target_pdf}")
    
    # 2. Digitize (extract noisy signal)
    print("Step 1: Digitizing Signal from PDF...")
    digitizer = ECGDigitizer(target_pdf)
    try:
        # Extract Lead II (Rhythm Strip)
        noisy_signal = digitizer.extract_lead_ii()
        if noisy_signal is None:
            print("❌ Failed to extract signal from PDF.")
            return
            
        print(f"✅ Extracted. Shape: {noisy_signal.shape}")
        
        # Ensure length 2500
        import scipy.signal
        if len(noisy_signal) != 2500:
            print(f"Resampling from {len(noisy_signal)} to 2500...")
            noisy_signal = scipy.signal.resample(noisy_signal, 2500)
            
    except Exception as e:
        print(f"❌ Digitization Error: {e}")
        return

    # 3. Denoise (Diffusion)
    print("Step 2: Denoising with Diffusion Model...")
    pipeline = ECGDenoisingPipeline() # Loads weights automatically
    
    clean_signal = pipeline.process_signal(noisy_signal)
    print("✅ Denoising Complete.")
    
    # 4. Visualize Results
    print("Step 3: Generating Clinical Report (Plot)...")
    plt.figure(figsize=(15, 6))
    
    t = np.arange(2500) / 500
    
    # Normalize Noisy for display comparison
    noisy_disp = (noisy_signal - np.mean(noisy_signal)) / np.std(noisy_signal)
    
    plt.subplot(2, 1, 1)
    plt.plot(t, noisy_disp, color='gray', linewidth=0.8, label='Raw Scan Input')
    plt.title(f"Input: Raw Scan ({os.path.basename(target_pdf)})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, clean_signal, color='blue', linewidth=1.5, label='Diffusion Reconstruction')
    plt.title("CardioEquation 2.0 Output (Clean Digital Twin)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "final_clinical_result.png"
    plt.savefig(output_path)
    print(f"✅ Result saved to {output_path}")

if __name__ == '__main__':
    main()
