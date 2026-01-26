import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ecg_digitizer import ECGDigitizer

class RealisticScanArtifacts:
    """
    Phase 2: Realistic Artifact Simulator.
    Simulates the physical process of printing and scanning an ECG.
    """
    def __init__(self, dpi=200):
        # DPI 200 is sufficient for training and faster than 300/600
        self.dpi = dpi
        self.digitizer = ECGDigitizer(pdf_path=None)
        
    def add_artifacts(self, clean_ecg):
        """
        Input: 1D Clean ECG (normalized approx -1 to 1?)
        Output: 1D Noisy ECG (digitized from simulated scan)
        """
        # 1. Render to Image
        img = self.render_ecg_image(clean_ecg)
        
        # 2. Add Visual Artifacts
        img = self.apply_paper_defects(img)
        img = self.apply_scan_noise(img)
        
        # 3. Digitize back to 1D
        # Preprocess (Grid Removal)
        mask = self.digitizer.preprocess_image(img)
        
        # Extract
        noisy_signal = self.digitizer.extract_signal_from_mask(mask, dpi=self.dpi)
        
        # Handle cases where extraction fails (e.g. too much noise)
        if noisy_signal is None:
            # Fallback to simple noise if digitization fails?
            # Or return None and let data loader retry?
            # For now, return simple noisy version to avoid crashing
            return clean_ecg + np.random.normal(0, 0.1, len(clean_ecg))
            
        # Resizing/Padding to original length (2500)
        target_len = len(clean_ecg)
        current_len = len(noisy_signal)
        
        if current_len != target_len:
            # Resample strictly to target length
            noisy_signal = scipy.signal.resample(noisy_signal, target_len)
            
        return noisy_signal

    def render_ecg_image(self, ecg_signal):
        """
        Draws the ECG signal on a grid.
        Assumes signal is roughly in mV.
        """
        height_mm = 40  # 4cm strip height
        width_mm = len(ecg_signal) / 500 * 25 # 25mm/s
        
        h_px = int(height_mm * self.dpi / 25.4)
        w_px = int(width_mm * self.dpi / 25.4)
        
        # White Background
        img = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255
        
        # 1. Draw Grid
        # 1mm grid (Light Pink/Red)
        grid_color = (200, 200, 255) # BGR (Light Rednish)
        major_grid_color = (150, 150, 255) # Darker Red
        
        pixels_per_mm = self.dpi / 25.4
        
        # Minor lines (1mm)
        for x in range(0, w_px, int(pixels_per_mm)):
            cv2.line(img, (x, 0), (x, h_px), grid_color, 1)
        for y in range(0, h_px, int(pixels_per_mm)):
            cv2.line(img, (0, y), (w_px, y), grid_color, 1)
            
        # Major lines (5mm)
        for x in range(0, w_px, int(pixels_per_mm * 5)):
            cv2.line(img, (x, 0), (x, h_px), major_grid_color, 1)
        for y in range(0, h_px, int(pixels_per_mm * 5)):
            cv2.line(img, (0, y), (w_px, y), major_grid_color, 1)
            
        # 2. Draw Signal (Black)
        # Coordinate mapping
        # 10mm / mV
        pixels_per_mv = 10 * pixels_per_mm
        
        # Center Y
        center_y = h_px // 2
        
        # X step
        # Total samples = len(ecg_signal)
        # Total width = w_px
        # We draw line segments
        
        trace_color = (0, 0, 0) # Black
        thickness = max(1, int(self.dpi / 150)) # Thicker for higher DPI
        
        pts = []
        for i, val in enumerate(ecg_signal):
            x = int(i * w_px / len(ecg_signal))
            # Invert Y (Image coords are Top-Down, Signal is Bottom-Up)
            y = int(center_y - val * pixels_per_mv)
            pts.append([x, y])
            
        pts_np = np.array(pts, np.int32)
        pts_np = pts_np.reshape((-1, 1, 2))
        
        cv2.polylines(img, [pts_np], isClosed=False, color=trace_color, thickness=thickness, lineType=cv2.LINE_AA)
        
        return img
        
    def apply_paper_defects(self, img):
        # 1. Blur (Ink Bleed)
        if np.random.rand() < 0.5:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
            
        # 2. Add Noise (Paper Texture)
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img_int = img.astype(np.int16) + noise
        img = np.clip(img_int, 0, 255).astype(np.uint8)
        
        return img
        
    def apply_scan_noise(self, img):
        # 1. Rotation / Skew
        if np.random.rand() < 0.7:
            angle = np.random.uniform(-2, 2) # +/- 2 degrees
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
            
        return img

import scipy.signal

if __name__ == '__main__':
    # Test
    from ecg_synthesizer import generate_clean_ecg
    import matplotlib.pyplot as plt
    
    clean = generate_clean_ecg()
    sim = RealisticScanArtifacts(dpi=150)
    
    # Generate Sim Image
    img = sim.render_ecg_image(clean)
    cv2.imwrite("sim_test_pre.png", img)
    
    # Full pipeline
    noisy = sim.add_artifacts(clean)
    
    plt.plot(noisy)
    plt.title("Digitized Realistic Artifact Signal")
    plt.savefig("sim_test_signal.png")
    print("Test Complete. Check sim_test_pre.png and sim_test_signal.png")
