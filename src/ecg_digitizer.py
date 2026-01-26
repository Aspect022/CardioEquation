import fitz  # PyMuPDF
import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os

class ECGDigitizer:
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path
        if pdf_path:
            self.doc = fitz.open(pdf_path)
        else:
            self.doc = None
        self.sampling_rate = 500  # Target Hz
        
    def pdf_to_image(self, page_num=0, dpi=300):
        """Convert PDF page to numpy image."""
        page = self.doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    def preprocess_image(self, img):
        """
        Remove grid lines and isolate the ECG trace.
        Assumes standard red/pink grid and black trace.
        """
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define masks for Grid (Red/Pink/Orange)
        # Note: Grid colors vary, but valid traces are usually black/dark blue
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100]) # Dark pixels (Trace)
        
        # Create mask for the signal (dark pixels)
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Clean up noise with morphological operations
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def extract_signal_from_mask(self, mask, dpi=300):
        """
        Extract time-series signal from binary mask.
        Returns: normalized_signal (1D array)
        """
        # Find all non-zero points (y, x)
        points = np.column_stack(np.where(mask > 0))
        
        if len(points) == 0:
            return None
            
        # Sort by X coordinate (time)
        points = points[points[:, 1].argsort()]
        
        x_coords = points[:, 1]
        y_coords = points[:, 0]
        
        # Coordinate mapping: Image Y is top-down, Voltage is bottom-up
        # Invert Y so up is positive voltage
        y_coords = -y_coords
        
        # 1. Handle multiple Y values per X (thick lines/vertical segments)
        # Strategy: Take the mean Y for each unique X
        unique_x = np.unique(x_coords)
        averaged_y = []
        
        for x in unique_x:
            ys = y_coords[x_coords == x]
            averaged_y.append(np.mean(ys))
            
        averaged_y = np.array(averaged_y)
        
        # 2. Resample to target sampling rate
        # Calculate physical duration
        # Standard ECG speed: 25 mm/s
        # DPI = dots per inch (25.4 mm)
        pixels_per_mm = dpi / 25.4
        pixels_per_sec = pixels_per_mm * 25.0
        
        total_pixels = unique_x[-1] - unique_x[0]
        total_seconds = total_pixels / pixels_per_sec
        
        target_samples = int(total_seconds * self.sampling_rate)
        
        # Interpolate to target samples
        resampled_signal = scipy.signal.resample(averaged_y, target_samples)
        
        # 3. Normalize Voltage
        # Remove baseline wander (centering)
        resampled_signal = resampled_signal - np.median(resampled_signal)
        
        # Normalize amplitude (Robust scaling using IQR to handle spikes)
        q75, q25 = np.percentile(resampled_signal, [75 ,25])
        iqr = q75 - q25
        if iqr > 0:
            resampled_signal = resampled_signal / iqr
            
        return resampled_signal

    def extract_lead_ii(self, page_num=0):
        """
        Attempt to extract the long rhythm strip (Lead II) usually at bottom.
        """
        img = self.pdf_to_image(page_num)
        h, w, _ = img.shape
        
        # Heuristic: Rhythm strip is usually the bottom 15-20% of the page
        crop_start_y = int(h * 0.80)
        crop_img = img[crop_start_y:h-50, 50:w-50] # Crop margins
        
        mask = self.preprocess_image(crop_img)
        signal = self.extract_signal_from_mask(mask)
        
        return signal

if __name__ == '__main__':
    # Test on a file if it exists
    dataset_dir = r"d:\Projects\CardioEquation\Dataset"
    
    # Recursive scan for pdfs
    pdfs = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdfs.append(os.path.join(root, file))
    
    if not pdfs:
        print("No PDFs found in Dataset directory.")
    else:
        # Pick one at random or the first one
        test_pdf = pdfs[0]
        print(f"Found {len(pdfs)} PDFs.")
        print(f"Testing digitization on: {test_pdf}")
        
        digitizer = ECGDigitizer(test_pdf)
        # Try to plot the intermediate mask to debug
        # We can't easily show images in the terminal, but we can print stats
        
        try:
            signal = digitizer.extract_lead_ii()
            
            if signal is not None:
                print(f"Successfully extracted signal. Shape: {signal.shape}")
                print(f"Duration: {len(signal)/500:.2f} seconds")
                print(f"Amplitude Range: {signal.min():.3f} to {signal.max():.3f}")
                
                # Check for NaNs
                if np.isnan(signal).any():
                    print("WARNING: Signal contains NaNs!")
                
                plt.figure(figsize=(15, 4))
                t = np.arange(len(signal)) / 500
                plt.plot(t, signal)
                plt.title(f"Digitized ECG Signal from {os.path.basename(test_pdf)} (Lead II)")
                plt.xlabel("Time (s)")
                plt.ylabel("Normalized Amplitude")
                plt.grid(True)
                plt.savefig('digitization_test.png')
                print("Plot saved to digitization_test.png")
                # plt.show() # Blocking
            else:
                print("Failed to extract signal (returned None).")
        except Exception as e:
            print(f"Error during extraction: {e}")
            import traceback
            traceback.print_exc()
