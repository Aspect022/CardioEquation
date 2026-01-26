import numpy as np
import scipy.signal
from config import SIGNAL_LENGTH, FS

class ECGPreprocessor:
    def __init__(self, target_fs=500):
        self.fs = target_fs
        
    def bandpass_filter(self, signal, lowcut=0.5, highcut=40.0, order=4):
        """
        Apply Butterworth bandpass filter to remove baseline wander and HF noise.
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        return filtered_signal

    def normalize_signal(self, signal):
        """
        Min-Max normalization to [-1, 1] range.
        Or robust scaling. Let's stick to simple min-max for now as per trainer.
        """
        # Remove DC offset
        signal = signal - np.mean(signal)
        
        # Scale to [-1, 1] roughly, preserving relative amplitude
        # Using 99th percentile to avoid outlier spikes affecting scale too much
        abs_max = np.percentile(np.abs(signal), 99.5)
        if abs_max > 0:
            signal = signal / abs_max
            
        # Clip to ensure hard [-1, 1] bounds
        signal = np.clip(signal, -1.0, 1.0)
        return signal

    def segment_signal(self, signal, window_size=SIGNAL_LENGTH, overlap=0.5):
        """
        Slice long signal into chunks of `window_size`.
        Arguments:
            signal: 1D array
            window_size: samples (default 2500)
            overlap: fraction of overlap (0 to 1)
        Returns:
            segments: array of shape (N, window_size, 1)
        """
        step = int(window_size * (1 - overlap))
        segments = []
        
        length = len(signal)
        if length < window_size:
            # Pad if too short
            pad_width = window_size - length
            padded = np.pad(signal, (0, pad_width), 'constant')
            segments.append(padded)
        else:
            for start in range(0, length - window_size + 1, step):
                end = start + window_size
                segment = signal[start:end]
                segments.append(segment)
                
        if not segments:
            return np.array([])
            
        return np.array(segments)[..., np.newaxis] # Add channel dim

if __name__ == '__main__':
    # Test
    fake_signal = np.random.randn(5300)
    preprocessor = ECGPreprocessor()
    filtered = preprocessor.bandpass_filter(fake_signal)
    segments = preprocessor.segment_signal(filtered)
    print(f"Original: {len(fake_signal)}")
    print(f"Segments shape: {segments.shape}")
