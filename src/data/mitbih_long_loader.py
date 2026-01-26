import wfdb
import numpy as np
import os
import tqdm
import scipy.signal

class MITBIHLongLoader:
    """
    Phase 4: Personalized Forecasting Loader.
    Extracts (Context_10s, Future_10s) pairs from MIT-BIH.
    """
    def __init__(self, download_dir='data/mitbih_raw'):
        self.download_dir = download_dir
        self.fs = 500 # Target Hz
        self.records = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        os.makedirs(download_dir, exist_ok=True)
        
    def download(self):
        print(f"Downloading {len(self.records)} records from MIT-BIH...")
        for rec in tqdm.tqdm(self.records):
            if not os.path.exists(os.path.join(self.download_dir, rec + '.hea')):
                try:
                    wfdb.dl_database('mitdb', self.download_dir, [rec])
                except Exception as e:
                    print(f"Failed {rec}: {e}")
                    
    def process(self, context_sec=10, future_sec=10, stride_sec=10, output_file='data/mitbih_forecasting.npz'):
        """
        Creates (N, 5000) context and (N, 5000) future arrays.
        """
        print("Processing Records for Forecasting...")
        
        X_context = []
        Y_future = []
        
        window_len = int((context_sec + future_sec) * 360) # Native fs is 360
        split_idx = int(context_sec * 360)
        stride = int(stride_sec * 360)
        
        target_len = int(context_sec * self.fs) # 5000 samples
        
        for rec in tqdm.tqdm(self.records):
            try:
                # Read Signal
                record = wfdb.rdrecord(os.path.join(self.download_dir, rec))
                signal = record.p_signal[:, 0] # Lead I
                
                # Filter (Bandpass 0.5-40Hz)
                # Simple normalization for now
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                
                # Sliding Window
                for i in range(0, len(signal) - window_len, stride):
                    segment = signal[i : i + window_len]
                    
                    past = segment[:split_idx]
                    future = segment[split_idx:]
                    
                    # Resample to 500Hz
                    past_res = scipy.signal.resample(past, target_len)
                    future_res = scipy.signal.resample(future, target_len)
                    
                    X_context.append(past_res)
                    Y_future.append(future_res)
                    
            except Exception as e:
                print(f"Error processing {rec}: {e}")
                
        X_context = np.array(X_context, dtype=np.float32)
        Y_future = np.array(Y_future, dtype=np.float32)
        
        # Add Channel Dim
        X_context = X_context[..., np.newaxis]
        Y_future = Y_future[..., np.newaxis]
        
        print(f"✅ Created Dataset: {X_context.shape} Context, {Y_future.shape} Future")
        np.savez_compressed(output_file, context=X_context, future=Y_future)
        print(f"Saved to {output_file}")
        
if __name__ == '__main__':
    loader = MITBIHLongLoader()
    loader.download()
    loader.process()
