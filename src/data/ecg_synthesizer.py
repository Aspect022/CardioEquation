import numpy as np
import scipy.signal

def generate_clean_ecg(duration=5, fs=500, hr=60, noise=0):
    """
    Generate a CLEAN synthetic ECG using the dynamical model (McSharry et al.)
    or a simplified Gaussian sum for robustness and speed.
    Using Gaussian Sum here for controllability.
    """
    # Parameters for P, Q, R, S, T
    # (theta, a, b) -> Position (rad), Amplitude, Width
    # Standard values adapted for 2500 samples (5s)
    
    # HR=60 -> 1 beat per second. 5 beats.
    
    t = np.linspace(0, duration, duration*fs)
    signal = np.zeros_like(t)
    
    beats = int(duration * hr / 60)
    beat_dur = 60 / hr
    
    for i in range(beats):
        t_center = 0.5 + i * beat_dur # Center of beat
        
        # P wave
        signal += 0.15 * np.exp(-((t - (t_center - 0.2))**2) / (2 * 0.03**2))
        
        # Q wave
        signal -= 0.15 * np.exp(-((t - (t_center - 0.05))**2) / (2 * 0.015**2))
        
        # R wave
        signal += 1.0 * np.exp(-((t - t_center)**2) / (2 * 0.015**2))
        
        # S wave
        signal -= 0.25 * np.exp(-((t - (t_center + 0.05))**2) / (2 * 0.015**2))
        
        # T wave
        signal += 0.3 * np.exp(-((t - (t_center + 0.3))**2) / (2 * 0.06**2))
        
    return signal

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    s = generate_clean_ecg()
    plt.plot(s)
    plt.title("Clean Synthetic ECG")
    plt.savefig('clean_ecg_test.png')
