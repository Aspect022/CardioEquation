
import numpy as np
import matplotlib.pyplot as plt

def generate_ecg(params: dict, num_beats: int = 5, fs: int = 500, add_noise: bool = False, noise_level: float = 0.02) -> np.ndarray:
    """
    Generates a synthetic ECG waveform using a sum of Gaussian functions.

    Args:
        params (dict): A dictionary containing the parameters for the ECG waves (P, Q, R, S, T),
                       including amplitude (A), position (μ), and width (σ), plus heart rate (HR).
        num_beats (int): The number of heartbeats to generate.
        fs (int): The sampling frequency in Hz.
        add_noise (bool): If True, adds Gaussian noise to the signal.
        noise_level (float): The standard deviation of the Gaussian noise to add.

    Returns:
        np.ndarray: The generated ECG signal.
    """
    hr = params.get('HR', 75)  # Default to 75 bpm if not provided
    beat_duration = 60 / hr
    total_duration = num_beats * beat_duration
    t = np.arange(0, total_duration, 1/fs)

    # Time vector for a single beat
    t_beat = np.arange(0, beat_duration, 1/fs)
    
    ecg_beat = np.zeros_like(t_beat)

    for wave in ['p', 'q', 'r', 's', 't']:
        A = params[f'A_{wave}']
        mu = params[f'μ_{wave}']
        sigma = params[f'σ_{wave}']
        ecg_beat += A * np.exp(-((t_beat - mu * beat_duration)**2) / (2 * sigma**2))

    # Repeat the single beat for the desired number of beats
    ecg = np.tile(ecg_beat, num_beats)

    # Adjust the length of the generated ecg to match the total time vector
    if len(ecg) > len(t):
        ecg = ecg[:len(t)]
    elif len(ecg) < len(t):
        ecg = np.pad(ecg, (0, len(t) - len(ecg)), 'constant')

    if add_noise:
        noise = np.random.normal(0, noise_level, ecg.shape)
        ecg += noise

    return ecg

def plot_ecg(ecg: np.ndarray, fs: int):
    """
    Plots the ECG waveform.

    Args:
        ecg (np.ndarray): The ECG signal to plot.
        fs (int): The sampling frequency in Hz.
    """
    t = np.arange(0, len(ecg) / fs, 1/fs)
    plt.figure(figsize=(15, 5))
    plt.plot(t, ecg)
    plt.title("Generated ECG Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Default parameters for the ECG model
default_params = {
    'HR': 75,       # Heart Rate in bpm
    # P-wave parameters
    'A_p': 0.25,    # Amplitude
    'μ_p': 0.2,     # Position (as a fraction of beat duration)
    'σ_p': 0.025,   # Width
    # Q-wave parameters
    'A_q': -0.15,
    'μ_q': 0.35,
    'σ_q': 0.015,
    # R-wave parameters
    'A_r': 1.0,
    'μ_r': 0.4,
    'σ_r': 0.01,
    # S-wave parameters
    'A_s': -0.25,
    'μ_s': 0.45,
    'σ_s': 0.015,
    # T-wave parameters
    'A_t': 0.35,
    'μ_t': 0.65,
    'σ_t': 0.05,
}

if __name__ == '__main__':
    # Generate and plot a clean ECG
    print("Generating and plotting a clean 5-beat ECG...")
    ecg_signal = generate_ecg(default_params, num_beats=5, fs=500)
    plot_ecg(ecg_signal, fs=500)

    # Generate and plot a noisy ECG
    print("Generating and plotting a noisy 5-beat ECG...")
    ecg_noisy_signal = generate_ecg(default_params, num_beats=5, fs=500, add_noise=True, noise_level=0.05)
    plot_ecg(ecg_noisy_signal, fs=500)
