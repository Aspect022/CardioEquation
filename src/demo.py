#!/usr/bin/env python3
"""
CardioEquation Demo Script
==========================

This script demonstrates the key capabilities of the CardioEquation system:
1. Synthetic ECG generation with custom parameters
2. AI parameter estimation (if models are available)
3. ECG reconstruction and comparison

Usage:
    python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from ecg_generator import generate_ecg, plot_ecg, default_params
import os

def demonstrate_ecg_generation():
    """Demonstrate synthetic ECG generation with different parameter sets."""
    print("=" * 60)
    print("🫀 CardioEquation Demo: Synthetic ECG Generation")
    print("=" * 60)
    
    # Define three different "patients" with varying cardiac characteristics
    patients = {
        "👩‍⚕️ Healthy Young Adult": {
            'HR': 72,        # Normal resting heart rate
            'A_p': 0.25,     # Normal P-wave
            'μ_p': 0.20,     
            'σ_p': 0.025,    
            'A_q': -0.15,    # Normal Q-wave
            'μ_q': 0.35,     
            'σ_q': 0.015,    
            'A_r': 1.0,      # Standard R-wave
            'μ_r': 0.40,     
            'σ_r': 0.010,    
            'A_s': -0.25,    # Normal S-wave
            'μ_s': 0.45,     
            'σ_s': 0.015,    
            'A_t': 0.35,     # Normal T-wave
            'μ_t': 0.65,     
            'σ_t': 0.050,    
        },
        
        "🏃‍♂️ Athletic Heart": {
            'HR': 55,        # Lower resting HR (bradycardia)
            'A_p': 0.30,     # Slightly larger P-wave
            'μ_p': 0.18,     
            'σ_p': 0.030,    
            'A_q': -0.18,    
            'μ_q': 0.34,     
            'σ_q': 0.012,    
            'A_r': 1.4,      # Taller R-wave (left ventricular hypertrophy)
            'μ_r': 0.39,     
            'σ_r': 0.008,    # Sharper R-wave
            'A_s': -0.30,    
            'μ_s': 0.46,     
            'σ_s': 0.012,    
            'A_t': 0.45,     # Prominent T-wave
            'μ_t': 0.68,     
            'σ_t': 0.055,    
        },
        
        "⚡ Stressed Individual": {
            'HR': 95,        # Elevated heart rate (tachycardia)
            'A_p': 0.20,     # Smaller P-wave
            'μ_p': 0.22,     # Delayed P-wave
            'σ_p': 0.020,    
            'A_q': -0.12,    
            'μ_q': 0.37,     
            'σ_q': 0.018,    
            'A_r': 0.85,     # Shorter R-wave
            'μ_r': 0.41,     
            'σ_r': 0.012,    
            'A_s': -0.22,    
            'μ_s': 0.44,     
            'σ_s': 0.016,    
            'A_t': 0.28,     # Flattened T-wave (stress response)
            'μ_t': 0.62,     
            'σ_t': 0.045,    
        }
    }
    
    # Generate and plot ECGs for each patient type
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, (patient_name, params) in enumerate(patients.items()):
        print(f"\n📊 Generating ECG for: {patient_name}")
        print(f"   Heart Rate: {params['HR']} BPM")
        print(f"   R-wave amplitude: {params['A_r']:.2f}")
        
        # Generate ECG signal
        ecg_signal = generate_ecg(params, num_beats=8, fs=500, add_noise=True, noise_level=0.02)
        time_vector = np.arange(0, len(ecg_signal) / 500, 1/500)
        
        # Plot on subplot
        axes[i].plot(time_vector, ecg_signal, linewidth=1.5, color=['blue', 'red', 'green'][i])
        axes[i].set_title(f"{patient_name} (HR: {params['HR']} BPM)", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Amplitude", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        if i == 2:  # Last subplot
            axes[i].set_xlabel("Time (seconds)", fontsize=10)
    
    plt.suptitle("🫀 CardioEquation: Personalized ECG Generation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return patients

def demonstrate_parameter_analysis():
    """Show how different parameters affect the ECG waveform."""
    print("\n" + "=" * 60)
    print("🔬 Parameter Sensitivity Analysis")
    print("=" * 60)
    
    base_params = default_params.copy()
    
    # Test different R-wave amplitudes (simulating different heart conditions)
    r_amplitudes = [0.6, 0.8, 1.0, 1.2, 1.4]  # From weak to strong
    conditions = ["Weak Heart", "Below Average", "Normal", "Strong", "Hypertrophic"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (amp, condition) in enumerate(zip(r_amplitudes, conditions)):
        test_params = base_params.copy()
        test_params['A_r'] = amp
        
        ecg_signal = generate_ecg(test_params, num_beats=3, fs=500)
        time_vector = np.arange(0, len(ecg_signal) / 500, 1/500)
        
        axes[i].plot(time_vector, ecg_signal, linewidth=2, color=plt.cm.viridis(i/4))
        axes[i].set_title(f"{condition}\\nR-amplitude: {amp:.1f}", fontsize=11, fontweight='bold')
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)
        
        if i >= 3:
            axes[i].set_xlabel("Time (seconds)")
    
    # Hide the last subplot (we only have 5 conditions)
    axes[5].set_visible(False)
    
    plt.suptitle("📊 R-wave Amplitude Effects on ECG Morphology", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def demonstrate_equation_extraction():
    """Show the mathematical equation behind the ECG generation."""
    print("\n" + "=" * 60)
    print("🧮 Mathematical Equation Extraction")
    print("=" * 60)
    
    # Take the athletic heart as an example
    params = {
        'HR': 55, 'A_p': 0.30, 'μ_p': 0.18, 'σ_p': 0.030,
        'A_q': -0.18, 'μ_q': 0.34, 'σ_q': 0.012,
        'A_r': 1.4, 'μ_r': 0.39, 'σ_r': 0.008,
        'A_s': -0.30, 'μ_s': 0.46, 'σ_s': 0.012,
        'A_t': 0.45, 'μ_t': 0.68, 'σ_t': 0.055,
    }
    
    beat_duration = 60 / params['HR']  # seconds per beat
    
    print(f"\\n🏃‍♂️ Athletic Heart Equation (HR: {params['HR']} BPM):")
    print(f"Beat Duration: {beat_duration:.3f} seconds")
    print("\\n📐 Mathematical Model:")
    print("ECG(t) = Σ A_wave · exp(-((t - μ_wave · beat_duration)²)/(2σ_wave²))")
    print("         wave∈{P,Q,R,S,T}")
    
    print("\\n📊 Parameter Values:")
    print("-" * 50)
    
    waves = ['p', 'q', 'r', 's', 't']
    wave_names = ['P-wave', 'Q-wave', 'R-wave', 'S-wave', 'T-wave']
    
    for wave, name in zip(waves, wave_names):
        A = params[f'A_{wave}']
        mu = params[f'μ_{wave}']
        sigma = params[f'σ_{wave}']
        timing_sec = mu * beat_duration
        
        print(f"{name:8}: A={A:6.3f}, μ={mu:.3f} ({timing_sec:.3f}s), σ={sigma:.3f}")
    
    print("\\n🔢 Expanded Equation:")
    print("ECG(t) = ")
    
    for i, (wave, name) in enumerate(zip(waves, wave_names)):
        A = params[f'A_{wave}']
        mu = params[f'μ_{wave}']
        sigma = params[f'σ_{wave}']
        timing_sec = mu * beat_duration
        
        sign = "+" if A >= 0 else ""
        if i > 0:
            print("         ", end="")
        else:
            print("         ", end="")
            
        print(f"{sign}{A:.3f} · exp(-((t - {timing_sec:.3f})²)/(2·{sigma:.3f}²))")
    
    print("\\n✨ This equation generates the unique cardiac signature!")

def check_trained_models():
    """Check if trained models are available for AI demonstration."""
    model_files = [
        'best_ecg_model.weights.h5',
        'input_scaler.joblib', 
        'output_scaler.joblib'
    ]
    
    available = all(os.path.exists(f) for f in model_files)
    
    if available:
        print("\\n" + "=" * 60)
        print("🤖 AI Models Available!")
        print("=" * 60)
        print("✅ Trained neural network models detected")
        print("✅ Ready for parameter estimation and ECG reconstruction")
        print("\\n💡 To see AI capabilities in action, run:")
        print("   python ecg_model_trainer.py")
    else:
        print("\\n" + "=" * 60)
        print("⚠️  AI Models Not Found")
        print("=" * 60)
        print("📝 To train the neural network models, run:")
        print("   python ecg_model_trainer.py")
        print("\\n⏱️  This will take ~5-10 minutes and will generate:")
        for f in model_files:
            print(f"   • {f}")
    
    return available

def main():
    """Main demo function."""
    print("🫀" * 20)
    print(" CARDIOEQUATION SYSTEM DEMO")
    print("🫀" * 20)
    
    print("\\n🎯 CardioEquation generates personalized mathematical equations")
    print("   that reproduce individual ECG waveform patterns using AI!")
    
    try:
        # Phase 1: Demonstrate ECG generation
        patients = demonstrate_ecg_generation()
        
        # Phase 2: Show parameter effects
        demonstrate_parameter_analysis()
        
        # Phase 3: Extract mathematical equations
        demonstrate_equation_extraction()
        
        # Phase 4: Check for AI models
        models_available = check_trained_models()
        
        print("\\n" + "=" * 60)
        print("🎉 Demo Complete!")
        print("=" * 60)
        print("\\n📚 Key Takeaways:")
        print("   ✨ Each person can have a unique mathematical cardiac equation")
        print("   🧬 Parameters control P-QRS-T wave morphology")
        print("   🤖 AI can learn to predict these parameters from real ECGs")
        print("   ⚕️  Applications: diagnostics, simulation, biometric ID")
        
        if not models_available:
            print("\\n🚀 Next Steps:")
            print("   1. Run 'python ecg_model_trainer.py' to train AI models")
            print("   2. Explore real ECG integration with PhysioNet data")
            print("   3. Develop clinical applications")
        
    except Exception as e:
        print(f"\\n❌ Error during demo: {str(e)}")
        print("\\n🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check that ecg_generator.py is in the same directory")
        print("   • Verify matplotlib is working for plotting")

if __name__ == "__main__":
    main()
