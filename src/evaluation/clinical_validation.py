"""
Clinical ECG Validation using Real Hospital Data
==================================================
Loads ECG PDFs from the Dataset/ folder, digitizes them using
the ECGDigitizer, and validates the DiT-ECG model against
real clinical signals.

The Dataset/ folder contains 24 real ECG reports from a hospital:
- Each subfolder (ecg_report_XXXXXX) contains a single PDF
- Multiple records may belong to the same patient (same name prefix)
- These are used ONLY for validation/testing, never for training

Usage:
    python src/evaluation/clinical_validation.py --model_path checkpoints/dit_ecg_ema_final.pt
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def discover_clinical_pdfs(dataset_dir='Dataset'):
    """
    Discover all clinical ECG PDFs in the Dataset folder.

    Returns:
        list of dicts: [{
            'pdf_path': str,
            'patient_name': str,
            'report_id': str,
            'timestamp': str
        }]
    """
    records = []

    if not os.path.exists(dataset_dir):
        print(f"⚠️  Dataset directory not found: {dataset_dir}")
        return records

    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        pdfs = glob.glob(os.path.join(folder_path, '*.pdf'))
        for pdf_path in pdfs:
            filename = os.path.basename(pdf_path)
            name_parts = filename.replace('.pdf', '').split('_')

            # Parse: NAME_DATE_TIME.pdf
            patient_name = name_parts[0] if name_parts[0] else 'Unknown'
            timestamp = '_'.join(name_parts[1:]) if len(name_parts) > 1 else 'unknown'

            records.append({
                'pdf_path': pdf_path,
                'patient_name': patient_name,
                'report_id': folder,
                'timestamp': timestamp,
            })

    return records


def digitize_clinical_ecgs(records, target_length=2500, fs=500):
    """
    Digitize clinical ECG PDFs to 1D signals for validation.

    Uses the existing ECGDigitizer from the project.

    Args:
        records: list of record dicts from discover_clinical_pdfs
        target_length: target signal length (2500 = 5s at 500Hz)
        fs: target sampling rate
    Returns:
        signals: dict mapping patient_name → list of (T,) numpy arrays
    """
    try:
        from src.ecg_digitizer import ECGDigitizer
    except ImportError:
        print("⚠️  ECGDigitizer not available. Ensure src/ecg_digitizer.py exists.")
        return {}

    digitizer = ECGDigitizer()
    patient_signals = {}

    for rec in records:
        try:
            pages = digitizer.load_pdf(rec['pdf_path'])
            if not pages:
                print(f"   ⚠️ No pages in {rec['pdf_path']}")
                continue

            # Process first page
            signal = digitizer.extract_signal(pages[0])
            if signal is None or len(signal) < 100:
                print(f"   ⚠️ Failed to extract signal from {rec['pdf_path']}")
                continue

            # Resample to target length
            from scipy.signal import resample
            signal_resampled = resample(signal, target_length)

            # Normalize
            signal_resampled = (signal_resampled - signal_resampled.mean()) / (signal_resampled.std() + 1e-8)

            patient = rec['patient_name']
            if patient not in patient_signals:
                patient_signals[patient] = []
            patient_signals[patient].append(signal_resampled.astype(np.float32))

            print(f"   ✅ {rec['report_id']}: {patient} — {len(signal_resampled)} samples")

        except Exception as e:
            print(f"   ❌ Error processing {rec['pdf_path']}: {e}")

    return patient_signals


def run_clinical_validation(model_path, fe_path=None, dataset_dir='Dataset', output_dir='outputs/clinical_validation'):
    """
    Run full clinical validation:
    1. Digitize hospital PDFs
    2. Extract identity vectors from real ECGs
    3. Generate synthetic ECGs conditioned on real patient identity
    4. Compare using evaluation metrics
    5. Produce visual reports

    Args:
        model_path: Path to trained DiT-ECG weights
        fe_path: Path to feature extractor weights
        dataset_dir: Path to Dataset/ folder with hospital PDFs
        output_dir: Where to save validation reports
    """
    os.makedirs(output_dir, exist_ok=True)

    print("🏥 Clinical Validation: Real Hospital ECG Data")
    print("=" * 60)

    # 1. Discover and digitize
    records = discover_clinical_pdfs(dataset_dir)
    print(f"📋 Found {len(records)} ECG reports")

    patients = set(r['patient_name'] for r in records)
    print(f"👤 Unique patients: {len(patients)}")
    for p in sorted(patients):
        count = sum(1 for r in records if r['patient_name'] == p)
        print(f"   {p}: {count} records")

    print(f"\n📡 Digitizing ECG signals...")
    patient_signals = digitize_clinical_ecgs(records)

    if not patient_signals:
        print("❌ No signals could be digitized. Check ECGDigitizer.")
        return

    print(f"\n✅ Successfully digitized {sum(len(v) for v in patient_signals.values())} signals from {len(patient_signals)} patients")

    # 2. Try to load model and run inference
    try:
        import torch
        from src.inference.pipeline_v2 import ECGPipelineV2

        pipe = ECGPipelineV2(model_path, fe_path)

        # 3. For each patient with multiple records, use one as context
        # and compare generated ECG against the other(s)
        print(f"\n🎯 Running generation + validation...")
        all_real = []
        all_generated = []

        for patient_name, signals in patient_signals.items():
            if len(signals) < 2:
                # Only one recording — use it for identity extraction + generate
                context = signals[0]
                generated = pipe.generate(context, num_steps=50, guidance_scale=3.0, num_samples=1)
                all_real.append(context)
                all_generated.append(generated[0])
            else:
                # Multiple recordings — use first as context, validate against rest
                context = signals[0]
                for i, real_other in enumerate(signals[1:], 1):
                    generated = pipe.generate(context, num_steps=50, guidance_scale=3.0, num_samples=1)
                    all_real.append(real_other)
                    all_generated.append(generated[0])

        all_real = np.array(all_real)
        all_generated = np.array(all_generated)

        # 4. Evaluate
        from src.evaluation.eval_metrics import ECGEvaluator
        from src.models.feature_extractor_pt import FeatureExtractorPT

        encoder = FeatureExtractorPT()
        if fe_path and os.path.exists(fe_path):
            encoder.load_state_dict(torch.load(fe_path, weights_only=True))

        evaluator = ECGEvaluator(encoder=encoder)
        results = evaluator.evaluate_all(all_real, all_generated)

        print(f"\n📊 Clinical Validation Results:")
        print(f"   FFD (Fréchet ECG Distance): {results.get('FFD', 'N/A')}")
        print(f"   MMD: {results.get('MMD', 'N/A')}")
        print(f"   HR MAE: {results.get('HR_MAE', 'N/A')}")
        print(f"   Re-ID Top-1: {results.get('ReID_Top1', 'N/A')}")
        print(f"   Re-ID Top-5: {results.get('ReID_Top5', 'N/A')}")

        # 5. Save visual comparison
        _save_clinical_plots(patient_signals, all_generated, output_dir)

        # Save results
        import json
        results_clean = {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                         for k, v in results.items()}
        with open(os.path.join(output_dir, 'clinical_results.json'), 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"\n💾 Results saved to {output_dir}/clinical_results.json")

    except ImportError as e:
        print(f"\n⚠️  Cannot run model inference (PyTorch not available): {e}")
        print("   Run this on the GPU server after training.")

    # Always save the digitized signals for later use
    _save_digitized_signals(patient_signals, output_dir)


def _save_digitized_signals(patient_signals, output_dir):
    """Save digitized signals as .npz for later use."""
    all_signals = []
    all_labels = []
    label_map = {}
    idx = 0

    for patient_name, signals in patient_signals.items():
        if patient_name not in label_map:
            label_map[patient_name] = idx
            idx += 1
        for sig in signals:
            all_signals.append(sig)
            all_labels.append(label_map[patient_name])

    if all_signals:
        np.savez_compressed(
            os.path.join(output_dir, 'clinical_ecg_digitized.npz'),
            signals=np.array(all_signals),
            labels=np.array(all_labels),
            label_map=label_map,
        )
        print(f"💾 Digitized signals saved: {output_dir}/clinical_ecg_digitized.npz")


def _save_clinical_plots(patient_signals, generated_signals, output_dir):
    """Save visual comparison plots."""
    fig, axes = plt.subplots(min(6, len(patient_signals)), 2,
                             figsize=(16, 3 * min(6, len(patient_signals))))

    if len(patient_signals) == 1:
        axes = axes[np.newaxis, :]

    gen_idx = 0
    for i, (patient_name, signals) in enumerate(patient_signals.items()):
        if i >= 6:
            break

        # Plot real
        ax_real = axes[i, 0]
        ax_real.plot(signals[0], 'b-', linewidth=0.5)
        ax_real.set_title(f'{patient_name} — Real ECG', fontsize=10)
        ax_real.set_ylabel('Amplitude')

        # Plot generated (if available)
        ax_gen = axes[i, 1]
        if gen_idx < len(generated_signals):
            ax_gen.plot(generated_signals[gen_idx], 'r-', linewidth=0.5)
            ax_gen.set_title(f'{patient_name} — Generated ECG', fontsize=10)
            gen_idx += 1
        else:
            ax_gen.set_title('No generation available')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clinical_comparison.png'), dpi=150)
    plt.close()
    print(f"📊 Comparison plot saved: {output_dir}/clinical_comparison.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/dit_ecg_ema_final.pt')
    parser.add_argument('--fe_path', type=str, default='checkpoints/feature_extractor_contrastive.pt')
    parser.add_argument('--dataset_dir', type=str, default='Dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/clinical_validation')
    args = parser.parse_args()

    run_clinical_validation(args.model_path, args.fe_path, args.dataset_dir, args.output_dir)
