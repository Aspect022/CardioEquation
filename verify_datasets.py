"""
CardioEquation V2: Dataset Verification Script
=================================================
Run this AFTER download_all_datasets.py to verify everything
is properly downloaded and ready for training.

This is your ONE check before starting GPU training.

Usage:
    python verify_datasets.py

Exit code 0 = all good, you can start training
Exit code 1 = something is missing
"""

import os
import sys
import json
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')

# MIT-BIH: All 48 records expected
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]


class DatasetChecker:
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []

    def check(self, name, condition, detail=""):
        if condition:
            self.results.append(('✅', name, detail))
        else:
            self.results.append(('❌', name, detail))
            self.errors.append(name)

    def warn(self, name, detail=""):
        self.results.append(('⚠️ ', name, detail))
        self.warnings.append(name)

    def print_results(self):
        print("\n" + "=" * 70)
        print("  CardioEquation V2: Dataset Verification Report")
        print("=" * 70)

        for icon, name, detail in self.results:
            if detail:
                print(f"  {icon} {name}: {detail}")
            else:
                print(f"  {icon} {name}")

        print("\n" + "-" * 70)
        if not self.errors:
            print("  ✅ ALL CHECKS PASSED — Ready for training!")
        else:
            print(f"  ❌ {len(self.errors)} ERRORS found:")
            for e in self.errors:
                print(f"     • {e}")
            print(f"\n  Fix with: python download_all_datasets.py")

        if self.warnings:
            print(f"\n  ⚠️  {len(self.warnings)} warnings (non-blocking):")
            for w in self.warnings:
                print(f"     • {w}")

        print("=" * 70)
        return len(self.errors) == 0


def check_mitbih(checker):
    """Check MIT-BIH Arrhythmia Database completeness."""
    mitbih_dir = os.path.join(DATA_DIR, 'mitbih_raw')

    print("\n📂 1. MIT-BIH Arrhythmia Database")
    print(f"   Location: {mitbih_dir}")

    if not os.path.exists(mitbih_dir):
        checker.check("MIT-BIH directory exists", False, f"{mitbih_dir} not found")
        return

    checker.check("MIT-BIH directory exists", True)

    # Check each record has .hea, .dat, .atr
    present = []
    missing = []
    for rec in MITBIH_RECORDS:
        hea = os.path.exists(os.path.join(mitbih_dir, f'{rec}.hea'))
        dat = os.path.exists(os.path.join(mitbih_dir, f'{rec}.dat'))
        if hea and dat:
            present.append(rec)
        else:
            missing.append(rec)

    checker.check(
        f"MIT-BIH records ({len(present)}/48)",
        len(present) == 48,
        f"Present: {len(present)}, Missing: {len(missing)}"
    )

    if missing and len(missing) <= 10:
        print(f"   Missing records: {', '.join(missing)}")
    elif missing:
        print(f"   Missing {len(missing)} records (first 10): {', '.join(missing[:10])}...")

    # Check per-record file sizes (dat should be ~1.9MB each)
    if present:
        sizes = [os.path.getsize(os.path.join(mitbih_dir, f'{r}.dat')) for r in present[:5]]
        avg_size = sum(sizes) / len(sizes) / 1e6
        checker.check(
            "MIT-BIH file sizes look correct",
            avg_size > 1.0,
            f"Avg .dat size: {avg_size:.1f} MB"
        )


def check_mitbih_forecasting(checker):
    """Check processed forecasting dataset."""
    npz_path = os.path.join(DATA_DIR, 'mitbih_forecasting.npz')

    print("\n📂 2. MIT-BIH Forecasting Data (Processed)")
    print(f"   Location: {npz_path}")

    if not os.path.exists(npz_path):
        checker.check("mitbih_forecasting.npz exists", False)
        return

    size_mb = os.path.getsize(npz_path) / 1e6
    checker.check("mitbih_forecasting.npz exists", True, f"{size_mb:.1f} MB")

    try:
        data = np.load(npz_path)
        context = data['context']
        future = data['future']

        checker.check(
            "Forecasting data has 'context' and 'future' arrays", True,
            f"context: {context.shape}, future: {future.shape}"
        )

        checker.check(
            "Sufficient training samples (>100)",
            context.shape[0] > 100,
            f"{context.shape[0]} samples"
        )

        # Signal length check
        expected_dim = 2500  # Could be 2500 or 5000 depending on config
        sig_len = context.shape[1]
        checker.check(
            f"Signal length is {sig_len}",
            sig_len in [2500, 5000],
            f"Expected 2500 or 5000"
        )

        # NaN check
        has_nan = np.isnan(context).any() or np.isnan(future).any()
        checker.check("No NaN values in data", not has_nan)

        # Inf check
        has_inf = np.isinf(context).any() or np.isinf(future).any()
        checker.check("No Inf values in data", not has_inf)

        # Value range check
        ctx_std = context.std()
        checker.check(
            "Data is normalized",
            0.1 < ctx_std < 5.0,
            f"std={ctx_std:.3f}"
        )

    except Exception as e:
        checker.check("Can load forecasting data", False, str(e))


def check_ptbxl(checker):
    """Check PTB-XL dataset."""
    ptbxl_dir = os.path.join(DATA_DIR, 'ptbxl')

    print("\n📂 3. PTB-XL Dataset (21,837 records)")
    print(f"   Location: {ptbxl_dir}")

    if not os.path.exists(ptbxl_dir):
        checker.warn("PTB-XL not downloaded", "Run: python download_all_datasets.py --ptbxl")
        return

    # Check for main CSV
    csv_path = os.path.join(ptbxl_dir, 'ptbxl_database.csv')
    checker.check("PTB-XL metadata CSV exists", os.path.exists(csv_path))

    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            checker.check(
                "PTB-XL record count",
                len(df) > 20000,
                f"{len(df)} records, {df['patient_id'].nunique()} patients"
            )
        except ImportError:
            checker.warn("pandas not installed — cannot verify PTB-XL CSV")

    # Check for processed file
    processed_path = os.path.join(DATA_DIR, 'ptbxl_processed.npz')
    if os.path.exists(processed_path):
        data = np.load(processed_path)
        checker.check(
            "PTB-XL processed data",  True,
            f"signals: {data['signals'].shape}, patients: {len(np.unique(data['patient_ids']))}"
        )
    else:
        checker.warn("PTB-XL not processed yet",
                      "Run: python download_all_datasets.py --process")


def check_cpsc2018(checker):
    """Check CPSC2018 dataset."""
    cpsc_dir = os.path.join(DATA_DIR, 'cpsc2018')

    print("\n📂 4. CPSC2018 Dataset (6,877 records)")
    print(f"   Location: {cpsc_dir}")

    if not os.path.exists(cpsc_dir):
        checker.warn("CPSC2018 not downloaded (optional)", "Run: python download_all_datasets.py --cpsc")
        return

    files = os.listdir(cpsc_dir)
    data_files = [f for f in files if f.endswith(('.mat', '.hea', '.dat'))]
    checker.check(
        f"CPSC2018 files present",
        len(data_files) > 1000,
        f"{len(data_files)} files found"
    )


def check_clinical_data(checker):
    """Check hospital ECG PDFs for clinical validation."""
    print("\n📂 5. Clinical Hospital ECG Data (Validation)")
    print(f"   Location: {DATASET_DIR}")

    if not os.path.exists(DATASET_DIR):
        checker.warn("Dataset/ folder not found (needed for clinical validation)")
        return

    # Count PDFs
    pdf_count = 0
    patients = set()
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.pdf'):
                    pdf_count += 1
                    name = f.split('_')[0]
                    if name:
                        patients.add(name)

    checker.check(
        f"Hospital ECG PDFs",
        pdf_count > 0,
        f"{pdf_count} PDFs from {len(patients)} patients: {', '.join(sorted(patients)[:5])}{'...' if len(patients) > 5 else ''}"
    )


def check_model_code(checker):
    """Check that model source code is present and importable."""
    print("\n📂 6. Model Code Integrity")

    required_files = [
        'src/models/dit_ecg.py',
        'src/models/feature_extractor_pt.py',
        'src/training/train_dit.py',
        'src/training/train_contrastive.py',
        'src/training/losses_v2.py',
        'src/training/noise_scheduler.py',
        'src/training/ema.py',
        'src/inference/pipeline_v2.py',
        'src/evaluation/eval_metrics.py',
        'src/evaluation/clinical_validation.py',
    ]

    base = os.path.dirname(os.path.abspath(__file__))
    for f in required_files:
        path = os.path.join(base, f)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        checker.check(
            f"{f}",
            exists and size > 100,
            f"{size} bytes" if exists else "MISSING"
        )


def check_dependencies(checker):
    """Check that required Python packages are available."""
    print("\n📂 7. Python Dependencies")

    packages = {
        'torch': 'PyTorch (core training)',
        'numpy': 'NumPy (arrays)',
        'scipy': 'SciPy (resampling)',
        'wfdb': 'WFDB (PhysioNet data loading)',
        'tqdm': 'tqdm (progress bars)',
    }

    optional_packages = {
        'neurokit2': 'NeuroKit2 (ECG morphology analysis)',
        'pandas': 'pandas (PTB-XL metadata)',
        'matplotlib': 'matplotlib (plotting)',
    }

    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            checker.check(f"{desc}", True, "installed")
        except ImportError:
            checker.check(f"{desc}", False, f"pip install {pkg}")

    for pkg, desc in optional_packages.items():
        try:
            __import__(pkg)
            checker.check(f"{desc}", True, "installed")
        except ImportError:
            checker.warn(f"{desc} not installed", f"pip install {pkg}")


def check_gpu(checker):
    """Check GPU availability."""
    print("\n📂 8. GPU Availability")

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            bf16 = torch.cuda.is_bf16_supported()
            checker.check("GPU available", True, f"{name} ({mem:.1f} GB, BF16={bf16})")
        else:
            checker.warn("No GPU detected", "Training will be very slow without GPU")
    except ImportError:
        checker.check("PyTorch installed", False, "pip install torch")


def main():
    checker = DatasetChecker()

    print("🫀 CardioEquation V2: Pre-Training Verification")
    print("Run this before starting training to ensure everything is ready.\n")

    check_mitbih(checker)
    check_mitbih_forecasting(checker)
    check_ptbxl(checker)
    check_cpsc2018(checker)
    check_clinical_data(checker)
    check_model_code(checker)
    check_dependencies(checker)
    check_gpu(checker)

    success = checker.print_results()

    # Save report
    report_path = os.path.join(DATA_DIR, 'verification_report.json')
    report = {
        'status': 'PASS' if success else 'FAIL',
        'errors': checker.errors,
        'warnings': checker.warnings,
        'total_checks': len(checker.results),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
