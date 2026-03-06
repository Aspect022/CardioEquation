"""
CardioEquation V2: Download All Required Datasets
====================================================
Downloads and prepares all datasets needed for training.

Run this ONCE on the GPU server before starting training.
It will download everything to the data/ folder.

Usage:
    python download_all_datasets.py              # Download everything
    python download_all_datasets.py --mitbih      # Only MIT-BIH
    python download_all_datasets.py --ptbxl       # Only PTB-XL
    python download_all_datasets.py --cpsc        # Only CPSC2018
    python download_all_datasets.py --process     # Only process (skip download)

Estimated Download Sizes:
    MIT-BIH:    ~100 MB  (48 records, 30 min each, 360 Hz, 2 leads)
    PTB-XL:     ~2.5 GB  (21,837 records, 10s each, 500 Hz, 12 leads)
    CPSC2018:   ~4.0 GB  (6,877 records, 10s each, 500 Hz, 12 leads)

Total: ~6.6 GB disk space needed
"""

import os
import sys
import argparse
import time
import numpy as np

# ─── Constants ──────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# MIT-BIH: All 48 records
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

MITBIH_DIR = os.path.join(DATA_DIR, 'mitbih_raw')
PTBXL_DIR = os.path.join(DATA_DIR, 'ptbxl')
CPSC_DIR = os.path.join(DATA_DIR, 'cpsc2018')


def download_mitbih():
    """
    Download the full MIT-BIH Arrhythmia Database (48 records).
    Source: PhysioNet (physionet.org/content/mitdb/1.0.0/)
    """
    import wfdb

    os.makedirs(MITBIH_DIR, exist_ok=True)
    print("=" * 60)
    print("📥 Downloading MIT-BIH Arrhythmia Database")
    print(f"   Target: {MITBIH_DIR}")
    print(f"   Records: {len(MITBIH_RECORDS)}")
    print("=" * 60)

    downloaded = 0
    skipped = 0
    failed = []

    for rec in MITBIH_RECORDS:
        hea_file = os.path.join(MITBIH_DIR, f'{rec}.hea')
        dat_file = os.path.join(MITBIH_DIR, f'{rec}.dat')

        if os.path.exists(hea_file) and os.path.exists(dat_file):
            skipped += 1
            continue

        try:
            print(f"   Downloading record {rec}...", end=' ', flush=True)
            wfdb.dl_database('mitdb', MITBIH_DIR, records=[rec])
            downloaded += 1
            print("✅")
        except Exception as e:
            failed.append(rec)
            print(f"❌ ({e})")

    print(f"\n   Result: {downloaded} downloaded, {skipped} already existed, {len(failed)} failed")
    if failed:
        print(f"   ⚠️  Failed records: {failed}")
        print(f"   Try again or download manually from: https://physionet.org/content/mitdb/1.0.0/")

    return len(failed) == 0


def download_ptbxl():
    """
    Download PTB-XL dataset (21,837 12-lead ECGs, 18,885 patients).
    Source: PhysioNet (physionet.org/content/ptb-xl/1.0.3/)

    This is the LARGEST labeled ECG dataset — essential for:
    - Contrastive pre-training (18K unique patients)
    - Conditional generation with diagnostic labels (71 SCP codes)
    """
    import wfdb

    os.makedirs(PTBXL_DIR, exist_ok=True)
    print("=" * 60)
    print("📥 Downloading PTB-XL Dataset")
    print(f"   Target: {PTBXL_DIR}")
    print(f"   Size: ~2.5 GB (21,837 records, 12-lead, 500Hz)")
    print("=" * 60)

    # Check if already downloaded
    if os.path.exists(os.path.join(PTBXL_DIR, 'ptbxl_database.csv')):
        print("   ✅ PTB-XL already downloaded, skipping.")
        return True

    try:
        print("   Downloading full database (this may take 10-30 minutes)...")
        wfdb.dl_database('ptb-xl', PTBXL_DIR)
        print("   ✅ PTB-XL download complete!")
        return True
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print(f"   Manual download: https://physionet.org/content/ptb-xl/1.0.3/")
        print(f"   Extract to: {PTBXL_DIR}")
        return False


def download_cpsc2018():
    """
    Download CPSC2018 dataset (6,877 12-lead ECGs, 9 arrhythmia classes).
    Source: http://2018.icbeb.org/Challenge.html

    Note: CPSC2018 is hosted differently from PhysioNet datasets.
    We'll try the PhysioNet mirror first.
    """
    os.makedirs(CPSC_DIR, exist_ok=True)
    print("=" * 60)
    print("📥 Downloading CPSC2018 Dataset")
    print(f"   Target: {CPSC_DIR}")
    print(f"   Size: ~4 GB (6,877 records, 12-lead)")
    print("=" * 60)

    # Check if already downloaded
    existing_mats = [f for f in os.listdir(CPSC_DIR) if f.endswith('.mat')] if os.path.exists(CPSC_DIR) else []
    if len(existing_mats) > 1000:
        print(f"   ✅ CPSC2018 already downloaded ({len(existing_mats)} files), skipping.")
        return True

    try:
        import wfdb
        print("   Attempting PhysioNet mirror download...")
        wfdb.dl_database('cpsc2018', CPSC_DIR)
        print("   ✅ CPSC2018 download complete!")
        return True
    except Exception as e:
        print(f"   ⚠️  PhysioNet mirror failed: {e}")
        print(f"   Manual download required:")
        print(f"   1. Go to: http://2018.icbeb.org/Challenge.html")
        print(f"   2. Download the training data")
        print(f"   3. Extract to: {CPSC_DIR}")
        return False


def process_mitbih_forecasting(context_sec=10, future_sec=10, stride_sec=10):
    """
    Process MIT-BIH records into (context, future) pairs for forecasting training.
    Regenerates data/mitbih_forecasting.npz from the full 48-record set.
    """
    import scipy.signal
    import wfdb

    output_file = os.path.join(DATA_DIR, 'mitbih_forecasting.npz')
    target_fs = 500

    print("=" * 60)
    print("⚙️  Processing MIT-BIH for Forecasting")
    print(f"   Context: {context_sec}s, Future: {future_sec}s, Stride: {stride_sec}s")
    print(f"   Output: {output_file}")
    print("=" * 60)

    X_context = []
    Y_future = []

    native_fs = 360  # MIT-BIH native sampling rate
    window_len = int((context_sec + future_sec) * native_fs)
    split_idx = int(context_sec * native_fs)
    stride = int(stride_sec * native_fs)
    target_len = int(context_sec * target_fs)  # 5000 samples

    for rec in MITBIH_RECORDS:
        rec_path = os.path.join(MITBIH_DIR, rec)
        hea_file = rec_path + '.hea'

        if not os.path.exists(hea_file):
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            signal = record.p_signal[:, 0]  # Lead I

            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            # Sliding window
            for i in range(0, len(signal) - window_len, stride):
                segment = signal[i:i + window_len]
                past = segment[:split_idx]
                future = segment[split_idx:]

                # Resample to 500Hz
                past_res = scipy.signal.resample(past, target_len)
                future_res = scipy.signal.resample(future, target_len)

                X_context.append(past_res.astype(np.float32))
                Y_future.append(future_res.astype(np.float32))

        except Exception as e:
            print(f"   ⚠️  Error processing {rec}: {e}")

    X_context = np.array(X_context)[..., np.newaxis]  # (N, T, 1)
    Y_future = np.array(Y_future)[..., np.newaxis]

    print(f"   ✅ Created: {X_context.shape[0]} samples")
    print(f"   Context: {X_context.shape}, Future: {Y_future.shape}")

    np.savez_compressed(output_file, context=X_context, future=Y_future)
    print(f"   💾 Saved: {output_file}")

    return True


def process_ptbxl():
    """
    Process PTB-XL into patient-indexed segments for contrastive learning
    and conditional generation training.

    Outputs:
        data/ptbxl_processed.npz — {signals, patient_ids, labels}
    """
    import pandas as pd
    import wfdb

    output_file = os.path.join(DATA_DIR, 'ptbxl_processed.npz')

    if os.path.exists(output_file):
        print("   ✅ PTB-XL already processed, skipping.")
        return True

    csv_path = os.path.join(PTBXL_DIR, 'ptbxl_database.csv')
    if not os.path.exists(csv_path):
        print("   ❌ PTB-XL not downloaded yet. Run download first.")
        return False

    print("=" * 60)
    print("⚙️  Processing PTB-XL")
    print(f"   Output: {output_file}")
    print("=" * 60)

    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"   Records: {len(df)}")
    print(f"   Patients: {df['patient_id'].nunique()}")

    signals = []
    patient_ids = []
    record_count = 0

    for idx, row in df.iterrows():
        try:
            # PTB-XL stores records in subfolders like records500/00000/00001_hr
            filename = row['filename_hr']  # 500Hz version
            rec_path = os.path.join(PTBXL_DIR, filename)

            if not os.path.exists(rec_path + '.hea'):
                # Try without the _hr suffix or different path
                filename = row.get('filename_lr', filename)
                rec_path = os.path.join(PTBXL_DIR, filename)

            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal  # (T, 12)

            # Use Lead I for single-lead training (column 0)
            lead_I = sig[:, 0]

            # Normalize
            lead_I = (lead_I - lead_I.mean()) / (lead_I.std() + 1e-8)

            # Resample to 2500 samples (5s at 500Hz) if needed
            if len(lead_I) != 5000:
                from scipy.signal import resample
                lead_I = resample(lead_I, 5000)

            # Take first 5s (2500 samples) for identity
            signals.append(lead_I[:2500].astype(np.float32))
            patient_ids.append(int(row['patient_id']))
            record_count += 1

            if record_count % 5000 == 0:
                print(f"   Processed {record_count}/{len(df)} records...")

        except Exception as e:
            continue

    signals = np.array(signals)[:, np.newaxis, :]  # (N, 1, 2500)
    patient_ids = np.array(patient_ids)

    print(f"   ✅ Processed {len(signals)} records from {len(np.unique(patient_ids))} patients")
    np.savez_compressed(output_file, signals=signals, patient_ids=patient_ids)
    print(f"   💾 Saved: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Download CardioEquation datasets')
    parser.add_argument('--mitbih', action='store_true', help='Download MIT-BIH only')
    parser.add_argument('--ptbxl', action='store_true', help='Download PTB-XL only')
    parser.add_argument('--cpsc', action='store_true', help='Download CPSC2018 only')
    parser.add_argument('--process', action='store_true', help='Only process (skip download)')
    parser.add_argument('--all', action='store_true', help='Download everything (default)')
    args = parser.parse_args()

    # If no specific flag, download all
    download_all = not (args.mitbih or args.ptbxl or args.cpsc or args.process)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n🫀 CardioEquation V2: Dataset Downloader")
    print("=" * 60)

    results = {}
    start = time.time()

    # Downloads
    if not args.process:
        if download_all or args.mitbih:
            results['mitbih_download'] = download_mitbih()

        if download_all or args.ptbxl:
            results['ptbxl_download'] = download_ptbxl()

        if download_all or args.cpsc:
            results['cpsc_download'] = download_cpsc2018()

    # Processing
    if download_all or args.mitbih or args.process:
        print()
        results['mitbih_process'] = process_mitbih_forecasting()

    if download_all or args.ptbxl or args.process:
        print()
        results['ptbxl_process'] = process_ptbxl()

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"📊 Summary (took {elapsed:.0f}s):")
    for key, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {key}")

    if all(results.values()):
        print("\n✅ All datasets ready! You can start training.")
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
        print("   You can retry individual datasets with --mitbih, --ptbxl, --cpsc")

    print(f"\n   Next step: python verify_datasets.py")


if __name__ == '__main__':
    main()
