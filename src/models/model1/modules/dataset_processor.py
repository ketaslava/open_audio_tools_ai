import os
import csv
import numpy as np
import sys
import math
from matplotlib import pyplot as plt


# --------------------------------------------------------------------
# 1. COUNT SPEAKERS + VALIDATE DATASET
# --------------------------------------------------------------------
def check_data_in_dataset(dataset_path):

    # All directories (speakers) should be described in speakers_info.csv,
    # however speakers_info.csv can include speakers which are not present in directories

    # Load speakers from folders
    speaker_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    print(f"Speakers in folders: {len(speaker_dirs)}")

    # Load speakers from CSV
    csv_path = os.path.join(dataset_path, "speakers_info.csv")
    if not os.path.exists(csv_path):
        print("ERROR: speakers_info.csv is not found!")
        sys.exit(1)

    speakers_csv = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            name, fem, mas, atyp = row
            speakers_csv[name] = [float(fem), float(mas), float(atyp)]

    print(f"Speakers in CSV: {len(speakers_csv)}")

    # Check consistency
    missing_in_csv = [s for s in speaker_dirs if s not in speakers_csv]
    if missing_in_csv:
        print("ERROR: These folders exist but NOT in CSV:")
        for m in missing_in_csv:
            print("  ", m)
        sys.exit(1)

    print("Dataset consistency OK.")


def debug_data_for(input_tensor, target_tensor, sample):

    # Print tensors
    print("== input_tensor ==")
    print(input_tensor)
    print("== target_tensor ==")
    print(target_tensor)

    # Print tensor info
    print(f"== Tensor info ==")
    print(f"Loaded {len(input_tensor)} samples.")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")

    # Show target data [sample]
    print(f"== Target [sample] ==")
    print(target_tensor[sample])

    # Show input data [sample]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        input_tensor[sample].T,
        aspect='auto',
        origin='lower',
        cmap='magma'
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frequency Bin")
    plt.show()

    # exit
    exit()


# --------------------------------------------------------------------
# LOAD TENSORS FROM DATASET
# --------------------------------------------------------------------
def get_tensor_for_model1_from_dataset(dataset_path: str, data_portion: float = 1.0):
    """
    dataset_path  — path to train/, dev/, or test/ folder
    data_portion  — fraction of each speaker's files to load (0.0–1.0)
    """
    csv_path = os.path.join(dataset_path, "speakers_info.csv")

    # Load speaker labels
    speaker_info = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_info[row["speaker_name"]] = [
                float(row["femininity"]),
                float(row["masculinity"]),
                float(row["atypicality"]),
            ]

    speakers = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )

    # ----------------------------------------------------------------
    # First pass — collect file paths and labels to know total count
    # ----------------------------------------------------------------
    entries = []  # list of (npy_path, label)

    for spk in speakers:
        if spk not in speaker_info:
            print(f"WARNING: Speaker {spk} not in CSV, skipping.")
            continue

        spk_dir = os.path.join(dataset_path, spk)
        npy_files = sorted(f for f in os.listdir(spk_dir) if f.endswith(".npy"))

        if not npy_files:
            print(f"WARNING: Speaker {spk} has no .npy files, skipping.")
            continue

        count = max(1, math.floor(len(npy_files) * data_portion))
        label = [float(v) for v in speaker_info[spk]]  # ensures 1 → 1.0, 0 → 0.0

        for f in npy_files[:count]:
            entries.append((os.path.join(spk_dir, f), label))

    if not entries:
        print("ERROR: No data found.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Second pass — preallocate and fill (avoids list growth overhead)
    # ----------------------------------------------------------------
    sample_shape = np.load(entries[0][0]).shape  # (frames, freq_bins)
    n = len(entries)

    input_tensor  = np.empty((n, *sample_shape), dtype=np.float32)
    target_tensor = np.empty((n, 3),             dtype=np.float32)

    for i, (path, label) in enumerate(entries):
        input_tensor[i]  = np.load(path)
        target_tensor[i] = label

    print(f"Loaded {n} samples — "
          f"input: {input_tensor.shape}, target: {target_tensor.shape}")

    return input_tensor, target_tensor