import os
import shutil
import numpy as np
import librosa
import soundfile
import csv
import math
from pathlib import Path

# ------------------------------------------
# Spectrogram parameters
# ------------------------------------------
SAMPLE_RATE = 48000
WINDOW_SIZE = int(0.025 * SAMPLE_RATE)
HOP_SIZE    = int(0.010 * SAMPLE_RATE)

TARGET_BINS   = 128
TARGET_FRAMES = 200
MAX_FREQ      = 8192

# ------------------------------------------
# Data configuration
# ------------------------------------------

# Partial splitting
AMOUNT_OF_FILES_TO_USE = 1.0

# Speaker splitting
IS_ENABLE_SPEAKER_SPLITTING = True
MIN_FILES_PER_SPEAKER = 75

# Partition splitting
SPLIT_DATASET_RATIOS = {
    "train": 0.80,
    "dev":   0.10,
    "test":  0.10,
}


# ------------------------------------------
# Helpers
# ------------------------------------------
def normalize_csv_row(row):
    for key in ["femininity", "masculinity", "atypicality"]:
        val = float(row[key])
        row[key] = f"{val:.1f}"
    return row


def split_speaker_into_parts(files, min_files):
    total_files = len(files)

    # Only split if strictly more than 2x
    if total_files <= 2 * min_files:
        return [files]

    num_parts = total_files // min_files
    chunk_size = math.ceil(total_files / num_parts)

    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = start + chunk_size
        chunk = files[start:end]

        if chunk:
            chunks.append(chunk)

    return chunks


def split_files(files, ratios):
    n = len(files)
    ratio_items = list(ratios.items())
    num_splits = len(ratio_items)

    if n < num_splits:
        return {}

    reserved = {name: 1 for name, _ in ratio_items}
    remaining = n - num_splits

    extras = {}
    cursor_extra = 0

    for i, (name, ratio) in enumerate(ratio_items):
        if i == num_splits - 1:
            extras[name] = remaining - cursor_extra
        else:
            extras[name] = math.floor(remaining * ratio)
            cursor_extra += extras[name]

    splits = {}
    cursor = 0

    for name, _ in ratio_items:
        count = reserved[name] + extras[name]
        splits[name] = files[cursor: cursor + count]
        cursor += count

    return splits


def make_spectrogram(path: str) -> np.ndarray:
    audio, sr = soundfile.read(path)

    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} has SR {sr}, expected {SAMPLE_RATE}")

    stft = librosa.stft(
        audio,
        n_fft=MAX_FREQ * 2,
        hop_length=HOP_SIZE,
        win_length=WINDOW_SIZE,
        window="hann",
        center=True
    )

    mag = np.abs(stft)

    frequencies = np.linspace(0, sr / 2, mag.shape[0])
    max_bin = np.searchsorted(frequencies, MAX_FREQ)
    mag = mag[:max_bin]

    mag_resampled = librosa.resample(
        mag,
        orig_sr=mag.shape[0],
        target_sr=TARGET_BINS,
        axis=0
    )

    mag_resampled = mag_resampled.astype(np.float32).T

    if len(mag_resampled) > TARGET_FRAMES + 1:
        raise ValueError(f"Too long: {path}")

    if len(mag_resampled) < TARGET_FRAMES:
        raise ValueError(f"Too short: {path}")

    if len(mag_resampled) == TARGET_FRAMES + 1:
        mag_resampled = mag_resampled[:-1, :]

    return mag_resampled


def check_data(input_dataset: str, csv_path: str):
    disk_speakers = set(
        d for d in os.listdir(input_dataset)
        if os.path.isdir(os.path.join(input_dataset, d))
    )

    csv_speakers = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_speakers.add(row["speaker_name"])

    return sorted(disk_speakers - csv_speakers)


# ------------------------------------------
# Main
# ------------------------------------------
def main():
    input_dataset  = "data/2_structure_and_markup/dataset"
    output_dataset = "data/3_training_data/dataset"
    csv_path       = os.path.join(input_dataset, "speakers_info.csv")

    print("---- Data Check ----")
    missing = check_data(input_dataset, csv_path)

    if missing:
        print("Missing speakers in CSV:")
        for spk in missing:
            print(f"  - {spk}")
        return

    print("All speakers are marked up.")
    input("Press Enter to start processing...")

    # Create split folders
    for split in SPLIT_DATASET_RATIOS:
        os.makedirs(os.path.join(output_dataset, split), exist_ok=True)

    # Load CSV
    csv_rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows[row["speaker_name"]] = normalize_csv_row(row)

    # Prepare CSV writers
    csv_writers = {}
    for split in SPLIT_DATASET_RATIOS:
        path = os.path.join(output_dataset, split, "speakers_info.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker_name", "femininity", "masculinity", "atypicality"]
        )
        writer.writeheader()
        csv_writers[split] = (writer, f)

    # Get speakers
    speakers = sorted(
        d for d in os.listdir(input_dataset)
        if os.path.isdir(os.path.join(input_dataset, d))
    )

    print("\n---- Processing Speakers ----")

    for spk in speakers:
        print(f"\n[Speaker] {spk}")

        in_spk_path = os.path.join(input_dataset, spk)
        files = sorted(f for f in os.listdir(in_spk_path) if f.lower().endswith(".wav"))

        if not files:
            print("  -> No WAV files found, skipping.")
            continue

        print(f"  Original files: {len(files)}")

        # Apply AMOUNT
        capped = math.floor(len(files) * AMOUNT_OF_FILES_TO_USE)
        files = files[:capped]

        print(f"  After AMOUNT ({AMOUNT_OF_FILES_TO_USE}): {len(files)}")

        if len(files) < len(SPLIT_DATASET_RATIOS):
            print("  -> Not enough files for split, skipping.")
            continue

        # Speaker splitting
        if IS_ENABLE_SPEAKER_SPLITTING:
            parts = split_speaker_into_parts(files, MIN_FILES_PER_SPEAKER)
        else:
            parts = [files]

        print(f"  Parts created: {len(parts)}")

        # Process parts
        for idx, part_files in enumerate(parts):
            part_name = spk if idx == 0 else f"{spk}_pt{idx+1}"

            print(f"\n  [Part] {part_name}")
            print(f"    Files in part: {len(part_files)}")

            file_splits = split_files(part_files, SPLIT_DATASET_RATIOS)

            for split_name, split_files_list in file_splits.items():
                print(f"    -> {split_name}: {len(split_files_list)} files")

                if not split_files_list:
                    continue

                out_spk_path = os.path.join(output_dataset, split_name, part_name)
                os.makedirs(out_spk_path, exist_ok=True)

                for file_name in split_files_list:
                    in_file = os.path.join(in_spk_path, file_name)
                    spec = make_spectrogram(in_file)

                    out_file = os.path.join(out_spk_path, file_name.replace(".wav", ".npy"))
                    np.save(out_file, spec)

                row = dict(csv_rows[spk])
                row["speaker_name"] = part_name
                csv_writers[split_name][0].writerow(row)

    # Close CSV files AFTER processing everything
    for _, f in csv_writers.values():
        f.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
