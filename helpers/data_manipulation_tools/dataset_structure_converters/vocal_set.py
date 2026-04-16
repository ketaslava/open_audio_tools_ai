from pathlib import Path
import subprocess
import soundfile
from dataset_structure_converters.modules import dataset_utilities


# ------------------------------------------
# VocalSet
# ------------------------------------------
def process(datasets_dir, output_dir, sample_rate, chunk_samples, split_duration, hop_samples):
    print("Process VocalSet")

    max_time_of_dataset_sec = 43200
    max_time_per_speaker_sec = 999999999999 # Use all dataset

    input_dir = Path(datasets_dir, "VocalSet")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_dataset_time = 0

    for speaker_dir in input_dir.iterdir():
        if not speaker_dir.is_dir():
            continue

        if total_dataset_time >= max_time_of_dataset_sec:
            print("Dataset time limit reached. Stopping VocalSet.")
            break

        new_speaker_id = f"VocalSet_{speaker_dir.name}"
        output_speaker_dir = output_dir / new_speaker_id
        output_speaker_dir.mkdir(parents=True, exist_ok=True)

        print(f"{speaker_dir.name} >> {new_speaker_id}")

        speaker_time = 0

        for wav_file in speaker_dir.rglob("*.wav"):

            if speaker_time >= max_time_per_speaker_sec:
                print(f"Speaker {new_speaker_id} time limit reached.")
                break

            if total_dataset_time >= max_time_of_dataset_sec:
                print("Dataset time limit reached.")
                break

            audio, sr = soundfile.read(wav_file)

            # Resample if needed
            if sr != sample_rate:
                temp_wav = output_speaker_dir / f"temp.wav"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(wav_file),
                    "-ar", str(sample_rate),
                    str(temp_wav)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                audio, sr = soundfile.read(temp_wav)
                temp_wav.unlink()

            # Slice and save
            add_spk, add_ds = dataset_utilities.split_and_save_chunks(
                sample_rate,
                chunk_samples,
                split_duration,
                hop_samples,
                audio,
                output_speaker_dir,
                dataset_utilities.clean_file_name(wav_file),
                speaker_time,
                total_dataset_time,
                max_time_per_speaker_sec,
                max_time_of_dataset_sec
            )

            speaker_time += add_spk
            total_dataset_time += add_ds

        print(f"Speaker final time: {speaker_time:.2f} sec")

    print("Done VocalSet!")
    print(f"Total dataset time: {total_dataset_time:.2f} sec")