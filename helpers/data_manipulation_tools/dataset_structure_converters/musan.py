from pathlib import Path
import subprocess
import soundfile
from dataset_structure_converters.modules import dataset_utilities


# ------------------------------------------
# MUSAN
# ------------------------------------------
def process(datasets_dir, output_dir, sample_rate, chunk_samples, split_duration, hop_samples):
    print("Process MUSAN")

    max_time_of_dataset_sec = 999999999999
    max_time_per_source_sec = 4800

    input_dir = Path(datasets_dir, "MUSAN")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_dataset_time = 0

    for source_dir in input_dir.iterdir():
        if not source_dir.is_dir():
            continue

        if total_dataset_time >= max_time_of_dataset_sec:
            print("Dataset time limit reached. Stopping MUSAN.")
            break

        new_source_id = f"MUSAN_{source_dir.name}"
        output_source_dir = output_dir / new_source_id
        output_source_dir.mkdir(parents=True, exist_ok=True)

        print(f"{source_dir.name} >> {new_source_id}")

        source_time = 0

        for wav_file in source_dir.glob("*.wav"):

            if source_time >= max_time_per_source_sec:
                print(f"Source {new_source_id} time limit reached.")
                break

            if total_dataset_time >= max_time_of_dataset_sec:
                print("Dataset time limit reached.")
                break

            audio, sr = soundfile.read(wav_file)

            # Resample if needed
            if sr != sample_rate:
                temp_wav = output_source_dir / f"temp.wav"
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
                output_source_dir,
                dataset_utilities.clean_file_name(wav_file),
                source_time,
                total_dataset_time,
                max_time_per_source_sec,
                max_time_of_dataset_sec
            )

            source_time += add_spk
            total_dataset_time += add_ds

        print(f"Source final time: {source_time:.2f} sec")

    print("Done MUSAN!")
    print(f"Total dataset time: {total_dataset_time:.2f} sec")