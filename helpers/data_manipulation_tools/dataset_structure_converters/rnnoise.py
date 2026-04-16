from pathlib import Path
import subprocess
import soundfile
from dataset_structure_converters.modules import dataset_utilities


# ------------------------------------------
# RNNoise
# ------------------------------------------
def process(datasets_dir, output_dir, sample_rate, chunk_samples, split_duration, hop_samples):
    print("Process RNNoise")

    max_time_of_dataset_sec = 14400

    input_dir = Path(datasets_dir, "RNNoise")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir, "RNNoise")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_dataset_time = 0

    for wav_file in input_dir.glob("*.wav"):

        if total_dataset_time >= max_time_of_dataset_sec:
            print("Dataset time limit reached. Stopping RNNoise.")
            break

        audio, sr = soundfile.read(wav_file)

        # Resample if needed
        if sr != sample_rate:
            temp_wav = output_dir / f"temp.wav"
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
            output_dir,
            dataset_utilities.clean_file_name(wav_file),
            total_dataset_time,
            total_dataset_time,
            max_time_of_dataset_sec,
            max_time_of_dataset_sec
        )

        total_dataset_time += add_ds

    print("Done RNNoise!")
    print(f"Total dataset time: {total_dataset_time:.2f} sec")