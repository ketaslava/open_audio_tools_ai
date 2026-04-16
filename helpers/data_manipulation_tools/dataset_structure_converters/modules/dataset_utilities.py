from pathlib import Path
import soundfile


# ------------------------------------------
# Split audio and save chunks with time constraints
# ------------------------------------------
def split_and_save_chunks(sample_rate, chunk_samples, split_duration, hop_samples, audio, output_dir,
                          current_audio_name, current_speaker_time, current_dataset_time,
                          max_speaker_sec, max_dataset_sec):

    i = 0
    pos = 0
    added_speaker = 0
    added_dataset = 0

    while pos + chunk_samples <= len(audio):

        # Check limits before saving
        if current_dataset_time + added_dataset >= max_dataset_sec:
            return added_speaker, added_dataset

        if current_speaker_time + added_speaker >= max_speaker_sec:
            return added_speaker, added_dataset

        chunk = audio[pos:pos + chunk_samples]

        chunk_name = f"{current_audio_name}_{i}.wav"
        out_file = output_dir / chunk_name

        soundfile.write(out_file, chunk, sample_rate)

        # 2-second chunk added
        added_speaker += split_duration
        added_dataset += split_duration

        pos += hop_samples
        i += 1

    return added_speaker, added_dataset


# ------------------------------------------
# Clean the file name
# ------------------------------------------
def clean_file_name(file_name: Path):
    return str(file_name.name).replace(" ", "").replace(".", "")


# ------------------------------------------
# Measure time of existing dataset
# ------------------------------------------
def measure_total_current_time(dataset_root: Path):
    total_sec = 0

    if not dataset_root.exists():
        return 0

    for speaker_dir in dataset_root.iterdir():
        if not speaker_dir.is_dir():
            continue

        for wav_file in speaker_dir.glob("*.wav"):
            audio, sr = soundfile.read(wav_file)
            total_sec += len(audio) / sr

    return total_sec