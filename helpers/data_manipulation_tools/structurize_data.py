from pathlib import Path
import soundfile
from dataset_structure_converters import demand
from dataset_structure_converters import libri_speech
from dataset_structure_converters import musan
from dataset_structure_converters import rnnoise
from dataset_structure_converters import vctk_corpus
from dataset_structure_converters import vocal_set


DATASETS_DIR = Path("data/1_sources/")
OUTPUT_DIR = Path("data/2_structure_and_markup/dataset")
SPLIT_DURATION = 2.0  # seconds
SAMPLE_RATE = 48000
CHUNK_SAMPLES = int(SPLIT_DURATION * SAMPLE_RATE)
# HOP_SAMPLES = CHUNK_SAMPLES // 2  # 50% overlap
HOP_SAMPLES = CHUNK_SAMPLES  # 0% overlap


from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile
from mutagen.oggopus import OggOpus


def get_audio_duration(file: Path) -> float:
    try:
        suffix = file.suffix.lower()
        if suffix == ".opus":
            return OggOpus(file).info.length
        else:
            info = soundfile.info(file)
            return info.duration
    except Exception as e:
        print(f"Could not read {file}: {e}")
        return 0.0


def measure_audio_time_in_dir(root: Path) -> float:
    if not root.exists():
        return 0.0

    files = [
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".opus"}
    ]

    total_sec = 0.0
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_audio_duration, f): f for f in files}
        for future in as_completed(futures):
            total_sec += future.result()

    return total_sec


def measure_the_time_of_datasets():
    combined_root = Path("data/1_sources")

    print("---- Time Measurement ----")

    total_all_sec = 0

    for dataset_dir in combined_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        sec = measure_audio_time_in_dir(dataset_dir)
        hr = sec / 3600
        total_all_sec += sec

        print(f"{dataset_dir.name}: {hr:.2f} hours ({sec:.2f} sec)")

    print("---------------------------------")
    print(f"TOTAL: {total_all_sec/3600:.2f} hours ({total_all_sec:.2f} sec)")
    print("---------------------------------")


# ------------------------------------------
# Run everything
# ------------------------------------------
def main():
    #measure_the_time_of_datasets()
    input("Press enter to start processing...")
    demand.process(DATASETS_DIR, OUTPUT_DIR, SAMPLE_RATE, CHUNK_SAMPLES, SPLIT_DURATION, HOP_SAMPLES)
    #libri_speech.process(DATASETS_DIR, OUTPUT_DIR, SAMPLE_RATE, CHUNK_SAMPLES, SPLIT_DURATION, HOP_SAMPLES)
    #musan.process(DATASETS_DIR, OUTPUT_DIR, SAMPLE_RATE, CHUNK_SAMPLES, SPLIT_DURATION, HOP_SAMPLES)
    #rnnoise.process(DATASETS_DIR, OUTPUT_DIR, SAMPLE_RATE, CHUNK_SAMPLES, SPLIT_DURATION, HOP_SAMPLES)
    #vctk_corpus.process(DATASETS_DIR, OUTPUT_DIR, SAMPLE_RATE, CHUNK_SAMPLES, SPLIT_DURATION, HOP_SAMPLES)
    #vocal_set.process(DATASETS_DIR, OUTPUT_DIR, SAMPLE_RATE, CHUNK_SAMPLES, SPLIT_DURATION, HOP_SAMPLES)


main()
