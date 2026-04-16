import numpy as np
import librosa
from src.models.model1 import config


def normalize_row(row):
    mn = row.min()
    mx = row.max()
    if mx - mn == 0:
        return np.zeros_like(row)
    return (row - mn) / (mx - mn)


# ---------- constants (reuse your existing constants if available) ----------

WINDOW_SIZE = int(0.025 * config.SAMPLE_RATE)  # 25 ms
# For STFT hop, we will stick to 10ms as used in training - but we'll pad/trim frames later
HOP_SIZE = int(0.010 * config.SAMPLE_RATE)  # 10 ms
TARGET_BINS = 128
TARGET_FRAMES = 200
MAX_FREQ = 8192


# ---------- small utility: build spectrogram from raw 1D audio ----------
def make_spectrogram_from_audio(audio, sr=config.SAMPLE_RATE):
    """
    audio: 1D numpy array, length ~ 2*sr
    returns: spectrogram as float32 with shape (TARGET_FRAMES, TARGET_BINS)
    """
    # ensure correct rate and type
    if sr != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        sr = config.SAMPLE_RATE

    # STFT - use same parameters as training (n_fft large for frequency resolution)
    n_fft = MAX_FREQ * 2  # 16384
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=HOP_SIZE,
        win_length=WINDOW_SIZE,
        window="hann",
        center=True
    )
    mag = np.abs(stft)  # shape (n_bins, n_frames)

    # Frequency trimming: keep up to MAX_FREQ
    freqs = np.linspace(0, sr / 2, mag.shape[0])
    max_bin = np.searchsorted(freqs, MAX_FREQ)
    mag = mag[:max_bin, :]  # now shape (bins_up_to_MAX_FREQ, n_frames)

    # Resample frequency axis to TARGET_BINS
    mag_resampled = librosa.resample(mag, orig_sr=mag.shape[0], target_sr=TARGET_BINS, axis=0)

    # transpose to (frames, freq_bins)
    spec = mag_resampled.T.astype(np.float32)  # shape (n_frames, TARGET_BINS)

    # Ensure time axis is exactly TARGET_FRAMES: pad or trim
    n_frames = spec.shape[0]
    if n_frames > TARGET_FRAMES:
        spec = spec[:TARGET_FRAMES, :]
    elif n_frames < TARGET_FRAMES:
        pad_amount = TARGET_FRAMES - n_frames
        pad = np.zeros((pad_amount, TARGET_BINS), dtype=np.float32)
        spec = np.vstack([spec, pad])

    return spec  # shape (TARGET_FRAMES, TARGET_BINS)