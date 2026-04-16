from matplotlib import pyplot as plt
from src.models.model1 import model
from src.models.model1 import config
import numpy as np
import sounddevice as sd
import threading
from src.models.model1.modules import utilities
import time
from collections import deque
import librosa


def predict():
    """
    Runs a real-time analyzer:
      - records audio continuously
      - every 1s takes last 2s (50% overlap) and predicts
      - updates a live matplotlib plot of the three outputs over time
    """

    # Configs
    model_weights_path = "misc/models/model1/model.pth"
    history_seconds = 30
    record_device = None

    # rolling audio buffer (samples)
    buffer_len = config.SAMPLE_RATE * 2  # 2 seconds buffer (samples)
    audio_buffer = deque(maxlen=buffer_len)

    # rolling history for plotting
    max_points = int(history_seconds)  # one update per second
    history = deque(maxlen=max_points)  # will hold normalized [3] vectors
    times = deque(maxlen=max_points)

    # threading lock for buffer access
    buf_lock = threading.Lock()
    stop_event = threading.Event()

    # prepare model loader wrapper that calls your predict
    def predict_from_spec_array(spec_array):
        """
        spec_array: shape (TARGET_FRAMES, TARGET_BINS)
        predict expects (batch, frames, freq) -> we will pass a batch of 1
        """
        # model1.predict signature we've used: predict(tensor_input, weights_override=None)
        batch = np.expand_dims(spec_array, axis=0)  # (1, frames, freq)
        preds = model.predict(batch, model_weights_path)
        # print(preds)
        # preds shape (1,3)
        return preds[0]

    # audio callback (runs in sounddevice thread)
    def audio_callback(indata, frames, time_info, status):
        # indata shape: (frames, channels)
        if status:
            # you can log status if you want
            pass
        # take first channel
        samples = indata[:, 0]
        with buf_lock:
            audio_buffer.extend(samples.tolist())

    # background worker: every 1 second, take last 2 seconds and predict
    def worker_loop():
        last_pred_time = 0.0
        while not stop_event.is_set():
            now = time.time()
            # once per second
            if now - last_pred_time >= 1.0:
                with buf_lock:
                    if len(audio_buffer) < buffer_len:
                        # not enough data yet
                        pass
                    else:
                        # copy buffer into numpy array (most recent at the right)
                        arr = np.array(audio_buffer, dtype=np.float32)
                        # arr length == buffer_len
                        # compute spectrogram
                        spec = utilities.make_spectrogram_from_audio(arr, sr=config.SAMPLE_RATE)  # (200, 128)
                        # predict
                        try:
                            pred = predict_from_spec_array(spec)  # shape (3,)
                        except Exception as e:
                            print("Prediction error:", e)
                            pred = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                        # normalize prediction (per sample)
                        norm = utilities.normalize_row(pred)
                        # store history
                        history.append(norm)
                        times.append(time.time())
                last_pred_time = now
            time.sleep(0.05)

    # prepare sounddevice stream
    sd_kwargs = dict(samplerate=config.SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    if record_device is not None:
        sd_kwargs['device'] = record_device

    # Start audio stream and worker thread
    stream = sd.InputStream(**sd_kwargs)
    stream.start()
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    # Prepare live plot (main thread must run plotting)
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(-max_points + 1, 1, 1)  # relative seconds ago: -N+1 ... 0
    line_fem, = ax.plot(x, np.zeros_like(x), label='fem', color='C0')
    line_mas, = ax.plot(x, np.zeros_like(x), label='mas', color='C1')
    line_atyp, = ax.plot(x, np.zeros_like(x), label='atyp', color='C2')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-history_seconds + 1, 0)
    ax.set_xlabel("seconds ago")
    ax.set_ylabel("normalized value")
    ax.legend(loc='upper right')
    ax.grid(True)

    print("Real-time analyzer started. Press Ctrl+C in console to stop.")

    try:
        while True:
            # build arrays for plot
            with buf_lock:
                h = np.array(history) if len(history) > 0 else np.zeros((0, 3), dtype=np.float32)
            # pad left if we have fewer points than max
            if h.shape[0] < max_points:
                pad = np.zeros((max_points - h.shape[0], 3), dtype=np.float32)
                h_full = np.vstack([pad, h]) if h.shape[0] > 0 else pad
            else:
                h_full = h[-max_points:, :]

            line_fem.set_ydata(h_full[:, 0])
            line_mas.set_ydata(h_full[:, 1])
            line_atyp.set_ydata(h_full[:, 2])

            # update x if history window changed
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Stopping real-time analyzer...")
    finally:
        stop_event.set()
        stream.stop()
        stream.close()
        worker.join(timeout=1.0)
        plt.ioff()
        plt.close(fig)
