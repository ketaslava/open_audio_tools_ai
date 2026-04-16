import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER = "data/3_training_data/dataset/train/DEMAND_DKITCHEN"  # change this

files = [f for f in os.listdir(FOLDER) if f.endswith(".npy")]
files.sort()
index = 0

fig, ax = plt.subplots(figsize=(10, 4))

def show_file():
    spec = np.load(os.path.join(FOLDER, files[index]))

    ax.clear()
    ax.imshow(
        spec.T,              # <--- transpose for proper orientation
        aspect='auto',
        origin='lower',
        cmap='magma'
    )
    ax.set_title(files[index])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frequency Bin")
    fig.canvas.draw_idle()

def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(files)
    elif event.key == 'left':
        index = (index - 1) % len(files)
    show_file()

fig.canvas.mpl_connect('key_press_event', on_key)
show_file()
plt.show()
