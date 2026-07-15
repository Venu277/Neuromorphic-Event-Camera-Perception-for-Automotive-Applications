import os
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\DELL\Downloads\ncars_extracted\Prophesee_Dataset_n_cars"
SENSOR_W, SENSOR_H = 120, 100


def load_atis_dat(filepath):
    with open(filepath, 'rb') as f:
        # Skip ASCII header lines (start with '%')
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'%'):
                f.seek(pos)
                break

        # Skip the 2-byte event-type/event-size header that follows
        header_bytes = f.read(2)
        if len(header_bytes) < 2:
            f.seek(pos)  # no extra header present, rewind

        raw = np.fromfile(f, dtype=np.uint32)

    # Ensure even length before reshaping into [timestamp, data] pairs
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    raw = raw.reshape(-1, 2)

    timestamps = raw[:, 0]
    data = raw[:, 1]

    x = data & 0x3FFF
    y = (data >> 14) & 0x3FFF
    polarity = (data >> 28) & 0x1

    events = np.zeros(len(timestamps),
                       dtype=[('x', 'u2'), ('y', 'u2'), ('t', 'u4'), ('p', 'i1')])
    events['x'] = x
    events['y'] = y
    events['t'] = timestamps
    events['p'] = polarity
    return events


def events_to_frames(events, n_time_bins=10, width=SENSOR_W, height=SENSOR_H):
    frames = np.zeros((n_time_bins, 2, height, width), dtype=np.float32)
    if len(events) == 0:
        return frames

    t_min, t_max = events['t'].min(), events['t'].max()
    bin_edges = np.linspace(t_min, t_max + 1, n_time_bins + 1)
    bin_idx = np.clip(np.digitize(events['t'], bin_edges) - 1, 0, n_time_bins - 1)

    valid = (events['x'] < width) & (events['y'] < height)
    print(f"  -> {valid.sum()} / {len(events)} events within bounds "
          f"(x range: {events['x'].min()}-{events['x'].max()}, "
          f"y range: {events['y'].min()}-{events['y'].max()})")

    for i in np.where(valid)[0]:
        b = bin_idx[i]
        frames[b, events['p'][i], events['y'][i], events['x'][i]] += 1

    return frames


def find_split_dir(base, split_name):
    for c in [f"n-cars-{split_name}", f"n-cars_{split_name}", f"n-cars{split_name}"]:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"No '{split_name}' folder found in {base}: {os.listdir(base)}")


# ---- Run ----
train_dir = find_split_dir(DATA_ROOT, "train")
cars_dir = os.path.join(train_dir, "cars")

sample_files = sorted(os.listdir(cars_dir))
print("Number of car samples found:", len(sample_files))

sample_path = os.path.join(cars_dir, sample_files[0])
events = load_atis_dat(sample_path)
print("Loaded", len(events), "events from", sample_files[0])
print("Raw x range:", events['x'].min(), "-", events['x'].max())
print("Raw y range:", events['y'].min(), "-", events['y'].max())

frames = events_to_frames(events, n_time_bins=10)
print("Frames shape:", frames.shape)
print("Max value in frames (should be > 0):", frames.max())

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    frame_img = frames[i][0] + frames[i][1]
    ax.imshow(frame_img, cmap='gray')
    ax.set_title(f"t_bin {i}")
    ax.axis('off')

plt.suptitle("N-Cars sample — label: CAR")
plt.tight_layout()
plt.savefig("ncars_sample_visualization.png")
plt.show()
print("Done.")