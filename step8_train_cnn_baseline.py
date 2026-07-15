import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = r"C:\Users\DELL\Downloads\ncars_extracted\Prophesee_Dataset_n_cars"
SENSOR_W, SENSOR_H = 120, 100
N_TIME_BINS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---- Reuse your working loader/binning functions ----
def load_atis_dat(filepath):
    with open(filepath, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'%'):
                f.seek(pos)
                break
        header_bytes = f.read(2)
        if len(header_bytes) < 2:
            f.seek(pos)
        raw = np.fromfile(f, dtype=np.uint32)

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
    events['x'], events['y'], events['t'], events['p'] = x, y, timestamps, polarity
    return events


def events_to_frames(events, n_time_bins=N_TIME_BINS, width=SENSOR_W, height=SENSOR_H):
    frames = np.zeros((n_time_bins, 2, height, width), dtype=np.float32)
    if len(events) == 0:
        return frames
    valid = (events['x'] < width) & (events['y'] < height)
    events = events[valid]
    if len(events) == 0:
        return frames
    t_min, t_max = events['t'].min(), events['t'].max()
    if t_max == t_min:
        bin_idx = np.zeros(len(events), dtype=int)
    else:
        bin_edges = np.linspace(t_min, t_max + 1, n_time_bins + 1)
        bin_idx = np.clip(np.digitize(events['t'], bin_edges) - 1, 0, n_time_bins - 1)
    np.add.at(frames, (bin_idx, events['p'], events['y'], events['x']), 1)
    return frames


class NCarsDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        base = None
        for candidate in [f"n-cars-{split}", f"n-cars_{split}", f"n-cars{split}"]:
            p = os.path.join(root_dir, candidate)
            if os.path.isdir(p):
                base = p
                break
        if base is None:
            raise FileNotFoundError(f"Could not find '{split}' folder in {root_dir}")

        self.samples = []
        for label, cls in [(1, "cars"), (0, "background")]:
            cls_dir = os.path.join(base, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label))
        print(f"[{split}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        events = load_atis_dat(filepath)
        frames = events_to_frames(events)
        # For the CNN baseline: COLLAPSE all time bins into ONE static frame,
        # like a conventional camera would see (this is the key difference
        # from the SNN, which processes the 10 time bins sequentially)
        collapsed = frames.sum(axis=0)  # (2, H, W)
        return torch.from_numpy(collapsed), label


train_dataset = NCarsDataset(DATA_ROOT, split="train")
test_dataset = NCarsDataset(DATA_ROOT, split="test")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ---- Standard CNN, architecturally matched to the SNN (same conv sizes) ----
class NCarsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        flat_size = 32 * 12 * 15
        self.fc1 = nn.Linear(flat_size, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


model = NCarsCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 8
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device).float(), targets.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss {loss.item():.4f}")
    print(f"Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")

# ---- Test accuracy ----
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device).float(), targets.to(device)
        out = model(data)
        preds = out.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
print(f"CNN Test accuracy: {100*correct/total:.2f}%")
torch.save(model.state_dict(), "cnn_ncars_baseline.pth")

# ---- FLOPs estimate (rough, manual calculation) ----
def conv_flops(in_c, out_c, k, h_out, w_out):
    return 2 * in_c * out_c * k * k * h_out * w_out

flops = 0
flops += conv_flops(2, 16, 5, 50, 60)
flops += conv_flops(16, 32, 5, 25, 30)
flops += 2 * (32*12*15) * 64  # fc1
flops += 2 * 64 * 2            # fc2
print(f"Estimated CNN FLOPs per inference: {flops:,}")

# ---- Latency benchmark (CPU, single-sample inference) ----
model.eval()
sample_data, _ = next(iter(test_loader))
sample_data = sample_data[:1].to(device).float()

# warmup
for _ in range(5):
    _ = model(sample_data)

times = []
with torch.no_grad():
    for _ in range(50):
        start = time.perf_counter()
        _ = model(sample_data)
        times.append(time.perf_counter() - start)

print(f"CNN avg inference latency: {np.mean(times)*1000:.3f} ms "
      f"(std: {np.std(times)*1000:.3f} ms)")