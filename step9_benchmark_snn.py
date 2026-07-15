import os
import time
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = r"C:\Users\DELL\Downloads\ncars_extracted\Prophesee_Dataset_n_cars"
SENSOR_W, SENSOR_H = 120, 100
N_TIME_BINS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---- Same loader/dataset as before ----
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
    def __init__(self, root_dir, split="test"):
        base = None
        for candidate in [f"n-cars-{split}", f"n-cars_{split}", f"n-cars{split}"]:
            p = os.path.join(root_dir, candidate)
            if os.path.isdir(p):
                base = p
                break
        self.samples = []
        for label, cls in [(1, "cars"), (0, "background")]:
            cls_dir = os.path.join(base, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        events = load_atis_dat(filepath)
        frames = events_to_frames(events)
        return torch.from_numpy(frames), label


test_dataset = NCarsDataset(DATA_ROOT, split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch=1 for clean per-sample timing


# ---- Same conv SNN architecture as Step 7, but instrumented to count spikes ----
beta = 0.9
spike_grad = surrogate.fast_sigmoid()

class NCarsSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool = nn.MaxPool2d(2)
        flat_size = 32 * 12 * 15
        self.fc1 = nn.Linear(flat_size, 64)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 2)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, count_spikes=False):
        x = x.permute(1, 0, 2, 3, 4)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_rec = []
        total_spikes = 0

        for t in range(x.size(0)):
            xt = x[t]
            cur1 = self.conv1(xt)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            pooled = self.pool(spk2)
            flat = pooled.flatten(1)
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4_rec.append(spk4)

            if count_spikes:
                total_spikes += spk1.sum().item() + spk2.sum().item() + \
                                spk3.sum().item() + spk4.sum().item()

        if count_spikes:
            return torch.stack(spk4_rec), total_spikes
        return torch.stack(spk4_rec)


model = NCarsSNN().to(device)
model.load_state_dict(torch.load("snn_ncars_model_v2.pth", map_location=device))
model.eval()

# ---- 1. Accuracy check (confirm loaded model matches training result) ----
correct, total = 0, 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device).float(), targets.to(device)
        spk_out = model(data)
        preds = spk_out.sum(0).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
print(f"SNN Test accuracy (reloaded model): {100*correct/total:.2f}%")

# ---- 2. Spike count over full test set ----
total_spikes_all = 0
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device).float()
        _, spikes = model(data, count_spikes=True)
        total_spikes_all += spikes
avg_spikes_per_sample = total_spikes_all / len(test_dataset)
print(f"Average spikes fired per inference: {avg_spikes_per_sample:,.0f}")

# ---- 3. Latency benchmark (single sample, CPU) ----
sample_data, _ = next(iter(test_loader))
sample_data = sample_data.to(device).float()

for _ in range(5):
    _ = model(sample_data)

times = []
with torch.no_grad():
    for _ in range(50):
        start = time.perf_counter()
        _ = model(sample_data)
        times.append(time.perf_counter() - start)

print(f"SNN avg inference latency: {np.mean(times)*1000:.3f} ms "
      f"(std: {np.std(times)*1000:.3f} ms)")