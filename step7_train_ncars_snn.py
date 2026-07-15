import os
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


# ---- 1. Your working .dat loader (unchanged) ----
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


# ---- 2. Vectorized frame binning (fast version) ----
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

    # vectorized scatter-add using np.add.at instead of a Python loop
    np.add.at(frames, (bin_idx, events['p'], events['y'], events['x']), 1)
    return frames


# ---- 3. PyTorch Dataset wrapping cars/ and background/ folders ----
class NCarsDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        base = None
        for candidate in [f"n-cars-{split}", f"n-cars_{split}", f"n-cars{split}"]:
            p = os.path.join(root_dir, candidate)
            if os.path.isdir(p):
                base = p
                break
        if base is None:
            raise FileNotFoundError(
                f"Could not find '{split}' folder in {root_dir}. "
                f"Contents: {os.listdir(root_dir)}"
            )

        self.samples = []
        for label, cls in [(1, "cars"), (0, "background")]:
            cls_dir = os.path.join(base, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Expected class folder missing: {cls_dir}")
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label))

        print(f"[{split}] Loaded {len(self.samples)} samples "
              f"({sum(1 for _, l in self.samples if l == 1)} cars, "
              f"{sum(1 for _, l in self.samples if l == 0)} background)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        events = load_atis_dat(filepath)
        frames = events_to_frames(events)
        return torch.from_numpy(frames), label


# ---- 4. Datasets and loaders ----
train_dataset = NCarsDataset(DATA_ROOT, split="train")
test_dataset = NCarsDataset(DATA_ROOT, split="test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# ---- 5. Improved SNN model with convolutional first layer ----
beta = 0.9
spike_grad = surrogate.fast_sigmoid()

class NCarsSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer: input has 2 channels (polarity), sees local spatial patterns
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)  # -> (16, 50, 60)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  # -> (32, 25, 30)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.pool = nn.MaxPool2d(2)  # -> (32, 12, 15)

        flat_size = 32 * 12 * 15
        self.fc1 = nn.Linear(flat_size, 64)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(64, 2)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x: (batch, time_bins, 2, H, W) -> (time_bins, batch, 2, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_rec = []

        for t in range(x.size(0)):
            xt = x[t]  # (batch, 2, H, W)

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

        return torch.stack(spk4_rec)


model = NCarsSNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

# ---- 6. Train ----
num_epochs = 8

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device).float(), targets.to(device)
        optimizer.zero_grad()

        spk_out = model(data)
        spike_counts = spk_out.sum(0)
        loss = loss_fn(spike_counts, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss {loss.item():.4f}")

    scheduler.step()
    print(f"Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f} "
          f"(lr={optimizer.param_groups[0]['lr']:.6f})")

    # quick test check every couple epochs so you can watch progress without waiting till the end
    if epoch % 2 == 1:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device).float(), targets.to(device)
                spk_out = model(data)
                preds = spk_out.sum(0).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        print(f"  -> Test accuracy after epoch {epoch}: {100*correct/total:.2f}%")
        model.train()

# ---- 7. Final test accuracy ----
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device).float(), targets.to(device)
        spk_out = model(data)
        preds = spk_out.sum(0).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"Final test accuracy: {100*correct/total:.2f}%")
torch.save(model.state_dict(), "snn_ncars_model_v2.pth")
print("Model saved as snn_ncars_model_v2.pth")

