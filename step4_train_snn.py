import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

# ---- 1. Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)

trainset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=frame_transform)
testset  = tonic.datasets.NMNIST(save_to='./data', train=False, transform=frame_transform)

# tonic's collate function handles variable-length event data batching correctly
train_loader = DataLoader(trainset, batch_size=64, shuffle=True,
                           collate_fn=tonic.collation.PadTensors(batch_first=False))
test_loader = DataLoader(testset, batch_size=64, shuffle=False,
                          collate_fn=tonic.collation.PadTensors(batch_first=False))

# ---- 2. Define the Spiking Neural Network ----
beta = 0.9  # neuron "leak" rate — how much membrane potential decays each step
spike_grad = surrogate.fast_sigmoid()  # needed because spikes aren't differentiable

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(34*34*2, 256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(256, 10)  # 10 output classes (digits 0-9)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x shape: (time_bins, batch, channels, H, W)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        for t in range(x.size(0)):  # loop over time bins
            xt = x[t].flatten(1)  # flatten each frame to a vector
            cur1 = self.fc1(xt)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)  # (time_bins, batch, 10)

model = SNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---- 3. Train for a few epochs ----
num_epochs = 2  # start small just to confirm it works, increase later

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device).float(), targets.to(device)
        optimizer.zero_grad()

        spk_out = model(data)             # (time_bins, batch, 10)
        spike_counts = spk_out.sum(0)     # sum spikes over time -> (batch, 10)
        loss = loss_fn(spike_counts, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

    print(f"Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")

# ---- 4. Quick test accuracy check ----
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device).float(), targets.to(device)
        spk_out = model(data)
        spike_counts = spk_out.sum(0)
        preds = spike_counts.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"Test accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "snn_nmnist_model.pth")
print("Model saved as snn_nmnist_model.pth")