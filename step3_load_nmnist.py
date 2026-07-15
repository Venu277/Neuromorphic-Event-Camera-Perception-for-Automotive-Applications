import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt

# 1. Get the sensor resolution for N-MNIST (34x34 event camera output)
sensor_size = tonic.datasets.NMNIST.sensor_size
print("Sensor size:", sensor_size)

# 2. Define a transform that bins raw events into 10 time-step "frames"
#    This turns a sparse event stream into a tensor shape: (time_bins, polarity, H, W)
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)

# 3. Download and load the dataset (this will download ~few hundred MB on first run)
trainset = tonic.datasets.NMNIST(
    save_to='./data',
    train=True,
    transform=frame_transform
)

print("Number of samples in training set:", len(trainset))

# 4. Grab a single sample and inspect it
frames, label = trainset[0]
print("Label (digit):", label)
print("Frames shape:", frames.shape)   # expect (10, 2, 34, 34) -> (time_bins, polarity, H, W)

# 5. Visualize the event stream across time bins
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    # sum both polarity channels together just for visualization
    frame_img = frames[i][0] + frames[i][1]
    ax.imshow(frame_img, cmap='gray')
    ax.set_title(f"t_bin {i}")
    ax.axis('off')

plt.suptitle(f"N-MNIST sample — digit label: {label}")
plt.tight_layout()
plt.savefig("nmnist_sample_visualization.png")
plt.show()

print("Done. Check nmnist_sample_visualization.png to see the event frames.")