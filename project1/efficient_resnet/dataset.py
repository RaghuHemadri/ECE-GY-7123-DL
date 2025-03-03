import os
import pickle
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

# -------------------------------
# Dataset Definition (same as before)
# -------------------------------
class CIFAR10PickleDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        if train:
            # Load batches 1 to 5 for training
            for i in range(1, 6):
                batch_file = os.path.join(data_dir, f"data_batch_{i}")
                with open(batch_file, 'rb') as fo:
                    batch = pickle.load(fo, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.labels += batch[b'labels']
            self.data = np.concatenate(self.data, axis=0)
        else:
            # Load test batch (assumes test labels exist; adjust if not)
            batch_file = os.path.join(data_dir, "test_batch")
            with open(batch_file, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                self.data = batch[b'data']
                self.labels = batch.get(b'labels', None)
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform:
            img = self.transform(img)
        label = self.labels[index] if self.labels is not None else -1
        return img, label