import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import torch.nn.functional as F


class EmbeddingDataset(Dataset):
    def __init__(self, path, label, max_samples=None):
        self.path = path
        self.label = torch.tensor(label, dtype=torch.float32)
        self.files = sorted(os.listdir(path)[:max_samples] if max_samples else os.listdir(path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pt_file_path = os.path.join(self.path, self.files[idx])
        embeddings = self.load_embeddings(pt_file_path)
        return embeddings, self.label, self.files[idx]

    def load_embeddings(self, pt_file_path):
        embeddings = torch.load(pt_file_path).float()
        depth = embeddings.shape[0]

        if depth < 32:  # Pad if less than 32 slices
            padding = (0, 0, 0, 0, 0, 0, 0, 32 - depth)
            embeddings = F.pad(embeddings, padding, mode='constant', value=0)
        else:
            mid_idx = depth // 2
            start_idx = max(0, mid_idx - 16)
            end_idx = min(depth, mid_idx + 16)
            embeddings = embeddings[start_idx:end_idx]  # Only take middle 32 slices

        return embeddings


def segmentation_loader(dataset, batch_size=1, shuffle=True, val_split=0.01):
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def classification_loader(dataset, n_splits=5, batch_size=1, shuffle=True, fold_index=0):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    fold_size = len(dataset) // n_splits
    remainder = len(dataset) % n_splits
    current = 0
    datasets = []

    for i in range(n_splits):
        start = current
        end = start + fold_size + (1 if i < remainder else 0)
        datasets.append(Subset(dataset, indices[start:end]))
        current = end

    val_dataset = datasets[fold_index]
    train_datasets = [ds for i, ds in enumerate(datasets) if i != fold_index]
    train_dataset = ConcatDataset(train_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
