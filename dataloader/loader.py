import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import h5py
import math


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EmbeddingDataset(Dataset):
    """Dataset for train/val embeddings."""
    def __init__(self, path, label, max_samples=None):
        self.label = torch.tensor(label, dtype=torch.long)
        self.files = [os.path.join(path, f) for f in os.listdir(path)[:max_samples]] if max_samples else \
            [os.path.join(path, f) for f in os.listdir(path)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_file_path = self.files[idx]
        return h5_file_path, self.label


class TestEmbeddingDataset(Dataset):
    """Dataset for test embeddings."""
    def __init__(self, test_path, output_file, max_samples=None):
        self.file_labels = {}
        # Read label information from output file
        with open(output_file, 'r') as file:
            next(file)  # Skip header
            for line in file:
                if max_samples is not None and len(self.file_labels) >= max_samples:
                    break
                filename, label = line.strip().split(',')
                self.file_labels[filename] = int(label)

        # Prepare list of files with labels
        self.files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f in self.file_labels]
        if max_samples is not None and len(self.files) > max_samples:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.file_labels[os.path.basename(file_path)]
        return file_path, label


def create_datasets(paths, labels, max_samples=None):
    """Create datasets for each label."""
    datasets = {}
    for label, path in zip(labels, paths):
        datasets[label] = EmbeddingDataset(path, label, max_samples)
    return datasets


def split_datasets(datasets, split_ratios):
    split_data = {}
    for label, dataset in datasets.items():
        total_len = len(dataset)
        train_size = math.ceil(split_ratios[0] * total_len)
        val_size = total_len - train_size
        split_data[label] = random_split(dataset, [train_size, val_size])
    return split_data


def create_dataloaders(paths, test_path, output_file, split_ratios=[0.9, 0.1], batch_size=1, seed=42, max_samples=None):
    """Create train, val, and test loaders."""
    set_seed(seed)
    labels = [0, 1]  # Define class labels

    # Create train/val datasets
    datasets = create_datasets(paths, labels, max_samples)
    split_data = split_datasets(datasets, split_ratios)

    # Combine train/val datasets for each split
    train_dataset = ConcatDataset([split_data[label][0] for label in labels])
    val_dataset = ConcatDataset([split_data[label][1] for label in labels])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create test dataset and loader
    test_dataset = TestEmbeddingDataset(test_path, output_file, max_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Save train/val splits to file for reproducibility
    with open("train_val_split.csv", "w") as f:
        f.write("Category,Filename,Label\n")
        for category, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
            for idx in range(len(dataset)):
                file_path, label = dataset[idx]
                f.write(f"{category},{file_path},{label.item()}\n")

    return train_loader, val_loader, test_loader


