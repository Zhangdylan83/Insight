import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from tqdm import tqdm
import numpy as np
from losses.Insight_loss import InsightValLoss, InsightTrainLoss
from sklearn.metrics import roc_auc_score
import nibabel as nib
import torch.nn.functional as F
import argparse
import importlib
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        if depth < 32: #pad if less than 32 slices
            padding = (0, 0, 0, 0, 0, 0, 0, 32 - depth)
            embeddings = F.pad(embeddings, padding, mode='constant', value=0)
        else:
            mid_idx = depth // 2
            start_idx = max(0, mid_idx - 16)
            end_idx = min(depth, mid_idx + 16)
            embeddings = embeddings[start_idx:end_idx] #only take middle 32 slices

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


def calculate_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def train_model(config):
    set_seed(config['seed'])

    # Dataset loading
    datasets = [
        EmbeddingDataset(path, label, max_samples=config['training']['max_samples'])
        for path, label in zip(config['paths'], config['labels'])
    ]
    combined_dataset = ConcatDataset(datasets)

    if config['task'] == "segmentation":
        train_loader, val_loader = segmentation_loader(
            combined_dataset,
            batch_size=config['training']['batch_size'],
            val_split=config['training']['split_ratios'][1]
        )
    else:  # Classification
        train_loader, val_loader = classification_loader(
            combined_dataset,
            n_splits=config['cross_validation']['n_splits'],
            batch_size=config['training']['batch_size'],
            fold_index=config['cross_validation']['fold_index']
        )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module_name, class_name = config['model'].rsplit('.', 1)
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)
    model = ModelClass(alpha=config['model_params']['alpha'], lambda_sd=config['model_params']['lambda_sd']).to(device)

    optimizer = getattr(optim, config['training']['optimizer'])(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        model.train()
        running_loss = 0.0

        for features, label, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}"):
            features, label = features.to(device), label.float().to(device)
            optimizer.zero_grad()

            raw_predictions, _, sd_penalty = model(features)
            train_loss = InsightTrainLoss(
                pos_weight=config['loss_weights']['pos_weight'],
                neg_weight=config['loss_weights']['neg_weight']
            )(raw_predictions, label)
            loss = train_loss.mean() + sd_penalty
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, label, _ in val_loader:
                features, label = features.to(device), label.float().to(device)
                raw_predictions, _, _ = model(features)
                val_loss += InsightValLoss(
                    pos_weight=config['loss_weights']['pos_weight'],
                    neg_weight=config['loss_weights']['neg_weight']
                )(raw_predictions, label).mean().item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['save']['path'], f"best_model_epoch{epoch + 1}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_model(config)
