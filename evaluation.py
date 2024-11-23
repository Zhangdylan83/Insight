import os
import yaml
import numpy as np
import torch
import h5py
import tifffile
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn.functional import interpolate
from dataloader.dataloader_WSI import create_dataloaders
from models.clam import CLAM_MB, CLAM_SB
from models.abmil import GatedAttention
from models.TransMIL import TransMIL
from models.Insight import INSIGHT, SmoothMax 
import argparse


def load_h5(h5_file_path, avg_pool=True, batch_size=1000):
    """Load HDF5 file with optimized memory usage and batch loading."""
    with h5py.File(h5_file_path, 'r') as file:
        features_dataset = file['features']
        coords = file['coords'][:]

        num_samples = features_dataset.shape[0]
        embeddings_list = []
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i in range(0, num_samples, batch_size):
            batch_features = torch.tensor(features_dataset[i:i + batch_size], dtype=torch.float32)
            if avg_pool:
                batch_embeddings_pooled = avgpool(batch_features).squeeze(-1).squeeze(-1)
            else:
                batch_embeddings_pooled = batch_features
            embeddings_list.append(batch_embeddings_pooled)
        embeddings_pooled = torch.cat(embeddings_list, dim=0)

    return embeddings_pooled, coords


def match_volume_to_mask(h5_filename, mask_dir):
    """Matches an H5 file to its corresponding mask file."""
    base_name = os.path.splitext(os.path.basename(h5_filename))[0]
    mask_filename = f"{base_name}_mask.tif"
    return os.path.join(mask_dir, mask_filename)


def initialize_model(model_class, model_ckpt, device):
    """Initializes the appropriate model based on the model class."""
    if model_class == "clam-mb":
        model = CLAM_MB(gate=True, size_arg="small", dropout=0.0, k_sample=8, n_classes=2,
                        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024)
    elif model_class == "clam-sb":
        model = CLAM_SB()
    elif model_class == "abmil":
        model = GatedAttention(embed_size=1024)
    elif model_class == "transmil":
        model = TransMIL(n_classes=2)
    elif model_class == "Insight":
        model = INSIGHT()
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def calculate_auc(y_true, y_scores):
    auc_value = roc_auc_score(y_true, y_scores)
    print("AUC value:", auc_value)
    return auc_value


def evaluate_auc(loader, model, model_class, device):
    """Evaluates the model and calculates the AUC."""
    all_labels = []
    all_probs = []
    smoothmax = SmoothMax(8.0)

    for h5_file_path, label in tqdm(loader, desc="Evaluating AUC"):
        h5_file_path = h5_file_path[0]
        features, _ = load_h5(h5_file_path, avg_pool=(model_class != "Insight"))
        features = features.to(device)

        with torch.no_grad():
            if model_class in ["clam-mb", "clam-sb"]:
                _, Y_prob, _, _, _ = model(features, label=label)
                probs = Y_prob[:, 1].detach().cpu().numpy()
            elif model_class == "abmil":
                result_dict = model(features)
                probs = result_dict["Y_prob"].detach().cpu().numpy().flatten()
            elif model_class == "transmil":
                result_dict = model(features)
                probs = result_dict["Y_prob"][:, 1].detach().cpu().numpy()
            elif model_class == "Insight":
                patch_heatmaps, _ = model(features)
                final_output = smoothmax(patch_heatmaps).float()
                probs = final_output.detach().cpu().numpy().flatten()
            else:
                raise ValueError(f"Unsupported model class: {model_class}")

        all_labels.extend(label.cpu().numpy())
        all_probs.extend(probs)

    auc_score = calculate_auc(all_labels, all_probs)
    return auc_score


def evaluate(loader, model_class, model_ckpt, mask_dir, evaluation_type, threshold, output_filename):
    """Evaluates the model based on the specified evaluation type (AUC or Dice)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_class, model_ckpt, device)

    os.makedirs("evaluation_results", exist_ok=True)
    output_filepath = os.path.join("evaluation_results", output_filename)

    if evaluation_type == "auc":
        auc_score = evaluate_auc(loader, model, model_class, device)
        with open(output_filepath, 'w') as f:
            f.write(f"AUC Score: {auc_score:.4f}\n")
        print(f"AUC score saved to {output_filepath}")

    elif evaluation_type == "dice":
        overall_true_positives, overall_pred_positives, overall_intersections = 0, 0, 0

        for h5_file_path, label in tqdm(loader, desc="Evaluating Dice"):
            if label == 0:
                continue  # Skip negative samples for Dice
            h5_file_path = h5_file_path[0]
            mask_file_path = match_volume_to_mask(h5_file_path, mask_dir)
            with tifffile.TiffFile(mask_file_path) as tif:
                true_mask = tif.pages[1].asarray()

            features, coords = load_h5(h5_file_path, avg_pool=(model_class != "Insight"))
            features = features.to(device)

            with torch.no_grad():
                if model_class == "clam-mb":
                    _, _, _, A_raw, _ = model(features, label=label)
                    patch_heatmaps = A_raw[1].squeeze(0).cpu().numpy()
                elif model_class == "clam-sb":
                    _, _, _, A_raw, _ = model(features, label=label)
                    patch_heatmaps = A_raw.squeeze(0).cpu().numpy()
                elif model_class == "abmil":
                    _, _, attention_weights = model(features)
                    patch_heatmaps = attention_weights.squeeze(0).cpu().numpy()
                elif model_class == "transmil":
                    result_dict = model(features)
                    attention_weights = result_dict["attention_weights"].mean(dim=1).cpu().numpy().squeeze(0)
                    cls_attention = attention_weights[0, 1:len(coords) + 1]
                    patch_heatmaps = cls_attention
                elif model_class == "Insight":
                    patch_heatmaps, _ = model(features)
                    resized_heatmaps_list = []
                    for i in range(len(features)):
                            resized_heatmap = interpolate(
                                patch_heatmaps[i].unsqueeze(0),
                                size=(224, 224),
                                mode='bilinear',
                                align_corners=True
                            ) 
                            
                            resized_heatmaps_list.append(resized_heatmap.squeeze(0))  # Shape: (1, 224, 224)
                    patch_heatmaps = torch.cat(resized_heatmaps_list, dim=0)  # Final shape: (N, 224, 224)
                else:
                    raise ValueError(f"Unsupported model class: {model_class}")

            if (model_class != "Insight"): #Insight is already well calibrated and doesn't need normalization
                patch_heatmaps = (patch_heatmaps - patch_heatmaps.min()) / (patch_heatmaps.max() - patch_heatmaps.min())

            pred_patch = patch_heatmaps > threshold
            intersection = np.sum(pred_patch & (true_mask == 2))
            pred_positive = np.sum(pred_patch)
            true_positive = np.sum(true_mask == 2)

            overall_true_positives += true_positive
            overall_pred_positives += pred_positive
            overall_intersections += intersection

        dice_score = (2 * overall_intersections) / (overall_true_positives + overall_pred_positives) if (
                overall_true_positives + overall_pred_positives) > 0 else 0
        with open(output_filepath, 'w') as f:
            f.write(f"Dice Score at threshold {threshold}: {dice_score:.4f}\n")
        print(f"Dice score saved to {output_filepath}")
    else:
        raise ValueError("Evaluation type must be 'auc' or 'dice'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models with AUC or Dice calculation using config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    normal_path = config["paths"]["normal"]
    tumor_path = config["paths"]["tumor"]
    test_path = config["paths"]["test"]
    mask_dir = config["paths"]["mask_dir"]
    output_file = config["paths"]["output_file"]
    paths = [normal_path, tumor_path]

    # Create train, val, and test dataloaders
    split_ratios = config["dataloader"]["split_ratios"]
    batch_size = config["dataloader"]["batch_size"]
    seed = config["dataloader"]["seed"]

    train_loader, val_loader, test_loader = create_dataloaders(
        paths=paths,
        test_path=test_path,
        output_file=output_file,
        split_ratios=split_ratios,
        batch_size=batch_size,
        seed=seed,
        max_samples=None,
    )

    evaluate(
        test_loader,
        config["model"]["class"],
        config["model"]["checkpoint"],
        mask_dir,
        config["evaluation"]["type"],
        config["evaluation"].get("threshold", 0.5),
        "test_results.csv"
    )



