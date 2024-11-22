import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score
import yaml
from dataloader.loader import create_dataloaders
from torch.utils.checkpoint import checkpoint
from ranger import Ranger
import importlib
import h5py
import argparse


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

#Comparison models loading vector embeddings
def load_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as file:
        embeddings = torch.tensor(file['features'][:], dtype=torch.float32)
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        embeddings_pooled = avgpool(embeddings)
        return embeddings_pooled.squeeze(-1).squeeze(-1)

#INSIGHT specific loading in chunks methods due to large memory inputs.
def load_chunk(h5_file_path, chunk_idx, chunk_size):
    with h5py.File(h5_file_path, 'r') as file:
        embeddings = file['features']
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, embeddings.shape[0])
        return torch.tensor(embeddings[start:end], dtype=torch.float32)


def calculate_num_chunks(file_path, chunk_size):
    with h5py.File(file_path, 'r') as file:
        total_data_points = file['features'].shape[0]
        return total_data_points // chunk_size + (1 if total_data_points % chunk_size != 0 else 0), total_data_points


def validate_config(config):
    """Validate the structure and content of the YAML config."""
    required_top_keys = ["model", "loss", "training", "save", "paths"]
    required_model_keys = ["name", "chunk_size", "pooling"]
    required_training_keys = ["epochs", "patience", "lr", "optimizer"]

    # Check for required top-level keys
    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: '{key}'")

    # Validate model section
    model = config["model"]
    for key in required_model_keys:
        if key not in model:
            raise ValueError(f"Missing required model key: '{key}'")
    if model["pooling"] == "smoothmax" and "smoothmax_class" not in model:
        raise ValueError("Missing 'smoothmax_class' for smoothmax pooling in model config.")

    # Validate training section
    training = config["training"]
    for key in required_training_keys:
        if key not in training:
            raise ValueError(f"Missing required training key: '{key}'")

    # Validate paths
    if not isinstance(config["paths"], list) or not config["paths"]:
        raise ValueError("Invalid or missing 'paths'. It should be a non-empty list.")

    print("Config validation passed!")


def validate_data(dataloaders):
    """Check that dataloaders provide valid data."""
    for phase, loader in dataloaders.items():
        if not loader:
            raise ValueError(f"Dataloader for '{phase}' is missing or empty.")
        sample = next(iter(loader), None)
        if not sample:
            raise ValueError(f"Dataloader for '{phase}' contains no data.")
        print(f"Dataloader validation passed for '{phase}'.")


def load_class(full_class_name):
    """
    Dynamically load a class from a string path.
    :param full_class_name: Full path to the class (e.g., 'utils.pooling.SmoothMax').
    :return: Class object.
    """
    try:
        module_name, class_name = full_class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not load class '{full_class_name}': {e}")


def train_model(model, dataloaders, optimizer, loss_fn, num_epochs, device, smoothmax, save_dir, model_args=None):
    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = model_args.get("patience", 8)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_probs = []

            for h5_file_path, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}"):
                h5_file_path = h5_file_path[0]  
                labels = labels.to(device)

                if "INSIGHT" in model.__class__.__name__:
                    # INSIGHT-specific logic
                    num_chunks, _ = model_args["num_chunks_func"](h5_file_path, model_args["chunk_size"])
                    chunk_results = []
                    all_sd_penalty = 0

                    for chunk_idx in range(num_chunks):
                        chunk_inputs = model_args["load_chunk_func"](h5_file_path, chunk_idx, model_args["chunk_size"]).to(device)
                        chunk_inputs = chunk_inputs.to(device).requires_grad_(phase == 'train')
                        with torch.set_grad_enabled(phase == 'train'):
                            final_heatmap, sd_penalty = checkpoint(lambda x: model(x), chunk_inputs, use_reentrant=False)
                            chunk_results.append(final_heatmap)
                            all_sd_penalty += sd_penalty

                    chunk_results_tensor = torch.cat(chunk_results, dim=0)
                    if model_args["pooling"] == "smoothmax":
                        final_output = smoothmax(chunk_results_tensor)
                    else:
                        max_across_n = torch.max(chunk_results_tensor, dim=0)[0]  # [c, h, w]
                        max_across_h = torch.max(max_across_n, dim=1)[0]  # [c, w]
                        final_output = torch.max(max_across_h, dim=1)[0]  # [c]

                    loss = loss_fn(final_output, labels) + model.lambda_sd * all_sd_penalty
                    Y_prob = final_output
                 
                else:
                    # General case for CLAM, ABMIL, and TransMIL
                    inputs = load_h5(h5_file_path).to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        if "CLAM-SB" or "CLAM-MB" in model.__class__.__name__:
                            def create_custom_forward(module): #need to wrap up here because requiring additional arguments.
                                def custom_forward(x, label):
                                    return module(x, label=label, instance_eval=True, return_features=True)
                                return custom_forward

                            outputs = checkpoint(create_custom_forward(model), inputs, labels, use_reentrant=False)
                            logits, Y_prob, _, _, result_dict = outputs
                            instance_loss = result_dict.get('instance_loss')
                            bag_loss = nn.CrossEntropyLoss()(logits, labels)
                            bag_weight = model_args["bag_weight"]
                            loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss

                        elif "TransMIL" in model.__class__.__name__:

                            outputs = checkpoint(create_custom_forward(model), inputs, use_reentrant=False)
                            logits = outputs["logits"]
                            Y_prob = outputs["Y_prob"]  
                            loss = loss_fn(logits, labels)

                        elif "ABMIL" in model.__class__.__name__:

                            outputs = checkpoint(create_custom_forward(model), inputs, use_reentrant=False)
                            Y_prob = outputs["Y_prob"] 
                            loss = model.calculate_objective(Y_prob, labels)
                        else:
                            raise ValueError(f"Unsupported model class: {model.__class__.__name__}")

                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(Y_prob.detach().cpu().numpy())

            
            epoch_loss = running_loss / len(dataloaders[phase])
            auc_score = calculate_auc(all_labels, all_probs)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} AUC: {auc_score:.4f}")

            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_auc": auc_score, "epoch": epoch})
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))#save ckpts of all epochs.
            #save least validation loss ckpt too
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                no_improve_epochs = 0
                torch.save(model.state_dict(), os.path.join(save_dir, "best_ckpt.pth"))
                print("Best model saved.")
            elif phase == 'val':
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                return model

    return model



def main(args):
    set_seed(42)
    #wandb.init(project=args["wandb"]["project_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normal_path = args["paths"]["normal"]
    tumor_path = args["paths"]["tumor"]
    test_path = args["paths"]["test"]
    output_file = args["paths"]["output_file"]
    paths = [normal_path, tumor_path]

    # Create train, val, and test dataloaders
    split_ratios = args["training"].get("split_ratios", [0.9, 0.1]) #second value is default
    batch_size = args["training"].get("batch_size", 1)
    max_samples = args["training"].get("max_samples", None)

    train_loader, val_loader, test_loader = create_dataloaders(
        paths=paths,
        test_path=test_path,
        output_file=output_file,
        split_ratios=split_ratios,
        batch_size=batch_size,
        seed=42,
        max_samples=max_samples
    )

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    validate_data(dataloaders)

    ModelClass = load_class(args["model"]["name"])
    lambda_sd = args["model"].get("lambda_sd", 0.01)
    model = ModelClass(lambda_sd=lambda_sd).to(device)

    optimizer = {
        "adam": lambda: optim.Adam(model.parameters(), lr=args["training"]["lr"], weight_decay=args["training"]["weight_decay"]),
        "adamw": lambda: optim.AdamW(model.parameters(), lr=args["training"]["lr"], weight_decay=args["training"]["weight_decay"]),
        "ranger": lambda: Ranger(model.parameters(), lr=args["training"]["lr"], weight_decay=args["training"]["weight_decay"], k=6, alpha=0.5)
    }[args["training"]["optimizer"]]()
    
    SmoothMaxClass = load_class(args["model"].get("smoothmax_class")) if args["model"].get("smoothmax_class") else None
    smoothmax = SmoothMaxClass(args["model"]["alpha"]) if SmoothMaxClass else None

    model_args = {
    "num_chunks_func": calculate_num_chunks,
    "load_chunk_func": load_chunk,
    "chunk_size": args["model"]["chunk_size"],
    "pooling": args["model"]["pooling"],
}

    LossClass = load_class(args["loss"]["type"])
    loss_fn = LossClass(**args["loss"].get("params", {}))

    train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args["training"]["epochs"],
        device=device,
        smoothmax=smoothmax,
        save_dir=args["save"]["path"],
        model_args=model_args
    )

    #wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Script with YAML Support")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    #print(config)
    main(config)



