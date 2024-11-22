import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse

class CTDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        processed_volume = self.process_volume(file_path)
        return processed_volume, file_path

    def load_volume(self, volume_path):
        nifti_img = nib.load(volume_path)
        volume = nifti_img.get_fdata()
        return volume

    def resize_volume(self, volume):
        desired_width = 518
        desired_height = 518

        current_depth = volume.shape[2]
        current_width = volume.shape[1]
        current_height = volume.shape[0]

        depth_factor = 1  # Keep depth unchanged
        width_factor = desired_width / current_width
        height_factor = desired_height / current_height

        resized_volume = ndimage.zoom(volume, (height_factor, width_factor, depth_factor), order=1)
        return resized_volume

    def normalize_slice(self, slice_data):
        """Clip HU values and normalize a single 2D slice to [0, 1]."""
        min_val = -1000
        max_val = 400
        slice_data[slice_data < min_val] = min_val
        slice_data[slice_data > max_val] = max_val
        slice_data = (slice_data - min_val) / (max_val - min_val)
        return slice_data.astype("float32")

    def process_volume(self, file_path):
        """Process the volume: load, resize, and normalize each slice."""
        volume = self.load_volume(file_path)
        resized_volume = self.resize_volume(volume)

        normalized_volume = np.zeros_like(resized_volume, dtype=np.float32)
        for i in range(resized_volume.shape[2]):
            normalized_volume[:, :, i] = self.normalize_slice(resized_volume[:, :, i])
   
        return normalized_volume


# DINOv2FeatureExtractor class to extract embeddings
class DINOv2FeatureExtractor(nn.Module):
    def __init__(self):
        super(DINOv2FeatureExtractor, self).__init__()
        # Load the DiNOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', force_reload=True)


    def convert_to_3channel(self, slices_2d):
        '''
        Convert a stack of grayscale 2D slices to 3-channel.
        Input: (num_slices, 224, 224)
        Output: (num_slices, 224, 224, 3)
        '''
        return np.stack([slices_2d] * 3, axis=-1)  # [num_slices, 224, 224] -> [num_slices, 224, 224, 3]

    def forward(self, scan_volumes):
        """
        Process the stack of 2D slices, normalize, and extract features in a batch.
        scan_volumes shape: (1, 224, 224, num_slices)
        """
        scan_volumes = scan_volumes.squeeze(0)  # Remove unnecessary dimension: (224, 224, num_slices)

        # Convert to 3 channels to meet DINOv2 criteria: (224, 224, num_slices) -> (num_slices, 224, 224, 3)
        slice_3channel = self.convert_to_3channel(scan_volumes).astype(np.float32)
        slice_tensor = torch.tensor(slice_3channel).permute(2, 3, 0, 1)  # (num_slices, 3, 224, 224)
      

        # Extract features for all slices in the batch
        with torch.no_grad():
            features = self.dino.forward_features(slice_tensor)["x_norm_patchtokens"]  # (num_slices, 256, 1024)
            features = features.transpose(1, 2).view(-1, 1024, 37, 37)  # (num_slices, 1024, 16, 16)

        return features


def main(file_paths, save_folder):
    file_paths = file_paths
    files = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.nii')]

    print(f"Total files: {len(files)}")

    ct_dataset = CTDataset(files)
    dinov2 = DINOv2FeatureExtractor()

    loader = DataLoader(ct_dataset, batch_size=1, shuffle=False)
    save_folder = save_folder
    os.makedirs(save_folder, exist_ok=True)

    # Iterate through volumes in the dataset
    for volume, file_path in tqdm(loader, desc="Extracting features"):
        
        embeddings = dinov2(volume)

        
        file_name = os.path.basename(file_path[0].replace('.nii', '.pt'))
        save_path = os.path.join(save_folder, file_name)
        torch.save(embeddings, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yes")
    parser.add_argument("--file_paths", type=str, help="yes")
    parser.add_argument("--save_folder", type=str, help="yes")
    args = parser.parse_args()
    main(args.file_paths, args.save_folder)

