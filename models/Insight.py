import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

class SmoothMax(nn.Module):
    def __init__(self, alpha=8.0):
        super(SmoothMax, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        # x should have shape [N, C, H, W]
        # Reshape x to [C, N*H*W] to apply SmoothMax over N, H, W for each channel
        original_shape = x.shape
        x = x.permute(1, 0, 2, 3).reshape(original_shape[1], -1)  # Shape becomes [C, N*H*W]

        weighted_exp = torch.exp(self.alpha * x)
        pooled_value = torch.sum(x * weighted_exp, dim=1) / torch.sum(weighted_exp, dim=1)  # Sum over N*H*W, result shape [C]

        return pooled_value


class INSIGHT(nn.Module):
    def __init__(self, embedding_dim=1024, attention_dim=128, output_dim=1, lambda_sd=0.00):
        super(INSIGHT, self).__init__()

        self.reduce_dimension_block = nn.Sequential(
            nn.Conv2d(embedding_dim, attention_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([128, 14, 14]),
        )

        self.backbone_block = nn.Sequential(
            nn.Conv2d(attention_dim, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([128, 14, 14]),

            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([64, 14, 14]),

            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([64, 14, 14]),

            nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0),
        )

        self.context_block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(attention_dim, 128, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([128, 14, 14]),

            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([64, 14, 14]),

            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([64, 14, 14]),

            nn.ReplicationPad2d(1),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=0),
        )

        
        self.lambda_sd = lambda_sd

    def otsu_threshold_4d(self, heatmap):
        """Apply Otsu threshold and return a mask where foreground values are kept."""
        thresholds = []
        masks = []
        for c in range(heatmap.shape[1]):  # iterate through channels independently
            channel_data = heatmap[:, c, :, :]  # N, H, W
            flattened_channel = channel_data.flatten()
            threshold_value = threshold_otsu(flattened_channel)
            channel_mask = channel_data > threshold_value
            masks.append(channel_mask)
        masks = np.stack(masks, axis=1)  # stack along channel dimension
        return masks

    def apply_spectrum_decoupling(self, logits):
        penalty = self.lambda_sd / 2 * (logits ** 2).mean()
        return penalty

    def forward(self, x):
        total_sd_penalty = 0
        x = x.float()
        num_slices, channels, heights, widths = x.shape

        x_reduced = self.reduce_dimension_block(x)
        backbone_out = self.backbone_block(x_reduced)

        # Apply spectrum decoupling penalties to logits
        total_sd_penalty += self.apply_spectrum_decoupling(backbone_out)

        context_out = self.context_block(x_reduced)
        context_out_sigmoid = torch.sigmoid(context_out)

        backbone_heatmap = torch.sigmoid(backbone_out)
        context_heatmap = backbone_heatmap * context_out_sigmoid
        final_heatmap = torch.relu(backbone_heatmap - context_heatmap)

        final_heatmap_np = final_heatmap.detach().cpu().numpy()
        mask = self.otsu_threshold_4d(final_heatmap_np)
        final_heatmap = final_heatmap * torch.from_numpy(mask).to(final_heatmap.device).float()  # N, C, H, W

        return final_heatmap, total_sd_penalty


class Baseline(nn.Module):
    def __init__(self, embedding_dim=1024, attention_dim=128, output_dim=1, lambda_sd=0.00):
        super(Baseline, self).__init__()

        self.reduce_dimension_block = nn.Sequential(
            nn.Conv2d(embedding_dim, attention_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([128, 14, 14]),
        )

        self.backbone_block = nn.Sequential(
            nn.Conv2d(attention_dim, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([128, 14, 14]),

            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([64, 14, 14]),

            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.LayerNorm([64, 14, 14]),

            nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0),
        )
        
        self.lambda_sd = lambda_sd

    def otsu_threshold_4d(self, heatmap):
        thresholds = []
        masks = []
        for c in range(heatmap.shape[1]):  # iterate through channels independently
            channel_data = heatmap[:, c, :, :]  # N, H, W
            flattened_channel = channel_data.flatten()
            threshold_value = threshold_otsu(flattened_channel)
            channel_mask = channel_data > threshold_value
            masks.append(channel_mask)
        masks = np.stack(masks, axis=1)  # stack along channel dimension
        return masks

    def apply_spectrum_decoupling(self, logits):
        penalty = self.lambda_sd / 2 * (logits ** 2).mean()
        return penalty

    def forward(self, x):
        total_sd_penalty = 0
        x = x.float()
        num_slices, channels, heights, widths = x.shape

        x_reduced = self.reduce_dimension_block(x)
        backbone_out = self.backbone_block(x_reduced)

        # Apply spectrum decoupling penalties to logits
        total_sd_penalty += self.apply_spectrum_decoupling(backbone_out)

        backbone_heatmap = torch.sigmoid(backbone_out)
        final_heatmap = torch.relu(backbone_heatmap)

        final_heatmap_np = final_heatmap.detach().cpu().numpy()
        mask = self.otsu_threshold_4d(final_heatmap_np)
        final_heatmap = final_heatmap * torch.from_numpy(mask).to(final_heatmap.device).float()  # N, C, H, W

        return final_heatmap, total_sd_penalty

