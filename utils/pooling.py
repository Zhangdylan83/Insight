import torch
import torch.nn as nn

class SmoothMax(nn.Module):
    def __init__(self, alpha=1.0):
        """
        SmoothMax pooling over the dimensions (N, H, W) for each channel C.
        Args:
            alpha: Smoothing factor. Larger alpha gives results closer to MaxPooling.
        """
        super(SmoothMax, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [N, C, H, W]
        Returns:
            pooled_value: Tensor of shape [C], SmoothMax pooled values over [N, H, W] for each channel.
        """
        # x should have shape [N, C, H, W]
        original_shape = x.shape
        # Reshape x to [C, N*H*W] to apply SmoothMax over N, H, W for each channel
        x = x.permute(1, 0, 2, 3).reshape(original_shape[1], -1)  # Shape becomes [C, N*H*W]

        # Compute SmoothMax
        weighted_exp = torch.exp(self.alpha * x)
        pooled_value = torch.sum(x * weighted_exp, dim=1) / torch.sum(weighted_exp, dim=1)  # Sum over N*H*W, result shape [C]

        return pooled_value


def max_pooling(x):
    """
    MaxPooling over the dimensions (N, H, W) for each channel C.
    Args:
        x: Tensor of shape [N, C, H, W]
    Returns:
        final_output: Tensor of shape [C], Max pooled values over [N, H, W] for each channel.
    """
    # Max pooling over N, H, W
    max_across_n = torch.max(x, dim=0)[0]  # Shape becomes [C, H, W]
    max_across_h = torch.max(max_across_n, dim=1)[0]  # Shape becomes [C, W]
    final_output = torch.max(max_across_h, dim=1)[0]  # Shape becomes [C]

    return final_output



