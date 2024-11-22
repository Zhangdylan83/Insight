import torch
import torch.nn as nn
import torch.nn.functional as F

class InsightTrainLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, pos_weight=1.0, neg_weight=1.0):
        super(InsightTrainLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        assert 0 <= self.label_smoothing < 1
        pred = torch.clamp(pred, min=0.0, max=0.999)

        # Apply label smoothing
        target_smooth = target.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Calculate weights based on the original targets
        weight = target * (self.pos_weight - 1) + (1 - target) * (self.neg_weight - 1) + 1

        # Compute the binary cross-entropy loss with weights
        loss = F.binary_cross_entropy(pred, target_smooth, weight=weight, reduction='mean')
        return loss


class InsightValLoss(nn.Module):
    def __init__(self):
        super(InsightValLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0.0, max=0.999)

        # Compute the binary cross-entropy loss
        loss = F.binary_cross_entropy(pred, target.float(), reduction='mean')
        return loss



