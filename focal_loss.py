import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        weights.append(weight)
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


#Implementation of focal loss by:
# https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/
