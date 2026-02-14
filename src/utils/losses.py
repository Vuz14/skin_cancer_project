import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLossBCE(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
     
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
   
        targets = targets.float()
        logits = logits.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
