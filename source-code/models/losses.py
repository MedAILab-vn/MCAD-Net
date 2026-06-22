import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareLossSmoothing(nn.Module):
    def __init__(self, alpha=0.1, label_smoothing=0.0):
        super(UncertaintyAwareLossSmoothing, self).__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, stacked_logits, labels):
        if stacked_logits.dim() == 2:
            stacked_logits = stacked_logits.unsqueeze(0)

        mean_probs = torch.softmax(stacked_logits, dim=-1).mean(dim=0)
        log_mean_probs = torch.log(mean_probs + 1e-8)

        if self.label_smoothing > 0.0:
            num_classes = mean_probs.size(-1)
            true_dist = torch.zeros_like(log_mean_probs)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = torch.mean(-torch.sum(true_dist * log_mean_probs, dim=-1))
        else:
            ce_loss = F.nll_loss(log_mean_probs, labels)

        entropy = -torch.sum(mean_probs * log_mean_probs, dim=-1)
        mean_entropy = torch.mean(entropy)

        total_loss = ce_loss + (self.alpha * mean_entropy)
        return total_loss

def build_loss_function(loss_type, device, label_smoothing=0.1, alpha=0.1):
    if loss_type == 'CE':
        print(f"-> Activating Standard Cross Entropy Loss (Smoothing: {label_smoothing})")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    elif loss_type == 'UAL':
        print(f"-> Activating Uncertainty-Aware Loss (Requires enabling MC-Dropout during Training | Alpha: {alpha})")
        return UncertaintyAwareLossSmoothing(alpha=alpha, label_smoothing=label_smoothing).to(device)
    else:
        raise ValueError(f"[ERROR] Loss architecture not found: '{loss_type}'")