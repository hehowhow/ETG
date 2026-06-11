import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialTransferHead(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features):
        evidence = F.softplus(self.mlp(features))
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / strength
        uncertainty = self.num_classes / strength.squeeze(-1)
        return alpha, probs, uncertainty


class PlainTransferHead(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features):
        logits = self.mlp(features)
        probs = F.softmax(logits, dim=-1)
        uncertainty = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)
        return logits, probs, uncertainty


class RegressionTransferHead(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        return self.mlp(features).squeeze(-1)


def dirichlet_kl_to_uniform(alpha):
    num_classes = alpha.size(-1)
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_beta = beta.sum(dim=-1, keepdim=True)

    ln_b_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(sum_alpha.squeeze(-1))
    ln_b_beta = torch.lgamma(beta).sum(dim=-1) - torch.lgamma(sum_beta.squeeze(-1))
    digamma_term = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1)
    return ln_b_beta - ln_b_alpha + digamma_term


def evidential_classification_loss(alpha, labels, epoch=0, anneal_epochs=20, kl_weight=1.0):
    strength = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / strength
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels, reduction='mean')
    anneal = min(1.0, float(epoch + 1) / max(float(anneal_epochs), 1.0))
    kl = dirichlet_kl_to_uniform(alpha).mean()
    return nll + anneal * kl_weight * kl
