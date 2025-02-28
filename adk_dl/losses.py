import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.manifold import trustworthiness
from torchmetrics.regression import SpearmanCorrCoef
from adk_dl.utils import HyperParameterScheduler

class CMLLoss(torch.nn.Module):
    def __init__(self):
        super(CMLLoss, self).__init__()
    
    def forward(self, embeddings, labels):
        """
        Compute the Log-ratio Loss.

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings tensor of shape (batch_size, embedding_dim).
        labels : torch.Tensor
            The continuous labels tensor of shape (batch_size, label_dim).

        Returns
        -------
        torch.Tensor
            The Log-ratio Loss.

        """
        # Compute pairwise distances for embeddings and labels
        emb_distances = torch.cdist(embeddings, embeddings, p=2) + 1e-9
        label_distances = torch.cdist(labels, labels, p=2) + 1e-9

        # Expand dimensions to create triplets
        emb_distances_a = emb_distances.unsqueeze(2)
        emb_distances_i = emb_distances.unsqueeze(1)
        label_distances_a = label_distances.unsqueeze(2)
        label_distances_i = label_distances.unsqueeze(1)
        
        # Compute log-ratios for embeddings and labels
        log_ratio_emb = torch.log(emb_distances_a / (emb_distances_i))
        log_ratio_label = torch.log(label_distances_a / (label_distances_i))

        # Create mask to exclude invalid triplets (where a == i or a == j or i == j)
        mask = torch.eye(embeddings.size(0), device=embeddings.device).unsqueeze(2) + \
            torch.eye(embeddings.size(0), device=embeddings.device).unsqueeze(1)
        mask = mask.expand(embeddings.size(0), embeddings.size(0), embeddings.size(0))
        mask = (mask == 0)

        # Compute loss using valid triplets
        loss = ((log_ratio_emb - log_ratio_label) ** 2)[mask].mean()
        return loss

# def masked_cml_loss(embeddings, labels, k=3):
#     """
#     Compute the Log-ratio Loss based on top-k closest and furthest labels.

#     Parameters
#     ----------
#     embeddings : torch.Tensor
#         The embeddings tensor of shape (batch_size, embedding_dim).
#     labels : torch.Tensor
#         The continuous labels tensor of shape (batch_size, label_dim).
#     k : int
#         The number of closest and furthest instances to consider.

#     Returns
#     -------
#     torch.Tensor
#         The Log-ratio Loss.

#     """
#     # Compute pairwise distances for embeddings and labels
#     emb_distances = torch.cdist(embeddings, embeddings, p=2) + 1e-9
#     label_distances = torch.cdist(labels, labels, p=2) + 1e-9

#     # Sort label distances and get top-k closest and furthest indices
#     sorted_label_distances, sorted_indices = torch.sort(label_distances, dim=1)
#     closest_indices = sorted_indices[:, 1:k+1]  # Exclude self (first index)
#     furthest_indices = sorted_indices[:, -k:]

#     # Create mask for top-k closest and furthest
#     closest_mask = torch.zeros_like(label_distances, dtype=torch.bool)
#     closest_mask.scatter_(1, closest_indices, True)

#     furthest_mask = torch.zeros_like(label_distances, dtype=torch.bool)
#     furthest_mask.scatter_(1, furthest_indices, True)

#     k_mask = closest_mask | furthest_mask  # Combine masks

#     # Expand dimensions to create triplets
#     emb_distances_a = emb_distances.unsqueeze(2)
#     emb_distances_i = emb_distances.unsqueeze(1)
#     label_distances_a = label_distances.unsqueeze(2)
#     label_distances_i = label_distances.unsqueeze(1)

#     #print(emb_distances_a, emb_distances_i, label_distances_a, label_distances_i   )
#     # Compute log-ratios for embeddings and labels
#     log_ratio_emb = torch.log(emb_distances_a / (emb_distances_i))
#     log_ratio_label = torch.log(label_distances_a / (label_distances_i))

#     # Create mask to exclude invalid triplets (where a == i or a == j or i == j)
#     identity_mask = torch.eye(embeddings.size(0), device=embeddings.device).unsqueeze(2) + \
#                     torch.eye(embeddings.size(0), device=embeddings.device).unsqueeze(1)
#     identity_mask = (identity_mask == 0)

#     # Expand k_mask to match the dimensionality and combine with identity_mask
#     final_mask = k_mask.unsqueeze(2) & k_mask.unsqueeze(1) & identity_mask

#     # Compute loss using valid triplets
#     loss = ((log_ratio_emb - log_ratio_label) ** 2)[final_mask].mean()

#     return loss