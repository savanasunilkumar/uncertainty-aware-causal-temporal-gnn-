"""Learnable multi-modal fusion components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMultiModalFusion(nn.Module):
    """Learnable fusion of multi-modal similarities using attention."""
    
    def __init__(self, modalities, embedding_dim):
        super().__init__()
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        
        self.query = nn.Parameter(torch.randn(1, embedding_dim))

        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(1, embedding_dim) for modality in modalities
        })

        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, similarities):
        # Cast inputs to the parameter dtype/device so callers (cold start, etc.)
        # that build similarities from numpy / float64 intermediates do not hit a
        # dtype mismatch against the nn.Linear weights.
        target_dtype = self.query.dtype
        target_device = self.query.device

        projected_similarities = []
        any_similarity = None
        for modality, similarity in similarities.items():
            if not torch.is_tensor(similarity):
                similarity = torch.as_tensor(similarity)
            similarity = similarity.to(device=target_device, dtype=target_dtype)
            if any_similarity is None:
                any_similarity = similarity
            if modality in self.modality_projections:
                projected = self.modality_projections[modality](similarity.unsqueeze(-1))
                projected_similarities.append(projected)

        if not projected_similarities:
            # Preserve autograd + device/dtype of the inputs if any were passed,
            # otherwise fall back to a zero tied to the fusion parameters.
            if any_similarity is not None:
                return torch.zeros_like(any_similarity)
            return self.query.sum() * 0.0

        stacked_similarities = torch.stack(projected_similarities, dim=1)

        query_expanded = self.query.expand(stacked_similarities.size(0), -1, -1)
        attention_scores = torch.sum(stacked_similarities * query_expanded, dim=2)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)

        weighted_similarities = torch.sum(stacked_similarities * attention_weights, dim=1)

        fused_similarity = self.output_layer(weighted_similarities).squeeze(-1)

        return fused_similarity

