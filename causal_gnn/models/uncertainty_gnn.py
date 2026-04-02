"""Uncertainty-Aware Causal Temporal GNN for Recommendations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Tuple, Optional, Dict, List
import numpy as np

from ..causal.bayesian_discovery import (
    UncertaintyAwareCausalLayer,
    compute_recommendation_confidence,
    get_confident_recommendations,
)


class UncertaintyAwareCausalTemporalGNN(nn.Module):
    """Uncertainty-Aware Causal Temporal Graph Neural Network."""

    def __init__(self, config, metadata):
        super().__init__()
        self.config = config
        self.metadata = metadata

        self.embedding_dim = config.embedding_dim
        self.num_users = metadata['num_users']
        self.num_items = metadata['num_items']
        self.num_nodes = self.num_users + self.num_items
        self.num_layers = config.num_layers
        self.time_steps = config.time_steps
        self.dropout = config.dropout

        self.mc_dropout_samples = getattr(config, 'mc_dropout_samples', 10)
        self.uncertainty_weight = getattr(config, 'uncertainty_weight', 0.1)
        self.min_variance = getattr(config, 'min_variance', 1e-6)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        self.user_embedding_log_var = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding_log_var = nn.Embedding(self.num_items, self.embedding_dim)

        self.temporal_embedding = nn.Embedding(self.time_steps, self.embedding_dim)
        self.temporal_embedding_log_var = nn.Embedding(self.time_steps, self.embedding_dim)

        self.causal_embedding = nn.Embedding(self.num_nodes, self.embedding_dim)

        self.causal_layers = nn.ModuleList([
            UncertaintyAwareCausalLayer(
                self.embedding_dim,
                self.embedding_dim,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])

        self.use_layer_norm = getattr(config, 'use_layer_norm', True)
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)
            ])
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.temporal_attention = UncertainTemporalAttention(
            self.embedding_dim,
            num_heads=4,
            dropout=self.dropout
        )

        self.output_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.output_log_var = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.confidence_calibrator = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        self._init_embeddings()

        self.edge_index = None
        self.edge_timestamps = None
        self.time_indices = None
        self.edge_weight_mean = None
        self.edge_weight_var = None

    def _init_embeddings(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.temporal_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.causal_embedding.weight, mean=0, std=0.1)

        initial_log_var = getattr(self.config, 'initial_log_variance', -2.0)
        nn.init.constant_(self.user_embedding_log_var.weight, initial_log_var)
        nn.init.constant_(self.item_embedding_log_var.weight, initial_log_var)
        nn.init.constant_(self.temporal_embedding_log_var.weight, initial_log_var)

    def set_causal_graph(
        self,
        edge_weight_mean: torch.Tensor,
        edge_weight_var: torch.Tensor
    ):
        """Set the uncertain causal graph weights."""
        self.edge_weight_mean = edge_weight_mean
        self.edge_weight_var = edge_weight_var

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
        time_indices: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        user_emb_mean = self.user_embedding.weight
        item_emb_mean = self.item_embedding.weight
        all_emb_mean = torch.cat([user_emb_mean, item_emb_mean], dim=0)

        user_emb_var = torch.exp(self.user_embedding_log_var.weight) + self.min_variance
        item_emb_var = torch.exp(self.item_embedding_log_var.weight) + self.min_variance
        all_emb_var = torch.cat([user_emb_var, item_emb_var], dim=0)

        temporal_emb_mean = self.temporal_embedding(time_indices)
        temporal_emb_var = torch.exp(self.temporal_embedding_log_var(time_indices)) + self.min_variance

        all_emb_mean = all_emb_mean + temporal_emb_mean
        all_emb_var = all_emb_var + temporal_emb_var

        causal_emb = self.causal_embedding.weight
        all_emb_mean = all_emb_mean + causal_emb

        if self.edge_weight_mean is None:
            num_edges = edge_index.size(1)
            default_var = getattr(self.config, 'default_edge_weight_var', 0.01)
            edge_weight_mean = torch.ones(num_edges, device=edge_index.device)
            edge_weight_var = torch.ones(num_edges, device=edge_index.device) * default_var
        else:
            edge_weight_mean = self.edge_weight_mean
            edge_weight_var = self.edge_weight_var

        h_mean, h_var = all_emb_mean, all_emb_var
        for i, layer in enumerate(self.causal_layers):
            h_mean, h_var = layer(h_mean, edge_index, edge_weight_mean, edge_weight_var)
            if self.use_layer_norm:
                h_mean = self.layer_norms[i](h_mean)

        if self.use_layer_norm:
            h_mean = self.final_layer_norm(h_mean)

        h_mean, h_var = self.temporal_attention(
            h_mean, h_var, edge_index, edge_timestamps
        )

        out_mean = self.output_mean(h_mean)
        out_log_var = self.output_log_var(h_mean)
        out_var = torch.exp(out_log_var) + self.min_variance

        user_emb_mean = out_mean[:self.num_users]
        item_emb_mean = out_mean[self.num_users:]
        user_emb_var = out_var[:self.num_users]
        item_emb_var = out_var[self.num_users:]

        if return_uncertainty:
            confidence_input = torch.cat([out_mean, torch.sqrt(out_var)], dim=-1)
            confidence = self.confidence_calibrator(confidence_input)
            return out_mean, user_emb_mean, item_emb_mean, out_var, confidence
        else:
            return out_mean, user_emb_mean, item_emb_mean, None, None

    def forward_with_mc_dropout(
        self,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
        time_indices: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_samples is None:
            n_samples = self.mc_dropout_samples

        self.train()
        samples = []

        for _ in range(n_samples):
            out_mean, _, _, out_var, _ = self.forward(
                edge_index, edge_timestamps, time_indices, return_uncertainty=True
            )
            samples.append(out_mean.unsqueeze(0))

        samples = torch.cat(samples, dim=0)

        epistemic_mean = samples.mean(dim=0)
        epistemic_var = samples.var(dim=0)

        self.eval()
        _, _, _, aleatoric_var, _ = self.forward(
            edge_index, edge_timestamps, time_indices, return_uncertainty=True
        )

        total_var = epistemic_var + aleatoric_var

        return epistemic_mean, epistemic_var, aleatoric_var, total_var

    def recommend_items_with_uncertainty(
        self,
        user_indices: torch.Tensor,
        top_k: int = 10,
        excluded_items: Optional[Dict[int, List[int]]] = None,
        confidence_threshold: float = 0.5,
        use_mc_dropout: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()

        with torch.no_grad():
            if use_mc_dropout:
                mean_emb, epistemic_var, aleatoric_var, total_var = self.forward_with_mc_dropout(
                    self.edge_index, self.edge_timestamps, self.time_indices
                )
                user_emb_mean = mean_emb[:self.num_users]
                item_emb_mean = mean_emb[self.num_users:]
                user_emb_var = total_var[:self.num_users]
                item_emb_var = total_var[self.num_users:]
            else:
                _, user_emb_mean, item_emb_mean, out_var, _ = self.forward(
                    self.edge_index, self.edge_timestamps, self.time_indices
                )
                user_emb_var = out_var[:self.num_users]
                item_emb_var = out_var[self.num_users:]

            batch_user_mean = user_emb_mean[user_indices]
            batch_user_var = user_emb_var[user_indices]

            scores_mean, scores_var = compute_recommendation_confidence(
                batch_user_mean, batch_user_var,
                item_emb_mean, item_emb_var
            )

            if excluded_items is not None:
                for i, user_idx in enumerate(user_indices.cpu().tolist()):
                    if user_idx in excluded_items:
                        for item_idx in excluded_items[user_idx]:
                            if item_idx < scores_mean.size(1):
                                scores_mean[i, item_idx] = float('-inf')

            top_indices, top_scores, confident_flags = get_confident_recommendations(
                scores_mean, scores_var, top_k, confidence_threshold
            )

            batch_size = user_indices.size(0)
            scores_std = torch.sqrt(scores_var + 1e-6)
            confidence = scores_mean / (scores_std + scores_mean.abs() + 1e-6)
            confidence = torch.sigmoid(confidence * 2)
            top_confidence = torch.gather(confidence, 1, top_indices)

            uncertain_flags = ~confident_flags

        return top_indices, top_scores, top_confidence, uncertain_flags

    def recommend_items(
        self,
        user_indices: torch.Tensor,
        top_k: int = 10,
        excluded_items: Optional[Dict[int, List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        top_indices, top_scores, _, _ = self.recommend_items_with_uncertainty(
            user_indices, top_k, excluded_items
        )
        return top_indices, top_scores

    def predict_with_uncertainty(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
        time_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            _, user_emb_mean, item_emb_mean, out_var, _ = self.forward(
                edge_index, edge_timestamps, time_indices
            )
            user_emb_var = out_var[:self.num_users]
            item_emb_var = out_var[self.num_users:]

            u_mean = user_emb_mean[user_indices]
            u_var = user_emb_var[user_indices]
            i_mean = item_emb_mean[item_indices]
            i_var = item_emb_var[item_indices]

            scores = torch.sum(u_mean * i_mean, dim=1)

            score_var = torch.sum(
                u_var * i_mean**2 + i_var * u_mean**2 + u_var * i_var,
                dim=1
            )
            uncertainties = torch.sqrt(score_var + self.min_variance)

        return scores, uncertainties

    def compute_uncertainty_loss(
        self,
        pos_scores_mean: torch.Tensor,
        pos_scores_var: torch.Tensor,
        neg_scores_mean: torch.Tensor,
        neg_scores_var: torch.Tensor
    ) -> torch.Tensor:
        score_diff = pos_scores_mean - neg_scores_mean
        bpr_loss = -F.logsigmoid(score_diff).mean()

        pos_var_penalty = pos_scores_var.mean()
        neg_var_bonus = -torch.log(neg_scores_var + 1e-6).mean() * 0.1

        total_loss = bpr_loss + self.uncertainty_weight * (pos_var_penalty + neg_var_bonus)

        return total_loss


class UncertainTemporalAttention(nn.Module):
    """Temporal attention layer with uncertainty propagation."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.var_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x_mean: torch.Tensor,
        x_var: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x_mean)
        k = self.k_proj(x_mean)
        v = self.v_proj(x_mean)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out_mean = torch.matmul(attn_weights, v)
        out_mean = self.out_proj(out_mean)

        attn_weights_sq = attn_weights ** 2
        v_var = self.var_proj(x_var)
        out_var = torch.matmul(attn_weights_sq, v_var)

        return out_mean, out_var


class UncertaintyCalibrator(nn.Module):
    """Calibrates uncertainty estimates to be well-calibrated."""

    def __init__(self, initial_temperature: float = 1.0, embed_dim: int = None):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temperature]))
        self.recalibrator = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            calibrated = scores / self.temperature
            loss = F.binary_cross_entropy_with_logits(calibrated, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad = False

    def forward(
        self,
        scores_mean: torch.Tensor,
        scores_var: torch.Tensor
    ) -> torch.Tensor:
        calibrated_mean = scores_mean / self.temperature

        scores_std = torch.sqrt(scores_var + 1e-6)
        raw_confidence = calibrated_mean / (scores_std + 1e-6)

        features = torch.stack([calibrated_mean, scores_std], dim=-1)
        calibrated_confidence = self.recalibrator(features).squeeze(-1)

        return calibrated_confidence
