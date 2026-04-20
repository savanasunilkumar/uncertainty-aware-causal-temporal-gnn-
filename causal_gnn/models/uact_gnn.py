"""Enhanced Universal Adaptive Causal Temporal GNN model with PyTorch Geometric.

See ``causal_gnn.causal.discovery`` for the caveat on "causal" naming: the
edges used for message passing are a temporal co-activity graph, not a
counterfactual causal graph.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint

from ..causal.discovery import CausalGraphConstructor
from .layers import CausalGNNLayer, TemporalAttentionLayer, GraphSAGELayer

logger = logging.getLogger(__name__)


class CausalTemporalGNN(nn.Module):
    """Causal Temporal Graph Neural Network for recommendations with PyTorch Geometric."""

    def __init__(self, config, metadata):
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.num_users = metadata['num_users']
        self.num_items = metadata['num_items']

        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.time_steps = config.time_steps
        self.dropout = config.dropout
        self.causal_strength = config.causal_strength

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        self.temporal_embedding = nn.Embedding(self.time_steps, self.embedding_dim)

        self.causal_embedding = nn.Embedding(
            self.num_users + self.num_items, self.embedding_dim
        )

        # Attention uses batch_first so inputs can be (nodes, time_steps, dim)
        # with the time axis as the attention sequence — which is the intent,
        # not attention over nodes.
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True,
        )

        encoder_layers = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)

        self.causal_layers = nn.ModuleList([
            CausalGNNLayer(
                self.embedding_dim,
                self.embedding_dim,
                dropout=self.dropout,
                use_edge_weight=True,
            ) for _ in range(self.num_layers)
        ])

        self.temporal_gnn_layers = nn.ModuleList([
            TemporalAttentionLayer(
                self.embedding_dim,
                self.embedding_dim,
                heads=4,
                dropout=self.dropout,
            ) for _ in range(2)
        ])

        self.sage_layers = nn.ModuleList([
            GraphSAGELayer(
                self.embedding_dim,
                self.embedding_dim,
                normalize=True,
            ) for _ in range(self.num_layers)
        ])

        self.use_gradient_checkpointing = getattr(
            config, 'use_gradient_checkpointing', False
        )

        # Buffers set by the trainer after precomputation.
        # Registered as non-persistent buffers so they follow .to(device).
        self.register_buffer(
            'cached_causal_edge_index',
            torch.empty(2, 0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'cached_causal_edge_weights',
            torch.empty(0, dtype=torch.float),
            persistent=False,
        )
        self._has_cached_causal_graph = False

        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 128),
        )

        self.output_layer = nn.Linear(self.embedding_dim, 1)

        self.causal_constructor = CausalGraphConstructor(config)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.temporal_embedding.weight, std=0.1)
        nn.init.normal_(self.causal_embedding.weight, std=0.1)

        self.device = torch.device(config.device)

    def set_causal_graph(
        self,
        causal_edge_index: torch.Tensor,
        causal_edge_weights: torch.Tensor = None,
    ) -> None:
        """Install a precomputed causal edge set on the model.

        Called once by the trainer after ``create_graph``. This avoids
        recomputing the (expensive) Granger F-test on every forward pass
        and avoids CPU round-trips inside the training loop.
        """
        self.cached_causal_edge_index = causal_edge_index.to(
            self.cached_causal_edge_index.device
        )
        if causal_edge_weights is None:
            causal_edge_weights = torch.ones(
                causal_edge_index.size(1),
                dtype=torch.float,
                device=causal_edge_index.device,
            )
        self.cached_causal_edge_weights = causal_edge_weights.to(
            self.cached_causal_edge_weights.device
        )
        self._has_cached_causal_graph = True

    def _get_ego_embeddings(self):
        """Get the initial embeddings for all users and items."""
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def _get_temporal_embeddings(self, time_indices):
        return self.temporal_embedding(time_indices)

    def _get_causal_embeddings(self):
        return self.causal_embedding.weight

    def _compute_causal_graph(self, edge_index, edge_timestamps, node_features):
        """Return (causal_edge_index, causal_edge_weights).

        If a causal graph was precomputed via ``set_causal_graph`` we use it.
        Otherwise, for ``causal_method == 'simple'`` we pass the interaction
        edges straight through with unit weights; for ``'advanced'`` we run
        the vectorized Granger constructor *once* and cache the result.

        The previous per-batch CPU round-trip and O(E^2) Python fallback
        have been removed — neither is acceptable inside a training loop.
        """
        if self._has_cached_causal_graph:
            return self.cached_causal_edge_index, self.cached_causal_edge_weights

        if self.config.causal_method == 'advanced':
            interaction_data = edge_index.detach().cpu().numpy()
            edge_timestamps_np = edge_timestamps.detach().cpu().numpy()
            node_features_np = node_features.detach().cpu().numpy()
            causal_edges, edge_weights = self.causal_constructor.compute_hybrid_causal_graph(
                interaction_data, node_features_np, edge_timestamps_np
            )
            if causal_edges:
                ci = torch.tensor(causal_edges, dtype=torch.long, device=edge_index.device).t().contiguous()
                cw = torch.tensor(edge_weights, dtype=torch.float, device=edge_index.device)
            else:
                ci = edge_index
                cw = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
            self.set_causal_graph(ci, cw)
            return ci, cw

        # 'simple': pass interaction edges through with unit weights.
        return edge_index, torch.ones(
            edge_index.size(1), dtype=torch.float, device=edge_index.device
        )

    def _propagate_causal_information(self, node_features, causal_edge_index, causal_edge_weights=None):
        causal_embeddings = self._get_causal_embeddings()
        combined_features = node_features + self.causal_strength * causal_embeddings

        for layer in self.causal_layers:
            if self.use_gradient_checkpointing and self.training:
                combined_features = checkpoint(
                    layer,
                    combined_features,
                    causal_edge_index,
                    causal_edge_weights,
                    use_reentrant=False,
                )
            else:
                combined_features = layer(
                    combined_features,
                    causal_edge_index,
                    edge_weight=causal_edge_weights,
                )
            combined_features = F.relu(combined_features)

        return combined_features

    def _apply_temporal_attention(self, node_features, time_indices):
        """Temporal self-attention across time buckets, per node.

        Input shape convention: (nodes, time_steps, dim) with batch_first=True,
        so attention runs across the time_steps axis (as intended), not across
        nodes.
        """
        time_emb = self._get_temporal_embeddings(time_indices)  # (nodes, dim)
        t_indices = torch.arange(self.time_steps, device=node_features.device)
        all_time_emb = self.temporal_embedding(t_indices)  # (T, dim)

        node_features_expanded = node_features.unsqueeze(1)  # (N, 1, D)
        time_features = all_time_emb.unsqueeze(0)  # (1, T, D)
        temporal_features = node_features_expanded + time_features  # (N, T, D)

        attended, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        # Weight the per-time features by whether that time step is this
        # node's most-recent activity bucket.
        t_range = torch.arange(self.time_steps, device=node_features.device)
        match = (t_range.unsqueeze(0) == time_indices.unsqueeze(1)).float()  # (N, T)
        match = match / match.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregated = (attended * match.unsqueeze(-1)).sum(dim=1)
        aggregated = aggregated + time_emb
        return aggregated

    def _contrastive_loss(self, embeddings, positive_pairs, negative_pairs, temperature=0.1):
        embeddings = F.normalize(embeddings, dim=1)
        projections = self.contrastive_projection(embeddings)
        projections = F.normalize(projections, dim=1)

        similarity_matrix = torch.matmul(projections, projections.t()) / temperature
        labels = torch.arange(embeddings.size(0), device=embeddings.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def forward(self, edge_index, edge_timestamps, time_indices=None):
        """Forward pass of the UACT-GNN model."""
        all_embeddings = self._get_ego_embeddings()

        causal_edge_index, causal_edge_weights = self._compute_causal_graph(
            edge_index, edge_timestamps, all_embeddings
        )

        if time_indices is None:
            time_indices = torch.zeros(
                all_embeddings.size(0), dtype=torch.long, device=edge_index.device
            )

        temporal_features = self._apply_temporal_attention(all_embeddings, time_indices)
        all_embeddings = all_embeddings + temporal_features

        causal_features = self._propagate_causal_information(
            all_embeddings, causal_edge_index, causal_edge_weights
        )
        all_embeddings = all_embeddings + causal_features

        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]
        return all_embeddings, user_embeddings, item_embeddings

    def get_user_embeddings(self):
        _, user_embeddings, _ = self.forward(
            self.edge_index, self.edge_timestamps, self.time_indices
        )
        return user_embeddings

    def get_item_embeddings(self):
        _, _, item_embeddings = self.forward(
            self.edge_index, self.edge_timestamps, self.time_indices
        )
        return item_embeddings

    def predict(self, user_indices, item_indices):
        _, user_embeddings, item_embeddings = self.forward(
            self.edge_index, self.edge_timestamps, self.time_indices
        )
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        return torch.sum(user_emb * item_emb, dim=1)

    def recommend_items(self, user_indices, top_k=10, excluded_items=None):
        _, user_embeddings, item_embeddings = self.forward(
            self.edge_index, self.edge_timestamps, self.time_indices
        )
        user_emb = user_embeddings[user_indices]
        scores = torch.matmul(user_emb, item_embeddings.t())

        if excluded_items is not None:
            for i, user_idx in enumerate(user_indices.tolist()):
                if user_idx in excluded_items:
                    scores[i, excluded_items[user_idx]] = -float('inf')

        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        return top_indices, top_scores
