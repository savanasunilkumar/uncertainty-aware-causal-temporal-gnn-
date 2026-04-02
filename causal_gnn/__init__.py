"""Causal Temporal Graph Neural Network (UACT-GNN) for recommendation systems."""

__version__ = "1.0.0"

from .config import Config
from .models.uact_gnn import CausalTemporalGNN
from .training.trainer import RecommendationSystem

__all__ = [
    'Config',
    'CausalTemporalGNN',
    'RecommendationSystem',
]

