"""Bayesian causal discovery with uncertainty quantification."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from scipy.stats import f as f_distribution, invgamma, norm


@dataclass
class CausalEdgeDistribution:
    """Represents a distribution over causal edge strength."""
    mean: float
    variance: float
    confidence: float  # 1 - p_value or posterior probability

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    @property
    def lower_bound(self, alpha: float = 0.05) -> float:
        """Lower bound of credible interval."""
        return self.mean - norm.ppf(1 - alpha/2) * self.std

    @property
    def upper_bound(self, alpha: float = 0.05) -> float:
        """Upper bound of credible interval."""
        return self.mean + norm.ppf(1 - alpha/2) * self.std

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the posterior distribution."""
        return np.random.normal(self.mean, self.std, n_samples)

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Check if the causal relationship is significant."""
        return self.confidence >= threshold and self.mean > 0


@dataclass
class UncertainCausalGraph:
    """Represents a causal graph with uncertainty on edges."""
    edge_distributions: Dict[Tuple[int, int], CausalEdgeDistribution]
    num_nodes: int

    def get_mean_adjacency(self) -> np.ndarray:
        """Get adjacency matrix of mean causal strengths."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in self.edge_distributions.items():
            adj[i, j] = dist.mean
        return adj

    def get_variance_adjacency(self) -> np.ndarray:
        """Get adjacency matrix of causal strength variances."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in self.edge_distributions.items():
            adj[i, j] = dist.variance
        return adj

    def get_confidence_adjacency(self) -> np.ndarray:
        """Get adjacency matrix of confidence scores."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in self.edge_distributions.items():
            adj[i, j] = dist.confidence
        return adj

    def sample_adjacency(self) -> np.ndarray:
        """Sample an adjacency matrix from the posterior."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in self.edge_distributions.items():
            sampled = dist.sample(1)[0]
            adj[i, j] = max(0, sampled)  # Clip to non-negative
        return adj

    def get_high_confidence_edges(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """Get edges with high confidence."""
        edges = []
        for (i, j), dist in self.edge_distributions.items():
            if dist.confidence >= threshold:
                edges.append((i, j, dist.mean))
        return edges


class BayesianCausalGraphConstructor:
    """Bayesian causal discovery with uncertainty quantification."""

    def __init__(self, config):
        self.config = config
        self.significance_level = getattr(config, 'significance_level', 0.05)
        self.max_lag = getattr(config, 'max_lag', 3)
        self.min_causal_strength = getattr(config, 'min_causal_strength', 0.1)
        self.n_bootstrap = getattr(config, 'n_bootstrap_samples', 100)
        self.prior_precision = getattr(config, 'causal_prior_precision', 1.0)

    def compute_bayesian_granger_causality(
        self,
        time_series: Dict[int, List[float]],
        max_lag: Optional[int] = None
    ) -> UncertainCausalGraph:
        if max_lag is None:
            max_lag = self.max_lag

        nodes = list(time_series.keys())
        n_nodes = len(nodes)
        edge_distributions = {}

        scaler = StandardScaler()
        standardized_series = {}
        for node in nodes:
            if len(time_series[node]) > max_lag:
                standardized_series[node] = scaler.fit_transform(
                    np.array(time_series[node]).reshape(-1, 1)
                ).flatten()

        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i == j or source not in standardized_series or target not in standardized_series:
                    continue

                source_series = standardized_series[source]
                target_series = standardized_series[target]

                if len(source_series) <= max_lag or len(target_series) <= max_lag:
                    continue

                edge_dist = self._compute_bayesian_edge_strength(
                    source_series, target_series, max_lag
                )

                if edge_dist is not None and edge_dist.mean > self.min_causal_strength:
                    edge_distributions[(i, j)] = edge_dist

        return UncertainCausalGraph(
            edge_distributions=edge_distributions,
            num_nodes=n_nodes
        )

    def _compute_bayesian_edge_strength(
        self,
        source_series: np.ndarray,
        target_series: np.ndarray,
        max_lag: int
    ) -> Optional[CausalEdgeDistribution]:
        X_target_only = []
        X_both = []
        y = []

        for t in range(max_lag, len(target_series)):
            y.append(target_series[t])

            row_target = [target_series[t - lag] for lag in range(1, max_lag + 1)]
            X_target_only.append(row_target)

            row_both = []
            for lag in range(1, max_lag + 1):
                row_both.append(target_series[t - lag])
                row_both.append(source_series[t - lag])
            X_both.append(row_both)

        X_target_only = np.array(X_target_only)
        X_both = np.array(X_both)
        y = np.array(y)

        if len(y) < 2 * max_lag + 5:
            return None

        try:
            model_target_only = BayesianRidge(
                alpha_1=self.prior_precision,
                alpha_2=self.prior_precision,
                lambda_1=self.prior_precision,
                lambda_2=self.prior_precision,
                compute_score=True
            )
            model_target_only.fit(X_target_only, y)
            model_both = BayesianRidge(
                alpha_1=self.prior_precision,
                alpha_2=self.prior_precision,
                lambda_1=self.prior_precision,
                lambda_2=self.prior_precision,
                compute_score=True
            )
            model_both.fit(X_both, y)

            pred_target_only, std_target_only = model_target_only.predict(
                X_target_only, return_std=True
            )
            pred_both, std_both = model_both.predict(X_both, return_std=True)

            mse_target_only = np.mean((y - pred_target_only) ** 2)
            mse_both = np.mean((y - pred_both) ** 2)

            mse_var_target, mse_var_both = self._bootstrap_mse_variance(
                y, pred_target_only, pred_both
            )

            if mse_target_only > 1e-6:
                causal_strength_mean = (mse_target_only - mse_both) / mse_target_only
                causal_strength_mean = max(0, min(1, causal_strength_mean))

                df_dx = 1 / mse_target_only
                df_dy = mse_both / (mse_target_only ** 2)
                causal_strength_var = (df_dx ** 2) * mse_var_both + (df_dy ** 2) * mse_var_target
                causal_strength_var = max(1e-6, causal_strength_var)

                n_obs = len(y)
                df1 = max_lag
                df2 = n_obs - 2 * max_lag

                if df2 > 0 and mse_both > 1e-6:
                    f_stat = ((mse_target_only - mse_both) / df1) / (mse_both / df2)
                    if f_stat > 0:
                        p_value = f_distribution.sf(f_stat, df1, df2)
                        confidence = 1 - p_value
                    else:
                        confidence = 0.0
                else:
                    confidence = 0.0

                return CausalEdgeDistribution(
                    mean=causal_strength_mean,
                    variance=causal_strength_var,
                    confidence=confidence
                )

        except Exception as e:
            print(f"Error in Bayesian edge computation: {e}")
            return None

        return None

    def _bootstrap_mse_variance(
        self,
        y_true: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        n_bootstrap: int = 50
    ) -> Tuple[float, float]:
        """Estimate variance of MSE using bootstrap."""
        n = len(y_true)
        mse1_samples = []
        mse2_samples = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            mse1_samples.append(np.mean((y_true[idx] - pred1[idx]) ** 2))
            mse2_samples.append(np.mean((y_true[idx] - pred2[idx]) ** 2))

        return np.var(mse1_samples), np.var(mse2_samples)

    def compute_bootstrap_pc_algorithm(
        self,
        data: np.ndarray,
        n_bootstrap: Optional[int] = None
    ) -> UncertainCausalGraph:
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        n_samples, n_nodes = data.shape

        edge_counts = defaultdict(int)
        edge_strengths = defaultdict(list)

        for b in range(n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = data[idx]

            adj_matrix = self._run_pc_single(bootstrap_data)

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and adj_matrix[i, j] > 0:
                        edge_counts[(i, j)] += 1
                        edge_strengths[(i, j)].append(adj_matrix[i, j])

        edge_distributions = {}
        for (i, j), count in edge_counts.items():
            if count > 0:
                strengths = edge_strengths[(i, j)]
                all_strengths = strengths + [0.0] * (n_bootstrap - count)

                mean_strength = np.mean(all_strengths)
                var_strength = np.var(all_strengths)
                confidence = count / n_bootstrap

                if mean_strength > self.min_causal_strength:
                    edge_distributions[(i, j)] = CausalEdgeDistribution(
                        mean=mean_strength,
                        variance=var_strength,
                        confidence=confidence
                    )

        return UncertainCausalGraph(
            edge_distributions=edge_distributions,
            num_nodes=n_nodes
        )

    def _run_pc_single(self, data: np.ndarray) -> np.ndarray:
        from scipy.stats import pearsonr

        n_nodes = data.shape[1] if len(data.shape) > 1 else data.shape[0]
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    try:
                        if len(data.shape) > 1:
                            corr, p_value = pearsonr(data[:, i], data[:, j])
                        else:
                            corr, p_value = 0.0, 1.0

                        if p_value < self.significance_level and not np.isnan(corr):
                            adj_matrix[i, j] = abs(corr)
                    except Exception:
                        pass

        return adj_matrix

    def compute_hybrid_uncertain_causal_graph(
        self,
        interaction_data: torch.Tensor,
        node_features: np.ndarray,
        edge_timestamps: torch.Tensor
    ) -> UncertainCausalGraph:
        time_series = self._extract_time_series(interaction_data, edge_timestamps)

        granger_graph = self.compute_bayesian_granger_causality(time_series)

        pc_graph = self.compute_bootstrap_pc_algorithm(node_features)

        combined_distributions = {}
        all_edges = set(granger_graph.edge_distributions.keys()) | set(pc_graph.edge_distributions.keys())

        for edge in all_edges:
            granger_dist = granger_graph.edge_distributions.get(edge)
            pc_dist = pc_graph.edge_distributions.get(edge)

            if granger_dist is not None and pc_dist is not None:
                w1 = 1 / (granger_dist.variance + 1e-6)
                w2 = 1 / (pc_dist.variance + 1e-6)

                combined_mean = (w1 * granger_dist.mean + w2 * pc_dist.mean) / (w1 + w2)
                combined_var = 1 / (w1 + w2)
                combined_conf = max(granger_dist.confidence, pc_dist.confidence)

            elif granger_dist is not None:
                combined_mean = granger_dist.mean
                combined_var = granger_dist.variance
                combined_conf = granger_dist.confidence * 0.8

            else:
                combined_mean = pc_dist.mean
                combined_var = pc_dist.variance
                combined_conf = pc_dist.confidence * 0.8

            if combined_mean > self.min_causal_strength:
                combined_distributions[edge] = CausalEdgeDistribution(
                    mean=combined_mean,
                    variance=combined_var,
                    confidence=combined_conf
                )

        return UncertainCausalGraph(
            edge_distributions=combined_distributions,
            num_nodes=max(granger_graph.num_nodes, pc_graph.num_nodes)
        )

    def _extract_time_series(
        self,
        interaction_data: torch.Tensor,
        edge_timestamps: torch.Tensor
    ) -> Dict[int, List[float]]:
        """Extract time series data from interactions."""
        time_step_interactions = defaultdict(list)

        if isinstance(edge_timestamps, torch.Tensor):
            edge_timestamps = edge_timestamps.detach().cpu().numpy()
        if isinstance(interaction_data, torch.Tensor):
            interaction_data = interaction_data.detach().cpu().numpy()

        min_time = edge_timestamps.min()
        max_time = edge_timestamps.max()
        time_range = max_time - min_time
        if time_range == 0:
            time_range = 1

        time_steps = getattr(self.config, 'time_steps', 10)
        time_step_size = time_range / time_steps

        for i, timestamp in enumerate(edge_timestamps):
            time_step = int((timestamp - min_time) / time_step_size)
            time_step = min(time_step, time_steps - 1)

            user_idx = interaction_data[0, i] if len(interaction_data.shape) > 1 else i
            item_idx = interaction_data[1, i] if len(interaction_data.shape) > 1 else i

            time_step_interactions[time_step].append((user_idx, item_idx))

        # Create time series
        n_nodes = int(interaction_data.max()) + 1 if len(interaction_data) > 0 else 0
        node_time_series = {}

        for node_idx in range(min(n_nodes, 1000)):  # Limit for efficiency
            time_series = []
            for time_step in sorted(time_step_interactions.keys()):
                count = sum(
                    1 for user, item in time_step_interactions[time_step]
                    if user == node_idx or item == node_idx
                )
                time_series.append(float(count))
            node_time_series[node_idx] = time_series

        return node_time_series


class UncertaintyAwareCausalLayer(nn.Module):
    """GNN layer that propagates uncertainty through causal message passing."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_mean = nn.Linear(in_channels, out_channels)
        self.lin_log_var = nn.Linear(in_channels, out_channels)
        self.edge_mean_transform = nn.Linear(1, out_channels)
        self.edge_var_transform = nn.Linear(1, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_mean: torch.Tensor,
        edge_weight_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_mean = self.lin_mean(x)
        h_log_var = self.lin_log_var(x)
        h_var = torch.exp(h_log_var)

        row, col = edge_index

        msg_mean = h_mean[col] * edge_weight_mean.unsqueeze(-1)

        msg_var = (edge_weight_mean.unsqueeze(-1) ** 2) * h_var[col] + \
                  (h_mean[col] ** 2) * edge_weight_var.unsqueeze(-1)

        num_nodes = x.size(0)
        out_mean = torch.zeros(num_nodes, self.out_channels, device=x.device)
        out_var = torch.zeros(num_nodes, self.out_channels, device=x.device)

        out_mean.scatter_add_(0, row.unsqueeze(-1).expand_as(msg_mean), msg_mean)
        out_var.scatter_add_(0, row.unsqueeze(-1).expand_as(msg_var), msg_var)

        degree = torch.zeros(num_nodes, device=x.device)
        degree.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        degree = degree.clamp(min=1).unsqueeze(-1)

        out_mean = out_mean / degree
        out_var = out_var / (degree ** 2)

        out_mean = self.layer_norm(out_mean)
        out_mean = self.dropout(out_mean)

        return out_mean, out_var


def compute_recommendation_confidence(
    user_embedding_mean: torch.Tensor,
    user_embedding_var: torch.Tensor,
    item_embedding_mean: torch.Tensor,
    item_embedding_var: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores_mean = torch.matmul(user_embedding_mean, item_embedding_mean.t())

    scores_var = torch.matmul(user_embedding_var, item_embedding_mean.t() ** 2) + \
                 torch.matmul(user_embedding_mean ** 2, item_embedding_var.t()) + \
                 torch.matmul(user_embedding_var, item_embedding_var.t())

    return scores_mean, scores_var


def get_confident_recommendations(
    scores_mean: torch.Tensor,
    scores_var: torch.Tensor,
    top_k: int = 10,
    confidence_threshold: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores_std = torch.sqrt(scores_var + 1e-6)
    confidence = scores_mean / (scores_std + 1e-6)
    confidence = torch.sigmoid(confidence)

    top_scores, top_indices = torch.topk(scores_mean, top_k, dim=-1)

    batch_size = scores_mean.size(0)
    top_confidence = torch.gather(confidence, 1, top_indices)

    confident_flags = top_confidence >= confidence_threshold

    return top_indices, top_scores, confident_flags
