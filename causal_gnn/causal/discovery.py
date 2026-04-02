"""Causal discovery techniques for recommendation systems."""

import numpy as np
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, f as f_distribution

# For causal discovery
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    print("Warning: causal-learn not installed. Causal discovery will be disabled.")
    print("Install with: pip install causal-learn")
    CAUSAL_LEARN_AVAILABLE = False


class CausalGraphConstructor:
    """Implements causal discovery techniques for recommendation systems."""
    
    def __init__(self, config):
        self.config = config
        self.causal_method = config.causal_method
        self.significance_level = config.significance_level
        self.max_lag = config.max_lag
        self.min_causal_strength = config.min_causal_strength
        
    def compute_granger_causality(self, time_series, max_lag=None):
        if not CAUSAL_LEARN_AVAILABLE:
            print("Warning: causal-learn not available. Skipping Granger causality.")
            return np.zeros((len(time_series), len(time_series)))

        if max_lag is None:
            max_lag = self.max_lag
            
        nodes = list(time_series.keys())
        n_nodes = len(nodes)
        causal_matrix = np.zeros((n_nodes, n_nodes))

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

                X = []
                y = []

                for t in range(max_lag, len(target_series)):
                    y.append(target_series[t])

                    row = []
                    for lag in range(1, max_lag + 1):
                        row.append(target_series[t - lag])
                        row.append(source_series[t - lag])
                    X.append(row)
                
                X = np.array(X)
                y = np.array(y)
                
                if len(X) == 0:
                    continue

                X_target_only = X[:, ::2]
                if X_target_only.shape[1] > 0:
                    model_target_only = LinearRegression().fit(X_target_only, y)
                    mse_target_only = np.mean((model_target_only.predict(X_target_only) - y) ** 2)
                else:
                    mse_target_only = np.mean((y - np.mean(y))**2)

                model_both = LinearRegression().fit(X, y)
                mse_both = np.mean((model_both.predict(X) - y) ** 2)

                n_obs = len(y)
                df1 = max_lag
                df2 = n_obs - 2 * max_lag

                if mse_target_only > 1e-6 and mse_both > 1e-6 and df2 > 0:
                    f_stat = ((mse_target_only - mse_both) / df1) / (mse_both / df2)

                    if f_stat > 0:
                        p_value = f_distribution.sf(f_stat, df1, df2)
                    else:
                        p_value = 1.0

                    if p_value < self.significance_level:
                        partial_r2 = (mse_target_only - mse_both) / mse_target_only
                        causal_strength = min(1.0, max(0.0, partial_r2))
                        if causal_strength > self.min_causal_strength:
                            causal_matrix[i, j] = causal_strength
        
        return causal_matrix
    
    def compute_pc_algorithm(self, data):
        if not CAUSAL_LEARN_AVAILABLE:
            print("Warning: causal-learn not available. Skipping PC algorithm.")
            return np.zeros((data.shape[0], data.shape[0]))

        try:
            cg = pc(data, alpha=self.significance_level, indep_test="fisherz")

            n_nodes = data.shape[0]
            causal_matrix = np.zeros((n_nodes, n_nodes))

            G = cg.G
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if G.has_edge(i, j):
                        correlation, _ = pearsonr(data[i], data[j])
                        if not np.isnan(correlation):
                            causal_matrix[i, j] = abs(correlation)

            return causal_matrix
        except Exception as e:
            print(f"Error in PC algorithm: {e}. Falling back to correlation-based causality.")
            n_nodes = data.shape[0]
            causal_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        correlation, _ = pearsonr(data[i], data[j])
                        if not np.isnan(correlation) and abs(correlation) > self.min_causal_strength:
                            causal_matrix[i, j] = abs(correlation)
            return causal_matrix

    def compute_hybrid_causal_graph(self, interaction_data, node_features, edge_timestamps):
        time_series = self._extract_time_series(interaction_data, edge_timestamps)

        granger_matrix = self.compute_granger_causality(time_series)

        pc_matrix = self.compute_pc_algorithm(node_features)

        combined_matrix = 0.6 * granger_matrix + 0.4 * pc_matrix

        causal_edges = []
        edge_weights = []

        for i in range(combined_matrix.shape[0]):
            for j in range(combined_matrix.shape[1]):
                if i != j and combined_matrix[i, j] > self.min_causal_strength:
                    causal_edges.append((i, j))
                    edge_weights.append(combined_matrix[i, j])
        
        return causal_edges, edge_weights
    
    def _extract_time_series(self, interaction_data, edge_timestamps):
        """Extract time series data for each node from interactions."""
        node_time_series = {}
        time_step_interactions = defaultdict(list)

        min_time = edge_timestamps.min()
        max_time = edge_timestamps.max()
        time_range = max_time - min_time
        if time_range == 0:
            time_range = 1

        time_step_size = time_range / self.config.time_steps

        for i, timestamp in enumerate(edge_timestamps):
            time_step = int((timestamp - min_time) / time_step_size)
            time_step = min(time_step, self.config.time_steps - 1)
            
            user_idx = interaction_data[0, i]
            item_idx = interaction_data[1, i]
            
            time_step_interactions[time_step].append((user_idx, item_idx))

        n_nodes = interaction_data.max() + 1
        for node_idx in range(n_nodes):
            time_series = []
            
            for time_step in sorted(time_step_interactions.keys()):
                count = sum(
                    1 for user, item in time_step_interactions[time_step]
                    if user == node_idx or item == node_idx
                )
                time_series.append(count)
            
            node_time_series[node_idx] = time_series
        
        return node_time_series

