"""Temporal co-activity graph construction for recommendation systems.

NOTE on naming
--------------
The public API still uses the word "causal" for backward compatibility,
but what this module actually computes is a *temporal co-activity graph*:
a pairwise score on node activity time series using Granger's F-test.
Granger causality is a test of *predictive* precedence, not of causation;
on user/item activity counts, it mostly picks up co-popularity and shared
temporal trends. Treat the resulting edges as a data-driven regularizer
that encodes "when node A was active, node B's activity changed in a
way the data predicts," not as ground-truth causal structure.

References:
- Luo et al. (2024), "A Survey on Causal Inference for Recommendation"
  (arXiv:2303.11666) — why "causal" in recsys needs IV/backdoor/uplift
  style procedures, not Granger on aggregate counts.
- The repo previously applied the PC algorithm to node *feature* rows
  using Pearson correlation as edge strength; that is a misuse
  (PC expects i.i.d. samples of variables) and has been removed.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import f as f_distribution

logger = logging.getLogger(__name__)

try:
    from causallearn.search.ConstraintBased.PC import pc  # noqa: F401
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False


class CausalGraphConstructor:
    """Builds a temporal co-activity adjacency over active nodes.

    Despite the name (kept for backward compatibility), the edges are *not*
    causal in the counterfactual sense; see module docstring.
    """

    def __init__(self, config):
        self.config = config
        self.causal_method = getattr(config, 'causal_method', 'advanced')
        self.significance_level = getattr(config, 'significance_level', 0.05)
        self.max_lag = getattr(config, 'max_lag', 3)
        self.min_causal_strength = getattr(config, 'min_causal_strength', 0.05)
        self.max_nodes = getattr(config, 'max_causal_nodes', 512)

    def _extract_time_series(
        self,
        interaction_data: np.ndarray,
        edge_timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized per-node time-bucketed interaction counts.

        Returns
        -------
        series : (n_nodes, T) int32 array of activity counts per time bucket.
        node_totals : (n_nodes,) int32 array of total interaction counts.
        """
        T = int(self.config.time_steps)
        ts = np.asarray(edge_timestamps, dtype=np.float64)
        if ts.size == 0:
            return np.zeros((0, T), dtype=np.int32), np.zeros(0, dtype=np.int32)

        min_t = ts.min()
        max_t = ts.max()
        trange = max(float(max_t - min_t), 1.0)
        step = trange / float(T)
        bucket = np.clip(((ts - min_t) / step).astype(np.int64), 0, T - 1)

        src = np.asarray(interaction_data[0], dtype=np.int64)
        dst = np.asarray(interaction_data[1], dtype=np.int64)
        n_nodes = int(max(src.max(), dst.max()) + 1) if src.size else 0

        series = np.zeros((n_nodes, T), dtype=np.int32)
        np.add.at(series, (src, bucket), 1)
        np.add.at(series, (dst, bucket), 1)
        node_totals = series.sum(axis=1).astype(np.int32)
        return series, node_totals

    def compute_granger_causality(
        self,
        series: np.ndarray,
        max_lag: int = None,
    ) -> np.ndarray:
        """Vectorized pairwise Granger F-test on activity time series.

        For each pair (i, j) we compare:
          reduced:  y_t = sum_l a_l y_{t-l}                    (target-only)
          full:     y_t = sum_l a_l y_{t-l} + sum_l b_l x_{t-l} (target + source)
        and return an F-statistic-derived strength in [0, 1] thresholded at
        ``min_causal_strength`` and filtered by ``significance_level``.

        This is an O(N^2 * T) numpy implementation (no Python inner loops).
        """
        if max_lag is None:
            max_lag = self.max_lag
        if series.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        n_nodes, T = series.shape
        if T <= 2 * max_lag + 2:
            return np.zeros((n_nodes, n_nodes), dtype=np.float32)

        # Standardize each node's series.
        x = series.astype(np.float64)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - mu) / sd

        # Build lag matrices: y = values at t in [max_lag, T), shape (n_nodes, T-max_lag)
        # Y_lag[i, l, t] = x[i, t - (l+1)] for l in [0, max_lag)
        n_obs = T - max_lag
        lags = np.stack(
            [x[:, max_lag - l - 1:T - l - 1] for l in range(max_lag)],
            axis=1,
        )  # (n_nodes, max_lag, n_obs)
        y = x[:, max_lag:]  # (n_nodes, n_obs)

        # Reduced OLS (target-only) per node: design = [y lags, 1]
        reduced_design = np.concatenate(
            [np.transpose(lags, (0, 2, 1)), np.ones((n_nodes, n_obs, 1))],
            axis=-1,
        )  # (n_nodes, n_obs, max_lag+1)
        # Solve per-node reduced model: beta_i = (X_i^T X_i)^-1 X_i^T y_i
        # Use np.linalg.lstsq in a loop would be Python; instead batched via einsum.
        xtx_red = np.einsum('nij,nik->njk', reduced_design, reduced_design)
        xty_red = np.einsum('nij,ni->nj', reduced_design, y)
        try:
            beta_red = np.linalg.solve(
                xtx_red + 1e-8 * np.eye(max_lag + 1)[None, :, :],
                xty_red[:, :, None],
            )[:, :, 0]
        except np.linalg.LinAlgError:
            return np.zeros((n_nodes, n_nodes), dtype=np.float32)
        resid_red = y - np.einsum('nij,nj->ni', reduced_design, beta_red)
        rss_red = (resid_red ** 2).sum(axis=1) + 1e-12  # (n_nodes,)

        # Full model adds source lags. To keep memory bounded, restrict Granger
        # to the top-K most active nodes (by total activity). Others become
        # zero-strength.
        if n_nodes > self.max_nodes:
            totals = series.sum(axis=1)
            keep = np.argpartition(-totals, self.max_nodes)[:self.max_nodes]
            keep.sort()
        else:
            keep = np.arange(n_nodes)
        k = keep.size
        if k == 0:
            return np.zeros((n_nodes, n_nodes), dtype=np.float32)

        lags_k = lags[keep]       # (k, max_lag, n_obs)
        y_k = y[keep]             # (k, n_obs)
        rss_red_k = rss_red[keep] # (k,)

        # For target j, design = [y_j lags, x_i lags, 1]; run i over k candidates.
        # Solve full-model RSS for all (i, j) pairs in vectorized form.
        # Shape construction: full design for pair (i, j) has 2*max_lag + 1 columns.
        # We assemble an (k_j, k_i, n_obs, 2L+1) tensor. With k=max_nodes=512 and L=3
        # this is 512*512*~T*7 floats — capped by T.
        L = max_lag
        # y-lags: (k_j, n_obs, L) — repeated over source axis below
        y_lag_mat = np.transpose(lags_k, (0, 2, 1))  # (k, n_obs, L)
        ones = np.ones((k, 1, n_obs, 1), dtype=np.float64)

        # Source lags tiled over target: (k_j=1, k_i, n_obs, L)
        src_lag_mat = np.transpose(lags_k, (0, 2, 1))[None, :, :, :]

        # Target lags tiled over source: (k_j, k_i=1, n_obs, L)
        tgt_lag_mat = y_lag_mat[:, None, :, :]

        # Broadcast to (k_j, k_i, n_obs, L) each, then concat on last axis.
        # Intentionally not materializing an (k_j, k_i, n_obs, 2L+1) float64 tensor
        # for very large k: fall back to a per-target loop when it would exceed
        # ~1 GiB.
        approx_bytes = k * k * n_obs * (2 * L + 1) * 8
        f_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        strength_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        def _update_from_chunk(j_idx_global, i_idx_global, rss_full):
            rss_red_val = rss_red[j_idx_global]
            df_num = L
            df_den = n_obs - (2 * L + 1)
            if df_den <= 0 or rss_red_val <= 0:
                return
            num = (rss_red_val - rss_full) / df_num
            den = np.maximum(rss_full, 1e-12) / df_den
            f_stat = np.maximum(num / den, 0.0)
            p = f_distribution.sf(f_stat, df_num, df_den)
            strength = np.clip(
                (rss_red_val - rss_full) / rss_red_val,
                0.0,
                1.0,
            )
            mask = (p < self.significance_level) & (strength > self.min_causal_strength)
            f_matrix[j_idx_global, i_idx_global] = np.where(mask, f_stat, 0.0).astype(
                np.float32
            )
            strength_matrix[j_idx_global, i_idx_global] = np.where(
                mask, strength, 0.0
            ).astype(np.float32)

        if approx_bytes > 1_000_000_000:
            # Per-target loop; still fully vectorized over source.
            for jj_local in range(k):
                jj_global = int(keep[jj_local])
                # Design for all i at this target j:
                # (k_i, n_obs, 2L+1) = [y_j lags, x_i lags, 1]
                y_lags_j = np.broadcast_to(
                    y_lag_mat[jj_local][None, :, :], (k, n_obs, L)
                )
                x_lags_i = np.transpose(lags_k, (0, 2, 1))  # (k, n_obs, L)
                ones_i = np.ones((k, n_obs, 1), dtype=np.float64)
                design = np.concatenate([y_lags_j, x_lags_i, ones_i], axis=-1)
                xtx = np.einsum('kij,kil->kjl', design, design)
                y_tgt = np.broadcast_to(y_k[jj_local][None, :], (k, n_obs))
                xty = np.einsum('kij,ki->kj', design, y_tgt)
                try:
                    beta = np.linalg.solve(
                        xtx + 1e-8 * np.eye(2 * L + 1)[None, :, :],
                        xty[:, :, None],
                    )[:, :, 0]
                except np.linalg.LinAlgError:
                    continue
                resid = y_tgt - np.einsum('kij,kj->ki', design, beta)
                rss_full = (resid ** 2).sum(axis=1)
                _update_from_chunk(jj_global, keep, rss_full)
        else:
            # Full (k_j, k_i, n_obs, 2L+1) tensor: one big solve.
            tgt_lag_bc = np.broadcast_to(tgt_lag_mat, (k, k, n_obs, L))
            src_lag_bc = np.broadcast_to(src_lag_mat, (k, k, n_obs, L))
            ones_bc = np.ones((k, k, n_obs, 1), dtype=np.float64)
            design = np.concatenate([tgt_lag_bc, src_lag_bc, ones_bc], axis=-1)
            xtx = np.einsum('jinl,jinm->jilm', design, design)
            y_tgt = np.broadcast_to(y_k[:, None, :], (k, k, n_obs))
            xty = np.einsum('jinl,jin->jil', design, y_tgt)
            try:
                beta = np.linalg.solve(
                    xtx + 1e-8 * np.eye(2 * L + 1)[None, None, :, :],
                    xty[:, :, :, None],
                )[:, :, :, 0]
            except np.linalg.LinAlgError:
                return strength_matrix
            resid = y_tgt - np.einsum('jinl,jil->jin', design, beta)
            rss_full = (resid ** 2).sum(axis=-1)  # (k, k)
            for jj_local in range(k):
                _update_from_chunk(int(keep[jj_local]), keep, rss_full[jj_local])

        np.fill_diagonal(strength_matrix, 0.0)
        return strength_matrix

    def compute_pc_algorithm(self, data: np.ndarray) -> np.ndarray:
        """Deprecated. Kept as a stub that returns a zero matrix with a warning.

        Previously this ran PC on node-feature rows using Pearson correlation
        as the edge strength. PC expects i.i.d. samples of variables; passing
        embedding rows is a misuse. If you want actual constraint-based
        structure discovery, run it on a properly-shaped variable matrix at
        a higher level in your pipeline.
        """
        logger.warning(
            "compute_pc_algorithm is deprecated and returns zeros; "
            "prior implementation was a misuse of the PC algorithm."
        )
        n = int(data.shape[0]) if data.size else 0
        return np.zeros((n, n), dtype=np.float32)

    def compute_hybrid_causal_graph(
        self,
        interaction_data: np.ndarray,
        node_features: np.ndarray,
        edge_timestamps: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build the co-activity edge set used by the GNN.

        The hybrid was previously 0.6 * Granger + 0.4 * PC(node features).
        PC was misapplied; this now returns Granger-only edges thresholded at
        ``min_causal_strength``. See module docstring.
        """
        if not CAUSAL_LEARN_AVAILABLE and self.causal_method == 'pc':
            raise ImportError(
                "causal_method='pc' requires `causal-learn`. "
                "Install with `pip install causal-learn` or use "
                "causal_method='advanced' (Granger) / 'simple' (identity)."
            )

        series, _ = self._extract_time_series(interaction_data, edge_timestamps)
        strength = self.compute_granger_causality(series)

        mask = strength > self.min_causal_strength
        src_idx, dst_idx = np.nonzero(mask)
        edges = list(zip(src_idx.tolist(), dst_idx.tolist()))
        weights = strength[src_idx, dst_idx].astype(np.float32).tolist()
        return edges, weights
