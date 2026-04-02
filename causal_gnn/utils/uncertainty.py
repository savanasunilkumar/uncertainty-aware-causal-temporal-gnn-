"""Uncertainty quantification utilities for recommendations."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class UncertaintyMetrics:
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    negative_log_likelihood: float
    mean_confidence: float
    mean_uncertainty: float
    uncertainty_correlation: float


def compute_calibration_metrics(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10
) -> UncertaintyMetrics:
    confidences = 1 / (1 + uncertainties)

    if predictions.ndim > 1:
        correct = (predictions.argmax(axis=1) == ground_truth).astype(float)
    else:
        correct = (predictions > 0.5) == ground_truth
        correct = correct.astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        in_bin = bin_indices == i
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = correct[in_bin].mean()
            bin_size = in_bin.sum() / len(confidences)

            calibration_error = abs(bin_accuracy - bin_confidence)
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)

    if predictions.ndim > 1:
        brier = np.mean((predictions.max(axis=1) - correct) ** 2)
    else:
        brier = np.mean((predictions - correct) ** 2)

    eps = 1e-7
    if predictions.ndim > 1:
        probs = predictions[np.arange(len(ground_truth)), ground_truth]
    else:
        probs = np.where(ground_truth == 1, predictions, 1 - predictions)
    nll = -np.mean(np.log(np.clip(probs, eps, 1 - eps)))

    errors = 1 - correct
    correlation = np.corrcoef(uncertainties, errors)[0, 1]

    return UncertaintyMetrics(
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        brier_score=brier,
        negative_log_likelihood=nll,
        mean_confidence=float(confidences.mean()),
        mean_uncertainty=float(uncertainties.mean()),
        uncertainty_correlation=correlation if not np.isnan(correlation) else 0.0
    )


def temperature_scaling_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_iterations: int = 100,
    lr: float = 0.01
) -> float:
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=n_iterations)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature.item()


def compute_prediction_intervals(
    mean: np.ndarray,
    variance: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    std = np.sqrt(variance)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    lower = mean - z_score * std
    upper = mean + z_score * std

    return lower, upper


def selective_prediction(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    selection_mask = uncertainties <= threshold
    coverage = selection_mask.mean()

    selected_predictions = predictions.copy()
    selected_predictions[~selection_mask] = np.nan

    return selected_predictions, selection_mask, coverage


def uncertainty_decomposition(
    mc_samples: np.ndarray
) -> Dict[str, np.ndarray]:
    mean_prediction = mc_samples.mean(axis=0)
    epistemic = mc_samples.var(axis=0)

    if mc_samples.shape[-1] > 1:
        probs = np.exp(mc_samples) / np.exp(mc_samples).sum(axis=-1, keepdims=True)
        mean_probs = probs.mean(axis=0)
        total = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=-1)
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        aleatoric = entropies.mean(axis=0)
    else:
        total = epistemic
        aleatoric = np.zeros_like(epistemic)

    return {
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total': total,
        'mean': mean_prediction
    }


def compute_recommendation_uncertainty_metrics(
    user_embeddings_mean: torch.Tensor,
    user_embeddings_var: torch.Tensor,
    item_embeddings_mean: torch.Tensor,
    item_embeddings_var: torch.Tensor,
    interactions: List[Tuple[int, int, float]]
) -> Dict[str, float]:
    metrics = {}

    user_uncertainty = user_embeddings_var.mean(dim=-1)
    metrics['avg_user_uncertainty'] = user_uncertainty.mean().item()
    metrics['max_user_uncertainty'] = user_uncertainty.max().item()

    item_uncertainty = item_embeddings_var.mean(dim=-1)
    metrics['avg_item_uncertainty'] = item_uncertainty.mean().item()
    metrics['max_item_uncertainty'] = item_uncertainty.max().item()

    if interactions:
        score_uncertainties = []
        for user_idx, item_idx, _ in interactions:
            u_mean = user_embeddings_mean[user_idx]
            u_var = user_embeddings_var[user_idx]
            i_mean = item_embeddings_mean[item_idx]
            i_var = item_embeddings_var[item_idx]

            score_var = (u_var * i_mean**2 + i_var * u_mean**2 + u_var * i_var).sum()
            score_uncertainties.append(score_var.item())

        metrics['avg_interaction_uncertainty'] = np.mean(score_uncertainties)
        metrics['std_interaction_uncertainty'] = np.std(score_uncertainties)

    return metrics


class UncertaintyAwareEvaluator:
    """Evaluator that considers uncertainty in recommendation metrics."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def evaluate_with_uncertainty(
        self,
        recommendations: Dict[int, List[Tuple[int, float, float]]],
        ground_truth: Dict[int, set],
        k: int = 10
    ) -> Dict[str, float]:
        metrics = {
            'precision': [],
            'recall': [],
            'confident_precision': [],
            'confident_coverage': [],
            'uncertainty_correlation': [],
        }

        for user, recs in recommendations.items():
            if user not in ground_truth:
                continue

            true_items = ground_truth[user]
            top_k = recs[:k]

            rec_items = [r[0] for r in top_k]
            hits = len(set(rec_items) & true_items)
            metrics['precision'].append(hits / k)
            metrics['recall'].append(hits / len(true_items) if true_items else 0)

            confident_recs = [r for r in top_k if r[2] >= self.confidence_threshold]
            if confident_recs:
                confident_items = [r[0] for r in confident_recs]
                confident_hits = len(set(confident_items) & true_items)
                metrics['confident_precision'].append(confident_hits / len(confident_recs))
            else:
                metrics['confident_precision'].append(0.0)

            metrics['confident_coverage'].append(len(confident_recs) / k)

            if len(top_k) > 1:
                correct = [1 if r[0] in true_items else 0 for r in top_k]
                confidence = [r[2] for r in top_k]
                if np.std(correct) > 0 and np.std(confidence) > 0:
                    corr = np.corrcoef(confidence, correct)[0, 1]
                    if not np.isnan(corr):
                        metrics['uncertainty_correlation'].append(corr)

        return {
            'precision@k': np.mean(metrics['precision']),
            'recall@k': np.mean(metrics['recall']),
            'confident_precision@k': np.mean(metrics['confident_precision']),
            'confident_coverage': np.mean(metrics['confident_coverage']),
            'uncertainty_calibration': np.mean(metrics['uncertainty_correlation']) if metrics['uncertainty_correlation'] else 0.0,
        }


def should_abstain(
    uncertainty: float,
    threshold: float,
    min_confidence: float = 0.3
) -> Tuple[bool, str]:
    confidence = 1 / (1 + uncertainty)

    if uncertainty > threshold:
        return True, f"High uncertainty ({uncertainty:.3f}). Need more data."
    elif confidence < min_confidence:
        return True, f"Low confidence ({confidence:.3f}). Prediction unreliable."
    else:
        return False, f"Confident prediction (confidence: {confidence:.3f})"
