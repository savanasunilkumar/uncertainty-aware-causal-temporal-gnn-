"""Publication-ready evaluation metrics for uncertainty-aware recommendations."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PublicationMetrics:
    ndcg: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    precision: Dict[int, float] = field(default_factory=dict)
    hit_ratio: Dict[int, float] = field(default_factory=dict)
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    brier_score: float = 0.0
    coverage_at_90_accuracy: float = 0.0
    coverage_at_95_accuracy: float = 0.0
    auc_accuracy_coverage: float = 0.0
    uncertainty_error_correlation: float = 0.0
    auroc_uncertainty: float = 0.0
    calibration_bins: List[float] = field(default_factory=list)
    calibration_accuracy: List[float] = field(default_factory=list)
    calibration_confidence: List[float] = field(default_factory=list)
    calibration_counts: List[int] = field(default_factory=list)
    coverage_thresholds: List[float] = field(default_factory=list)
    accuracy_at_coverage: List[float] = field(default_factory=list)


def compute_calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)

        if mask.sum() > 0:
            bin_accuracies.append(accuracies[mask].mean())
            bin_confidences.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)

    return (
        np.array(bin_centers),
        np.array(bin_accuracies),
        np.array(bin_confidences),
        np.array(bin_counts)
    )


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_curve(
        confidences, accuracies, n_bins
    )

    total = sum(bin_counts)
    if total == 0:
        return 0.0

    ece = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        ece += (count / total) * abs(acc - conf)

    return ece


def compute_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_curve(
        confidences, accuracies, n_bins
    )

    mce = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if count > 0:
            mce = max(mce, abs(acc - conf))

    return mce


def compute_brier_score(
    confidences: np.ndarray,
    accuracies: np.ndarray
) -> float:
    return np.mean((confidences - accuracies) ** 2)


def compute_selective_prediction_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(-confidences)
    sorted_accuracies = accuracies[sorted_indices]

    coverages = []
    accuracies_at_coverage = []

    n = len(confidences)
    for i in range(1, n + 1, max(1, n // n_points)):
        coverage = i / n
        acc = sorted_accuracies[:i].mean()
        coverages.append(coverage)
        accuracies_at_coverage.append(acc)

    if coverages[-1] != 1.0:
        coverages.append(1.0)
        accuracies_at_coverage.append(sorted_accuracies.mean())

    return np.array(coverages), np.array(accuracies_at_coverage)


def compute_coverage_at_accuracy(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    target_accuracy: float = 0.9
) -> float:
    coverages, accs = compute_selective_prediction_curve(confidences, accuracies)

    for i in range(len(coverages) - 1, -1, -1):
        if accs[i] >= target_accuracy:
            return coverages[i]

    return 0.0


def compute_auc_accuracy_coverage(
    confidences: np.ndarray,
    accuracies: np.ndarray
) -> float:
    coverages, accs = compute_selective_prediction_curve(confidences, accuracies)
    auc = np.trapz(accs, coverages)
    return auc


def compute_uncertainty_error_correlation(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> float:
    if len(uncertainties) < 2:
        return 0.0

    corr = np.corrcoef(uncertainties, errors)[0, 1]

    if np.isnan(corr):
        return 0.0

    return corr


def compute_auroc_uncertainty(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> float:
    is_error = (errors > 0).astype(int)

    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return 0.5

    sorted_indices = np.argsort(-uncertainties)
    sorted_errors = is_error[sorted_indices]

    n_pos = is_error.sum()
    n_neg = len(is_error) - n_pos

    rank_sum = 0
    for i, is_err in enumerate(sorted_errors):
        if is_err:
            rank_sum += i + 1

    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return auroc


def generate_publication_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    uncertainties: np.ndarray,
    recommendation_metrics: Optional[Dict] = None,
    n_calibration_bins: int = 10
) -> PublicationMetrics:
    metrics = PublicationMetrics()

    if recommendation_metrics:
        metrics.ndcg = recommendation_metrics.get('ndcg', {})
        metrics.recall = recommendation_metrics.get('recall', {})
        metrics.precision = recommendation_metrics.get('precision', {})
        metrics.hit_ratio = recommendation_metrics.get('hit_ratio', {})

    accuracies = (predictions == labels).astype(float)
    errors = 1 - accuracies

    bin_centers, bin_accs, bin_confs, bin_counts = compute_calibration_curve(
        confidences, accuracies, n_calibration_bins
    )

    metrics.calibration_bins = bin_centers.tolist()
    metrics.calibration_accuracy = bin_accs.tolist()
    metrics.calibration_confidence = bin_confs.tolist()
    metrics.calibration_counts = [int(c) for c in bin_counts]

    metrics.expected_calibration_error = compute_ece(confidences, accuracies, n_calibration_bins)
    metrics.maximum_calibration_error = compute_mce(confidences, accuracies, n_calibration_bins)
    metrics.brier_score = compute_brier_score(confidences, accuracies)

    coverages, accs_at_cov = compute_selective_prediction_curve(confidences, accuracies)
    metrics.coverage_thresholds = coverages.tolist()
    metrics.accuracy_at_coverage = accs_at_cov.tolist()

    metrics.coverage_at_90_accuracy = compute_coverage_at_accuracy(confidences, accuracies, 0.9)
    metrics.coverage_at_95_accuracy = compute_coverage_at_accuracy(confidences, accuracies, 0.95)
    metrics.auc_accuracy_coverage = compute_auc_accuracy_coverage(confidences, accuracies)

    metrics.uncertainty_error_correlation = compute_uncertainty_error_correlation(uncertainties, errors)
    metrics.auroc_uncertainty = compute_auroc_uncertainty(uncertainties, errors)

    return metrics


def format_latex_table(metrics: PublicationMetrics, model_name: str = "Ours") -> str:
    latex = f"""
% Add to your LaTeX table:
{model_name} & {metrics.ndcg.get(10, 0):.4f} & {metrics.recall.get(10, 0):.4f} & """
    latex += f"{metrics.expected_calibration_error:.4f} & {metrics.coverage_at_90_accuracy:.2%} & "
    latex += f"{metrics.uncertainty_error_correlation:.3f} \\\\"

    return latex
