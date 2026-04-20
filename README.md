# UACT-GNN: Uncertainty-Aware Causal Temporal Graph Neural Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-oriented recommendation system that combines **temporal
co-activity graphs**, **temporal attention**, **graph neural networks**,
and **uncertainty quantification**. Measured on MovieLens-100K; scaling
to larger datasets is an open item (see "Known Limitations").

> ⚠️ A note on "causal" in this project's name and APIs.
> The graph that message passing runs over is built from pairwise
> Granger F-tests on per-node time-bucketed activity counts. Granger
> is a test of *predictive precedence* — not counterfactual causation —
> and on aggregate activity signals it primarily captures co-popularity
> and shared temporal trends. Treat "causal edges" here as a learned
> co-activity regularizer, not as ground-truth causal structure. See
> `causal_gnn/causal/discovery.py` for details.

## Key Features

| Feature | Description |
|---------|-------------|
| **Temporal co-activity graph** | Vectorized Granger F-test over time-bucketed node activity; built once, cached on the model. |
| **Temporal modeling** | Transformer-based attention over per-node time buckets. |
| **Uncertainty quantification** | Optional Bayesian embeddings + MC dropout + calibration (ECE / Brier / NLL). |
| **Multi-modal cold start** | Optional text (DistilBERT) / image (ResNet18) / numeric / categorical fusion. |
| **Baselines for benchmarking** | Popular, BPR, NCF, LightGCN share the same `fit/predict` API. |

## Installation

```bash
git clone https://github.com/savanasunilkumar/uncertainty-aware-causal-temporal-gnn-.git
cd uncertainty-aware-causal-temporal-gnn-

pip install -r requirements.txt

# Install PyTorch Geometric (CUDA 11.8)
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Optional dependencies

```bash
pip install causal-learn       # Only needed for causal_method='pc'
pip install transformers       # Only needed for cold-start text features
pip install wandb tensorboard  # Experiment tracking
```

If an optional dependency is missing but your `Config` requires it (e.g.
`causal_method='pc'`, or a `ColdStartSolver` with
`cold_start_require_text=True`), the code now raises a loud `ImportError`
instead of silently falling back to zero outputs.

## Quick Start

### Training

```python
from causal_gnn.config import Config
from causal_gnn.training.trainer import RecommendationSystem

config = Config(
    embedding_dim=64,
    num_layers=3,
    learning_rate=0.001,
    batch_size=1024,
    num_epochs=20,
    use_amp=True,
)

rec_system = RecommendationSystem(config)
rec_system.load_data('./data/interactions.csv')
rec_system.preprocess_data()
rec_system.split_data()
rec_system.create_graph()        # also precomputes the co-activity graph
rec_system.initialize_model()

history = rec_system.train()
metrics = rec_system.evaluate('test', k_values=[5, 10, 20])
```

### Command line

```bash
python causal_gnn/scripts/train.py \
    --data_path ./data/interactions.csv \
    --embedding_dim 64 \
    --num_epochs 20 \
    --use_amp
```

## Benchmarking

```bash
python benchmark_m3.py
```

Benchmarks five models on MovieLens-100K: Popular, BPR, NCF, LightGCN,
and UACT-GNN, with consistent Precision / Recall / NDCG / Hit / MRR @k.

## Project Structure

```
causal_gnn/
├── models/
│   ├── uact_gnn.py              # Main UACT-GNN model
│   ├── layers.py                # CausalGNN / TemporalAttention / SAGE / sparse GCN
│   ├── fusion.py                # Multi-modal fusion
│   └── uncertainty_gnn.py       # Uncertainty-aware model
├── causal/
│   ├── discovery.py             # Vectorized temporal co-activity graph (Granger)
│   └── bayesian_discovery.py    # Bayesian edge-strength variant
├── data/
│   ├── processor.py             # Schema auto-detection + IO
│   ├── dataset.py               # PyTorch datasets
│   └── samplers.py              # Random / popularity / hard / mixed negatives
├── training/
│   ├── trainer.py               # Training loop + causal-graph precompute
│   ├── evaluator.py             # Evaluation metrics
│   └── uncertainty_trainer.py   # Uncertainty-aware training
├── baselines/                   # popular.py, bpr.py, ncf.py, lightgcn.py
├── evaluation/                  # publication_metrics.py, visualizations.py
└── utils/                       # uncertainty, cold_start, checkpointing, logging
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 128 | Embedding dimension |
| `num_layers` | 3 | Number of GNN layers |
| `time_steps` | 16 | Number of temporal buckets |
| `dropout` | 0.2 | Dropout rate |
| `learning_rate` | 0.001 | Learning rate |
| `batch_size` | 2048 | Batch size |
| `num_epochs` | 50 | Training epochs |
| `causal_method` | `advanced` | `simple` (identity) / `advanced` (Granger) / `pc` (requires `causal-learn`) |
| `max_causal_nodes` | 512 | Cap on nodes in the Granger solve; others get 0 edges |
| `use_amp` | False | Mixed precision training (CUDA) |

## Supported Data Formats

- CSV, JSON, Parquet, TSV
- Auto-detection of user, item, rating, and timestamp columns

## Known Limitations

- **Scale**: this repo has been exercised end-to-end on MovieLens-100K
  (~100K ratings, ~1K users, ~2K items). Claims of MovieLens-25M /
  Amazon-Reviews-233M scale from earlier versions of this README were
  aspirational and have been removed. The O(N²) Granger solve is
  bounded by `config.max_causal_nodes` but true billion-interaction
  scale will require further work.
- **"Causal" edges**: Granger on per-node activity counts captures
  co-popularity and shared temporal trends. This is not counterfactual
  causal discovery. See `causal_gnn/causal/discovery.py`.
- **Neighbor sampling / DDP**: `Config.distributed`, `Config.world_size`,
  `Config.local_rank`, and any neighbor-sampling flags are **unused** in
  the current training loop. They are preserved only as config
  placeholders for future work.
- **Per-batch full-graph forward**: training runs a full-graph forward
  pass per batch. This is fine for MovieLens-100K but will not scale to
  much larger graphs without neighbor sampling.

## References

The "causal" framing and the temporal-GNN design are informed by:

- Luo et al. 2024, *A Survey on Causal Inference for Recommendation*
  (arXiv:[2303.11666](https://arxiv.org/abs/2303.11666)).
- Wang et al. 2023, *Neural Causal Graph Collaborative Filtering*
  (arXiv:[2307.04384](https://arxiv.org/abs/2307.04384)).
- Poursafaei et al. 2024, *Towards Better Evaluation for Dynamic Link
  Prediction*-style TGN analyses for recsys
  (arXiv:[2403.16066](https://arxiv.org/abs/2403.16066)).
- Wu et al. 2022, *Uncertainty in GNNs: A Survey*
  (arXiv:[2205.09968](https://arxiv.org/abs/2205.09968)).

## License

MIT License

## Citation

```bibtex
@software{uact_gnn,
  title={UACT-GNN: Uncertainty-Aware Causal Temporal Graph Neural Network},
  author={Savana Sunil Kumar},
  year={2024},
  url={https://github.com/savanasunilkumar/uncertainty-aware-causal-temporal-gnn-}
}
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
