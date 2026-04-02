# UACT-GNN: Uncertainty-Aware Causal Temporal Graph Neural Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready recommendation system that combines **causal discovery**, **temporal modeling**, **uncertainty quantification**, and **graph neural networks** for large-scale datasets (100M-1B+ interactions).

## Key Features

| Feature | Description |
|---------|-------------|
| **Causal Discovery** | Granger causality and PC algorithm for causal relationship discovery |
| **Temporal Modeling** | Transformer-based attention for capturing temporal dynamics |
| **Uncertainty Quantification** | Bayesian inference with Monte Carlo dropout for prediction confidence |
| **Multi-Modal Learning** | Processes text, images, numeric, and categorical features |
| **Cold Start Handling** | Zero-shot recommendations for new users/items |
| **Scalable Architecture** | Neighbor sampling and sparse operations for large graphs |

## Installation

```bash
# Clone the repository
git clone https://github.com/savanasunilkumar/uncertainty-aware-causal-temporal-gnn-.git
cd uncertainty-aware-causal-temporal-gnn-

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (CUDA 11.8)
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Optional Dependencies

```bash
pip install causal-learn      # Advanced causal discovery
pip install transformers      # Text feature extraction
pip install wandb tensorboard # Experiment tracking
```

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
    use_amp=True
)

rec_system = RecommendationSystem(config)
rec_system.load_data('./data/interactions.csv')
rec_system.preprocess_data()
rec_system.split_data()
rec_system.create_graph()
rec_system.initialize_model()

history = rec_system.train()
metrics = rec_system.evaluate('test', k_values=[5, 10, 20])
```

### Generate Recommendations

```python
recommendations, scores = rec_system.generate_recommendations(user_id=1, top_k=10)
```

### Command Line

```bash
# Basic training
python causal_gnn/scripts/train.py \
    --data_path ./data/interactions.csv \
    --embedding_dim 64 \
    --num_epochs 20 \
    --use_amp

# Large-scale training (100M+ interactions)
python causal_gnn/scripts/train.py \
    --data_path ./data/interactions.csv \
    --embedding_dim 128 \
    --batch_size 4096 \
    --use_gradient_checkpointing \
    --use_neighbor_sampling \
    --use_amp
```

## Benchmarking

Run the benchmark suite on MovieLens dataset:

```bash
python benchmark_m3.py
```

Benchmarks 5 models: Popular, BPR, NCF, LightGCN, and UACT-GNN with comprehensive metrics.

## Project Structure

```
causal_gnn/
├── models/
│   ├── uact_gnn.py          # Main UACT-GNN model
│   ├── layers.py            # GNN layers (GAT, GCN, GraphSAGE)
│   ├── fusion.py            # Multi-modal fusion
│   └── uncertainty_gnn.py   # Uncertainty-aware model
├── causal/
│   ├── discovery.py         # Granger causality, PC algorithm
│   └── bayesian_discovery.py # Bayesian causal discovery
├── data/
│   ├── processor.py         # Universal data processor
│   ├── dataset.py           # PyTorch datasets
│   └── samplers.py          # Negative sampling strategies
├── training/
│   ├── trainer.py           # Training loop
│   ├── evaluator.py         # Evaluation metrics
│   └── uncertainty_trainer.py # Uncertainty-aware training
├── baselines/
│   ├── popular.py           # Popularity baseline
│   ├── bpr.py               # BPR model
│   ├── ncf.py               # Neural Collaborative Filtering
│   └── lightgcn.py          # LightGCN model
├── evaluation/
│   ├── publication_metrics.py # Comprehensive metrics
│   └── visualizations.py    # Result visualization
└── utils/
    ├── uncertainty.py       # Uncertainty utilities
    ├── cold_start.py        # Cold start handling
    ├── checkpointing.py     # Model checkpointing
    └── logging.py           # Experiment logging
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Embedding dimension |
| `num_layers` | 3 | Number of GNN layers |
| `time_steps` | 10 | Temporal time steps |
| `dropout` | 0.1 | Dropout rate |
| `learning_rate` | 0.001 | Learning rate |
| `batch_size` | 1024 | Batch size |
| `num_epochs` | 20 | Training epochs |
| `causal_method` | advanced | Causal discovery method |
| `use_amp` | False | Mixed precision training |

## Supported Data Formats

- CSV, JSON, Parquet, TSV
- Auto-detection of user, item, rating, and timestamp columns
- Handles sparse and dense interaction matrices

## Performance

Tested on:
- MovieLens-25M (25M ratings, 162K users, 62K movies)
- Amazon Reviews (233M ratings, 43M users, 3M products)

Optimizations:
- Mixed precision training (2-3x memory reduction)
- Gradient checkpointing for deep models
- Sparse tensor operations
- Multi-GPU distributed training

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
