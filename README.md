# GNN–RL Disaster-Aware Healthcare WSN Routing

Reproducibility code and experiment outputs for **“A Graph Neural Network and Reinforcement Learning Framework for Resilient Self-Healing Routing in Disaster-Aware Healthcare Wireless Sensor Networks”** by Ibrahim Salman and Jaspreet Kaur.

> **Publication status:** manuscript submitted for publication. The manuscript itself is not included here while publisher and preprint permissions are being confirmed.

## Overview

This project evaluates self-healing routing strategies for healthcare wireless sensor networks under disaster-induced failures. The simulator models a 74-node network—8 patient sensors, 65 relay nodes, and one sink—and compares:

- shortest-path routing;
- energy-aware routing;
- weighted failure-risk-aware routing;
- LSTM-based routing;
- LSTM with reinforcement learning; and
- graph neural network (GNN) with reinforcement learning.

The final regression-GNN experiments show the clearest advantage under extreme conditions: a 96.0% mean route-delivery ratio for GNN–RL compared with 89.6% for LSTM–RL across 100 trials (`p = 0.0267`). Results are simulation-based and should not be interpreted as clinical validation.

## Repository layout

```text
.
├── main_simulation_clean.py              # Network simulation and routing methods
├── gnn_regression.py                     # Final continuous-risk GNN
├── gnn_failure_prediction.py             # Earlier classification GNN baseline
├── run_full_experiment.py                # Multi-severity baseline experiment
├── run_ablation_experiment.py            # Classification-GNN ablation
├── run_ablation_regression.py            # Final regression-GNN ablation
├── run_extreme_experiment_regression.py  # 100-trial extreme-severity experiment
├── run_computational_cost_comparison.py  # LSTM/GNN efficiency comparison
├── results/                               # Reported CSV outputs
└── tests/                                 # Lightweight regression tests
```

Generated model weights, synthetic datasets, caches, exploratory prototypes, and manuscript files are intentionally excluded from version control.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the experiments

Run commands from the repository root.

```bash
# Main comparison across Low, Medium, and High severity
python run_full_experiment.py

# Final regression-GNN ablation study (30 trials per severity)
python run_ablation_regression.py

# Final Extreme-severity comparison (100 trials)
python run_extreme_experiment_regression.py

# Parameter count and timing comparison
python run_computational_cost_comparison.py
```

The scripts use fixed seed ranges for paired comparisons. Training and inference timings depend on hardware, operating system, and PyTorch build. The reference outputs used in the submitted manuscript are preserved in [`results/`](results/).

## Key reference results

| Experiment | Comparison | Result |
|---|---|---:|
| Extreme severity, 100 trials | GNN–RL route delivery | 96.0% |
| Extreme severity, 100 trials | LSTM–RL route delivery | 89.6% |
| Paired significance test | Route-delivery ratio | `p = 0.0267` |
| Computational profile | GNN trainable parameters | 1,313 |
| Computational profile | LSTM trainable parameters | 20,545 |

See [`results/extreme_experiment_results_regression.csv`](results/extreme_experiment_results_regression.csv), [`results/extreme_significance_results_regression.csv`](results/extreme_significance_results_regression.csv), and [`results/computational_cost_comparison.csv`](results/computational_cost_comparison.csv) for the underlying summaries.

## Reproducibility and limitations

- Experiments use simulated network data rather than deployment traces.
- GNN training targets are derived from the weighted risk formula, so the model does not yet learn from independently observed node failures.
- The regression-GNN version is the final model used to address the continuous-risk mismatch identified in the earlier classification ablation.
- Real-world validation, mobility, hardware energy measurements, and failure-labelled datasets remain future work.

## Citation

If this code supports your work, use the metadata in [`CITATION.cff`](CITATION.cff). Replace the submitted-manuscript citation with the final DOI and venue metadata once the paper is published.

## License

This project is available under the [MIT License](LICENSE).
