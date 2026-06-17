
# Federated Learning — Phase 1: FedAvg Baseline

Implements the **Federated Averaging (FedAvg)** algorithm from scratch, faithful to
[McMahan et al. 2017](https://arxiv.org/abs/1602.05629).

## Quick Start

```bash
# 1. Install dependencies (requires uv — https://docs.astral.sh/uv/)
uv sync

# 2. Run FedAvg on MNIST (10 clients, 20 rounds, IID split)
uv run python fedavg_baseline.py
```

Results (accuracy/loss curves) are saved to `results/`.

## Project Structure

| File | Description |
|------|-------------|
| `data_utils.py` | MNIST download + IID partitioning |
| `model.py` | 2-Conv CNN (paper architecture) + weight helpers |
| `client.py` | Local trainer — SGD for E epochs |
| `server.py` | FedAvg server — weighted aggregation + broadcast |
| `fedavg_baseline.py` | Main entry point — training loop + plots |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Clients | 10 |
| Selected/round | 10 (C=1.0) |
| Local epochs (E) | 5 |
| Learning rate (η) | 0.01 |
| Batch size (B) | 64 |
| Rounds | 20 |
| Optimizer | SGD (no momentum) |

## Reference

McMahan, H.B., Moore, E., Ramage, D., Hampson, S. and Agüera y Arcas, B., 2017.
*Communication-efficient learning of deep networks from decentralized data.* AISTATS.



