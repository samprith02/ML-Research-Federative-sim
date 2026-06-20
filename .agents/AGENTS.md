# Project Context: Federated Learning Simulator

## Overview
This project is a PyTorch-based Federated Learning simulation environment built from scratch. It aims to reproduce and extend the foundational "Federated Averaging" (FedAvg) algorithm by McMahan et al. (2017).

## Architecture & Phases
- **Phase 1 (Completed)**: Baseline FedAvg on MNIST with an IID data distribution and a 2-layer CNN. Proved the federated aggregation loop works (achieved ~99% accuracy).
- **Phase 2 (Completed)**: Transition to CIFAR-10 with ResNet-8 and Non-IID data partitioning (Dirichlet, Pathological, Quantity-skew). Demonstrated the "Client Drift" problem where FedAvg fails to converge (stalls at ~10% accuracy) under high data skew.
- **Phase 3 (Upcoming)**: Implement advanced FL algorithms (e.g., FedProx, Scaffold, Server Momentum) to fix the client drift issue observed in Phase 2.

## Conventions & Rules
- **Environment**: Python >= 3.11, PyTorch (CUDA 12.4), managed by `uv`.
- **Reproducibility**: Use `uv.lock` and `requirements.txt` to pin dependencies.
- **Git Strategy**: Commit after every major milestone or completed step. Exclude large data files (`data/`) and raw results (`results/*.json`, `results/*.csv`) via `.gitignore`.
- **Coding Style**: Clear, documented Python code with strict typing where possible. Separated components (Data, Model, Client, Server, Logger).
