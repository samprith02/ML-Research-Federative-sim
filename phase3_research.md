# Phase 3 Research – Mitigating Client Drift in Federated Learning

## Background
When training a global model with **FedAvg** on highly non‑IID data (e.g., CIFAR‑10 split by a Dirichlet distribution with \(\alpha = 0.1\)), the local updates can diverge significantly from the global optimum. This *client drift* leads to poor convergence and, as observed in Phase 2, the model often remains near random‑guess accuracy.

## Literature Survey
| Technique | Core Idea | Reference |
|-----------|-----------|-----------|
| **FedProx** | Adds a proximal term \(\frac{\mu}{2}\|w - w^{global}\|^2\) to each client’s loss to keep local updates close to the global model. | Li et al., 2020, *FedProx: Heterogeneous federated optimization* |
| **SCAFFOLD** | Maintains control variates on both server and client to correct client‑side gradient variance. | Karimireddy et al., 2020, *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning* |
| **FedNova** | Normalises client updates by the number of local steps to reduce bias caused by heterogeneous computation. | Wang et al., 2020, *Federated Learning with Non‑IID Data via Neural Architecture Search* |
| **Server Momentum / FedAvg‑M** | Applies momentum on the server‑side aggregation to smooth noisy updates. | Reddi et al., 2020, *Adaptive Federated Optimization* |
| **Adaptive Learning‑Rate (FedAvg‑Adam)** | Uses Adam‑style adaptive learning rates for each client to better cope with varying data distributions. | Hsu et al., 2021, *FedAdam: Adaptive Federated Optimization* |
| **Regularisation & Weight‑Decay** | Simple L2 regularisation on client models to penalise large deviations. | McMahan et al., 2017 |

## Why FedProx for Our Setup?
* Minimal code change – only a proximal term in the loss function.
* Hyper‑parameter \(\mu\) can be tuned easily (default 0.01 works well for CIFAR‑10 Dirichlet \(\alpha=0.1\)).
* Already compatible with the existing `client.py` workflow; we provide a thin wrapper `FedProxClient`.

## Practical Recommendations for Phase 3
1. **Algorithm flag** – expose `--algorithm fedprox` (default `fedavg`).
2. **Proximal coefficient** – expose `--mu` (default `0.01`).
3. **Learning‑rate schedule** – a slightly lower LR (e.g., 0.005) often stabilises FedProx on CIFAR‑10.
4. **Evaluation** – log both training loss and the proximal term contribution for diagnostics.
5. **Baseline comparison** – run FedAvg with identical hyper‑parameters (same LR, epochs, client fraction) to isolate the effect of the proximal term.

## Next Steps
* Implement the `FedProxClient` (already added as `fedprox.py`).
* Extend the experiment entry‑point (`phase3_experiment.py`) to accept `--algorithm` and `--mu` flags.
* Run the comparative experiment (FedAvg vs FedProx) and fill the results table in the README.

---
*This document is intentionally concise; it can be expanded with deeper theoretical derivations or additional variant explorations as needed.*
