# phase3_experiment.py
"""Phase 3 – FedAvg vs FedProx comparison on CIFAR-10.

Runs two experiments with identical data partitioning (Dirichlet α=0.1) and
hyper‑parameters:
- 100 clients, C=0.1 (10 selected per round)
- Local epochs = 2, batch = 64, lr = 0.01
- 30 communication rounds
- μ = 0.0 for FedAvg, μ = 0.01 for FedProx

Results are logged as CSV/JSON (via RoundLogger) in ``results/phase3`` and a
single accuracy comparison plot ``results/phase3/comparison_accuracy.png`` is
generated.
"""

from __future__ import annotations

import copy
import os
import random
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from data_partitioner import DataPartitioner
from logger import RoundLogger
from model_cifar import ResNet8, get_weights, set_weights
from phase2_experiment import (
    N_CLIENTS,
    FRACTION,
    N_SELECTED,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    N_ROUNDS,
    SEED,
    DEVICE,
    set_seed,
    load_cifar10,
    train_client,
    evaluate_global,
    fedavg_aggregate,
)

# Override output directory for Phase 3
OUTPUT_DIR = os.path.join("results", "phase3")


# ------------------------------------------------------------------
# Phase 3 comparison plot
# ------------------------------------------------------------------

def plot_phase3_comparison(
    loggers: Dict[str, RoundLogger],
    n_rounds: int,
    output_dir: str,
    filename_prefix: str = "comparison",
) -> None:
    """Generate accuracy + loss comparison curves for FedAvg vs FedProx."""
    plt.style.use("dark_background")
    PALETTE = {
        "FedAvg": "#F87171",   # red — baseline
        "FedProx": "#4ADE80",  # green — improved
    }

    rounds = list(range(1, n_rounds + 1))

    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    for label, lg in loggers.items():
        acc_curve = lg.get_acc_curve()
        color = PALETTE.get(label, "#CCCCCC")
        ax.plot(
            rounds[:len(acc_curve)], acc_curve,
            label=label, color=color,
            linewidth=2.2,
            marker="o", markersize=4, markevery=5,
        )

    ax.set_xlabel("Communication Round", color="white", fontsize=11)
    ax.set_ylabel("Global Test Accuracy (%)", color="white", fontsize=11)
    ax.set_title(
        "Phase 3: FedAvg vs FedProx on CIFAR-10 (Dirichlet α=0.1)",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(color="#2A2D3E", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=10, facecolor="#1A1D2E", edgecolor="#444466", labelcolor="white")
    ax.set_xlim(1, n_rounds)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    acc_path = os.path.join(output_dir, f"{filename_prefix}_accuracy.png")
    fig.savefig(acc_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {acc_path}")

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    for label, lg in loggers.items():
        loss_curve = lg.get_loss_curve()
        color = PALETTE.get(label, "#CCCCCC")
        ax.plot(
            rounds[:len(loss_curve)], loss_curve,
            label=label, color=color,
            linewidth=2.2,
            marker="s", markersize=4, markevery=5,
        )

    ax.set_xlabel("Communication Round", color="white", fontsize=11)
    ax.set_ylabel("Global Test Loss", color="white", fontsize=11)
    ax.set_title(
        "Phase 3: Loss Curves — FedAvg vs FedProx",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(color="#2A2D3E", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=10, facecolor="#1A1D2E", edgecolor="#444466", labelcolor="white")
    ax.set_xlim(1, n_rounds)

    plt.tight_layout()
    loss_path = os.path.join(output_dir, f"{filename_prefix}_loss.png")
    fig.savefig(loss_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {loss_path}")


# ------------------------------------------------------------------
# Single experiment runner
# ------------------------------------------------------------------

def run_experiment(
    experiment_name: str,
    indices: Dict[int, List[int]],
    trainset,
    testset,
    dp: DataPartitioner,
    rng: random.Random,
    algorithm: str = "fedavg",
    proximal_mu: float = 0.0,
) -> RoundLogger:
    """Run a single FedAvg/FedProx experiment and return the logger."""
    print(f"\n{'='*65}")
    print(f"Experiment: {experiment_name} (algorithm={algorithm}, mu={proximal_mu})")
    print(f"  Clients: {N_CLIENTS}  |  Selected/round: {N_SELECTED}")
    print(f"  Local epochs: {LOCAL_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Rounds: {N_ROUNDS}  |  Device: {DEVICE}")
    print(f"{'='*65}")

    logger = RoundLogger(experiment_name=experiment_name, output_dir=OUTPUT_DIR)
    subsets = dp.get_subsets(indices)
    all_stats = dp.all_stats(indices)
    global_model = ResNet8(num_classes=10).to(DEVICE)
    init_acc, init_loss = evaluate_global(global_model, testset, BATCH_SIZE, DEVICE)
    print(f"  Round  0 | Acc: {init_acc:6.2f}% | Loss: {init_loss:.4f} (random init)")

    for round_idx in range(1, N_ROUNDS + 1):
        round_start = time.time()
        selected_ids = rng.sample(range(N_CLIENTS), N_SELECTED)
        global_weights = get_weights(global_model)
        client_weights_list: List[List[np.ndarray]] = []
        client_sizes: List[int] = []
        client_metrics = {}
        for cid in selected_ids:
            local_model = ResNet8(num_classes=10).to(DEVICE)
            set_weights(local_model, copy.deepcopy(global_weights))
            updated_w, c_loss, c_acc, n_samples = train_client(
                local_model,
                subsets[cid],
                LOCAL_EPOCHS,
                BATCH_SIZE,
                LEARNING_RATE,
                DEVICE,
                algorithm=algorithm,
                proximal_mu=proximal_mu,
            )
            client_weights_list.append(updated_w)
            client_sizes.append(n_samples)
            client_metrics[cid] = {"acc": c_acc, "loss": c_loss, "n_samples": n_samples}
        new_weights = fedavg_aggregate(client_weights_list, client_sizes)
        set_weights(global_model, new_weights)
        global_acc, global_loss = evaluate_global(global_model, testset, BATCH_SIZE, DEVICE)
        round_time = time.time() - round_start
        logger.log_round(
            round_idx=round_idx,
            global_acc=global_acc,
            global_loss=global_loss,
            selected_clients=selected_ids,
            round_time=round_time,
            client_metrics=client_metrics,
        )
        print(
            f"  Round {round_idx:2d} | Acc: {global_acc:6.2f}% | "
            f"Loss: {global_loss:.4f} | Time: {round_time:.1f}s"
        )
    logger.save()
    return logger


if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 65)
    print("Phase 3 -- FedAvg vs FedProx on CIFAR-10 (Dirichlet alpha=0.1)")
    print("=" * 65)

    trainset, testset = load_cifar10()
    dp = DataPartitioner(trainset, n_clients=N_CLIENTS, seed=SEED)
    # Dirichlet α=0.1 partition used for both experiments
    partition = dp.dirichlet_split(alpha=0.1)
    rng_fedavg = random.Random(SEED)
    rng_fedprox = random.Random(SEED)

    fedavg_logger = run_experiment(
        experiment_name="fedavg_dirichlet_alpha0.1",
        indices=partition,
        trainset=trainset,
        testset=testset,
        dp=dp,
        rng=rng_fedavg,
        algorithm="fedavg",
        proximal_mu=0.0,
    )
    fedprox_logger = run_experiment(
        experiment_name="fedprox_dirichlet_alpha0.1",
        indices=partition,
        trainset=trainset,
        testset=testset,
        dp=dp,
        rng=rng_fedprox,
        algorithm="fedprox",
        proximal_mu=0.01,
    )

    # Plot comparison
    loggers = {
        "FedAvg": fedavg_logger,
        "FedProx": fedprox_logger,
    }
    plot_phase3_comparison(loggers, n_rounds=N_ROUNDS, output_dir=OUTPUT_DIR)

    # Print final summary for README update
    print(f"\n{'='*65}")
    print("PHASE 3 RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Experiment':<25}  {'Final Acc':>10}  {'Best Acc':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}")
    for label, lg in loggers.items():
        acc_curve = lg.get_acc_curve()
        final_acc = acc_curve[-1] if acc_curve else 0.0
        best_acc = max(acc_curve) if acc_curve else 0.0
        print(f"  {label:<25}  {final_acc:>9.2f}%  {best_acc:>9.2f}%")
    print("=" * 65)

