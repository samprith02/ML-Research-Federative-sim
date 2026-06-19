"""
phase2_experiment.py
====================
Phase 2 -- FedAvg Experiment Runner (CIFAR-10, 100 clients)

Runs three back-to-back experiments and generates a comparison plot:
  1. IID baseline
  2. Dirichlet non-IID alpha=0.5
  3. Dirichlet non-IID alpha=0.1

Each experiment:
  - 100 clients, C=0.1 (10 selected/round), E=2, B=64, 30 rounds
  - Uses ResNet-8 on CIFAR-10
  - Logs per-round CSV + per-client JSON via RoundLogger
  - Saves accuracy and loss curves

Final output: results/phase2/comparison_accuracy.png
              results/phase2/comparison_loss.png

Usage
-----
    uv run python phase2_experiment.py
"""

from __future__ import annotations

import copy
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms

from data_partitioner import DataPartitioner
from logger import RoundLogger
from model_cifar import ResNet8, get_weights, set_weights

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------

N_CLIENTS = 100
FRACTION = 0.1          # C: fraction of clients selected per round
N_SELECTED = max(1, int(N_CLIENTS * FRACTION))  # = 10
LOCAL_EPOCHS = 2         # E: local training epochs per round
BATCH_SIZE = 64          # B: local mini-batch size
LEARNING_RATE = 0.01     # eta: SGD learning rate
N_ROUNDS = 30            # total communication rounds
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join("results", "phase2")

# CIFAR-10 normalization (mean/std of training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_cifar10() -> Tuple:
    """Download CIFAR-10 and return (trainset, testset)."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    return trainset, testset


# ------------------------------------------------------------------
# Local training (client)
# ------------------------------------------------------------------

def train_client(
    model: nn.Module,
    subset: Subset,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[List[np.ndarray], float, float, int]:
    """Train a local model on a client's data for E epochs.

    Parameters
    ----------
    model : nn.Module
        The global model (already loaded with server weights).
    subset : Subset
        Client's local dataset.
    n_epochs : int
        Number of local training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for SGD.
    device : torch.device

    Returns
    -------
    (updated_weights, avg_loss, avg_acc, n_samples)
    """
    model.train()
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    # Fresh optimizer each round -- no stale momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(n_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += loss.item() * batch_size_actual
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size_actual

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return get_weights(model), avg_loss, avg_acc, len(subset)


# ------------------------------------------------------------------
# Global evaluation
# ------------------------------------------------------------------

def evaluate_global(
    model: nn.Module,
    testset,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the global model on the full test set.

    Returns
    -------
    (accuracy_percent, avg_loss)
    """
    model.eval()
    loader = DataLoader(
        testset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_sz = labels.size(0)
            total_loss += loss.item() * batch_sz
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_sz

    acc_pct = 100.0 * total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    return acc_pct, avg_loss


# ------------------------------------------------------------------
# FedAvg aggregation (weighted by sample count)
# ------------------------------------------------------------------

def fedavg_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
) -> List[np.ndarray]:
    """Weighted average of client model weights.

    w_global = sum_k (n_k / n_total) * w_k

    Parameters
    ----------
    client_weights : list of weight lists
        One weight list per participating client.
    client_sizes : list of ints
        Number of local samples for each client.

    Returns
    -------
    Aggregated weight list (same structure as a single client's weights).
    """
    total = sum(client_sizes)
    aggregated = [
        sum(
            (n_k / total) * w_k[i]
            for w_k, n_k in zip(client_weights, client_sizes)
        )
        for i in range(len(client_weights[0]))
    ]
    return aggregated


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
    n_rounds: int = N_ROUNDS,
) -> RoundLogger:
    """Run a complete FedAvg experiment and return the populated logger.

    Parameters
    ----------
    experiment_name : str
        Short label used for file names and print output.
    indices : dict
        Partition mapping (client_id -> list of dataset indices).
    trainset, testset :
        CIFAR-10 torchvision datasets.
    dp : DataPartitioner
        Used to retrieve client_stats for logging.
    rng : random.Random
        Seeded RNG for client selection (separate from global seed).
    n_rounds : int

    Returns
    -------
    RoundLogger with all rounds recorded.
    """
    print(f"\n{'='*65}")
    print(f"Experiment: {experiment_name}")
    print(f"  Clients: {N_CLIENTS}  |  Selected/round: {N_SELECTED}")
    print(f"  Local epochs: {LOCAL_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Rounds: {n_rounds}  |  Device: {DEVICE}")
    print(f"{'='*65}")

    logger = RoundLogger(experiment_name=experiment_name, output_dir=OUTPUT_DIR)

    # Build client subsets
    subsets = dp.get_subsets(indices)

    # Precompute client_stats for logging (label distributions)
    all_stats = dp.all_stats(indices)

    # Initialise global model
    global_model = ResNet8(num_classes=10).to(DEVICE)

    # Evaluate random init
    init_acc, init_loss = evaluate_global(global_model, testset, BATCH_SIZE, DEVICE)
    print(f"\n  Round  0 | Acc: {init_acc:6.2f}% | Loss: {init_loss:.4f} (random init)")

    for round_idx in tqdm(range(1, n_rounds + 1), desc=f"  {experiment_name}", ncols=70):
        round_start = time.time()

        # --- Select clients ---
        selected_ids = rng.sample(range(N_CLIENTS), N_SELECTED)

        # --- Broadcast global weights & train each client ---
        global_weights = get_weights(global_model)
        client_weights_list = []
        client_sizes = []
        client_metrics = {}

        for cid in selected_ids:
            # Clone model and load global weights
            local_model = ResNet8(num_classes=10).to(DEVICE)
            set_weights(local_model, copy.deepcopy(global_weights))

            # Local training
            updated_w, c_loss, c_acc, n_samples = train_client(
                local_model,
                subsets[cid],
                LOCAL_EPOCHS,
                BATCH_SIZE,
                LEARNING_RATE,
                DEVICE,
            )
            client_weights_list.append(updated_w)
            client_sizes.append(n_samples)

            # Store per-client metrics
            client_metrics[cid] = {
                "acc": c_acc,
                "loss": c_loss,
                "n_samples": n_samples,
                "label_dist": all_stats[cid]["label_counts"],
            }

        # --- Aggregate ---
        new_weights = fedavg_aggregate(client_weights_list, client_sizes)
        set_weights(global_model, new_weights)

        # --- Global evaluation ---
        global_acc, global_loss = evaluate_global(global_model, testset, BATCH_SIZE, DEVICE)

        round_time = time.time() - round_start

        # Print progress
        tqdm.write(
            f"  Round {round_idx:2d} | Acc: {global_acc:6.2f}% | "
            f"Loss: {global_loss:.4f} | Time: {round_time:.1f}s | "
            f"Clients: {sorted(selected_ids)}"
        )

        # Log
        logger.log_round(
            round_idx=round_idx,
            global_acc=global_acc,
            global_loss=global_loss,
            selected_clients=selected_ids,
            round_time=round_time,
            client_metrics=client_metrics,
        )

    # Save logs
    logger.save()
    logger.print_summary()
    return logger


# ------------------------------------------------------------------
# Comparison plots
# ------------------------------------------------------------------

def plot_comparison(
    loggers: Dict[str, RoundLogger],
    n_rounds: int,
    filename_prefix: str = "comparison",
) -> None:
    """Generate accuracy + loss comparison curves for multiple experiments.

    Parameters
    ----------
    loggers : dict
        {experiment_label -> RoundLogger}
    n_rounds : int
    filename_prefix : str
    """
    # Style
    plt.style.use("dark_background")
    PALETTE = {
        "IID": "#4ADE80",           # green
        "Dirichlet alpha=0.5": "#FACC15",  # yellow
        "Dirichlet alpha=0.1": "#F87171",  # red
    }
    LINE_STYLES = ["solid", "dashed", "dotted"]

    rounds = list(range(1, n_rounds + 1))

    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    for (label, lg), ls in zip(loggers.items(), LINE_STYLES):
        acc_curve = lg.get_acc_curve()
        color = PALETTE.get(label, "#CCCCCC")
        ax.plot(
            rounds[:len(acc_curve)], acc_curve,
            label=label, color=color,
            linewidth=2.2, linestyle=ls,
            marker="o", markersize=4, markevery=5,
        )

    ax.set_xlabel("Communication Round", color="white", fontsize=11)
    ax.set_ylabel("Global Test Accuracy (%)", color="white", fontsize=11)
    ax.set_title(
        "FedAvg on CIFAR-10: IID vs Non-IID (100 clients, C=0.1, E=2)",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(color="#2A2D3E", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=10, facecolor="#1A1D2E", edgecolor="#444466", labelcolor="white")
    ax.set_xlim(1, n_rounds)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    acc_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_accuracy.png")
    fig.savefig(acc_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {acc_path}")

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    for (label, lg), ls in zip(loggers.items(), LINE_STYLES):
        loss_curve = lg.get_loss_curve()
        color = PALETTE.get(label, "#CCCCCC")
        ax.plot(
            rounds[:len(loss_curve)], loss_curve,
            label=label, color=color,
            linewidth=2.2, linestyle=ls,
            marker="s", markersize=4, markevery=5,
        )

    ax.set_xlabel("Communication Round", color="white", fontsize=11)
    ax.set_ylabel("Global Test Loss", color="white", fontsize=11)
    ax.set_title(
        "FedAvg on CIFAR-10: Loss Curves IID vs Non-IID",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(color="#2A2D3E", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=10, facecolor="#1A1D2E", edgecolor="#444466", labelcolor="white")
    ax.set_xlim(1, n_rounds)

    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_loss.png")
    fig.savefig(loss_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {loss_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 65)
    print("FedAvg Phase 2 -- CIFAR-10, Non-IID Experiments")
    print("=" * 65)
    print(f"  Device:    {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    print(f"  Clients:   {N_CLIENTS}")
    print(f"  Selected:  {N_SELECTED}/round  (C={FRACTION})")
    print(f"  Epochs:    {LOCAL_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Rounds:    {N_ROUNDS}")
    print()

    # Load CIFAR-10
    print("[1/5] Loading CIFAR-10...")
    trainset, testset = load_cifar10()
    print(f"  Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    # Partition data
    print("\n[2/5] Partitioning data for 100 clients...")
    dp = DataPartitioner(trainset, n_clients=N_CLIENTS, seed=SEED)

    partitions = {
        "IID": dp.iid_split(),
        "Dirichlet alpha=0.5": dp.dirichlet_split(alpha=0.5),
        "Dirichlet alpha=0.1": dp.dirichlet_split(alpha=0.1),
    }
    for name, idxs in partitions.items():
        sizes = [len(v) for v in idxs.values()]
        avg_classes = sum(
            dp.client_stats(i, idxs)["classes_present"] for i in range(N_CLIENTS)
        ) / N_CLIENTS
        print(f"  {name}: sizes [{min(sizes)}-{max(sizes)}], "
              f"avg classes/client={avg_classes:.1f}")

    # Visualize distributions
    print("\n[3/5] Generating distribution charts...")
    from visualize_distribution import plot_distribution_grid, plot_comparison_panel
    for name, idxs in partitions.items():
        safe_name = name.lower().replace(" ", "_").replace("=", "")
        plot_distribution_grid(
            idxs, dp,
            title=f"CIFAR-10 Label Distribution -- {name} (100 clients)",
            filename=f"label_dist_{safe_name}.png",
        )

    # Run experiments
    print("\n[4/5] Running FedAvg experiments...")
    loggers = {}
    for exp_name, idxs in partitions.items():
        exp_rng = random.Random(SEED)  # same seed for fair client sampling comparison
        safe = exp_name.lower().replace(" ", "_").replace("=", "").replace(".", "")
        logger = run_experiment(
            experiment_name=safe,
            indices=idxs,
            trainset=trainset,
            testset=testset,
            dp=dp,
            rng=exp_rng,
            n_rounds=N_ROUNDS,
        )
        loggers[exp_name] = logger

    # Comparison plots
    print("\n[5/5] Generating comparison plots...")
    plot_comparison(loggers, n_rounds=N_ROUNDS)

    # Print final summary table
    print("\n" + "=" * 65)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Experiment':<25}  {'Final Acc':>10}  {'Best Acc':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}")
    for label, lg in loggers.items():
        acc_curve = lg.get_acc_curve()
        final_acc = acc_curve[-1] if acc_curve else 0.0
        best_acc = max(acc_curve) if acc_curve else 0.0
        print(f"  {label:<25}  {final_acc:>9.2f}%  {best_acc:>9.2f}%")
    print("=" * 65)
    print("\n[OK] Phase 2 experiment complete!")
    print(f"     Results saved to: {OUTPUT_DIR}/")
