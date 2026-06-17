"""
fedavg_baseline.py -- Single-file entry point for FedAvg on MNIST.

Runs the complete Federated Averaging experiment:
  - 10 clients, IID data split
  - 20 communication rounds
  - Produces accuracy/loss curves saved to results/

Reference: McMahan et al. "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (AISTATS 2017). arXiv:1602.05629
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from client import Client
from data_utils import get_client_loader, get_mnist, get_test_loader, iid_partition
from model import CNNMnist, get_model_params
from server import FedAvgServer

# --- Hyperparameters ----------------------------------------------------------------
NUM_CLIENTS = 10          # K: total number of clients
FRACTION_SELECTED = 1.0   # C: fraction selected per round (all for IID)
LOCAL_EPOCHS = 5           # E: local training epochs
LEARNING_RATE = 0.01       # eta: SGD learning rate
BATCH_SIZE = 64            # B: local minibatch size
NUM_ROUNDS = 20            # T: communication rounds
SEED = 42                  # reproducibility
RESULTS_DIR = "./results"
DATA_DIR = "./data"
# --------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_results(
    rounds: list[int],
    accuracies: list[float],
    losses: list[float],
    save_dir: str,
) -> None:
    """Generate and save accuracy + loss curves."""
    os.makedirs(save_dir, exist_ok=True)

    # --- Accuracy curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, accuracies, "o-", color="#2196F3", linewidth=2, markersize=5)
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    ax.set_title("FedAvg -- MNIST (10 Clients, IID)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.axhline(y=95, color="red", linestyle="--", alpha=0.5, label="95% target")
    ax.legend(fontsize=10)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {acc_path}")

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, losses, "o-", color="#FF5722", linewidth=2, markersize=5)
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Test Loss", fontsize=12)
    ax.set_title("FedAvg -- MNIST Loss Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {loss_path}")


def main() -> None:
    """Run the complete FedAvg experiment."""
    set_seed(SEED)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("FedAvg Baseline -- MNIST")
    print("=" * 60)
    print(f"Device:           {device}")
    print(f"Clients:          {NUM_CLIENTS}")
    print(f"Selected/round:   {int(FRACTION_SELECTED * NUM_CLIENTS)}")
    print(f"Local epochs (E): {LOCAL_EPOCHS}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Rounds:           {NUM_ROUNDS}")
    print("=" * 60)

    # -- Step 1: Load data + partition -------------------------------------------
    print("\n[1/3] Loading MNIST and partitioning data...")
    train_ds, test_ds = get_mnist(DATA_DIR)
    partitions = iid_partition(train_ds, NUM_CLIENTS)
    test_loader = get_test_loader(test_ds)

    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid}: {len(partitions[cid])} samples")

    # -- Step 2: Create clients --------------------------------------------------
    print("\n[2/3] Creating clients...")
    clients = []
    for cid in range(NUM_CLIENTS):
        loader = get_client_loader(partitions[cid], train_ds, batch_size=BATCH_SIZE)
        clients.append(Client(client_id=cid, dataloader=loader, device=device))

    # -- Step 3: Initialize server + train ---------------------------------------
    print("\n[3/3] Starting FedAvg training...\n")
    global_model = CNNMnist().to(device)
    server = FedAvgServer(
        global_model=global_model,
        clients=clients,
        fraction_selected=FRACTION_SELECTED,
        test_loader=test_loader,
        device=device,
    )

    # Initial evaluation
    init_loss, init_acc = server.evaluate_global()
    print(f"  Round  0 | Acc: {init_acc:6.2f}% | Loss: {init_loss:.4f} (random init)")

    # Training history
    round_nums = [0]
    accuracies = [init_acc]
    losses = [init_loss]

    total_start = time.time()

    for round_num in tqdm(range(1, NUM_ROUNDS + 1), desc="FedAvg Rounds"):
        round_start = time.time()

        # Run one federated round
        selected = server.run_round(lr=LEARNING_RATE, local_epochs=LOCAL_EPOCHS)

        # Evaluate global model
        test_loss, test_acc = server.evaluate_global()
        round_time = time.time() - round_start

        # Log
        client_ids = sorted([c.client_id for c in selected])
        tqdm.write(
            f"  Round {round_num:2d} | Acc: {test_acc:6.2f}% | Loss: {test_loss:.4f} | "
            f"Time: {round_time:.1f}s | Clients: {client_ids}"
        )

        round_nums.append(round_num)
        accuracies.append(test_acc)
        losses.append(test_loss)

    total_time = time.time() - total_start

    # -- Results -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final accuracy:    {accuracies[-1]:.2f}%")
    print(f"Best accuracy:     {max(accuracies):.2f}% (round {accuracies.index(max(accuracies))})")
    print(f"Total time:        {total_time:.1f}s")
    print(f"Converged (>95%):  {'YES [OK]' if max(accuracies) > 95 else 'NO [FAIL]'}")

    # Plot and save
    print("\nGenerating plots...")
    plot_results(round_nums, accuracies, losses, RESULTS_DIR)

    print("\n[OK] FedAvg baseline experiment complete!")


if __name__ == "__main__":
    main()
