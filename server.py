"""
server.py -- FedAvg server for Federated Learning.

Implements the server-side logic from McMahan et al. 2017 (Algorithm 1):
  - Maintains the global model
  - Selects a fraction C of clients each round
  - Broadcasts global weights to selected clients
  - Collects updated weights and performs weighted averaging
  - Evaluates global model on the test set
"""

import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from client import Client
from model import CNNMnist, get_model_params, set_model_params


class FedAvgServer:
    """
    FedAvg server -- orchestrates federated training rounds.

    Each round:
    1. Select max(C * K, 1) clients at random
    2. Broadcast global model weights (deep copy)
    3. Each selected client runs local_train()
    4. Aggregate: weighted average by client sample count
    5. Update global model
    """

    def __init__(
        self,
        global_model: CNNMnist,
        clients: list[Client],
        fraction_selected: float,
        test_loader: DataLoader,
        device: torch.device,
    ):
        self.global_model = global_model
        self.clients = clients
        self.fraction_selected = fraction_selected
        self.test_loader = test_loader
        self.device = device

    def select_clients(self) -> list[Client]:
        """Randomly select max(C * K, 1) clients for this round."""
        num_selected = max(int(self.fraction_selected * len(self.clients)), 1)
        return random.sample(self.clients, num_selected)

    def broadcast(self) -> dict:
        """Return a deep copy of the global model weights for distribution."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates: list[tuple[dict, int]]) -> dict:
        """
        Weighted average of client model weights by sample count.

        w_global = Sum (n_k / n_total) * w_k

        Args:
            client_updates: List of (state_dict, num_samples) from each client

        Returns:
            Aggregated state_dict
        """
        # Total samples across all participating clients
        total_samples = sum(n for _, n in client_updates)

        # Initialize aggregated weights with zeros
        aggregated = {}
        for key in client_updates[0][0]:
            aggregated[key] = torch.zeros_like(client_updates[0][0][key], dtype=torch.float32)

        # Weighted sum
        for client_params, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in aggregated:
                aggregated[key] += weight * client_params[key].float()

        return aggregated

    def run_round(self, lr: float, local_epochs: int) -> list[Client]:
        """
        Execute one full FedAvg round.

        1. Select clients
        2. Broadcast global weights
        3. Each client trains locally
        4. Aggregate and update global model

        Returns:
            List of clients that participated this round
        """
        # 1. Select clients
        selected = self.select_clients()

        # 2. Broadcast (deep copy for each client)
        global_params = self.broadcast()

        # 3. Local training on each selected client
        client_updates = []
        for client in selected:
            updated_params, n_samples = client.local_train(
                global_params, lr=lr, local_epochs=local_epochs
            )
            client_updates.append((updated_params, n_samples))

        # 4. Aggregate and update global model
        aggregated_params = self.aggregate(client_updates)
        set_model_params(self.global_model, aggregated_params)

        return selected

    @torch.no_grad()
    def evaluate_global(self) -> tuple[float, float]:
        """
        Evaluate the current global model on the test set.

        Returns:
            (average_loss, accuracy_percentage)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            output = self.global_model(batch_x)
            total_loss += criterion(output, batch_y).item() * batch_y.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_utils import get_mnist, iid_partition, get_client_loader, get_test_loader

    print("=" * 60)
    print("Server -- Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare data
    train_ds, test_ds = get_mnist()
    partitions = iid_partition(train_ds, num_clients=10)
    test_loader = get_test_loader(test_ds)

    # Create clients
    clients = []
    for cid in range(10):
        loader = get_client_loader(partitions[cid], train_ds, batch_size=64)
        clients.append(Client(client_id=cid, dataloader=loader, device=device))

    # --- Test 1: Aggregation of identical models should return the same model ---
    print("\nTest 1: Aggregation of identical models...")
    model_a = CNNMnist().to(device)
    params_a = get_model_params(model_a)
    # Simulate 3 clients returning identical weights
    updates = [(copy.deepcopy(params_a), 100), (copy.deepcopy(params_a), 200),
               (copy.deepcopy(params_a), 300)]

    global_model = CNNMnist().to(device)
    server = FedAvgServer(global_model, clients, fraction_selected=1.0,
                          test_loader=test_loader, device=device)
    aggregated = server.aggregate(updates)

    for key in params_a:
        assert torch.allclose(params_a[key].float(), aggregated[key], atol=1e-6), \
            f"Aggregation of identical models failed on {key}"
    print("  [OK] Identical models aggregate to same model")

    # --- Test 2: One round of FedAvg ---
    print("\nTest 2: Single FedAvg round...")
    set_model_params(global_model, params_a)
    _, acc_before = server.evaluate_global()
    print(f"  Before round -- Acc: {acc_before:.2f}%")

    selected = server.run_round(lr=0.01, local_epochs=1)
    _, acc_after = server.evaluate_global()
    print(f"  After round  -- Acc: {acc_after:.2f}% "
          f"({len(selected)} clients participated)")

    print("[OK] server smoke test passed")
