"""
client.py -- Local client trainer for Federated Learning.

Implements the ClientUpdate procedure from McMahan et al. 2017:
  - Receives global model weights
  - Trains locally for E epochs using vanilla SGD
  - Returns updated weights and local sample count
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CNNMnist, get_model_params, set_model_params


class Client:
    """
    Simulates a single FL client.

    Each round:
    1. Receives global weights via set_weights()
    2. Trains locally for `local_epochs` using SGD (no momentum)
    3. Returns updated weights and dataset size via local_train()
    """

    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.num_samples = len(dataloader.dataset)

    def local_train(
        self,
        global_params: dict,
        lr: float = 0.01,
        local_epochs: int = 5,
    ) -> tuple[dict, int]:
        """
        Run E local epochs of SGD starting from global_params.

        IMPORTANT implementation details (from paper research):
        - Creates a FRESH model + optimizer each round (no stale momentum)
        - Uses vanilla SGD without momentum or weight decay
        - Returns deep-copied weights to prevent reference sharing

        Args:
            global_params: Global model state_dict to start from
            lr: Learning rate for local SGD
            local_epochs: Number of local training epochs (E)

        Returns:
            (updated_state_dict, num_local_samples)
        """
        # Fresh model each round -- avoids any stale state
        model = CNNMnist().to(self.device)
        set_model_params(model, copy.deepcopy(global_params))

        # Fresh optimizer each round -- no stale momentum buffers
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(local_epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Return deep copy to prevent reference sharing across clients
        return get_model_params(model), self.num_samples

    @torch.no_grad()
    def evaluate(
        self,
        model_params: dict,
        test_loader: DataLoader,
    ) -> tuple[float, float]:
        """
        Evaluate model on a test set.

        Args:
            model_params: Model state_dict to evaluate
            test_loader: DataLoader for test data

        Returns:
            (average_loss, accuracy_percentage)
        """
        model = CNNMnist().to(self.device)
        set_model_params(model, model_params)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            output = model(batch_x)
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
    print("Client -- Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare data
    train_ds, test_ds = get_mnist()
    partitions = iid_partition(train_ds, num_clients=10)
    test_loader = get_test_loader(test_ds)

    # Create one client with partition 0
    client_loader = get_client_loader(partitions[0], train_ds, batch_size=64)
    client = Client(client_id=0, dataloader=client_loader, device=device)
    print(f"Client 0: {client.num_samples} samples")

    # Initialize model and get baseline accuracy
    model = CNNMnist().to(device)
    init_params = get_model_params(model)

    loss_before, acc_before = client.evaluate(init_params, test_loader)
    print(f"Before training -- Loss: {loss_before:.4f}, Acc: {acc_before:.2f}%")

    # Train locally for 5 epochs
    updated_params, n_samples = client.local_train(init_params, lr=0.01, local_epochs=5)
    print(f"Trained on {n_samples} samples for 5 epochs")

    loss_after, acc_after = client.evaluate(updated_params, test_loader)
    print(f"After training  -- Loss: {loss_after:.4f}, Acc: {acc_after:.2f}%")

    assert acc_after > acc_before, "Accuracy should improve after training!"
    print("[OK] Accuracy improved after local training")
    print("[OK] client smoke test passed")
