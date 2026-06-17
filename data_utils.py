"""
data_utils.py -- MNIST download + IID partitioning for Federated Learning.

Provides utilities to:
- Download MNIST via torchvision
- Partition training data into IID client splits
- Create per-client DataLoaders and a global test DataLoader
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist(data_dir: str = "./data") -> tuple[datasets.MNIST, datasets.MNIST]:
    """Download MNIST and return (train_dataset, test_dataset) with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean/std
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def iid_partition(dataset: datasets.MNIST, num_clients: int) -> dict[int, list[int]]:
    """
    Partition dataset indices into IID splits for `num_clients` clients.

    Each client gets an equal, randomly-shuffled share of the data,
    ensuring a balanced representation of all 10 classes.

    Returns:
        dict mapping client_id -> list of dataset indices
    """
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    # Split into num_clients roughly-equal chunks
    splits = np.array_split(indices, num_clients)
    return {i: split.tolist() for i, split in enumerate(splits)}


def get_client_loader(
    client_indices: list[int],
    dataset: datasets.MNIST,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for a single client's data partition."""
    subset = Subset(dataset, client_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_test_loader(
    test_dataset: datasets.MNIST,
    batch_size: int = 128,
) -> DataLoader:
    """Create a DataLoader for the full MNIST test set."""
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Data Utils -- Smoke Test")
    print("=" * 60)

    train_ds, test_ds = get_mnist()
    print(f"Training samples: {len(train_ds)}")
    print(f"Test samples:     {len(test_ds)}")

    num_clients = 10
    partitions = iid_partition(train_ds, num_clients)

    print(f"\nIID Partition -- {num_clients} clients:")
    for cid, indices in partitions.items():
        # Count class distribution for this client
        labels = [train_ds.targets[i].item() for i in indices]
        class_counts = {d: labels.count(d) for d in range(10)}
        print(f"  Client {cid:2d}: {len(indices):5d} samples | "
              f"classes: {class_counts}")

    # Quick DataLoader check
    loader = get_client_loader(partitions[0], train_ds, batch_size=64)
    batch_x, batch_y = next(iter(loader))
    print(f"\nSample batch -- x: {batch_x.shape}, y: {batch_y.shape}")
    print("[OK] data_utils smoke test passed")
