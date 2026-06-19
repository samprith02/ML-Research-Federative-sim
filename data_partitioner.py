"""
data_partitioner.py
====================
Phase 2 -- Data Partitioning Engine

Implements four partitioning strategies for federated learning experiments:
  1. IID          -- Uniform random shuffle (baseline)
  2. Dirichlet    -- Label distribution sampled from Dirichlet(alpha)
                     alpha->0: extreme non-IID, alpha->inf: IID
  3. Pathological -- Each client receives exactly K classes (McMahan 2017 style)
  4. Quantity skew-- Unequal dataset sizes drawn from a log-normal distribution

Usage
-----
    from data_partitioner import DataPartitioner
    import torchvision

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    dp = DataPartitioner(trainset, n_clients=100, seed=42)

    # IID
    iid_indices = dp.iid_split()

    # Dirichlet non-IID
    dir_indices = dp.dirichlet_split(alpha=0.1)

    # Pathological (2 classes per client)
    path_indices = dp.pathological_split(n_classes_per_client=2)

    # Quantity skew
    qty_indices = dp.quantity_skew_split(sigma=1.0)

    # Stats for a specific client
    stats = dp.client_stats(client_id=0, indices=dir_indices)
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class DataPartitioner:
    """Partition a dataset among N federated clients using various strategies.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch Dataset whose targets are integer class labels.
    n_clients : int
        Number of clients to partition data among.
    seed : int
        Random seed for reproducibility across all strategies.
    """

    def __init__(self, dataset: Dataset, n_clients: int, seed: int = 42) -> None:
        self.dataset = dataset
        self.n_clients = n_clients
        self.seed = seed

        # Extract labels once -- works for torchvision datasets and custom ones
        if hasattr(dataset, "targets"):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            self.labels = np.array(dataset.labels)
        else:
            # Fallback: iterate (slow for large datasets)
            self.labels = np.array([dataset[i][1] for i in range(len(dataset))])

        self.n_samples = len(self.labels)
        self.classes = sorted(set(self.labels.tolist()))
        self.n_classes = len(self.classes)

    # ------------------------------------------------------------------
    # Strategy 1: IID
    # ------------------------------------------------------------------

    def iid_split(self) -> Dict[int, List[int]]:
        """Uniformly random shuffle -- each client gets n_samples/n_clients samples
        with an approximately balanced class distribution.

        Returns
        -------
        dict mapping client_id -> list of dataset indices
        """
        rng = np.random.default_rng(self.seed)
        all_indices = np.arange(self.n_samples)
        rng.shuffle(all_indices)
        shards = np.array_split(all_indices, self.n_clients)
        return {i: shard.tolist() for i, shard in enumerate(shards)}

    # ------------------------------------------------------------------
    # Strategy 2: Dirichlet non-IID
    # ------------------------------------------------------------------

    def dirichlet_split(self, alpha: float) -> Dict[int, List[int]]:
        """Sample label distributions from a Dirichlet(alpha) distribution.

        Low alpha (e.g. 0.1) -> extreme non-IID (each client dominated by 1 class)
        High alpha (e.g. 10.0) -> near-IID (uniform distribution across classes)

        Parameters
        ----------
        alpha : float
            Concentration parameter of the Dirichlet distribution.

        Returns
        -------
        dict mapping client_id -> list of dataset indices
        """
        rng = np.random.default_rng(self.seed)

        # Group indices by class
        class_indices: Dict[int, np.ndarray] = {}
        for cls in self.classes:
            class_indices[cls] = np.where(self.labels == cls)[0]
            rng.shuffle(class_indices[cls])

        # Sample Dirichlet proportions for each class across clients
        client_indices: Dict[int, List[int]] = {i: [] for i in range(self.n_clients)}

        for cls in self.classes:
            # Proportions for this class across all clients
            proportions = rng.dirichlet(alpha=np.repeat(alpha, self.n_clients))
            # Convert proportions to cumulative sample counts
            cls_idx = class_indices[cls]
            n_cls = len(cls_idx)
            splits = (np.cumsum(proportions) * n_cls).astype(int)
            splits = np.minimum(splits, n_cls)  # clamp to available samples

            prev = 0
            for client_id, end in enumerate(splits):
                client_indices[client_id].extend(cls_idx[prev:end].tolist())
                prev = end
            # Give any remainder to the last client
            client_indices[self.n_clients - 1].extend(cls_idx[prev:].tolist())

        # Shuffle each client's indices so class labels are not ordered
        for client_id in range(self.n_clients):
            rng.shuffle(np.array(client_indices[client_id]))

        return client_indices

    # ------------------------------------------------------------------
    # Strategy 3: Pathological (K classes per client)
    # ------------------------------------------------------------------

    def pathological_split(self, n_classes_per_client: int) -> Dict[int, List[int]]:
        """Each client receives data from exactly K classes (McMahan 2017 style).

        The dataset is sorted by label, split into shards, and shards are
        distributed so each client gets n_classes_per_client shards.

        Parameters
        ----------
        n_classes_per_client : int
            Number of distinct classes each client will see. Must satisfy
            n_classes_per_client * n_clients <= n_classes * n_shards_per_class.

        Returns
        -------
        dict mapping client_id -> list of dataset indices
        """
        rng = random.Random(self.seed)

        # Determine number of shards: each client needs n_classes_per_client shards
        n_shards = self.n_clients * n_classes_per_client
        # n_shards must be divisible across n_classes evenly (or we pad)
        shards_per_class = max(1, n_shards // self.n_classes)

        # Build sorted-by-label index list
        sorted_indices = np.argsort(self.labels, kind="stable")

        # Split sorted indices into shards of equal size
        shard_size = math.ceil(self.n_samples / n_shards)
        shards = []
        for start in range(0, self.n_samples, shard_size):
            shards.append(sorted_indices[start : start + shard_size].tolist())

        # Shuffle shards before distribution
        rng.shuffle(shards)

        # Assign n_classes_per_client shards to each client
        client_indices: Dict[int, List[int]] = {}
        for client_id in range(self.n_clients):
            start = client_id * n_classes_per_client
            assigned = shards[start : start + n_classes_per_client]
            client_indices[client_id] = [idx for shard in assigned for idx in shard]

        return client_indices

    # ------------------------------------------------------------------
    # Strategy 4: Quantity skew
    # ------------------------------------------------------------------

    def quantity_skew_split(self, sigma: float = 1.0) -> Dict[int, List[int]]:
        """Unequal dataset sizes drawn from a log-normal distribution.

        Some clients receive many samples, others very few, but all clients
        still see a class-balanced subset (IID label distribution per client).

        Parameters
        ----------
        sigma : float
            Log-normal sigma parameter. Higher = more skewed size distribution.

        Returns
        -------
        dict mapping client_id -> list of dataset indices
        """
        rng = np.random.default_rng(self.seed)

        # Sample relative sizes from log-normal
        raw_sizes = rng.lognormal(mean=0.0, sigma=sigma, size=self.n_clients)
        # Normalize so total equals n_samples
        sizes = (raw_sizes / raw_sizes.sum() * self.n_samples).astype(int)
        # Fix rounding errors: add/subtract from largest client
        diff = self.n_samples - sizes.sum()
        sizes[np.argmax(sizes)] += diff

        # Shuffle all indices and assign by size
        all_indices = np.arange(self.n_samples)
        rng.shuffle(all_indices)

        client_indices: Dict[int, List[int]] = {}
        ptr = 0
        for client_id, size in enumerate(sizes):
            client_indices[client_id] = all_indices[ptr : ptr + size].tolist()
            ptr += size

        return client_indices

    # ------------------------------------------------------------------
    # client_stats()
    # ------------------------------------------------------------------

    def client_stats(
        self, client_id: int, indices: Dict[int, List[int]]
    ) -> Dict:
        """Return label distribution statistics for a given client.

        Parameters
        ----------
        client_id : int
            ID of the client to query.
        indices : dict
            Partition mapping returned by any split method.

        Returns
        -------
        dict with keys:
            n_samples       -- number of samples assigned to this client
            label_counts    -- {class_label: count} mapping
            label_fractions -- {class_label: fraction} mapping
            dominant_class  -- class label with the most samples
            imbalance_ratio -- max_count / min_count (1.0 = perfectly balanced)
            classes_present -- number of distinct classes with >0 samples
        """
        client_idx = indices[client_id]
        client_labels = self.labels[client_idx]
        counter = Counter(client_labels.tolist())

        # Fill in 0 for missing classes so fractions are well-defined
        label_counts = {cls: counter.get(cls, 0) for cls in self.classes}
        n = len(client_labels)
        label_fractions = {
            cls: (count / n if n > 0 else 0.0)
            for cls, count in label_counts.items()
        }

        counts_nonzero = [c for c in label_counts.values() if c > 0]
        imbalance = (
            max(counts_nonzero) / min(counts_nonzero) if len(counts_nonzero) > 1 else 1.0
        )

        return {
            "n_samples": n,
            "label_counts": label_counts,
            "label_fractions": label_fractions,
            "dominant_class": max(label_counts, key=label_counts.get),
            "imbalance_ratio": round(imbalance, 4),
            "classes_present": len(counts_nonzero),
        }

    # ------------------------------------------------------------------
    # Convenience: get Subset objects instead of raw indices
    # ------------------------------------------------------------------

    def get_subsets(
        self, indices: Dict[int, List[int]]
    ) -> Dict[int, Subset]:
        """Wrap raw index lists in torch Subset objects for use with DataLoader.

        Parameters
        ----------
        indices : dict
            Partition mapping returned by any split method.

        Returns
        -------
        dict mapping client_id -> torch.utils.data.Subset
        """
        return {cid: Subset(self.dataset, idx) for cid, idx in indices.items()}

    # ------------------------------------------------------------------
    # Convenience: summarize all clients
    # ------------------------------------------------------------------

    def all_stats(self, indices: Dict[int, List[int]]) -> Dict[int, Dict]:
        """Return client_stats() for every client.

        Parameters
        ----------
        indices : dict
            Partition mapping returned by any split method.

        Returns
        -------
        dict mapping client_id -> stats dict
        """
        return {cid: self.client_stats(cid, indices) for cid in range(self.n_clients)}


# ------------------------------------------------------------------
# Smoke test (run directly to verify)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    print("=" * 60)
    print("DataPartitioner -- Smoke Test (CIFAR-10, 100 clients)")
    print("=" * 60)

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    dp = DataPartitioner(trainset, n_clients=100, seed=42)
    print(f"  Dataset: CIFAR-10, {dp.n_samples} samples, {dp.n_classes} classes")

    # --- IID ---
    iid = dp.iid_split()
    sizes = [len(v) for v in iid.values()]
    print(f"\n[IID]")
    print(f"  Sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.0f}")
    s = dp.client_stats(0, iid)
    print(f"  Client 0: {s['n_samples']} samples, "
          f"{s['classes_present']} classes present, "
          f"imbalance={s['imbalance_ratio']:.2f}x")

    # --- Dirichlet alpha=0.1 ---
    dir01 = dp.dirichlet_split(alpha=0.1)
    avg_classes = sum(
        dp.client_stats(i, dir01)["classes_present"] for i in range(100)
    ) / 100
    print(f"\n[Dirichlet alpha=0.1]")
    print(f"  Avg classes per client: {avg_classes:.2f}  (expect ~1-3)")

    # --- Dirichlet alpha=1.0 ---
    dir10 = dp.dirichlet_split(alpha=1.0)
    avg_classes_10 = sum(
        dp.client_stats(i, dir10)["classes_present"] for i in range(100)
    ) / 100
    print(f"\n[Dirichlet alpha=1.0]")
    print(f"  Avg classes per client: {avg_classes_10:.2f}  (expect ~7-10)")

    # --- Pathological (2 classes per client) ---
    path = dp.pathological_split(n_classes_per_client=2)
    avg_path = sum(
        dp.client_stats(i, path)["classes_present"] for i in range(100)
    ) / 100
    print(f"\n[Pathological K=2]")
    print(f"  Avg classes per client: {avg_path:.2f}  (expect ~2)")

    # --- Quantity skew ---
    qty = dp.quantity_skew_split(sigma=1.0)
    qty_sizes = [len(v) for v in qty.values()]
    print(f"\n[Quantity Skew sigma=1.0]")
    print(f"  Sizes: min={min(qty_sizes)}, max={max(qty_sizes)}, "
          f"mean={sum(qty_sizes)/len(qty_sizes):.0f}")

    print("\n[OK] All strategies verified.")
