"""
visualize_distribution.py
==========================
Phase 2 -- Label Distribution Visualization

Generates a grid of bar charts showing the label distribution per client
for each partitioning strategy. This is the "hero image" for the README
that immediately shows the reviewer you understand what non-IID means.

Strategies visualized:
  - IID
  - Dirichlet alpha=0.1 (severe non-IID)
  - Dirichlet alpha=0.5 (moderate non-IID)
  - Dirichlet alpha=1.0 (mild non-IID)
  - Pathological K=2 (each client sees exactly 2 classes)
  - Quantity skew sigma=1.0

Output
------
    results/phase2/label_dist_iid.png
    results/phase2/label_dist_dirichlet_0.1.png
    results/phase2/label_dist_dirichlet_0.5.png
    results/phase2/label_dist_dirichlet_1.0.png
    results/phase2/label_dist_pathological_k2.png
    results/phase2/label_dist_quantity_skew.png
    results/phase2/label_dist_comparison.png  <- 4-panel summary figure

Usage
-----
    uv run python visualize_distribution.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torchvision
import torchvision.transforms as transforms

from data_partitioner import DataPartitioner

# CIFAR-10 class names for axis labels
CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Color palette: one color per class, consistent across all charts
CLASS_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

OUTPUT_DIR = os.path.join("results", "phase2")


def plot_distribution_grid(
    indices: Dict[int, List[int]],
    dp: DataPartitioner,
    title: str,
    filename: str,
    n_clients_shown: int = 100,
    grid_cols: int = 10,
) -> str:
    """Generate a grid of bar charts showing per-client label distribution.

    Parameters
    ----------
    indices : dict
        Partition mapping (client_id -> list of indices).
    dp : DataPartitioner
        The partitioner used to generate indices.
    title : str
        Super-title for the entire figure.
    filename : str
        Output filename (saved inside OUTPUT_DIR).
    n_clients_shown : int
        How many clients to show (clipped to available clients).
    grid_cols : int
        Number of columns in the grid layout.

    Returns
    -------
    str : full path to the saved PNG file.
    """
    n_show = min(n_clients_shown, dp.n_clients)
    grid_rows = (n_show + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(grid_cols * 1.4, grid_rows * 1.3),
        dpi=120,
    )
    fig.patch.set_facecolor("#0F1117")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for client_id in range(n_show):
        ax = axes_flat[client_id]
        stats = dp.client_stats(client_id, indices)
        fracs = [stats["label_fractions"].get(c, 0.0) for c in range(dp.n_classes)]

        bars = ax.bar(
            range(dp.n_classes), fracs,
            color=CLASS_COLORS,
            width=0.85,
            edgecolor="none",
        )
        ax.set_xlim(-0.6, dp.n_classes - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#1A1D2E")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5, 0.97, f"C{client_id}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=4.5, color="#AAAAAA",
        )

    # Hide unused axes
    for idx in range(n_show, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Super-title
    fig.suptitle(title, fontsize=13, color="white", fontweight="bold", y=1.01)

    # Class legend at the bottom
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CIFAR10_CLASSES[i])
        for i in range(10)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=10,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7,
        framealpha=0.0,
        labelcolor="white",
    )

    plt.tight_layout(pad=0.4)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def plot_comparison_panel(
    partitions: Dict[str, Dict[int, List[int]]],
    dp: DataPartitioner,
    filename: str = "label_dist_comparison.png",
    clients_per_panel: int = 20,
    grid_cols: int = 10,
) -> str:
    """Generate a multi-row comparison panel showing all strategies side by side.

    Shows the first `clients_per_panel` clients for each strategy
    stacked vertically, so it is easy to visually compare how heterogeneous
    each strategy is.

    Parameters
    ----------
    partitions : dict
        {strategy_name -> indices_dict}
    dp : DataPartitioner
    filename : str
    clients_per_panel : int
        Number of clients to show per strategy (default 20 = 2 rows of 10).
    grid_cols : int
        Columns per strategy panel.

    Returns
    -------
    str : path to saved PNG.
    """
    n_strategies = len(partitions)
    rows_per_strategy = (clients_per_panel + grid_cols - 1) // grid_cols
    total_rows = n_strategies * rows_per_strategy

    fig, axes = plt.subplots(
        total_rows, grid_cols,
        figsize=(grid_cols * 1.4, total_rows * 1.3),
        dpi=120,
    )
    fig.patch.set_facecolor("#0F1117")

    if total_rows == 1:
        axes = axes[np.newaxis, :]

    strategy_names = list(partitions.keys())
    row_offset = 0

    for s_idx, strategy_name in enumerate(strategy_names):
        indices = partitions[strategy_name]
        for local_row in range(rows_per_strategy):
            for col in range(grid_cols):
                client_id = local_row * grid_cols + col
                ax = axes[row_offset + local_row, col]
                ax.set_facecolor("#1A1D2E")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

                if client_id >= min(clients_per_panel, dp.n_clients):
                    ax.set_visible(False)
                    continue

                stats = dp.client_stats(client_id, indices)
                fracs = [stats["label_fractions"].get(c, 0.0) for c in range(dp.n_classes)]
                ax.bar(
                    range(dp.n_classes), fracs,
                    color=CLASS_COLORS, width=0.85, edgecolor="none",
                )
                ax.set_xlim(-0.6, dp.n_classes - 0.4)
                ax.set_ylim(0, 1.0)
                ax.text(
                    0.5, 0.97, f"C{client_id}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=4.5, color="#AAAAAA",
                )

            # Add strategy label on the left of the first column
            if local_row == 0:
                axes[row_offset, 0].set_ylabel(
                    strategy_name,
                    fontsize=8, color="white", labelpad=4,
                    rotation=90, va="center",
                )

        row_offset += rows_per_strategy

    # Super-title
    fig.suptitle(
        "CIFAR-10 Label Distribution per Client (first 20 shown)\n"
        "Each bar = one class (0-9). Height = fraction of client data.",
        fontsize=11, color="white", fontweight="bold", y=1.01,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CIFAR10_CLASSES[i])
        for i in range(10)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=10,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7, framealpha=0.0, labelcolor="white",
    )

    plt.tight_layout(pad=0.4)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Label Distribution Visualizer -- CIFAR-10, 100 clients")
    print("=" * 60)

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    dp = DataPartitioner(trainset, n_clients=100, seed=42)

    print("\nGenerating individual strategy charts...")

    partitions = {
        "IID": dp.iid_split(),
        "Dirichlet alpha=0.1": dp.dirichlet_split(alpha=0.1),
        "Dirichlet alpha=0.5": dp.dirichlet_split(alpha=0.5),
        "Dirichlet alpha=1.0": dp.dirichlet_split(alpha=1.0),
        "Pathological K=2": dp.pathological_split(n_classes_per_client=2),
        "Quantity Skew": dp.quantity_skew_split(sigma=1.0),
    }

    file_map = {
        "IID": "label_dist_iid.png",
        "Dirichlet alpha=0.1": "label_dist_dirichlet_0.1.png",
        "Dirichlet alpha=0.5": "label_dist_dirichlet_0.5.png",
        "Dirichlet alpha=1.0": "label_dist_dirichlet_1.0.png",
        "Pathological K=2": "label_dist_pathological_k2.png",
        "Quantity Skew": "label_dist_quantity_skew.png",
    }

    for name, idxs in partitions.items():
        plot_distribution_grid(
            idxs, dp,
            title=f"CIFAR-10 Label Distribution -- {name} (100 clients)",
            filename=file_map[name],
        )

    print("\nGenerating comparison panel (README hero image)...")
    # Use a subset of 4 strategies for the comparison panel
    comparison_partitions = {
        "IID": partitions["IID"],
        "Dirichlet\nalpha=1.0": partitions["Dirichlet alpha=1.0"],
        "Dirichlet\nalpha=0.5": partitions["Dirichlet alpha=0.5"],
        "Dirichlet\nalpha=0.1": partitions["Dirichlet alpha=0.1"],
    }
    plot_comparison_panel(
        comparison_partitions, dp,
        filename="label_dist_comparison.png",
        clients_per_panel=20,
    )

    print("\n[OK] All distribution charts generated in", OUTPUT_DIR)
