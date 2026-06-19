"""
model_cifar.py
==============
Phase 2 -- ResNet-8 for CIFAR-10

A compact 8-layer residual network suited for CIFAR-10 (32x32 RGB images).
Architecture:
  - Initial conv: 3->16 channels, 3x3, BN, ReLU
  - 3 residual stages (16->32->64 channels), 2 blocks each
  - Global Average Pooling
  - Fully-connected output (10 classes)

Total parameters: ~272K (fast to train, standard FL benchmark size)

The module exposes the same get_weights() / set_weights() interface as
model.py from Phase 1, making it a drop-in replacement for server/client code.

Usage
-----
    from model_cifar import ResNet8, get_weights, set_weights

    model = ResNet8(num_classes=10)
    weights = get_weights(model)        # list of numpy arrays
    set_weights(model, weights)         # restore from numpy arrays
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """A standard residual block with optional projection shortcut.

    Two 3x3 conv layers, each followed by BatchNorm.
    If in_channels != out_channels or stride != 1, a 1x1 conv
    projection is used on the skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# ------------------------------------------------------------------
# ResNet-8
# ------------------------------------------------------------------


class ResNet8(nn.Module):
    """ResNet-8 for CIFAR-10.

    8 weighted layers total:
        1 initial conv
        2 blocks x 3 stages = 6 residual conv layers (+ projection convs)
        1 fully-connected output layer

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 10 for CIFAR-10).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Initial conv: 3 -> 16 channels
        self.conv_init = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(16)

        # Stage 1: 16 -> 16 channels, 2 blocks, no downsampling
        self.stage1 = nn.Sequential(
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 16, stride=1),
        )

        # Stage 2: 16 -> 32 channels, 2 blocks, stride=2 downsampling
        self.stage2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32, stride=1),
        )

        # Stage 3: 32 -> 64 channels, 2 blocks, stride=2 downsampling
        self.stage3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )

        # Global average pooling + linear classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        # Weight initialization (Kaiming for conv, 1/0 for BN)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_init(self.conv_init(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ------------------------------------------------------------------
# Weight helpers (same interface as Phase 1 model.py)
# ------------------------------------------------------------------


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract all trainable parameters as a list of numpy arrays.

    Parameters
    ----------
    model : nn.Module
        The model to extract weights from.

    Returns
    -------
    list of numpy arrays, one per parameter tensor
    """
    return [param.data.cpu().numpy().copy() for param in model.parameters()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Restore model parameters from a list of numpy arrays.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be overwritten.
    weights : list of numpy arrays
        Must have the same length and shapes as model.parameters().
    """
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.data = torch.tensor(w, dtype=param.dtype).to(param.device)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------
# Smoke test (run directly to verify)
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ResNet-8 -- Smoke Test (CIFAR-10)")
    print("=" * 60)

    model = ResNet8(num_classes=10)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Forward pass with a batch of 4 CIFAR-10 images (3x32x32)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10), f"Expected (4,10), got {logits.shape}"
    print(f"  Forward pass: input {tuple(x.shape)} -> output {tuple(logits.shape)}  [OK]")

    # Weight roundtrip
    w_before = get_weights(model)
    set_weights(model, w_before)
    w_after = get_weights(model)
    max_diff = max(np.abs(a - b).max() for a, b in zip(w_before, w_after))
    assert max_diff < 1e-6, f"Weight roundtrip error: {max_diff}"
    print(f"  Weight roundtrip max diff: {max_diff:.2e}  [OK]")

    # GPU check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gpu = ResNet8().to(device)
    x_gpu = torch.randn(4, 3, 32, 32).to(device)
    out_gpu = model_gpu(x_gpu)
    print(f"  GPU forward pass on {device}: {tuple(out_gpu.shape)}  [OK]")

    print(f"\n[OK] ResNet-8 ready. {n_params:,} parameters.")
