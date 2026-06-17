"""
model.py -- CNN architecture for MNIST (McMahan et al. 2017).

Implements the exact "CNN" model from Table 1 of the FedAvg paper:
  Conv(1->32, 5x5) -> ReLU -> MaxPool(2)
  Conv(32->64, 5x5) -> ReLU -> MaxPool(2)
  Flatten(1024) -> FC(512) -> ReLU -> FC(10)

Total parameters: ~1.66M
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    """
    2-Conv + 1-FC CNN for MNIST, faithful to McMahan et al. 2017.

    Architecture (no padding -- valid convolution):
        28x28x1 -> Conv(32, 5x5) -> 24x24x32 -> MaxPool(2) -> 12x12x32
        -> Conv(64, 5x5) -> 8x8x64 -> MaxPool(2) -> 4x4x64
        -> Flatten(1024) -> FC(512) -> ReLU -> FC(10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)   # no padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)   # no padding
        self.fc1 = nn.Linear(64 * 4 * 4, 512)           # 1024 -> 512
        self.fc2 = nn.Linear(512, 10)                    # 512 -> 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.conv1(x))       # (B, 32, 24, 24)
        x = F.max_pool2d(x, 2)          # (B, 32, 12, 12)
        # Conv block 2
        x = F.relu(self.conv2(x))       # (B, 64, 8, 8)
        x = F.max_pool2d(x, 2)          # (B, 64, 4, 4)
        # Classifier
        x = x.view(x.size(0), -1)       # (B, 1024)
        x = F.relu(self.fc1(x))         # (B, 512)
        x = self.fc2(x)                 # (B, 10)
        return x


def get_model_params(model: nn.Module) -> OrderedDict:
    """Return a deep copy of the model's state_dict (safe for distribution)."""
    return copy.deepcopy(model.state_dict())


def set_model_params(model: nn.Module, params: OrderedDict) -> None:
    """Load parameters into a model."""
    model.load_state_dict(params)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Model -- Smoke Test")
    print("=" * 60)

    model = CNNMnist()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Forward pass test
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"

    # Weight get/set roundtrip
    params = get_model_params(model)
    model2 = CNNMnist()
    set_model_params(model2, params)

    # Verify weights match after roundtrip
    for key in params:
        assert torch.equal(model.state_dict()[key], model2.state_dict()[key]), \
            f"Mismatch on {key}"

    print("[OK] Forward pass shape correct")
    print("[OK] Weight get/set roundtrip correct")
    print("[OK] model smoke test passed")
