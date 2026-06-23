# FedProx implementation for CIFAR-10.

"""FedProx implementation for CIFAR‑10.

Provides a `FedProxClient` class that extends the existing client logic with a proximal
regularization term controlled by `mu`. The class implements `local_train` that adds the term
`mu/2 * ||w - w_global||^2` to the loss.

Usage example::

    from fedprox import FedProxClient
    client = FedProxClient(model, train_loader, test_loader, device, mu=0.01)
    client.local_train(global_weights)

The implementation mirrors the structure of `client.py` used for FedAvg, so integration with the
existing training script is straightforward.
"""

import copy
import torch
from torch import nn

class FedProxClient:
    def __init__(self, model: nn.Module, train_loader, test_loader, device, mu: float = 0.01, lr: float = 0.01, epochs: int = 1):
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.mu = mu
        self.lr = lr
        self.epochs = epochs
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, global_weights):
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        for _ in range(self.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                # Proximal term
                prox_term = 0.0
                for w, w_g in zip(self.model.parameters(), global_weights.values()):
                    prox_term += (w - w_g).norm(2) ** 2
                loss += (self.mu / 2) * prox_term
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()
