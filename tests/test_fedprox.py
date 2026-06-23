import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fedprox import FedProxClient


def test_fedprox_returns_state_dict_and_proximal_term():
    # Simple linear model
    model = nn.Linear(10, 2)
    # Dummy data: 20 samples
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5)
    device = torch.device('cpu')

    # Global weights (initial model)
    global_weights = model.state_dict()

    # FedProx client with a non‑zero mu
    client = FedProxClient(model, loader, None, device, mu=0.01, lr=0.1, epochs=1)
    updated_weights = client.local_train(global_weights)

    # Verify that the returned dict has the same keys as the model state dict
    assert set(updated_weights.keys()) == set(model.state_dict().keys())

    # Verify that the proximal term influences training: loss with mu>0 should be higher than without mu
    # Run a second client with mu=0 (standard FedAvg) on the same data
    client_no_prox = FedProxClient(model, loader, None, device, mu=0.0, lr=0.1, epochs=1)
    updated_weights_no_prox = client_no_prox.local_train(global_weights)

    # Compute loss values for a single batch to compare
    criterion = nn.CrossEntropyLoss()
    batch_x, batch_y = next(iter(loader))
    model.load_state_dict(updated_weights)
    out = model(batch_x)
    loss_prox = criterion(out, batch_y)
    model.load_state_dict(updated_weights_no_prox)
    out2 = model(batch_x)
    loss_no_prox = criterion(out2, batch_y)

    assert loss_prox.item() >= loss_no_prox.item()
