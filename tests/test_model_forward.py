import torch
from iac_core.models.mlp import MLPClassifier

def test_forward():
    model = MLPClassifier(input_dim=16, hidden_dims=[8], num_classes=3, lr=1e-3, weight_decay=0.0)
    x = torch.randn(4, 16)
    y = model(x)
    assert y.shape == (4, 3)