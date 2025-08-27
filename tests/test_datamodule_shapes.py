from omegaconf import OmegaConf
from iac_core.data.loader import SyntheticConfig, SyntheticClassifDM

def test_dm_shapes():
    cfg = OmegaConf.create({
        "n_samples": 200, "n_features": 16, "n_informative": 8, "n_redundant": 0, "n_repeated": 0,
        "n_classes": 3, "class_sep": 1.0, "random_state": 42, "test_size": 0.2, "val_size": 0.1,
        "batch_size": 32, "num_workers": 0
    })
    dm = SyntheticClassifDM(SyntheticConfig(**cfg))
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[1] == 16
    assert y.ndim == 1