from __future__ import annotations
from pathlib import Path
import pytorch_lightning as pl
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from iac_core.utils.seed import set_seed
from iac_core.data.loader import SyntheticConfig, SyntheticClassifDM
from iac_core.models.mlp import MLPClassifier

@hydra_main(version_base=None, config_path="../../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print("== CONFIG ==")
    print(OmegaConf.to_yaml(cfg, resolve=False))

    set_seed(int(cfg.seed))

    # Filtra chaves extras (ex.: "name") e resolve interpolações
    ds_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(ds_dict, dict):
        raise TypeError("dataset config must be a mapping")
    ds_dict.pop("name", None)
    dcfg = SyntheticConfig(**ds_dict)
    dm = SyntheticClassifDM(dcfg)

    mcfg = cfg.model
    model = MLPClassifier(
        input_dim=int(mcfg.input_dim),
        hidden_dims=[int(h) for h in list(mcfg.hidden_dims)],
        num_classes=int(mcfg.num_classes),
        lr=float(mcfg.lr),
        weight_decay=float(mcfg.weight_decay),
        dropout=float(getattr(mcfg, "dropout", 0.0)),
    )

    tcfg = cfg.train
    trainer = pl.Trainer(
        max_epochs=int(tcfg.max_epochs),
        accelerator=str(tcfg.accelerator),
        devices=int(tcfg.devices),
        precision=str(tcfg.precision),
        log_every_n_steps=int(tcfg.log_every_n_steps),
        enable_progress_bar=bool(tcfg.enable_progress_bar),
        default_root_dir=str(Path.cwd()),
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    run()